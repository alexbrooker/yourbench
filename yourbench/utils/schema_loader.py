"""Dynamic schema loader for custom Pydantic models."""

import sys
import inspect
import importlib.util
from typing import List, Type, Optional
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


class SchemaLoadError(Exception):
    """Raised when schema loading fails."""

    pass


def load_schema_from_file(
    file_path: str, class_name: Optional[str] = None, auto_batch: bool = False
) -> Type[BaseModel]:
    """Load a Pydantic schema from an external Python file.

    Args:
        file_path: Path to the Python file containing the schema
        class_name: Name of the class to load (if None, uses first BaseModel found)
        auto_batch: If True, automatically create a batch wrapper for single items

    Returns:
        The loaded Pydantic model class

    Raises:
        SchemaLoadError: If loading or validation fails
    """

    path = Path(file_path)
    if not path.exists():
        raise SchemaLoadError(f"Schema file not found: {file_path}")

    if not path.suffix == """.py""":
        raise SchemaLoadError(f"Schema file must be a Python file (.py): {file_path}")

    # Load the module dynamically
    try:
        spec = importlib.util.spec_from_file_location("custom_schema", file_path)
        if spec is None or spec.loader is None:
            raise SchemaLoadError(f"Failed to load module spec from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_schema"] = module
        spec.loader.exec_module(module)

    except Exception as e:
        raise SchemaLoadError(f"Failed to import schema from {file_path}: {e}")

    # Find the Pydantic model(s)
    models = []
    for name, obj in inspect.getmembers(module):
        if (
            inspect.isclass(obj)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
            and not name.startswith("""_""")
        ):
            models.append((name, obj))

    if not models:
        raise SchemaLoadError(f"No Pydantic models found in {file_path}")

    # Select the model
    if class_name:
        # Find specific class
        for name, model in models:
            if name == class_name:
                logger.info(f"Loaded schema class '''{class_name}''' from {file_path}")
                return _maybe_create_batch(model, auto_batch)

        available = ", ".join([name for name, _ in models])
        raise SchemaLoadError(f"Class '''{class_name}''' not found in {file_path}. Available: {available}")

    else:
        # Use first model found
        name, model = models[0]
        if len(models) > 1:
            logger.warning(
                f"Multiple models found in {file_path}, using '''{name}'''. Specify class_name to choose explicitly."
            )
        else:
            logger.info(f"Loaded schema class '''{name}''' from {file_path}")

        return _maybe_create_batch(model, auto_batch)


def _maybe_create_batch(model: Type[BaseModel], auto_batch: bool) -> Type[BaseModel]:
    """Optionally wrap a model in a batch container.

    Args:
        model: The base Pydantic model
        auto_batch: If True, create a batch wrapper

    Returns:
        Either the original model or a batch wrapper
    """

    if not auto_batch:
        return model

    # Check if it'''s already a batch-like structure
    fields = model.model_fields
    if len(fields) == 1:
        field_name = list(fields.keys())[0]
        field_info = fields[field_name]
        # Check if it'''s already a list type
        if hasattr(field_info.annotation, """__origin__"""):
            if field_info.annotation.__origin__ in (list, List):
                logger.debug("Model already has a list field, not creating batch wrapper")
                return model

    # Create a dynamic batch model
    from pydantic import Field, create_model

    batch_model = create_model(
        f"{model.__name__}Batch", items=(List[model], Field(default_factory=list)), __base__=BaseModel
    )

    logger.debug(f"Created batch wrapper for {model.__name__}")
    return batch_model


def validate_schema_for_generation(schema: Type[BaseModel]) -> None:
    """Validate that a schema is suitable for question generation.

    Args:
        schema: The Pydantic model to validate

    Raises:
        SchemaLoadError: If schema is not suitable
    """

    # Check that it'''s a proper Pydantic model
    if not issubclass(schema, BaseModel):
        raise SchemaLoadError(f"Schema must be a Pydantic BaseModel, got {type(schema)}")

    # Check fields
    fields = schema.model_fields
    if not fields:
        raise SchemaLoadError("Schema must have at least one field")

    # Warn about complex nested structures
    for field_name, field_info in fields.items():
        annotation = field_info.annotation

        # Check for deeply nested structures that might confuse LLMs
        if hasattr(annotation, """__origin__"""):
            origin = annotation.__origin__
            if origin is dict:
                logger.warning(
                    f"Field '''{field_name}''' uses dict type which may be hard for LLMs "
                    f"to generate consistently. Consider using a structured model instead."
                )


def generate_schema_description(schema: Type[BaseModel]) -> str:
    """Generate a human-readable description of a schema for prompting.

    Args:
        schema: The Pydantic model

    Returns:
        A description string suitable for including in prompts
    """

    lines = [f"Schema: {schema.__name__}"]
    lines.append("Fields:")

    for field_name, field_info in schema.model_fields.items():
        # Get type name
        annotation = field_info.annotation
        if hasattr(annotation, """__name__"""):
            type_name = annotation.__name__
        else:
            type_name = str(annotation).replace("""typing.""", """""")

        # Get description if available
        description = field_info.description or "No description"

        # Check if required
        required = field_info.is_required()
        req_str = "required" if required else "optional"

        lines.append(f"  - {field_name}: {type_name} ({req_str}) - {description}")

    return "\n".join(lines)


def create_example_from_schema(schema: Type[BaseModel]) -> dict:
    """Create an example instance from a schema for prompting.

    Args:
        schema: The Pydantic model

    Returns:
        A dictionary with example values
    """

    example = {}

    for field_name, field_info in schema.model_fields.items():
        annotation = field_info.annotation

        # Generate example based on type
        if annotation is str:
            if """question""" in field_name.lower():
                example[field_name] = "What is the main concept discussed in this text?"
            elif """answer""" in field_name.lower():
                example[field_name] = "The main concept is [detailed explanation based on the document]"
            else:
                example[field_name] = f"Example {field_name} text"

        elif annotation is int:
            example[field_name] = 5

        elif annotation is float:
            example[field_name] = 0.85

        elif annotation is bool:
            example[field_name] = True

        elif hasattr(annotation, """__origin__"""):
            origin = annotation.__origin__
            if origin in (list, List):
                # List type
                if hasattr(annotation, """__args__"""):
                    inner_type = annotation.__args__[0]
                    if inner_type is str:
                        example[field_name] = ["Item 1", "Item 2"]
                    else:
                        example[field_name] = []
                else:
                    example[field_name] = []
            else:
                # Other generic type
                example[field_name] = None

        else:
            # Complex type or unknown
            example[field_name] = None

    return example
