"""Comprehensive tests for custom Pydantic schema support."""

import json
import tempfile
from pathlib import Path
from typing import List

import pytest
from pydantic import BaseModel, Field

from yourbench.schemas.default_schemas import (
    SingleHopQuestion,
    MultiHopQuestion,
    MultiChoiceQuestion,
)
from yourbench.utils.schema_loader import (
    load_schema_from_file,
    validate_schema_for_generation,
    generate_schema_description,
    create_example_from_schema,
)


def test_default_single_hop_schema():
    """Test the default single-hop question schema."""
    question = SingleHopQuestion(
        question="What is YourBench",  # Should add ? automatically
        self_answer="YourBench is an automated framework for generating evaluation benchmarks.",
        estimated_difficulty=5,
        self_assessed_question_type="factual",
        thought_process="Testing basic understanding of the system.",
        citations=["YourBench is an automated framework"],
    )

    # Check question mark was added
    assert question.question.endswith("?")
    assert question.question == "What is YourBench?"

    # Check fields
    assert question.estimated_difficulty == 5
    assert question.self_assessed_question_type == "factual"
    assert len(question.citations) == 1


def test_default_multi_hop_schema():
    """Test the default multi-hop question schema."""
    question = MultiHopQuestion(
        question="How does YourBench compare to static benchmarks in terms of temporal relevance?",
        answer="YourBench generates dynamic benchmarks from recent documents, avoiding the temporal irrelevance of static benchmarks.",
        estimated_difficulty=7,
        question_type="comparison",
        reasoning_steps=[
            "Understand static benchmark limitations",
            "Understand YourBench's dynamic generation",
            "Compare temporal aspects",
        ],
        integration_type="comparison",
    )

    # Check aliases work
    assert (
        question.self_answer
        == "YourBench generates dynamic benchmarks from recent documents, avoiding the temporal irrelevance of static benchmarks."
    )
    assert question.self_assessed_question_type == "comparison"
    assert len(question.reasoning_steps) == 3


def test_multi_choice_schema():
    """Test the multiple-choice question schema."""
    question = MultiChoiceQuestion(
        question="What is the primary goal of YourBench?",
        self_answer="To generate dynamic evaluation benchmarks",
        choices=[
            "Generate static benchmarks",
            "Generate dynamic evaluation benchmarks",
            "Train language models",
            "Annotate documents manually",
        ],
        correct_answer="b",
        estimated_difficulty=3,
    )

    # Check normalization
    assert question.correct_answer == "B"
    assert len(question.choices) == 4

    # Test validation
    with pytest.raises(Exception):  # Pydantic raises ValidationError
        MultiChoiceQuestion(
            question="Test?",
            self_answer="Answer",
            choices=["A", "B"],  # Only 2 choices
            correct_answer="A",
        )


def test_load_custom_schema_from_file():
    """Test loading a custom schema from an external file."""

    # Create a temporary Python file with a custom schema
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write('''
from pydantic import BaseModel, Field
from typing import List, Optional

class CustomQuestion(BaseModel):
    """A custom question format for specialized evaluation."""
    
    query: str = Field(..., description="The main question")
    response: str = Field(..., description="Expected response")
    difficulty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[dict] = None
    
class AnotherModel(BaseModel):
    """Another model in the same file."""
    name: str
''')
        temp_file = f.name

    try:
        # Load the specific class
        schema = load_schema_from_file(temp_file, class_name="CustomQuestion")
        assert schema.__name__ == "CustomQuestion"

        # Validate it's suitable for generation
        validate_schema_for_generation(schema)

        # Generate description
        description = generate_schema_description(schema)
        assert "query" in description
        assert "response" in description
        assert "difficulty_score" in description

        # Generate example
        example = create_example_from_schema(schema)
        assert "query" in example
        assert "response" in example
        assert isinstance(example["difficulty_score"], float)
        assert isinstance(example["tags"], list)

        # Test auto-batch
        batch_schema = load_schema_from_file(temp_file, class_name="CustomQuestion", auto_batch=True)
        assert "Batch" in batch_schema.__name__
        assert "items" in batch_schema.model_fields

    finally:
        Path(temp_file).unlink()


def test_schema_with_nested_structure():
    """Test a schema with nested Pydantic models."""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
from pydantic import BaseModel, Field
from typing import List

class Citation(BaseModel):
    text: str
    page: int
    confidence: float = 0.9

class ComplexQuestion(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    follow_up: List[str] = Field(default_factory=list)
""")
        temp_file = f.name

    try:
        schema = load_schema_from_file(temp_file, class_name="ComplexQuestion")

        # This should work fine
        validate_schema_for_generation(schema)

        # Create example
        example = create_example_from_schema(schema)
        assert "citations" in example
        assert isinstance(example["citations"], list)

    finally:
        Path(temp_file).unlink()


def test_invalid_schema_detection():
    """Test that invalid schemas are properly detected."""

    # Test empty schema
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
from pydantic import BaseModel

class EmptyModel(BaseModel):
    pass
""")
        temp_file = f.name

    try:
        schema = load_schema_from_file(temp_file)

        # Should raise error for empty schema
        with pytest.raises(Exception, match="at least one field"):
            validate_schema_for_generation(schema)

    finally:
        Path(temp_file).unlink()


def test_schema_field_type_handling():
    """Test different field types in schema generation."""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class QuestionType(str, Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARISON = "comparison"

class RichQuestion(BaseModel):
    text: str
    score: int = Field(ge=1, le=10)
    confidence: float = Field(ge=0.0, le=1.0)
    is_valid: bool = True
    question_type: QuestionType = QuestionType.FACTUAL
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, str]] = None
""")
        temp_file = f.name

    try:
        schema = load_schema_from_file(temp_file, class_name="RichQuestion")

        # Generate example
        example = create_example_from_schema(schema)

        # Check types
        assert isinstance(example["text"], str)
        assert isinstance(example["score"], int)
        assert isinstance(example["confidence"], float)
        assert isinstance(example["is_valid"], bool)
        assert isinstance(example["tags"], list)

        # Description should include all fields
        description = generate_schema_description(schema)
        assert "text" in description
        assert "score" in description
        assert "confidence" in description

    finally:
        Path(temp_file).unlink()


if __name__ == "__main__":
    # Run tests
    test_default_single_hop_schema()
    test_default_multi_hop_schema()
    test_multi_choice_schema()
    test_load_custom_schema_from_file()
    test_schema_with_nested_structure()
    test_invalid_schema_detection()
    test_schema_field_type_handling()
    print("✅ All custom schema tests passed!")
