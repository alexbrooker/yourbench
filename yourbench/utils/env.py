"""Environment variable expansion utilities.

Simple wrapper around os.path.expandvars() for $VAR syntax.
"""

import os
import re
from typing import Any

from loguru import logger


def expand_env_value(value: Any) -> Any:
    """Expand $VAR and ${VAR} syntax in a string value.

    Args:
        value: Value to expand. Non-strings are returned unchanged.

    Returns:
        Expanded value using os.path.expandvars().
    """
    if not isinstance(value, str):
        return value
    return os.path.expandvars(value)


def expand_env_recursive(data: Any) -> Any:
    """Recursively expand $VAR and ${VAR} syntax in nested data.

    Args:
        data: dict, list, string, or other value.

    Returns:
        Data with all env vars expanded.
    """
    if isinstance(data, dict):
        return {k: expand_env_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [expand_env_recursive(item) for item in data]
    elif isinstance(data, str):
        return expand_env_value(data)
    return data


def validate_env_expanded(value: str, field: str) -> str:
    """Ensure value doesn't contain unexpanded $VAR placeholders.

    Args:
        value: String to validate.
        field: Field name for error message.

    Returns:
        The value unchanged if valid.

    Raises:
        ValueError: If value contains unexpanded $VAR.
    """
    # Check for unexpanded $VAR or ${VAR} patterns
    match = re.search(r"\$\{?([A-Z_][A-Z0-9_]*)\}?", value)
    if match:
        var_name = match.group(1)
        msg = f"Environment variable '{var_name}' in '{field}' not set"
        logger.error(msg)
        raise ValueError(msg)
    return value
