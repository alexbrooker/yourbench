"""Environment variable expansion utilities.

Centralized utilities for expanding $VAR syntax in configuration values.
"""

import os
from typing import Any

from loguru import logger


def expand_env_value(value: Any) -> Any:
    """Expand $VAR syntax in a single value.

    Args:
        value: Value to expand. Non-strings are returned unchanged.

    Returns:
        Expanded value. If env var not set, returns empty string.
        Special case: $HF_ORGANIZATION falls back to HF user lookup.
    """
    if not isinstance(value, str):
        return value

    if not value.startswith("$") or value.startswith("${"):
        return value

    var_name = value[1:]
    env_value = os.getenv(var_name)

    if env_value is not None:
        return env_value

    # Special case: auto-resolve HF_ORGANIZATION from token
    if var_name == "HF_ORGANIZATION":
        return _resolve_hf_organization()

    return ""


def expand_env_recursive(data: Any) -> Any:
    """Recursively expand $VAR syntax in nested data structures.

    Args:
        data: dict, list, string, or other value.

    Returns:
        Data with all $VAR strings expanded.
    """
    if isinstance(data, dict):
        return {k: expand_env_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [expand_env_recursive(item) for item in data]
    elif isinstance(data, str):
        return expand_env_value(data)
    return data


def validate_env_expanded(value: str, field: str) -> str:
    """Ensure value is not an unexpanded $VAR placeholder.

    Args:
        value: String to validate.
        field: Field name for error message.

    Returns:
        The value unchanged if valid.

    Raises:
        ValueError: If value still contains unexpanded $VAR.
    """
    if value.startswith("$"):
        var_name = value[1:].split("/")[0]
        msg = f"Environment variable '{var_name}' in '{field}' not set"
        logger.error(msg)
        raise ValueError(msg)
    return value


def _resolve_hf_organization() -> str:
    """Resolve HF organization from token."""
    token = os.getenv("HF_TOKEN")
    if not token:
        return ""

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        user_info = api.whoami()
        return user_info.get("name", "")
    except Exception as e:
        logger.warning(f"Failed to resolve HF organization: {e}")
        return ""
