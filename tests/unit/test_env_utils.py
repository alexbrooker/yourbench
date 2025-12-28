"""Tests for environment variable expansion utilities."""

import os
from unittest.mock import patch

import pytest

from yourbench.utils.env import (
    expand_env_value,
    expand_env_recursive,
    validate_env_expanded,
)


class TestEnvExpansion:
    """Tests for expand_env_value and expand_env_recursive."""

    def test_non_string_unchanged(self):
        """Non-string values pass through unchanged."""
        for val in [123, None, [1, 2], {"k": "v"}, 3.14, True]:
            assert expand_env_value(val) == val
            assert expand_env_recursive(val) == val

    @patch.dict(os.environ, {"TEST_VAR": "value", "OTHER": "other"}, clear=False)
    def test_expansion_syntax(self):
        """Both $VAR and ${VAR} syntax expand correctly."""
        assert expand_env_value("$TEST_VAR") == "value"
        assert expand_env_value("${TEST_VAR}/path") == "value/path"
        assert expand_env_value("plain string") == "plain string"
        # Missing vars stay as-is
        assert expand_env_value("$NOT_SET_XYZ") == "$NOT_SET_XYZ"

    @patch.dict(os.environ, {"A": "1", "B": "2"}, clear=False)
    def test_recursive_expansion(self):
        """Nested dicts and lists are expanded recursively."""
        data = {
            "key": "$A",
            "nested": {"inner": ["$B", "static"]},
            "num": 42,
        }
        result = expand_env_recursive(data)
        assert result == {
            "key": "1",
            "nested": {"inner": ["2", "static"]},
            "num": 42,
        }


class TestValidateEnvExpanded:
    """Tests for validate_env_expanded."""

    def test_valid_strings_pass(self):
        """Normal strings pass validation."""
        assert validate_env_expanded("hello", "f") == "hello"
        assert validate_env_expanded("/path/to/file", "f") == "/path/to/file"
        assert validate_env_expanded("", "f") == ""

    def test_unexpanded_var_raises(self):
        """Unexpanded $VAR patterns raise ValueError."""
        with pytest.raises(ValueError, match="HF_TOKEN.*not set"):
            validate_env_expanded("$HF_TOKEN", "field")
        with pytest.raises(ValueError, match="VAR.*not set"):
            validate_env_expanded("prefix/$VAR/suffix", "field")
