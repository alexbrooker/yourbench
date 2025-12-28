"""Tests for environment variable expansion utilities."""

import os
from unittest.mock import patch

import pytest

from yourbench.utils.env import (
    expand_env_value,
    expand_env_recursive,
    validate_env_expanded,
)


class TestExpandEnvValue:
    """Tests for expand_env_value function."""

    def test_non_string_returned_unchanged(self):
        """Non-string values should be returned as-is."""
        assert expand_env_value(123) == 123
        assert expand_env_value(None) is None
        assert expand_env_value([1, 2, 3]) == [1, 2, 3]
        assert expand_env_value({"key": "value"}) == {"key": "value"}

    def test_string_without_dollar_returned_unchanged(self):
        """Regular strings without $ prefix should be unchanged."""
        assert expand_env_value("hello") == "hello"
        assert expand_env_value("") == ""
        assert expand_env_value("some/path/to/file.txt") == "some/path/to/file.txt"

    @patch.dict(os.environ, {"TEST_VAR": "test_value"}, clear=False)
    def test_env_var_expanded(self):
        """$VAR should be replaced with its environment value."""
        assert expand_env_value("$TEST_VAR") == "test_value"

    @patch.dict(os.environ, {"HOME_TEST": "/home/user"}, clear=False)
    def test_brace_syntax_expanded(self):
        """${VAR} syntax should also be expanded."""
        assert expand_env_value("${HOME_TEST}/data") == "/home/user/data"

    @patch.dict(os.environ, {"MULTI_WORD": "hello world"}, clear=False)
    def test_env_var_with_spaces(self):
        """Env vars with spaces in values should work."""
        assert expand_env_value("$MULTI_WORD") == "hello world"

    def test_missing_env_var_not_expanded(self):
        """Missing env vars are left as-is by os.path.expandvars."""
        # os.path.expandvars leaves undefined vars as-is
        result = expand_env_value("$DEFINITELY_NOT_SET_XYZ123")
        assert result == "$DEFINITELY_NOT_SET_XYZ123"


class TestExpandEnvRecursive:
    """Tests for expand_env_recursive function."""

    @patch.dict(os.environ, {"API_KEY": "secret123", "BASE_URL": "http://example.com"}, clear=False)
    def test_dict_expansion(self):
        """Dict values should be expanded recursively."""
        data = {
            "api_key": "$API_KEY",
            "base_url": "$BASE_URL",
            "static": "no_expansion",
        }
        result = expand_env_recursive(data)
        assert result == {
            "api_key": "secret123",
            "base_url": "http://example.com",
            "static": "no_expansion",
        }

    @patch.dict(os.environ, {"MODEL_A": "gpt-4", "MODEL_B": "gpt-3.5"}, clear=False)
    def test_list_expansion(self):
        """List elements should be expanded."""
        data = ["$MODEL_A", "$MODEL_B", "static"]
        result = expand_env_recursive(data)
        assert result == ["gpt-4", "gpt-3.5", "static"]

    @patch.dict(os.environ, {"NESTED_VAL": "deep_value"}, clear=False)
    def test_nested_dict_list_expansion(self):
        """Deeply nested structures should be fully expanded."""
        data = {
            "outer": {
                "inner": ["$NESTED_VAL", "static"],
                "plain": 123,
            }
        }
        result = expand_env_recursive(data)
        assert result == {
            "outer": {
                "inner": ["deep_value", "static"],
                "plain": 123,
            }
        }

    def test_non_container_types_passed_through(self):
        """Non-container types should be passed through."""
        assert expand_env_recursive(42) == 42
        assert expand_env_recursive(3.14) == 3.14
        assert expand_env_recursive(True) is True
        assert expand_env_recursive(None) is None


class TestValidateEnvExpanded:
    """Tests for validate_env_expanded function."""

    def test_regular_string_passes(self):
        """Regular strings should pass validation."""
        assert validate_env_expanded("hello", "field") == "hello"
        assert validate_env_expanded("/path/to/file", "path") == "/path/to/file"
        assert validate_env_expanded("", "empty") == ""

    def test_unexpanded_var_raises_error(self):
        """Unexpanded $VAR should raise ValueError."""
        with pytest.raises(ValueError, match="HF_TOKEN.*not set"):
            validate_env_expanded("$HF_TOKEN", "token_field")

    def test_unexpanded_var_in_middle_raises(self):
        """Unexpanded $VAR in middle of string should also raise."""
        with pytest.raises(ValueError, match="SOME_VAR.*not set"):
            validate_env_expanded("prefix/$SOME_VAR/suffix", "path")
