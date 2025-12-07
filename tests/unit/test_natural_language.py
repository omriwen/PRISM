"""Tests for natural language configuration parsing."""
import argparse
from unittest.mock import MagicMock, patch

import pytest

from prism.config.natural_language import (
    ParsedConfig,
    apply_parsed_config,
    parse_instruction,
    show_confirmation,
)


class TestApplyParsedConfig:
    """Tests for apply_parsed_config function."""

    def test_applies_simple_params(self):
        """Test applying simple parameters to args."""
        args = argparse.Namespace(lr=0.001, n_samples=100, obj_name="europa")
        parsed = ParsedConfig(
            parameters={"lr": 0.01, "n_samples": 200},
            raw_response="{}",
            model_used="test",
        )

        result = apply_parsed_config(args, parsed)

        assert result.lr == 0.01
        assert result.n_samples == 200
        assert result.obj_name == "europa"  # unchanged

    def test_handles_use_cuda_mapping(self):
        """Test that use_cuda maps to no_cuda correctly."""
        args = argparse.Namespace(no_cuda=False)
        parsed = ParsedConfig(
            parameters={"use_cuda": False},
            raw_response="{}",
            model_used="test",
        )

        result = apply_parsed_config(args, parsed)

        assert result.no_cuda is True

    def test_ignores_unknown_params(self):
        """Test that unknown parameters are ignored."""
        args = argparse.Namespace(lr=0.001)
        parsed = ParsedConfig(
            parameters={"unknown_param": "value"},
            raw_response="{}",
            model_used="test",
        )

        result = apply_parsed_config(args, parsed)

        assert not hasattr(result, "unknown_param")


class TestParseInstruction:
    """Tests for parse_instruction function."""

    @patch("prism.config.natural_language.ollama")
    def test_parses_simple_instruction(self, mock_ollama):
        """Test parsing a simple instruction."""
        mock_ollama.chat.return_value = {
            "message": {"content": '{"lr": 0.01, "obj_name": "europa"}'}
        }

        result = parse_instruction("train europa with lr 0.01")

        assert result.parameters == {"lr": 0.01, "obj_name": "europa"}
        assert result.model_used == "llama3.2:3b"

    @patch("prism.config.natural_language.ollama")
    def test_handles_empty_response(self, mock_ollama):
        """Test handling when no parameters extracted."""
        mock_ollama.chat.return_value = {"message": {"content": "{}"}}

        result = parse_instruction("hello world")

        assert result.parameters == {}


class TestShowConfirmation:
    """Tests for show_confirmation function."""

    @patch("prism.config.natural_language.Confirm")
    @patch("prism.config.natural_language.Console")
    def test_returns_false_on_empty_params(self, mock_console, mock_confirm):
        """Test that empty parameters return False."""
        parsed = ParsedConfig(parameters={}, raw_response="{}", model_used="test")

        result = show_confirmation(parsed, "test instruction")

        assert result is False
        mock_confirm.ask.assert_not_called()
