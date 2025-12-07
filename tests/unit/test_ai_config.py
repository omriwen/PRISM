"""Unit tests for AI configuration system."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prism.cli.parser import create_main_parser
from prism.config.ai_config import AIConfigurator, ArgumentSchema, ConfigDelta


class TestSchemaExtraction:
    """Tests for _extract_schema method."""

    def test_extracts_all_arguments(self) -> None:
        """Verify all parser arguments are extracted."""
        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        # Should have 70+ arguments
        assert len(configurator.schema) > 50

    def test_captures_types(self) -> None:
        """Verify argument types are captured correctly."""
        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        n_samples = next(s for s in configurator.schema if s.name == "n_samples")
        assert n_samples.arg_type == "int"
        assert n_samples.default == 200

    def test_captures_choices(self) -> None:
        """Verify choices are captured for enum-like args."""
        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        log_level = next(s for s in configurator.schema if s.name == "log_level")
        assert log_level.choices is not None
        assert "INFO" in log_level.choices

    def test_captures_flags(self) -> None:
        """Verify boolean flags are identified."""
        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        fermat = next(s for s in configurator.schema if s.name == "fermat_sample")
        assert fermat.is_flag is True

    def test_schema_to_prompt_text(self) -> None:
        """Verify prompt text generation."""
        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        prompt_text = configurator._schema_to_prompt_text()
        assert "## Available Parameters" in prompt_text
        assert "n_samples" in prompt_text


class TestLoadBase:
    """Tests for load_base method."""

    def test_load_yaml(self, tmp_path: Path) -> None:
        """Test loading YAML configuration."""
        yaml_content = """
telescope:
  n_samples: 100
  fermat_sample: true
training:
  lr: 0.01
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        config = configurator.load_base(str(yaml_file))

        assert config.n_samples == 100
        assert config.fermat_sample is True
        assert config.lr == 0.01

    def test_load_json(self, tmp_path: Path) -> None:
        """Test loading JSON configuration."""
        json_content = '{"telescope": {"n_samples": 150}}'
        json_file = tmp_path / "config.json"
        json_file.write_text(json_content)

        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        config = configurator.load_base(str(json_file))

        assert config.n_samples == 150

    def test_load_shell(self, tmp_path: Path) -> None:
        """Test loading shell script configuration."""
        sh_content = """#!/bin/bash
# Run PRISM experiment
python main.py --n_samples 200 --fermat --lr 0.001
"""
        sh_file = tmp_path / "run.sh"
        sh_file.write_text(sh_content)

        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        config = configurator.load_base(str(sh_file))

        assert config.n_samples == 200

    def test_unsupported_format(self, tmp_path: Path) -> None:
        """Test that unsupported formats raise ValueError."""
        txt_file = tmp_path / "config.txt"
        txt_file.write_text("n_samples = 100")

        parser = create_main_parser()
        configurator = AIConfigurator(parser)

        with pytest.raises(ValueError, match="Unsupported config format"):
            configurator.load_base(str(txt_file))

    def test_file_not_found(self) -> None:
        """Test that missing files raise FileNotFoundError."""
        parser = create_main_parser()
        configurator = AIConfigurator(parser)

        with pytest.raises(FileNotFoundError):
            configurator.load_base("/nonexistent/path.yaml")


class TestGetDelta:
    """Tests for get_delta method."""

    @patch("prism.config.ai_config.ollama.chat")
    def test_simple_instruction(self, mock_chat: MagicMock) -> None:
        """Test parsing simple instruction."""
        mock_chat.return_value = {
            "message": {
                "content": '{"changes": {"n_samples": 200}, "explanation": "Set samples to 200", "warnings": []}'
            }
        }

        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        current = parser.parse_args([])

        delta = configurator.get_delta("use 200 samples", current)

        assert delta.changes == {"n_samples": 200}
        assert "200" in delta.explanation

    @patch("prism.config.ai_config.ollama.chat")
    def test_empty_instruction(self, mock_chat: MagicMock) -> None:
        """Test that irrelevant instruction returns empty delta."""
        mock_chat.return_value = {
            "message": {
                "content": '{"changes": {}, "explanation": "No changes", "warnings": []}'
            }
        }

        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        current = parser.parse_args([])

        delta = configurator.get_delta("hello", current)

        assert delta.changes == {}

    @patch("prism.config.ai_config.ollama.chat")
    def test_validates_choices(self, mock_chat: MagicMock) -> None:
        """Test that invalid choices are rejected."""
        mock_chat.return_value = {
            "message": {
                "content": '{"changes": {"log_level": "INVALID"}, "explanation": "", "warnings": []}'
            }
        }

        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        current = parser.parse_args([])

        with pytest.raises(ValueError, match="Invalid value"):
            configurator.get_delta("use invalid log level", current)

    @patch("prism.config.ai_config.ollama.chat")
    def test_type_conversion(self, mock_chat: MagicMock) -> None:
        """Test that values are type-converted correctly."""
        mock_chat.return_value = {
            "message": {
                "content": '{"changes": {"lr": "0.01"}, "explanation": "", "warnings": []}'
            }
        }

        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        current = parser.parse_args([])

        delta = configurator.get_delta("set lr to 0.01", current)

        assert delta.changes["lr"] == 0.01
        assert isinstance(delta.changes["lr"], float)

    @patch("prism.config.ai_config.ollama.chat")
    def test_handles_markdown_json(self, mock_chat: MagicMock) -> None:
        """Test extraction from markdown code blocks."""
        mock_chat.return_value = {
            "message": {
                "content": '```json\n{"changes": {"n_samples": 100}, "explanation": "", "warnings": []}\n```'
            }
        }

        parser = create_main_parser()
        configurator = AIConfigurator(parser)
        current = parser.parse_args([])

        delta = configurator.get_delta("100 samples", current)

        assert delta.changes == {"n_samples": 100}


class TestApplyDelta:
    """Tests for apply_delta method."""

    def test_applies_changes(self) -> None:
        """Test that changes are applied to config."""
        parser = create_main_parser()
        configurator = AIConfigurator(parser)

        current = parser.parse_args([])
        delta = {"n_samples": 300, "lr": 0.005}

        result = configurator.apply_delta(current, delta)

        assert result.n_samples == 300
        assert result.lr == 0.005

    def test_preserves_other_values(self) -> None:
        """Test that non-changed values are preserved."""
        parser = create_main_parser()
        configurator = AIConfigurator(parser)

        current = parser.parse_args(["--obj_name", "titan"])
        delta = {"n_samples": 300}

        result = configurator.apply_delta(current, delta)

        assert result.n_samples == 300
        assert result.obj_name == "titan"

    def test_does_not_modify_original(self) -> None:
        """Test that original config is not modified."""
        parser = create_main_parser()
        configurator = AIConfigurator(parser)

        current = parser.parse_args([])
        original_samples = current.n_samples
        delta = {"n_samples": 999}

        configurator.apply_delta(current, delta)

        assert current.n_samples == original_samples


class TestIntegration:
    """Integration tests with mocked ollama."""

    @patch("prism.config.ai_config.ollama.chat")
    def test_full_workflow(self, mock_chat: MagicMock, tmp_path: Path) -> None:
        """Test complete workflow: load -> delta -> apply."""
        mock_chat.return_value = {
            "message": {
                "content": '{"changes": {"max_epochs": 50}, "explanation": "Increased epochs", "warnings": []}'
            }
        }

        yaml_content = """
telescope:
  n_samples: 100
training:
  max_epochs: 25
"""
        yaml_file = tmp_path / "base.yaml"
        yaml_file.write_text(yaml_content)

        parser = create_main_parser()
        configurator = AIConfigurator(parser)

        # Full workflow
        base = configurator.load_base(str(yaml_file))
        delta = configurator.get_delta("double the max epochs", base)
        final = configurator.apply_delta(base, delta.changes)

        assert base.max_epochs == 25
        assert delta.changes == {"max_epochs": 50}
        assert final.max_epochs == 50
