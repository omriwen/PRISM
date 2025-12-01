"""
Interactive Mode Tests for SPIDS Configuration

Tests the interactive wizard with mocked user input.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from prism.config.interactive import (
    configure_telescope_params,
    configure_training_params,
    run_interactive_setup,
    select_object,
    select_preset,
)


class TestSelectObject:
    """Test object selection in interactive mode."""

    @patch("rich.prompt.Prompt.ask")
    @patch("prism.config.interactive.console")
    def test_select_object_europa(self, mock_console, mock_ask):
        """Test selecting Europa."""
        mock_ask.return_value = "europa"

        result = select_object()

        assert result == "europa"
        mock_ask.assert_called_once()

    @patch("rich.prompt.Prompt.ask")
    @patch("prism.config.interactive.console")
    def test_select_object_titan(self, mock_console, mock_ask):
        """Test selecting Titan."""
        mock_ask.return_value = "titan"

        result = select_object()

        assert result == "titan"

    @patch("rich.prompt.Prompt.ask")
    @patch("prism.config.interactive.console")
    def test_select_object_by_number(self, mock_console, mock_ask):
        """Test selecting object by number."""
        mock_ask.return_value = "2"  # Should select europa (2nd in sorted list)

        result = select_object()

        # Result should be a valid object name
        assert result in ["betelgeuse", "europa", "neptune", "titan"]


class TestSelectPreset:
    """Test preset selection in interactive mode."""

    @patch("rich.prompt.Prompt.ask")
    @patch("prism.config.interactive.console")
    def test_select_preset_quick_test(self, mock_console, mock_ask):
        """Test selecting quick_test preset."""
        mock_ask.return_value = "quick_test"

        result = select_preset(mode="prism")

        assert result == "quick_test"
        mock_ask.assert_called_once()

    @patch("rich.prompt.Prompt.ask")
    @patch("prism.config.interactive.console")
    def test_select_preset_production(self, mock_console, mock_ask):
        """Test selecting production preset."""
        mock_ask.return_value = "production"

        result = select_preset(mode="prism")

        assert result == "production"

    @patch("rich.prompt.Prompt.ask")
    @patch("prism.config.interactive.console")
    def test_select_preset_epie_mode(self, mock_console, mock_ask):
        """Test selecting preset in ePIE mode."""
        mock_ask.return_value = "epie_baseline"

        result = select_preset(mode="epie")

        assert result == "epie_baseline"


class TestConfigureTelescopeParams:
    """Test telescope parameter configuration."""

    def test_configure_telescope_function_exists(self):
        """Test that configure_telescope_params function exists."""
        assert callable(configure_telescope_params)


class TestConfigureTrainingParams:
    """Test training parameter configuration."""

    def test_configure_training_function_exists(self):
        """Test that configure_training_params function exists."""
        assert callable(configure_training_params)


class TestRunInteractiveSetup:
    """Test complete interactive setup wizard."""

    @patch("prism.config.interactive.select_object")
    @patch("prism.config.interactive.select_preset")
    @patch("prism.config.interactive.console")
    def test_interactive_setup_keyboard_interrupt(self, mock_console, mock_preset, mock_object):
        """Test Ctrl+C handling in interactive setup."""
        # Mock KeyboardInterrupt during preset selection
        mock_object.return_value = "europa"
        mock_preset.side_effect = KeyboardInterrupt()

        result = run_interactive_setup(mode="prism")

        # Should return None on Ctrl+C
        assert result is None

    def test_run_interactive_setup_function_exists(self):
        """Test that run_interactive_setup function exists."""
        assert callable(run_interactive_setup)


class TestInteractiveEdgeCases:
    """Test edge cases and error handling."""

    def test_functions_are_callable(self):
        """Test that all interactive functions are callable."""
        assert callable(select_object)
        assert callable(select_preset)
        assert callable(configure_telescope_params)
        assert callable(configure_training_params)
        assert callable(run_interactive_setup)


class TestInteractiveSmokeTests:
    """Smoke tests to ensure functions don't crash with mocked input."""

    @patch("rich.prompt.Prompt.ask")
    @patch("prism.config.interactive.console")
    def test_select_object_smoke(self, mock_console, mock_ask):
        """Smoke test for select_object."""
        mock_ask.return_value = "europa"

        try:
            result = select_object()
            assert result == "europa"
        except Exception as e:
            pytest.fail(f"select_object crashed: {e}")

    @patch("rich.prompt.Prompt.ask")
    @patch("prism.config.interactive.console")
    def test_select_preset_smoke(self, mock_console, mock_ask):
        """Smoke test for select_preset."""
        mock_ask.return_value = "quick_test"

        try:
            result = select_preset("prism")
            assert result == "quick_test"
        except Exception as e:
            pytest.fail(f"select_preset crashed: {e}")
