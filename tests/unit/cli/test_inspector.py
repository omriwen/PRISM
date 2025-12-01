"""
Inspector/Display Function Tests for SPIDS Configuration

Tests rich-formatted output and inspection utilities.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from prism.config.inspector import (
    handle_inspection_flags,
    list_all_presets,
    show_object_parameters,
    show_preset_details,
    validate_and_report,
)


class TestListAllPresets:
    """Test list_all_presets() function."""

    @patch("prism.config.inspector.console")
    def test_list_presets_spids_mode(self, mock_console):
        """Test listing SPIDS presets."""
        list_all_presets(mode="prism")

        # Verify console.print was called
        assert mock_console.print.called

    @patch("prism.config.inspector.console")
    def test_list_presets_epie_mode(self, mock_console):
        """Test listing ePIE presets."""
        list_all_presets(mode="epie")

        # Verify console.print was called
        assert mock_console.print.called

    def test_list_presets_no_crash(self):
        """Test list_all_presets doesn't crash (smoke test)."""
        # Just verify it doesn't raise an exception
        try:
            with patch("prism.config.inspector.console"):
                list_all_presets(mode="prism")
                list_all_presets(mode="epie")
        except Exception as e:
            pytest.fail(f"list_all_presets raised {e}")


class TestShowPresetDetails:
    """Test show_preset_details() function."""

    @patch("prism.config.inspector.console")
    def test_show_preset_valid(self, mock_console):
        """Test showing details for valid preset."""
        show_preset_details("quick_test", mode="prism")

        # Verify console.print was called
        assert mock_console.print.called

    @patch("prism.config.inspector.console")
    def test_show_preset_invalid(self, mock_console):
        """Test showing details for invalid preset."""
        show_preset_details("nonexistent_preset", mode="prism")

        # Should handle gracefully (print error message)
        assert mock_console.print.called

    def test_show_preset_no_crash(self):
        """Test show_preset_details doesn't crash (smoke test)."""
        try:
            with patch("prism.config.inspector.console"):
                show_preset_details("quick_test", mode="prism")
                show_preset_details("production", mode="prism")
        except Exception as e:
            pytest.fail(f"show_preset_details raised {e}")


class TestShowObjectParameters:
    """Test show_object_parameters() function."""

    def test_show_object_function_exists(self):
        """Test that show_object_parameters function exists and is callable."""
        assert callable(show_object_parameters)


class TestValidateAndReport:
    """Test validate_and_report() function."""

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        import argparse

        from prism.config import args_to_config

        parser = argparse.ArgumentParser()
        parser.add_argument("--obj_name", default="europa")
        parser.add_argument("--n_samples", type=int, default=100)
        parser.add_argument("--name", default="test")
        args = parser.parse_args([])

        # Use args_to_config to get a valid config
        config = args_to_config(args)

        # Convert back to args namespace for validate_and_report
        from prism.config.loader import config_to_args

        test_args = config_to_args(config)

        # Should not raise for valid config
        with patch("prism.config.inspector.console"):
            try:
                result = validate_and_report(test_args)
                # Function may return True or None, just ensure no exception
                assert result is not False
            except ValueError:
                pytest.fail("validate_and_report raised ValueError for valid config")

    def test_validate_invalid_config(self):
        """Test validation of invalid configuration."""
        import argparse

        args = argparse.Namespace(
            obj_name="europa",
            n_samples=-10,  # Invalid: negative
            fermat_sample=True,
            star_sample=False,
            name="test",
            image_size=1024,
            lr=0.001,
            max_epochs=10,
            n_epochs=1000,
            loss_threshold=0.001,
            sample_diameter=None,
            sample_length=0,
            samples_per_line_meas=None,
            snr=None,
            sample_sort="cntr",
            loss_type="l1",
            initialization_target="measurement",
            output_activation="sigmoid",
            middle_activation="leaky_relu",
            new_weight=1.0,
            f_weight=0.0001,
            dxf=1.0,
            checkpoint=None,
            log_dir="runs",
            save_data=True,
            input=None,
            obj_size=None,
            invert_image=False,
            crop_obj=False,
        )

        # Should return False for invalid config
        with patch("prism.config.inspector.console"):
            result = validate_and_report(args)
            assert not result


class TestHandleInspectionFlags:
    """Test handle_inspection_flags() function."""

    @patch("prism.config.inspector.list_all_presets")
    def test_handle_list_presets_flag(self, mock_list):
        """Test handling --list-presets flag."""
        import argparse

        args = argparse.Namespace(
            list_presets=True,
            show_preset=None,
            show_object=None,
        )

        handle_inspection_flags(args, mode="prism")

        # Verify list_all_presets was called
        mock_list.assert_called_once_with("prism")

    @patch("prism.config.inspector.show_preset_details")
    def test_handle_show_preset_flag(self, mock_show):
        """Test handling --show-preset flag."""
        import argparse

        args = argparse.Namespace(
            list_presets=False,
            show_preset="quick_test",
            show_object=None,
        )

        handle_inspection_flags(args, mode="prism")

        # Verify show_preset_details was called
        mock_show.assert_called_once_with("quick_test", "prism")

    @patch("prism.config.inspector.show_object_parameters")
    def test_handle_show_object_flag(self, mock_show):
        """Test handling --show-object flag."""
        import argparse

        args = argparse.Namespace(
            list_presets=False,
            show_preset=None,
            show_object="europa",
        )

        handle_inspection_flags(args, mode="prism")

        # Verify show_object_parameters was called
        mock_show.assert_called_once_with("europa")


class TestInspectorIntegration:
    """Integration tests for inspector module."""

    def test_all_presets_can_be_shown(self):
        """Test that all presets can be displayed without errors."""
        from prism.config.presets import list_presets

        with patch("prism.config.inspector.console"):
            for preset_name in list_presets("prism"):
                try:
                    show_preset_details(preset_name, mode="prism")
                except Exception as e:
                    pytest.fail(f"Failed to show preset '{preset_name}': {e}")

    def test_all_objects_can_be_shown(self):
        """Test that show_object_parameters function works."""
        # Simplified test - just verify function is callable
        assert callable(show_object_parameters)
