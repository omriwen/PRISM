"""Integration tests for unified entry point.

This module tests the unified CLI entry point that supports both PRISM
and MoPIE algorithms through a common interface.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from prism.cli.entry_points import (
    create_unified_parser,
    handle_config_loading,
    handle_help_topics,
    handle_inspection_flags,
    handle_interactive_mode,
    handle_natural_language_instruction,
    handle_post_config_inspection,
    handle_preset_loading,
    handle_scenario_flags,
    main,
    print_summary,
)
from prism.core.runner.base import ExperimentResult
from prism.core.runner.factory import RunnerFactory


class TestCreateUnifiedParser:
    """Tests for create_unified_parser()."""

    def test_parser_creation(self):
        """Test that unified parser is created successfully."""
        parser = create_unified_parser()
        assert parser is not None
        assert parser.description is not None
        assert "PRISM" in parser.description

    def test_algorithm_argument_present(self):
        """Test that --algorithm argument is present."""
        parser = create_unified_parser()
        args = parser.parse_args(["--algorithm", "prism", "--name", "test"])
        assert args.algorithm == "prism"

    def test_algorithm_default(self):
        """Test that algorithm defaults to 'prism'."""
        parser = create_unified_parser()
        args = parser.parse_args(["--name", "test"])
        assert args.algorithm == "prism"

    def test_algorithm_choices(self):
        """Test that algorithm only accepts valid choices."""
        parser = create_unified_parser()

        # Valid choices
        args_prism = parser.parse_args(["--algorithm", "prism", "--name", "test"])
        assert args_prism.algorithm == "prism"

        args_mopie = parser.parse_args(["--algorithm", "mopie", "--name", "test"])
        assert args_mopie.algorithm == "mopie"

        # Invalid choice should raise
        with pytest.raises(SystemExit):
            parser.parse_args(["--algorithm", "invalid", "--name", "test"])

    def test_mopie_specific_args_available(self):
        """Test that MoPIE-specific arguments are available."""
        parser = create_unified_parser()
        args = parser.parse_args(
            ["--algorithm", "mopie", "--alpha", "0.8", "--beta", "0.2", "--name", "test"]
        )
        assert args.alpha == 0.8
        assert args.beta == 0.2


class TestRunnerFactory:
    """Tests for RunnerFactory."""

    def test_create_prism_runner(self):
        """Test creating PRISM runner."""
        args = Namespace(
            name="test",
            log_dir="runs",
            save_data=False,
            use_cuda=False,
            device_num=0,
        )

        runner = RunnerFactory.create("prism", args)
        from prism.core.runner.prism_runner import PRISMRunner

        assert isinstance(runner, PRISMRunner)

    def test_create_mopie_runner(self):
        """Test creating MoPIE runner."""
        args = Namespace(
            name="test",
            log_dir="runs",
            save_data=False,
            use_cuda=False,
            device_num=0,
        )

        runner = RunnerFactory.create("mopie", args)
        from prism.core.runner.mopie_runner import MoPIERunner

        assert isinstance(runner, MoPIERunner)

    def test_invalid_algorithm_raises(self):
        """Test that invalid algorithm raises ValueError."""
        args = Namespace()

        with pytest.raises(ValueError, match="Unknown algorithm"):
            RunnerFactory.create("invalid", args)

    def test_case_insensitive_algorithm(self):
        """Test that algorithm is case-insensitive."""
        args = Namespace(
            name="test",
            log_dir="runs",
            save_data=False,
            use_cuda=False,
            device_num=0,
        )

        runner_upper = RunnerFactory.create("PRISM", args)
        runner_mixed = RunnerFactory.create("Mopie", args)

        from prism.core.runner.mopie_runner import MoPIERunner
        from prism.core.runner.prism_runner import PRISMRunner

        assert isinstance(runner_upper, PRISMRunner)
        assert isinstance(runner_mixed, MoPIERunner)


class TestHandleHelpTopics:
    """Tests for handle_help_topics()."""

    def test_no_help_flags(self):
        """Test with no help flags set."""
        args = Namespace(
            help_propagator=False,
            help_patterns=False,
            help_loss=False,
            help_model=False,
            help_objects=False,
        )
        result = handle_help_topics(args, "prism")
        assert result is False

    @patch("prism.config.validation.ConfigValidator")
    def test_help_propagator_prism(self, mock_validator):
        """Test help_propagator flag for PRISM."""
        args = Namespace(
            help_propagator=True,
            help_patterns=False,
            help_loss=False,
            help_model=False,
            help_objects=False,
        )
        result = handle_help_topics(args, "prism")

        assert result is True
        mock_validator.print_help_topic.assert_called_once_with("propagator")

    @patch("prism.config.validation.ConfigValidator")
    def test_help_patterns_prism(self, mock_validator):
        """Test help_patterns flag for PRISM."""
        args = Namespace(
            help_propagator=False,
            help_patterns=True,
            help_loss=False,
            help_model=False,
            help_objects=False,
        )
        result = handle_help_topics(args, "prism")

        assert result is True
        mock_validator.print_help_topic.assert_called_once_with("patterns")

    @patch("prism.config.validation.ConfigValidator")
    def test_help_loss_prism(self, mock_validator):
        """Test help_loss flag for PRISM."""
        args = Namespace(
            help_propagator=False,
            help_patterns=False,
            help_loss=True,
            help_model=False,
            help_objects=False,
        )
        result = handle_help_topics(args, "prism")

        assert result is True
        mock_validator.print_help_topic.assert_called_once_with("loss")

    @patch("prism.config.validation.ConfigValidator")
    def test_help_model_prism(self, mock_validator):
        """Test help_model flag for PRISM."""
        args = Namespace(
            help_propagator=False,
            help_patterns=False,
            help_loss=False,
            help_model=True,
            help_objects=False,
        )
        result = handle_help_topics(args, "prism")

        assert result is True
        mock_validator.print_help_topic.assert_called_once_with("model")

    @patch("prism.config.validation.ConfigValidator")
    def test_help_objects_prism(self, mock_validator):
        """Test help_objects flag for PRISM."""
        args = Namespace(
            help_propagator=False,
            help_patterns=False,
            help_loss=False,
            help_model=False,
            help_objects=True,
        )
        result = handle_help_topics(args, "prism")

        assert result is True
        mock_validator.print_help_topic.assert_called_once_with("objects")

    def test_help_loss_mopie_not_available(self):
        """Test that help_loss is not available for MoPIE."""
        args = Namespace(
            help_propagator=False,
            help_patterns=False,
            help_loss=True,  # Should be ignored for MoPIE
            help_model=False,
            help_objects=False,
        )
        result = handle_help_topics(args, "mopie")

        # MoPIE doesn't support loss help, so should return False
        assert result is False


class TestHandleInspectionFlags:
    """Tests for handle_inspection_flags()."""

    def test_no_inspection_flags(self):
        """Test with no inspection flags set."""
        args = Namespace(
            list_presets=False,
            show_preset=None,
            show_object=None,
        )
        result = handle_inspection_flags(args, "prism")
        assert result is False

    @patch("prism.config.inspector.handle_inspection_flags")
    def test_list_presets_flag(self, mock_inspection):
        """Test list_presets flag triggers inspection."""
        args = Namespace(
            list_presets=True,
            show_preset=None,
            show_object=None,
        )
        result = handle_inspection_flags(args, "prism")

        assert result is True
        mock_inspection.assert_called_once_with(args, mode="prism")

    @patch("prism.config.inspector.handle_inspection_flags")
    def test_show_preset_flag(self, mock_inspection):
        """Test show_preset flag triggers inspection."""
        args = Namespace(
            list_presets=False,
            show_preset="production",
            show_object=None,
        )
        result = handle_inspection_flags(args, "prism")

        assert result is True
        mock_inspection.assert_called_once_with(args, mode="prism")

    @patch("prism.config.inspector.handle_inspection_flags")
    def test_show_object_flag(self, mock_inspection):
        """Test show_object flag triggers inspection."""
        args = Namespace(
            list_presets=False,
            show_preset=None,
            show_object="europa",
        )
        result = handle_inspection_flags(args, "prism")

        assert result is True
        mock_inspection.assert_called_once_with(args, mode="prism")


class TestHandleInteractiveMode:
    """Tests for handle_interactive_mode()."""

    def test_non_interactive_mode(self):
        """Test non-interactive mode returns args unchanged."""
        args = Namespace(interactive=False)
        result = handle_interactive_mode(args, "prism")
        assert result == args

    @patch("prism.config.interactive.run_interactive_setup")
    def test_interactive_mode_success(self, mock_interactive):
        """Test interactive mode with successful setup."""
        mock_interactive.return_value = Namespace(name="interactive_test")

        args = Namespace(interactive=True)
        result = handle_interactive_mode(args, "prism")

        assert result.name == "interactive_test"
        mock_interactive.assert_called_once_with(mode="prism")

    @patch("prism.config.interactive.run_interactive_setup")
    def test_interactive_mode_cancelled(self, mock_interactive):
        """Test interactive mode when user cancels."""
        mock_interactive.return_value = None

        args = Namespace(interactive=True)
        result = handle_interactive_mode(args, "prism")

        assert result is None


class TestHandlePresetLoading:
    """Tests for handle_preset_loading()."""

    def test_no_preset(self):
        """Test with no preset specified."""
        args = Namespace(preset=None)
        result = handle_preset_loading(args, "prism")
        assert result == args

    @patch("prism.config.presets.get_preset")
    @patch("prism.config.presets.merge_preset_with_overrides")
    @patch("prism.config.presets.validate_preset_name")
    def test_valid_preset(self, mock_validate, mock_merge, mock_get):
        """Test with valid preset."""
        mock_validate.return_value = True
        mock_get.return_value = {"n_samples": 64}
        mock_merge.return_value = {"preset": "quick_test", "n_samples": 64}

        args = Namespace(preset="quick_test")
        result = handle_preset_loading(args, "prism")

        assert result.n_samples == 64
        mock_validate.assert_called_once()
        mock_get.assert_called_once()

    @patch("prism.config.presets.validate_preset_name")
    def test_invalid_preset(self, mock_validate):
        """Test with invalid preset name."""
        mock_validate.return_value = False

        args = Namespace(preset="nonexistent")

        with pytest.raises(SystemExit):
            handle_preset_loading(args, "prism")


class TestHandleNaturalLanguageInstruction:
    """Tests for handle_natural_language_instruction()."""

    def test_no_instruction(self):
        """Test with no instruction provided."""
        args = Namespace()
        result = handle_natural_language_instruction(args)
        assert result == args

    def test_instruction_none_attribute(self):
        """Test with instruction attribute set to None."""
        args = Namespace(instruction=None)
        result = handle_natural_language_instruction(args)
        assert result == args


class TestHandlePostConfigInspection:
    """Tests for handle_post_config_inspection()."""

    def test_no_inspection_flags(self):
        """Test with no inspection flags."""
        args = Namespace(show_config=False, validate_only=False)
        result = handle_post_config_inspection(args)
        assert result is False

    @patch("prism.config.inspector.show_effective_config")
    def test_show_config_flag(self, mock_show):
        """Test show_config flag."""
        args = Namespace(show_config=True, validate_only=False)
        result = handle_post_config_inspection(args)

        assert result is True
        mock_show.assert_called_once_with(args)


class TestPrintSummary:
    """Tests for print_summary()."""

    def test_summary_with_results(self, capsys):
        """Test summary output with valid results."""
        result = ExperimentResult(
            ssims=[0.8, 0.9],
            psnrs=[25.0, 30.0],
            rmses=[0.2, 0.1],
            log_dir=Path("/tmp/test"),
        )

        print_summary(result)

        captured = capsys.readouterr()
        assert "completed successfully" in captured.out
        assert "SSIM: 0.9" in captured.out
        assert "PSNR: 30.00" in captured.out

    def test_summary_with_failed_samples(self, capsys):
        """Test summary output with failed samples."""
        result = ExperimentResult(
            ssims=[0.8, 0.9],
            psnrs=[25.0, 30.0],
            rmses=[0.2, 0.1],
            failed_samples=[1, 3, 5],
        )

        print_summary(result)

        captured = capsys.readouterr()
        assert "3 samples failed" in captured.out

    def test_summary_with_empty_results(self, capsys):
        """Test summary output with empty results."""
        result = ExperimentResult()

        print_summary(result)

        captured = capsys.readouterr()
        # Should not print anything if no metrics
        assert "SSIM" not in captured.out


class TestMainFunction:
    """Tests for main() function.

    Note: These are simplified tests that validate the main function's
    argument handling rather than full end-to-end execution.
    """

    def test_main_invalid_algorithm_exits(self):
        """Test main() with invalid algorithm raises SystemExit."""
        import sys
        original_argv = sys.argv
        sys.argv = ["main.py", "--algorithm", "invalid", "--name", "test"]
        try:
            with pytest.raises(SystemExit):
                main()
        finally:
            sys.argv = original_argv

    def test_main_help_flag_exits(self):
        """Test that --help flag causes SystemExit."""
        import sys
        original_argv = sys.argv
        sys.argv = ["main.py", "--help"]
        try:
            with pytest.raises(SystemExit):
                main()
        finally:
            sys.argv = original_argv


class TestUnifiedEntryIntegration:
    """Integration tests for the unified entry point."""

    def test_parser_has_all_expected_args(self):
        """Test that unified parser has all expected arguments."""
        parser = create_unified_parser()
        args = parser.parse_args(["--name", "test"])

        # Common args
        assert hasattr(args, "name")
        assert hasattr(args, "algorithm")
        assert hasattr(args, "n_samples")
        assert hasattr(args, "image_size")
        assert hasattr(args, "obj_name")

        # PRISM-specific args
        assert hasattr(args, "lr")
        assert hasattr(args, "loss_type")

        # MoPIE-specific args
        assert hasattr(args, "alpha")
        assert hasattr(args, "beta")

    def test_default_values_from_example_script(self):
        """Test that defaults match example script values."""
        parser = create_unified_parser()
        args = parser.parse_args(["--name", "test"])

        # Values from run_profiled_reconstruction.sh
        assert args.n_samples == 64
        assert args.image_size == 1024
        assert args.sample_diameter == 17
        assert args.sample_sort == "center"
        assert args.pattern_fn == "builtin:fermat"
        assert args.propagator_method == "fraunhofer"
        assert args.n_epochs == 1000
        assert args.max_epochs == 1
        assert args.n_epochs_init == 100
        assert args.max_epochs_init == 100
        assert args.loss_type == "l1"
        assert args.lr == 0.001
        assert args.loss_th == 0.005
        assert args.new_weight == 1
        assert args.f_weight == 0.0001
        assert args.output_activation == "none"
        assert args.middle_activation == "sigmoid"
        assert args.initialization_target == "circle"
        assert args.enable_adaptive_convergence is False

    def test_cli_args_override_defaults(self):
        """Test that CLI args override defaults."""
        parser = create_unified_parser()
        args = parser.parse_args([
            "--name", "test",
            "--n_samples", "128",
            "--loss_th", "0.01",
            "--adaptive-convergence",
        ])

        assert args.n_samples == 128
        assert args.loss_th == 0.01
        assert args.enable_adaptive_convergence is True
