"""
Unified entry point for all PRISM algorithms.

This module provides a unified CLI entry point that handles both PRISM
(deep learning) and MoPIE (iterative phase retrieval) algorithms through
a common interface.

The entry point:
1. Parses arguments and determines the algorithm
2. Handles pre-run commands (inspection, help, interactive mode)
3. Creates the appropriate runner via RunnerFactory
4. Executes the experiment and reports results
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any

from torch.backends import cudnn

from prism.utils.logging_config import setup_logging


if TYPE_CHECKING:
    import argparse

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore")
# Enable cuDNN autotuner to find optimal algorithms for this hardware/config
cudnn.benchmark = True

# Configure logging with centralized configuration
setup_logging(level="INFO", show_time=True, show_level=True)


def create_unified_parser() -> argparse.ArgumentParser:
    """Create unified argument parser that supports all algorithms.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser supporting --algorithm flag
    """
    from prism.cli.parser import create_mopie_parser

    # Start with MoPIE parser (which extends main parser) to get all args
    parser = create_mopie_parser()

    # Update description for unified entry point
    parser.description = (
        "PRISM - Unified entry point for sparse phase imaging algorithms. "
        "Supports PRISM (deep learning) and MoPIE (iterative phase retrieval)."
    )

    # Add algorithm selection argument
    parser.add_argument(
        "--algorithm",
        type=str,
        default="prism",
        choices=["prism", "mopie"],
        help="Algorithm to use: 'prism' (deep learning) or 'mopie' (iterative)",
    )

    return parser


def handle_natural_language_instruction(args: Any) -> Any | None:
    """Handle natural language instruction processing.

    Parameters
    ----------
    args : Any
        Parsed arguments

    Returns
    -------
    Any | None
        Updated args or None if should exit
    """
    if not getattr(args, "instruction", None):
        return args

    from prism.config.natural_language import process_instruction

    args = process_instruction(args, interactive=not args.auto_confirm)
    if args is None:
        return None
    if getattr(args, "show_parse_only", False):
        return None

    return args


def handle_interactive_mode(args: Any, mode: str) -> Any | None:
    """Handle interactive setup mode.

    Parameters
    ----------
    args : Any
        Parsed arguments
    mode : str
        Algorithm mode ("prism" or "mopie")

    Returns
    -------
    Any | None
        Updated args or None if user cancelled
    """
    if not getattr(args, "interactive", False):
        return args

    from prism.config.interactive import run_interactive_setup

    args_interactive = run_interactive_setup(mode=mode)
    if args_interactive is None:
        return None

    return args_interactive


def handle_help_topics(args: Any, mode: str) -> bool:
    """Handle help topic flags.

    Parameters
    ----------
    args : Any
        Parsed arguments
    mode : str
        Algorithm mode ("prism" or "mopie")

    Returns
    -------
    bool
        True if a help topic was shown and should exit
    """
    from prism.config.validation import ConfigValidator

    # Check PRISM-specific help topics
    if mode == "prism":
        if getattr(args, "help_propagator", False):
            ConfigValidator.print_help_topic("propagator")
            return True
        if getattr(args, "help_patterns", False):
            ConfigValidator.print_help_topic("patterns")
            return True
        if getattr(args, "help_loss", False):
            ConfigValidator.print_help_topic("loss")
            return True
        if getattr(args, "help_model", False):
            ConfigValidator.print_help_topic("model")
            return True
        if getattr(args, "help_objects", False):
            ConfigValidator.print_help_topic("objects")
            return True
    else:
        # MoPIE only uses propagator, patterns, and objects (no loss/model)
        if getattr(args, "help_propagator", False):
            ConfigValidator.print_help_topic("propagator")
            return True
        if getattr(args, "help_patterns", False):
            ConfigValidator.print_help_topic("patterns")
            return True
        if getattr(args, "help_objects", False):
            ConfigValidator.print_help_topic("objects")
            return True

    return False


def handle_inspection_flags(args: Any, mode: str) -> bool:
    """Handle inspection flags (list presets, show preset, show object).

    Parameters
    ----------
    args : Any
        Parsed arguments
    mode : str
        Algorithm mode ("prism" or "mopie")

    Returns
    -------
    bool
        True if inspection was shown and should exit
    """
    if (
        getattr(args, "list_presets", False)
        or getattr(args, "show_preset", None)
        or getattr(args, "show_object", None)
    ):
        from prism.config.inspector import handle_inspection_flags as do_inspection

        do_inspection(args, mode=mode)
        return True

    return False


def handle_preset_loading(args: Any, mode: str) -> Any:
    """Handle preset loading.

    Parameters
    ----------
    args : Any
        Parsed arguments
    mode : str
        Algorithm mode ("prism" or "mopie")

    Returns
    -------
    Any
        Updated args with preset values merged
    """
    if not getattr(args, "preset", None):
        return args

    from loguru import logger

    from prism.config.presets import (
        get_preset,
        list_presets,
        merge_preset_with_overrides,
        validate_preset_name,
    )

    if not validate_preset_name(args.preset, mode=mode):
        available = ", ".join(list_presets(mode))
        print(f"Error: Unknown preset '{args.preset}'")
        print(f"Available presets: {available}")
        print("Use --list-presets to see details")
        sys.exit(1)

    preset_dict = get_preset(args.preset, mode=mode)
    if preset_dict is None:
        logger.error(f"Failed to load preset '{args.preset}'")
        sys.exit(1)

    # Merge preset with CLI args (CLI takes precedence)
    merged = merge_preset_with_overrides(preset_dict, vars(args))
    for key, value in merged.items():
        setattr(args, key, value)

    # For MoPIE, flatten nested parameters
    if mode == "mopie":
        if hasattr(args, "mopie") and isinstance(args.mopie, dict):
            for mopie_key, mopie_value in args.mopie.items():
                if not hasattr(args, mopie_key):
                    setattr(args, mopie_key, mopie_value)
        if hasattr(args, "training") and isinstance(args.training, dict):
            for train_key, train_value in args.training.items():
                if not hasattr(args, train_key):
                    setattr(args, train_key, train_value)
        if hasattr(args, "telescope") and isinstance(args.telescope, dict):
            for tele_key, tele_value in args.telescope.items():
                if not hasattr(args, tele_key):
                    setattr(args, tele_key, tele_value)

    logger.info(f"Loaded preset: {args.preset}")
    return args


def handle_scenario_flags(args: Any) -> Any | None:
    """Handle scenario-related flags.

    Parameters
    ----------
    args : Any
        Parsed arguments

    Returns
    -------
    Any | None
        Updated args or None if should exit
    """
    # List scenarios
    if getattr(args, "list_scenarios", False):
        from prism.scenarios import get_preset_description, list_scenario_presets

        print("\nAvailable Scenario Presets:")
        print("=" * 80)

        print("\nMicroscopy:")
        for name in list_scenario_presets("microscope"):
            desc = get_preset_description(name)
            print(f"  {name:30s} - {desc}")

        print("\nDrone Cameras:")
        for name in list_scenario_presets("drone"):
            desc = get_preset_description(name)
            print(f"  {name:30s} - {desc}")

        return None

    # Show specific scenario
    if getattr(args, "show_scenario", None):
        from prism.scenarios import get_scenario_preset

        try:
            scenario = get_scenario_preset(args.show_scenario)
            print(f"\nScenario: {scenario.name}")
            print("=" * 80)
            print(f"Type: {scenario.scenario_type}")
            print(f"Description: {scenario.description}")
            print("\nPhysics Parameters:")
            for key, value in scenario.get_info().items():
                print(f"  {key}: {value}")
            return None
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Load scenario and apply to args
    if getattr(args, "scenario", None):
        from loguru import logger

        from prism.scenarios import get_scenario_preset

        try:
            scenario = get_scenario_preset(args.scenario)
            logger.info(f"Loaded scenario: {scenario.name}")

            # Apply scenario-specific overrides from CLI
            if (
                hasattr(args, "objective")
                and args.objective
                and hasattr(scenario, "objective_spec")
            ):
                setattr(scenario, "objective_spec", args.objective)
            if (
                hasattr(args, "illumination")
                and args.illumination
                and hasattr(scenario, "illumination_mode")
            ):
                setattr(scenario, "illumination_mode", args.illumination)
            if hasattr(args, "lens") and args.lens and hasattr(scenario, "lens_spec"):
                setattr(scenario, "lens_spec", args.lens)
            if hasattr(args, "altitude") and args.altitude and hasattr(scenario, "altitude_m"):
                setattr(scenario, "altitude_m", args.altitude)
            if hasattr(args, "sensor") and args.sensor and hasattr(scenario, "sensor_spec"):
                setattr(scenario, "sensor_spec", args.sensor)

            # Convert to instrument config and merge with args
            instrument_config = scenario.to_instrument_config()

            # Update args with instrument config parameters
            for key, value in vars(instrument_config).items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)

            logger.info(
                f"Scenario applied: Resolution limit = {scenario.resolution_limit * 1e9:.0f} nm"
            )
        except ValueError as e:
            from loguru import logger

            logger.error(f"Failed to load scenario: {e}")
            sys.exit(1)

    return args


def handle_config_loading(args: Any) -> Any:
    """Handle configuration loading with checkpoint fallback.

    Parameters
    ----------
    args : Any
        Parsed arguments

    Returns
    -------
    Any
        Updated args with config loaded
    """
    from prism.config import load_config, merge_config_with_args
    from prism.utils.training_helpers import load_config_with_checkpoint_fallback

    return load_config_with_checkpoint_fallback(
        args,
        load_config,  # type: ignore[arg-type]
        merge_config_with_args,  # type: ignore[arg-type]
    )


def handle_post_config_inspection(args: Any) -> bool:
    """Handle post-config inspection flags.

    Parameters
    ----------
    args : Any
        Parsed arguments

    Returns
    -------
    bool
        True if inspection was shown and should exit
    """
    # Show effective config
    if getattr(args, "show_config", False):
        from prism.config.inspector import show_effective_config

        show_effective_config(args)
        return True

    # Validate only
    if getattr(args, "validate_only", False):
        from prism.config import args_to_config

        config = args_to_config(args)
        try:
            config.validate()
            print("Configuration is valid")
            return True
        except ValueError as e:
            print(f"Configuration error: {e}")
            sys.exit(1)

    return False


def handle_ai_configuration(args: Any, parser: argparse.ArgumentParser) -> Any | None:
    """Handle AI-powered configuration.

    Parameters
    ----------
    args : Any
        Parsed arguments
    parser : argparse.ArgumentParser
        The argument parser

    Returns
    -------
    Any | None
        Updated args or None if should exit
    """
    if not getattr(args, "instruction", None):
        return args

    from prism.config.ai_config import AIConfigurator

    try:
        configurator = AIConfigurator(parser, model=args.ai_model)

        # 1. Load base configuration
        if args.base:
            from loguru import logger

            base_config = configurator.load_base(args.base)
            logger.info(f"Loaded base config from: {args.base}")
        else:
            base_config = args

        # 2. Get delta from LLM
        from loguru import logger

        logger.info(f"Processing instruction: {args.instruction}")
        delta = configurator.get_delta(args.instruction, base_config)

        # 3. Show proposed changes
        configurator.show_delta(delta, base_config)

        # 4. Handle show-parse-only mode
        if args.show_parse_only:
            return None

        # 5. Confirm changes (unless auto-confirm)
        if delta.changes:
            if args.auto_confirm or configurator.confirm_delta(delta):
                args = configurator.apply_delta(base_config, delta.changes)
                logger.info("Applied AI configuration changes")
            else:
                logger.info("Changes rejected by user")
                return None

    except ConnectionError as e:
        from loguru import logger

        logger.error(str(e))
        logger.info("Hint: Start ollama with: ollama serve")
        sys.exit(1)
    except ValueError as e:
        from loguru import logger

        logger.error(f"AI configuration failed: {e}")
        sys.exit(1)

    return args


def print_summary(result: Any) -> None:
    """Print experiment summary.

    Parameters
    ----------
    result : Any
        Experiment result (ExperimentResult)
    """
    if result.ssims and result.psnrs:
        print("\nExperiment completed successfully!")
        print(f"  Final SSIM: {result.ssims[-1]:.4f}")
        print(f"  Final PSNR: {result.psnrs[-1]:.2f} dB")
        if result.log_dir:
            print(f"  Results saved to: {result.log_dir}")
        if result.failed_samples:
            print(f"  Warning: {len(result.failed_samples)} samples failed to converge")


def main(algorithm: str | None = None) -> None:
    """Unified entry point for all PRISM algorithms.

    This function handles the complete workflow:
    1. Parse arguments
    2. Handle pre-run commands
    3. Create and run the appropriate runner
    4. Report results

    Parameters
    ----------
    algorithm : str | None
        Optional algorithm override. If None, uses --algorithm flag.
    """
    # Parse arguments
    parser = create_unified_parser()
    args = parser.parse_args()

    # Override algorithm if specified programmatically
    if algorithm is not None:
        args.algorithm = algorithm

    # Determine mode
    mode = args.algorithm.lower()
    if mode not in ("prism", "mopie"):
        print(f"Error: Unknown algorithm '{mode}'. Use 'prism' or 'mopie'.")
        sys.exit(1)

    # Handle natural language instruction (first pass for initial processing)
    args = handle_natural_language_instruction(args)
    if args is None:
        sys.exit(0)

    # Handle interactive mode
    args = handle_interactive_mode(args, mode)
    if args is None:
        sys.exit(0)

    # Handle AI configuration (after interactive mode)
    args = handle_ai_configuration(args, parser)
    if args is None:
        sys.exit(0)

    # Handle help topics
    if handle_help_topics(args, mode):
        sys.exit(0)

    # Handle inspection flags
    if handle_inspection_flags(args, mode):
        sys.exit(0)

    # Handle preset loading
    args = handle_preset_loading(args, mode)

    # Configure logging based on args
    setup_logging(level=args.log_level, show_time=True, show_level=True)

    # Handle scenario flags (PRISM only)
    if mode == "prism":
        args = handle_scenario_flags(args)
        if args is None:
            sys.exit(0)

    # Load config
    args = handle_config_loading(args)

    # Handle post-config inspection
    if handle_post_config_inspection(args):
        sys.exit(0)

    # Create and run experiment
    from prism.core.runner.factory import RunnerFactory

    runner = RunnerFactory.create(mode, args)
    result = runner.run()

    # Print summary
    print_summary(result)


def main_prism() -> None:
    """Entry point specifically for PRISM algorithm.

    This function is a convenience wrapper that calls main() with
    algorithm="prism". Used for backward compatibility.
    """
    main(algorithm="prism")


def main_mopie() -> None:
    """Entry point specifically for MoPIE algorithm.

    This function is a convenience wrapper that calls main() with
    algorithm="mopie". Used for backward compatibility.
    """
    main(algorithm="mopie")


if __name__ == "__main__":
    main()
