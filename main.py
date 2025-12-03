"""
Module: main.py
Purpose: Main PRISM algorithm - progressive deep learning reconstruction
         from sparse telescope measurements
Dependencies: torch, prism.cli, prism.core, prism.config
Main Components:
    - Argument parsing (extracted to prism.cli.parser)
    - Training orchestration (extracted to prism.core.runner)
    - Progressive training (extracted to prism.core.trainers)

Algorithm Workflow:
    1. Setup: Parse arguments, load config, handle inspection flags
    2. Initialization: Train ProgressiveDecoder on first measurement
    3. Progressive Training: Iteratively add samples and refine reconstruction
    4. Save results: Checkpoints, figures, and metrics

Key Parameters:
    --obj_name: Predefined astronomical object (europa, titan, betelgeuse, neptune)
    --n_samples: Number of telescope positions (typical: 100-240)
    --fermat: Use Fermat spiral sampling (recommended)
    --sample_diameter: Telescope aperture size in pixels
    --max_epochs: Training repetitions per sample (1 for testing, 25 for production)
    --loss_th: Convergence threshold (default: 0.001)

Output Structure (runs/{name}/):
    - checkpoint.pt: Complete state (model, metrics, failed samples)
    - args.txt/args.pt: Experiment parameters
    - sample_points.pt: Sampling pattern
    - final_reconstruction.png: Ground truth vs reconstruction
    - synthetic_aperture.png: K-space coverage visualization
    - learning_curves.png: Loss/SSIM/PSNR progression
    - TensorBoard logs for real-time monitoring
"""

# %% Imports
import sys
import warnings

from torch.backends import cudnn

# PRISM modules
from prism.cli.parser import create_main_parser
from prism.config import load_config, merge_config_with_args
from prism.core.runner import PRISMRunner
from prism.utils.logging_config import setup_logging
from prism.utils.training_helpers import load_config_with_checkpoint_fallback


# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore")
# Enable cuDNN autotuner to find optimal algorithms for this hardware/config
cudnn.benchmark = True

# Configure logging with centralized configuration
setup_logging(level="INFO", show_time=True, show_level=True)


def main() -> None:
    """Main entry point for PRISM algorithm."""
    # %% Parse arguments
    parser = create_main_parser()
    args = parser.parse_args()

    # %% Handle natural language instruction
    if args.instruction:
        from prism.config.natural_language import process_instruction

        args = process_instruction(args, interactive=not args.auto_confirm)
        if args is None:
            sys.exit(0)
        if getattr(args, "show_parse_only", False):
            sys.exit(0)

    # %% Handle interactive mode (must happen first - replaces args entirely)
    if args.interactive:
        from prism.config.interactive import run_interactive_setup

        # Run wizard, get configured args
        args_interactive = run_interactive_setup(mode="prism")

        # If user cancelled or chose not to run, exit
        if args_interactive is None:
            sys.exit(0)

        # Continue with normal flow (args now contains user choices)
        args = args_interactive

    # %% Handle help topic flags (show detailed help and exit)
    if (
        args.help_propagator
        or args.help_patterns
        or args.help_loss
        or args.help_model
        or args.help_objects
    ):
        from prism.config.validation import ConfigValidator

        if args.help_propagator:
            ConfigValidator.print_help_topic("propagator")
        elif args.help_patterns:
            ConfigValidator.print_help_topic("patterns")
        elif args.help_loss:
            ConfigValidator.print_help_topic("loss")
        elif args.help_model:
            ConfigValidator.print_help_topic("model")
        elif args.help_objects:
            ConfigValidator.print_help_topic("objects")
        sys.exit(0)

    # %% Handle inspection flags (must happen before config loading)
    if args.list_presets or args.show_preset or args.show_object:
        # Import inspector module
        from prism.config.inspector import handle_inspection_flags

        handle_inspection_flags(args, mode="prism")
        sys.exit(0)

    # %% Handle preset loading BEFORE config file
    if args.preset:
        from prism.config.presets import (
            get_preset,
            list_presets,
            merge_preset_with_overrides,
            validate_preset_name,
        )

        if not validate_preset_name(args.preset, mode="prism"):
            available = ", ".join(list_presets("prism"))
            print(f"Error: Unknown preset '{args.preset}'")
            print(f"Available presets: {available}")
            print("Use --list-presets to see details")
            sys.exit(1)

        # Load preset
        preset_dict = get_preset(args.preset, mode="prism")

        # Preset should exist after validation, but check to satisfy mypy
        if preset_dict is None:
            from loguru import logger

            logger.error(f"Failed to load preset '{args.preset}'")
            sys.exit(1)

        # Convert args to dict for merging
        args_dict = vars(args)

        # Merge preset with CLI args (CLI takes precedence)
        merged = merge_preset_with_overrides(preset_dict, args_dict)

        # Convert back to namespace
        for key, value in merged.items():
            setattr(args, key, value)

        from loguru import logger

        logger.info(f"Loaded preset: {args.preset}")

    # %% Configure logger level based on args
    setup_logging(level=args.log_level, show_time=True, show_level=True)

    # %% Handle scenario presets (before config merge)
    if args.list_scenarios:
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

        sys.exit(0)

    if args.show_scenario:
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
            sys.exit(0)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Load scenario and apply to args
    if args.scenario:
        from loguru import logger

        from prism.scenarios import get_scenario_preset

        try:
            scenario = get_scenario_preset(args.scenario)
            logger.info(f"Loaded scenario: {scenario.name}")

            # Apply scenario-specific overrides from CLI (check attribute exists)
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

    # %% Load config with automatic checkpoint fallback
    args = load_config_with_checkpoint_fallback(
        args,
        load_config,  # type: ignore[arg-type]
        merge_config_with_args,  # type: ignore[arg-type]
    )

    # %% Handle post-config inspection flags
    if args.show_config:
        from prism.config.inspector import show_effective_config

        show_effective_config(args)
        sys.exit(0)

    if args.validate_only:
        from prism.config import args_to_config

        config = args_to_config(args)
        try:
            config.validate()
            print("✓ Configuration is valid")
            sys.exit(0)
        except ValueError as e:
            print(f"✗ Configuration error: {e}")
            sys.exit(1)

    # %% Run experiment
    runner = PRISMRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
