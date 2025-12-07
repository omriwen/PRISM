"""
Command-line argument parser for SPIDS experiments.

This module provides argument parsing for both SPIDS and Mo-PIE algorithms,
centralizing all CLI configuration in one place.
"""

from __future__ import annotations

import argparse


def create_main_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for main SPIDS algorithm.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for SPIDS

    Examples
    --------
    >>> parser = create_main_parser()
    >>> args = parser.parse_args(['--obj_name', 'europa', '--name', 'test'])
    """
    parser = argparse.ArgumentParser(
        description="SPIDS - Sparse Phase Imaging by Diffraction Spectroscopy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Image parameters
    parser.add_argument("--input", type=str, default=None, help="Path to the input image")
    parser.add_argument(
        "--obj_size",
        type=int,
        default=None,
        help="Size of the object. If None use the original image size",
    )
    parser.add_argument("--image_size", type=int, default=1024, help="Size of the recorded image")
    parser.add_argument(
        "--invert", dest="invert_image", action="store_true", help="invert the image"
    )
    parser.add_argument(
        "--crop", dest="crop_obj", action="store_true", help="Crop the object to the object size"
    )

    # Telescope parameters
    parser.add_argument(
        "--sample_diameter",
        type=float,
        default=17,
        help="Diameter of the telescope aperture [pix]",
    )
    parser.add_argument(
        "--sample_shape", type=str, default="circle", help="The shape of the sample: circle or line"
    )
    parser.add_argument(
        "--sample_length",
        type=float,
        default=0,
        help="Length of a line sample [pix]. If 0, use circles",
    )
    parser.add_argument(
        "--samples_per_line_meas",
        type=int,
        default=None,
        help="Number of samples per line in a measurement. "
        "If None, space the samples by half the sample diameter",
    )
    parser.add_argument(
        "--samples_per_line_rec",
        type=int,
        default=None,
        help="Number of samples per line in a reconstruction step. "
        "If None, use the same as for the measurement (set according to the line length)",
    )
    parser.add_argument(
        "--line_angle",
        type=float,
        default=None,
        help="Angle of line samples [rad]. If None, use random angles",
    )
    parser.add_argument(
        "--line-mode",
        type=str,
        default="fast",
        choices=["accurate", "fast"],
        help="Line acquisition mode: 'accurate' (1 sample/pixel) or 'fast' (half-diameter spacing)",
    )
    parser.add_argument(
        "--samples-per-pixel",
        type=float,
        default=1.0,
        help="Sampling density for accurate line mode (samples per pixel)",
    )
    parser.add_argument(
        "--roi_diameter",
        type=float,
        default=None,
        help="The diameter of the region of interest in k-space [pix]. "
        "If None, it is set to image_size",
    )
    parser.add_argument(
        "--samples_r_cutoff",
        type=float,
        default=None,
        help="The highest radius allowed for a sample center in Fermat spiral sampling",
    )
    parser.add_argument(
        "--roi_shape", type=str, default="circle", help="The shape of the sample: circle or square"
    )
    parser.add_argument(
        "--sample_sort",
        type=str,
        default="center",
        help="The sorting of the samples: "
        "'center' - sort by proximity to the center, "
        "'rand' - random sorting, "
        "'energy' - sort by energy",
    )
    parser.add_argument("--n_samples", type=int, default=64, help="number of samples")
    parser.add_argument("--n_angs", type=int, default=4, help="number of angles")

    # Pattern function system
    parser.add_argument(
        "--pattern-fn",
        type=str,
        default="builtin:fermat",
        help="Pattern function: builtin:name (fermat/star/random) or /path/to/pattern.py",
    )
    parser.add_argument(
        "--preview-pattern",
        action="store_true",
        help="Preview pattern and exit (no experiment run)",
    )

    # Legacy pattern flags (deprecated, kept for backward compatibility)
    parser.add_argument(
        "--star",
        dest="star_sample",
        action="store_true",
        help="DEPRECATED: Use --pattern-fn builtin:star instead",
    )
    parser.add_argument(
        "--fermat",
        dest="fermat_sample",
        action="store_true",
        help="DEPRECATED: Use --pattern-fn builtin:fermat instead",
    )

    parser.add_argument("--snr", type=float, default=None, help="Image SNR [dB]")
    parser.add_argument(
        "--blur", dest="blur_image", action="store_true", help="Blurring the telescope output"
    )

    # Model parameters
    parser.add_argument(
        "--no_bn", dest="use_bn", action="store_false", help="Dont use batch normalization"
    )
    parser.add_argument(
        "--output_activation",
        type=str,
        default="none",
        help="Network output activation function (ProgressiveDecoder)",
    )
    parser.add_argument(
        "--no_leaky", dest="use_leaky", action="store_false", help="Dont use LeakyReLU"
    )
    parser.add_argument(
        "--middle_activation",
        type=str,
        default="sigmoid",
        help="Network middle activation function (ProgressiveDecoder)",
    )
    parser.add_argument(
        "--complex", dest="complex_data", action="store_true", help="Allow complex-valued outputs"
    )

    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument(
        "--max_epochs", type=int, default=1, help="Max number of repetitions of epochs"
    )
    parser.add_argument(
        "--n_epochs_init",
        type=int,
        default=100,
        help="Number of epochs at the initialization stage",
    )
    parser.add_argument(
        "--max_epochs_init",
        type=int,
        default=100,
        help="Max number of repetitions of epochs at the initialization stage",
    )
    parser.add_argument(
        "--initialization_target",
        type=str,
        default="circle",
        help="The target of the initialization stage [measurement, circle, synthetic_aperture]",
    )
    parser.add_argument("--loss_type", type=str, default="l1", help="Loss function: L1 or L2")
    parser.add_argument(
        "--new_weight", type=float, default=1, help="The weight of the new sample loss"
    )
    parser.add_argument(
        "--f_weight", type=float, default=1e-4, help="The weight of the Fourier constraint loss"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--loss_th", type=float, default=0.005, help="Loss threshold for stopping the training"
    )
    parser.add_argument(
        "--no-normalize-loss",
        action="store_true",
        default=False,
        help="Disable zero-loss normalization (matches notebook behavior)",
    )
    parser.add_argument(
        "--amsgrad", dest="use_amsgrad", action="store_true", help="Use amsgrad in the optimizer"
    )
    parser.add_argument(
        "--mixed-precision",
        dest="use_mixed_precision",
        action="store_true",
        help="Use Automatic Mixed Precision (FP16/FP32) for 20-30%% speedup and 40-50%% memory reduction. "
        "Requires compatible GPU (Ampere, Hopper)",
    )

    # Adaptive convergence parameters
    parser.add_argument(
        "--adaptive-convergence",
        dest="enable_adaptive_convergence",
        action="store_true",
        default=False,
        help="Enable adaptive per-sample convergence (early exit, escalation, retries)",
    )
    parser.add_argument(
        "--no-adaptive-convergence",
        dest="enable_adaptive_convergence",
        action="store_false",
        help="Disable adaptive convergence (use fixed epochs for all samples)",
    )
    parser.add_argument(
        "--early-stop-patience",
        dest="early_stop_patience",
        type=int,
        default=10,
        help="Epochs of no improvement before considering early stop",
    )
    parser.add_argument(
        "--plateau-window",
        dest="plateau_window",
        type=int,
        default=50,
        help="Window size for plateau detection",
    )
    parser.add_argument(
        "--plateau-threshold",
        dest="plateau_threshold",
        type=float,
        default=0.01,
        help="Relative improvement threshold for plateau detection (<1%% = plateau)",
    )
    parser.add_argument(
        "--escalation-epochs",
        dest="escalation_epochs",
        type=int,
        default=200,
        help="Epochs before considering escalation to aggressive optimization",
    )
    parser.add_argument(
        "--aggressive-lr-multiplier",
        dest="aggressive_lr_multiplier",
        type=float,
        default=2.0,
        help="Learning rate multiplier in aggressive mode",
    )
    parser.add_argument(
        "--max-retries",
        dest="max_retries",
        type=int,
        default=2,
        help="Maximum retry attempts for failed samples",
    )
    parser.add_argument(
        "--retry-lr-multiplier",
        dest="retry_lr_multiplier",
        type=float,
        default=0.1,
        help="Learning rate multiplier on retry attempts",
    )
    parser.add_argument(
        "--retry-switch-loss",
        dest="retry_switch_loss",
        action="store_true",
        default=True,
        help="Switch loss function on retry attempts (cycles through L1, SSIM, L2, MS-SSIM)",
    )
    parser.add_argument(
        "--no-retry-switch-loss",
        dest="retry_switch_loss",
        action="store_false",
        help="Disable loss switching on retry attempts",
    )

    parser.add_argument("--device_num", type=int, default=0, help="Cuda device number")
    parser.add_argument(
        "--no_cuda", dest="use_cuda", action="store_false", help="Use CPU instead of GPU"
    )

    # Physics parameters
    parser.add_argument(
        "--wavelength", type=float, default=None, help="Wavelength of the light [m]"
    )
    parser.add_argument(
        "--dxf", type=float, default=1e-2, help="Pixel size on the detector plane [m]"
    )
    parser.add_argument(
        "--obj_diameter", type=float, default=None, help="The real diameter of the object [m]"
    )
    parser.add_argument(
        "--obj_distance",
        type=float,
        default=None,
        help="The distance of the object from the telescope [m]",
    )
    parser.add_argument("--obj_name", type=str, default="europa", help="Predefined object name")
    parser.add_argument("--obj", dest="obj_name", type=str, help="Shorthand for --obj_name")
    parser.add_argument(
        "--propagator-method",
        type=str,
        choices=["auto", "fraunhofer", "fresnel", "angular_spectrum"],
        default="fraunhofer",
        help="Propagator method to use: 'auto' (automatic selection based on physics), "
        "'fraunhofer' (far-field), 'fresnel' (near-field), or 'angular_spectrum' (general purpose)",
    )

    # Point source parameters
    parser.add_argument(
        "--point_source", dest="is_point_source", action="store_true", help="Use point source"
    )
    parser.add_argument(
        "--point_source_diameter", type=float, default=3, help="Diameter of the point source [pix]"
    )
    parser.add_argument(
        "--point_source_spacing", type=float, default=5, help="Spacing between point sources [pix]"
    )
    parser.add_argument(
        "--point_source_number", type=int, default=4, help="Number of point sources"
    )

    # Other parameters
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Use a built-in configuration preset (quick_test, production, high_quality, debug, "
        "line_sampling, europa, titan, betelgeuse, neptune)",
    )
    parser.add_argument(
        "--log_dir", type=str, default="runs", help="Directory for tensorboard logs and data"
    )
    parser.add_argument("--debug", dest="save_data", action="store_false", help="Dont save data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Load a checkpoint file to continue training from",
    )
    parser.add_argument("--name", type=str, help="Run name for saving the data")
    parser.add_argument("--comment", type=str, default="", help="Run name for saving the data")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args override config values)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )

    # Inspection and utility flags
    parser.add_argument(
        "--list-presets",
        dest="list_presets",
        action="store_true",
        help="List all available presets and exit",
    )
    parser.add_argument(
        "--show-preset",
        dest="show_preset",
        type=str,
        default=None,
        help="Show details of a specific preset and exit",
    )
    parser.add_argument(
        "--show-object",
        dest="show_object",
        type=str,
        default=None,
        help="Show parameters for a predefined object and exit",
    )
    parser.add_argument(
        "--show-config",
        dest="show_config",
        action="store_true",
        help="Show effective configuration and exit",
    )
    parser.add_argument(
        "--validate-only",
        dest="validate_only",
        action="store_true",
        help="Validate configuration without running",
    )
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        help="Interactive mode: guided parameter selection wizard",
    )

    # Dashboard integration flags
    parser.add_argument(
        "--dashboard",
        dest="dashboard",
        action="store_true",
        help="Launch web dashboard during training for real-time monitoring",
    )
    parser.add_argument(
        "--dashboard-port",
        dest="dashboard_port",
        type=int,
        default=8050,
        help="Port number for dashboard server (default: 8050)",
    )

    # Profiling flags
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        help="Enable profiling during training",
    )
    parser.add_argument(
        "--profile-output",
        dest="profile_output",
        type=str,
        default=None,
        help="Profile output path (default: runs/{name}/profile.pt)",
    )

    # Enhanced help flags for specific topics
    parser.add_argument(
        "--help-propagator",
        dest="help_propagator",
        action="store_true",
        help="Show detailed help for propagator methods and exit",
    )
    parser.add_argument(
        "--help-patterns",
        dest="help_patterns",
        action="store_true",
        help="Show detailed help for sampling patterns and exit",
    )
    parser.add_argument(
        "--help-loss",
        dest="help_loss",
        action="store_true",
        help="Show detailed help for loss functions and exit",
    )
    parser.add_argument(
        "--help-model",
        dest="help_model",
        action="store_true",
        help="Show detailed help for model configuration and exit",
    )
    parser.add_argument(
        "--help-objects",
        dest="help_objects",
        action="store_true",
        help="Show detailed help for predefined objects and exit",
    )

    # Scenario system arguments
    scenario_group = parser.add_argument_group("Scenario Configuration")
    scenario_group.add_argument(
        "--scenario",
        type=str,
        default=None,
        metavar="NAME",
        help="Use scenario preset: microscope_100x_oil, drone_50m_survey, etc.",
    )
    scenario_group.add_argument(
        "--list-scenarios",
        dest="list_scenarios",
        action="store_true",
        help="List available scenario presets and exit",
    )
    scenario_group.add_argument(
        "--show-scenario",
        dest="show_scenario",
        type=str,
        default=None,
        metavar="NAME",
        help="Show scenario preset details and exit",
    )

    # Microscope-specific overrides
    micro_group = parser.add_argument_group("Microscope Settings (overrides scenario)")
    micro_group.add_argument(
        "--objective",
        type=str,
        metavar="SPEC",
        help="Microscope objective: '100x_1.4NA_oil', '40x_0.9NA_air'",
    )
    micro_group.add_argument(
        "--illumination",
        type=str,
        choices=["brightfield", "darkfield", "phase", "dic"],
        help="Microscope illumination mode",
    )

    # Drone-specific overrides
    drone_group = parser.add_argument_group("Drone Camera Settings (overrides scenario)")
    drone_group.add_argument(
        "--lens",
        type=str,
        metavar="SPEC",
        help="Camera lens: '50mm_f1.8', '35mm_f2.8'",
    )
    drone_group.add_argument(
        "--altitude",
        type=float,
        metavar="METERS",
        help="Flight altitude in meters",
    )
    drone_group.add_argument(
        "--sensor",
        type=str,
        metavar="TYPE",
        help="Sensor type: 'full_frame', 'aps_c', '1_inch'",
    )

    # AI Configuration arguments
    ai_group = parser.add_argument_group("AI Configuration")
    ai_group.add_argument(
        "--base",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to base config file (.yaml, .json, or .sh) for AI configuration",
    )
    ai_group.add_argument(
        "-i",
        "--instruction",
        type=str,
        default=None,
        metavar="TEXT",
        help="Natural language instruction to modify configuration (requires ollama)",
    )
    ai_group.add_argument(
        "--auto-confirm",
        dest="auto_confirm",
        action="store_true",
        help="Skip confirmation prompt for AI-suggested changes",
    )
    ai_group.add_argument(
        "--show-parse-only",
        dest="show_parse_only",
        action="store_true",
        help="Display parsed configuration and exit (dry run)",
    )
    ai_group.add_argument(
        "--ai-model",
        type=str,
        default="llama3.2:3b",
        metavar="MODEL",
        help="Ollama model to use for AI configuration (default: llama3.2:3b)",
    )

    # Default parameters
    parser.set_defaults(
        invert_image=False,
        use_bn=True,
        use_leaky=True,
        complex_data=False,
        use_cuda=True,
        save_data=True,
        crop_obj=False,
        star_sample=False,
        use_amsgrad=False,
        fermat_sample=False,
        blur_image=False,
        is_point_source=False,
        dashboard=False,
        profile=False,
        auto_confirm=False,
        show_parse_only=False,
    )

    return parser


def create_mopie_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for Mo-PIE algorithm.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for Mo-PIE

    Examples
    --------
    >>> parser = create_mopie_parser()
    >>> args = parser.parse_args(['--obj_name', 'europa', '--name', 'test'])
    """
    # Start with main parser and add Mo-PIE-specific arguments
    parser = create_main_parser()
    parser.description = "Mo-PIE - Motion-aware Ptychographic Iterative Engine"

    # Add Mo-PIE-specific parameters
    mopie_group = parser.add_argument_group("Mo-PIE Specific Parameters")
    mopie_group.add_argument(
        "--alpha", type=float, default=0.9, help="Mo-PIE object update parameter"
    )
    mopie_group.add_argument(
        "--beta", type=float, default=0.1, help="Mo-PIE probe update parameter"
    )
    mopie_group.add_argument(
        "--lr_obj", type=float, default=1.0, help="Mo-PIE object learning rate"
    )
    mopie_group.add_argument(
        "--lr_probe", type=float, default=1.0, help="Mo-PIE probe learning rate"
    )

    return parser
