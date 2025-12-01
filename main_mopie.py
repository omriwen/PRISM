"""
Module: main_mopie.py
Purpose: Motion-aware Ptychographic Iterative Engine (Mo-PIE) algorithm for SPIDS reconstruction
Dependencies: torch, spids.cli, spids.core, spids.config
Main Components:
    - Argument parsing (extracted to spids.cli.parser)
    - Mo-PIE algorithm implementation (iterative phase retrieval)
    - Epochal training (iterate over all samples repeatedly)

Algorithm Workflow:
    1. Setup: Parse arguments, load config, create telescope
    2. Initialize object and probe estimates
    3. Mo-PIE Training Loop:
        - For each epoch:
            - For each sample: update object (and optionally probe)
            - Compute metrics (RMSE, SSIM, PSNR)
    4. Save results: Checkpoints and figures

Key Parameters:
    --lr_obj: Mo-PIE object learning rate (typical: 1.0)
    --lr_probe: Mo-PIE probe learning rate (typical: 1.0)
    --fix_probe: Keep probe fixed as known aperture (recommended: True)
    --n_epochs: Number of full passes through all samples (typical: 100-500)

Output Structure (runs/{name}/):
    - checkpoint.pt: Complete state (object, probe, metrics)
    - sample_points.pt: Sampling pattern
    - final_reconstruction.png: Results visualization
    - learning_curves.png: Metrics over epochs
"""

# %% Imports
import datetime
import os
import sys
import time
import warnings
from pathlib import Path

import torch
from loguru import logger
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

# PRISM modules
from prism.cli.parser import create_mopie_parser
from prism.config import args_to_config, load_config, merge_config_with_args, save_config
from prism.config.objects import get_obj_params
from prism.core import patterns
from prism.core.algorithms import MoPIE
from prism.core.instruments import Telescope, TelescopeConfig
from prism.core.measurement_system import MeasurementSystem
from prism.core.pattern_loader import load_and_generate_pattern
from prism.core.pattern_preview import preview_pattern
from prism.utils.image import crop_image, generate_point_sources, load_image
from prism.utils.io import save_args, save_checkpoint
from prism.utils.logging_config import setup_logging
from prism.utils.progress import ETACalculator, TrainingProgress
from prism.utils.training_helpers import (
    configure_line_sampling,
    load_config_with_checkpoint_fallback,
    setup_device,
)
from prism.visualization import (
    PUBLICATION,
    LearningCurvesPlotter,
    ReconstructionComparisonPlotter,
    SyntheticAperturePlotter,
)


# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore")
# Enable cuDNN autotuner
cudnn.benchmark = True

# Configure logging
setup_logging(level="INFO", show_time=True, show_level=True)


def main() -> None:
    """Main entry point for Mo-PIE algorithm."""
    # %% Parse arguments
    parser = create_mopie_parser()
    args = parser.parse_args()

    # %% Handle interactive mode
    if args.interactive:
        from prism.config.interactive import run_interactive_setup

        args_interactive = run_interactive_setup(mode="mopie")
        if args_interactive is None:
            sys.exit(0)
        args = args_interactive

    # %% Handle help topic flags (show detailed help and exit)
    # Note: Mo-PIE only uses propagator, patterns, and objects (no loss/model)
    if args.help_propagator or args.help_patterns or args.help_objects:
        from prism.config.validation import ConfigValidator

        if args.help_propagator:
            ConfigValidator.print_help_topic("propagator")
        elif args.help_patterns:
            ConfigValidator.print_help_topic("patterns")
        elif args.help_objects:
            ConfigValidator.print_help_topic("objects")
        sys.exit(0)

    # %% Handle inspection flags
    if args.list_presets or args.show_preset or args.show_object:
        from prism.config.inspector import handle_inspection_flags

        handle_inspection_flags(args, mode="mopie")
        sys.exit(0)

    # %% Handle preset loading
    if args.preset:
        from prism.config.presets import (
            get_preset,
            list_presets,
            merge_preset_with_overrides,
            validate_preset_name,
        )

        if not validate_preset_name(args.preset, mode="mopie"):
            available = ", ".join(list_presets("mopie"))
            print(f"Error: Unknown preset '{args.preset}'")
            print(f"Available presets: {available}")
            print("Use --list-presets to see details")
            sys.exit(1)

        preset_dict = get_preset(args.preset, mode="mopie")
        if preset_dict is None:
            logger.error(f"Failed to load preset '{args.preset}'")
            sys.exit(1)

        merged = merge_preset_with_overrides(preset_dict, vars(args))
        for key, value in merged.items():
            setattr(args, key, value)

        # Flatten nested Mo-PIE-specific parameters for easier access
        if hasattr(args, "mopie") and isinstance(args.mopie, dict):
            for mopie_key, mopie_value in args.mopie.items():
                if not hasattr(args, mopie_key):
                    setattr(args, mopie_key, mopie_value)
        # Flatten nested training parameters
        if hasattr(args, "training") and isinstance(args.training, dict):
            for train_key, train_value in args.training.items():
                if not hasattr(args, train_key):
                    setattr(args, train_key, train_value)
        # Flatten nested telescope parameters
        if hasattr(args, "telescope") and isinstance(args.telescope, dict):
            for tele_key, tele_value in args.telescope.items():
                if not hasattr(args, tele_key):
                    setattr(args, tele_key, tele_value)

        logger.info(f"Loaded preset: {args.preset}")

    # %% Configure logging
    setup_logging(level=args.log_level, show_time=True, show_level=True)

    # %% Load config
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
        config = args_to_config(args)
        try:
            config.validate()
            print("✓ Configuration is valid")
            sys.exit(0)
        except ValueError as e:
            print(f"✗ Configuration error: {e}")
            sys.exit(1)

    # %% Setup
    args.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~ Starting training ~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Run name: {args.name}")
    print(f"Start time: {args.start_time}")

    args = get_obj_params(args)

    if args.roi_diameter is None:
        args.roi_diameter = args.image_size
    if args.samples_r_cutoff is None:
        args.samples_r_cutoff = args.roi_diameter / 2

    device = setup_device(args)

    if args.name is None:
        args.name = args.start_time
    log_dir = os.path.join(args.log_dir, args.name)

    config = args_to_config(args)

    if args.save_data:
        os.makedirs(log_dir)
        log_file_path = Path(log_dir) / "training.log"
        setup_logging(level=args.log_level, log_file=log_file_path, show_time=True, show_level=True)
        logger.info(f"Logging to: {log_dir}/training.log")
        writer = SummaryWriter(log_dir)
        save_config(config, os.path.join(log_dir, "config.yaml"))
        logger.info(f"Configuration saved to: {log_dir}/config.yaml")
    else:
        writer = None

    # %% Load image and pattern
    args.dx = args.wavelength * args.obj_distance / (args.dxf * args.image_size)

    if args.obj_size is None:
        args.obj_size = int(args.obj_diameter / args.dx)

    print(f"Object size: {args.obj_size} pixels")
    print(f"Sample diameter: {args.sample_diameter} pixels")

    if args.is_point_source:
        image = generate_point_sources(
            image_size=args.image_size,
            number_of_sources=args.point_source_number,
            sample_diameter=args.point_source_diameter,
            spacing=args.point_source_spacing,
        )
        image_gt = image.sum(0)
    else:
        image = load_image(
            args.input,
            size=args.obj_size,
            padded_size=args.obj_size if args.crop_obj else args.image_size,
            invert=args.invert_image,
        )
        image_gt = image

    image = image.to(device)
    image_gt = image_gt.to(device)

    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)

    if image_gt.ndim == 2:
        image_gt = image_gt.unsqueeze(0).unsqueeze(0)
    elif image_gt.ndim == 3:
        image_gt = image_gt.unsqueeze(0)

    # Generate pattern
    pattern_spec = (
        args.pattern_fn
        if hasattr(args, "pattern_fn") and args.pattern_fn
        else config.telescope.pattern_fn
    )
    if pattern_spec is None:
        pattern_spec = "builtin:random"

    if hasattr(args, "preview_pattern") and args.preview_pattern:
        logger.info(f"Previewing pattern: {pattern_spec}")
        preview_save_path = Path(log_dir if args.save_data else ".") / "pattern_preview.png"
        preview_result = preview_pattern(
            pattern_spec, config.telescope, save_path=preview_save_path
        )
        logger.info("\nPattern Statistics:")
        for key, value in preview_result["statistics"].items():
            logger.info(
                f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}"
            )
        logger.info(f"\nPreview saved to: {preview_save_path}")
        sys.exit(0)

    logger.info(f"Generating pattern: {pattern_spec}")
    sample_centers, pattern_metadata = load_and_generate_pattern(pattern_spec, config.telescope)

    args.original_n_samples = args.n_samples
    args.n_samples = len(sample_centers)

    logger.info(f"Generated {sample_centers.shape[0]} sampling positions")
    if pattern_metadata["docstring"]:
        pattern_desc = pattern_metadata["docstring"].split("\n")[0].strip()
        logger.info(f"Pattern: {pattern_desc}")

    if args.snr is not None:
        logger.info(f"SNR: {args.snr:.1f} dB")
    logger.info(
        f"Training: lr_obj={args.lr_obj}, lr_probe={args.lr_probe}, n_epochs={args.n_epochs}"
    )

    # Configure line sampling
    samples_per_line_meas, dsnr = configure_line_sampling(
        args, args.sample_length, args.sample_diameter
    )
    args.samples_per_line_meas = samples_per_line_meas

    if args.save_data:
        torch.save(
            {"centers": sample_centers, "diameter": args.sample_diameter},
            os.path.join(log_dir, "sample_points.pt"),
        )

    # Create Mo-PIE model
    # Note: Mo-PIE internally creates its own telescope-like behavior
    # by inheriting from the legacy Telescope class. We pass parameters directly.
    # Calculate effective SNR (Mo-PIE requires float, not None)
    effective_snr = (args.snr + dsnr) if args.snr is not None else 100.0
    model = MoPIE(
        n=args.image_size,
        r=args.sample_diameter / 2,
        cropping=args.crop_obj,
        obj_size=args.obj_size,
        snr=effective_snr,
        ground_truth=image,
        fix_probe=getattr(args, "fix_probe", True),
        lr_obj=args.lr_obj,
        lr_probe=getattr(args, "lr_probe", 1.0),
        complex_data=getattr(args, "complex_data", False),
        parallel_update=getattr(args, "parallel_update", True),
        single_sample=getattr(args, "single_sample", False),
        blur_image=args.blur_image,
    )
    model = model.to(device)

    # Initialize metrics
    rmses = []
    ssims = []
    psnrs = []
    epoch_times = []

    # %% Training loop
    indices = torch.arange(args.n_samples)

    total_steps = args.n_epochs * args.n_samples
    epoch_eta = ETACalculator(args.n_epochs)
    sample_eta = ETACalculator(total_steps)
    steps_completed = 0

    logger.info(
        f"Starting Mo-PIE training: {args.n_epochs} epochs × {args.n_samples} samples = "
        f"{total_steps} iterations"
    )

    with TrainingProgress() as training_progress:
        epoch_task_id = training_progress.add_task("Epochs", total=args.n_epochs)
        sample_task_id = training_progress.add_task("Samples", total=total_steps)

        for epoch in range(args.n_epochs):
            t0 = time.time()

            if getattr(args, "rand_perm", True):
                indices = torch.randperm(args.n_samples)

            for center_idx, center_ends in enumerate(
                sample_centers[indices], start=getattr(args, "n_samples_0", 0)
            ):
                cntr, center_rec = patterns.create_patterns(
                    center_ends, samples_per_line_meas, args.samples_per_line_rec
                )

                model.update_cntr(cntr, center_rec, center_idx)
                model.update_step()

                steps_completed += 1
                sample_eta_seconds = sample_eta.update(steps_completed)
                training_progress.advance(
                    sample_task_id,
                    metrics={
                        "epoch": epoch + 1,
                        "sample": (center_idx % args.n_samples) + 1,
                    },
                    eta_seconds=sample_eta_seconds,
                    description=f"Epoch {epoch + 1}/{args.n_epochs}",
                )

            # Compute metrics
            rmse, ssim_val, psnr_val = model.errors()
            rmses.append(rmse)
            ssims.append(ssim_val)
            psnrs.append(psnr_val)
            epoch_times.append(time.time() - t0)

            epoch_eta_seconds = epoch_eta.update(epoch + 1)
            training_progress.advance(
                epoch_task_id,
                metrics={
                    "epoch": epoch + 1,
                    "ssim": ssim_val,
                    "rmse": rmse,
                    "psnr": psnr_val,
                },
                eta_seconds=epoch_eta_seconds,
                description=f"Epochs {epoch + 1}/{args.n_epochs}",
            )

            print(
                f"Epoch: {epoch + 1}/{args.n_epochs}, SSIM: {ssim_val:.2f}, "
                f"RMSE: {rmse:.2}, PSNR: {psnr_val:.1f} dB"
            )
            logger.debug(
                f"Epoch {epoch + 1}/{args.n_epochs}: SSIM={ssim_val:.3f}, "
                f"RMSE={rmse:.2e}, PSNR={psnr_val:.1f} dB"
            )

            # Log to TensorBoard
            if args.save_data and writer is not None:
                writer.add_scalar("SSIM", ssim_val, epoch)
                writer.add_scalar("RMSE", rmse, epoch)
                writer.add_scalar("PSNR", psnr_val, epoch)
                writer.add_scalar("Time/per_epoch", epoch_times[-1], epoch)

        training_progress.complete(epoch_task_id)
        training_progress.complete(sample_task_id)

    # %% Save results
    if args.save_data:
        args.end_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        save_args(args, log_dir)
        save_checkpoint(
            {
                "object": model.Og.detach(),
                "probe": model.Pg.detach(),
                "sample_centers": sample_centers,
                "ssims": torch.tensor(ssims),
                "rmses": torch.tensor(rmses),
                "psnrs": torch.tensor(psnrs),
                "epoch_times": torch.tensor(epoch_times),
                "args_dict": vars(args),
                "pattern_metadata": pattern_metadata,
                "pattern_spec": pattern_spec,
            },
            os.path.join(log_dir, "checkpoint.pt"),
        )

        # Save figures
        current_reconstruction = crop_image(model.Og, args.obj_size).abs()

        # Create telescope using unified API for visualization
        telescope_config = TelescopeConfig(
            n_pixels=args.image_size,
            aperture_radius_pixels=args.sample_diameter / 2,
            snr=args.snr + dsnr if args.snr is not None else None,
        )
        telescope = Telescope(telescope_config).to(device)

        with torch.no_grad():
            static_measurement = telescope.forward(image_gt, aperture_center=[0.0, 0.0])

        # Save figures using new visualization API
        with ReconstructionComparisonPlotter(PUBLICATION) as plotter:
            plotter.plot(
                ground_truth=image_gt,
                reconstruction=current_reconstruction,
                static_measurement=static_measurement,
                obj_size=args.obj_size,
            )
            plotter.save(os.path.join(log_dir, "final_reconstruction.png"))

        # Create MeasurementSystem for synthetic aperture visualization
        measurement_system = MeasurementSystem(telescope).to(device)
        # Populate cum_mask with all sample centers for visualization
        measurement_system.add_mask(sample_centers.tolist())

        with SyntheticAperturePlotter(PUBLICATION) as plotter:
            plotter.plot(
                tensor=image_gt,
                telescope_agg=measurement_system,  # MeasurementSystem is compatible
                roi_diameter=args.roi_diameter,
            )
            plotter.save(os.path.join(log_dir, "synthetic_aperture.png"))

        with LearningCurvesPlotter(PUBLICATION) as plotter:
            plotter.plot(
                losses=[],  # No per-sample losses in Mo-PIE
                ssims=ssims,
                psnrs=psnrs,
            )
            plotter.save(os.path.join(log_dir, "learning_curves.png"))

        logger.info(f"Final figures saved to: {log_dir}")

        if writer is not None:
            writer.close()

    logger.info(
        f"Training completed: {args.n_epochs} epochs, Final SSIM: {ssims[-1]:.3f}, "
        f"PSNR: {psnrs[-1]:.1f} dB"
    )
    print("~~~~~~~~~~~~~~~~~~~~~~~~~ Finished training ~~~~~~~~~~~~~~~~~~~~~~~~~")


if __name__ == "__main__":
    main()
