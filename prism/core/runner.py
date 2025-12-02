"""
Experiment runner for SPIDS algorithm.

This module orchestrates the complete experimental workflow including
setup, initialization, training, and result generation.
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from prism.config import args_to_config, save_config
from prism.config.objects import get_obj_params
from prism.core import patterns
from prism.core.instruments import Telescope, TelescopeConfig
from prism.core.measurement_system import MeasurementSystem
from prism.core.pattern_loader import load_and_generate_pattern
from prism.core.pattern_preview import preview_pattern
from prism.core.trainers import PRISMTrainer
from prism.models.networks import ProgressiveDecoder
from prism.utils.image import generate_point_sources, load_image
from prism.utils.io import save_args, save_checkpoint
from prism.utils.logging_config import setup_logging
from prism.utils.training_helpers import configure_line_sampling, setup_device
from prism.visualization import (
    PUBLICATION,
    LearningCurvesPlotter,
    ReconstructionComparisonPlotter,
    SyntheticAperturePlotter,
)


class PRISMRunner:
    """
    Orchestrates SPIDS experiment from setup to completion.

    Handles all phases of a SPIDS experiment:
    - Argument validation and setup
    - Image loading and preprocessing
    - Pattern generation
    - Model and telescope initialization
    - Training (initialization + progressive)
    - Checkpoint and figure generation

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Examples
    --------
    >>> from prism.cli.parser import create_main_parser
    >>> parser = create_main_parser()
    >>> args = parser.parse_args(['--obj_name', 'europa', '--name', 'test'])
    >>> runner = PRISMRunner(args)
    >>> runner.run()
    """

    def __init__(self, args: Any):
        self.args = args
        self.device: torch.device | None = None
        self.log_dir: str | None = None
        self.writer: SummaryWriter | None = None
        self.config: Any = None

        # Components
        self.telescope: Telescope | None = None
        self.measurement_system: MeasurementSystem | None = None
        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.scheduler: Any = None
        self.trainer: PRISMTrainer | None = None

        # Dashboard
        self.dashboard_launcher: Any = None

        # Profiling
        self.profiler: Any = None

        # Data
        self.image: torch.Tensor | None = None
        self.image_gt: torch.Tensor | None = None
        self.sample_centers: torch.Tensor | None = None
        self.pattern_metadata: dict[str, Any] | None = None
        self.pattern_spec: str | None = None

    def setup(self) -> None:
        """Setup experiment: device, logging, directories, and configuration."""
        # Set start time
        self.args.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        print("~~~~~~~~~~~~~~~~~~~~~~~~~ Starting training ~~~~~~~~~~~~~~~~~~~~~~~~~")  # noqa: T201
        print(f"Run name: {self.args.name}")  # noqa: T201
        print(f"Start time: {self.args.start_time}")  # noqa: T201

        # Set object parameters
        self.args = get_obj_params(self.args)

        # Set ROI diameter
        if self.args.roi_diameter is None:
            self.args.roi_diameter = self.args.image_size
        if self.args.samples_r_cutoff is None:
            self.args.samples_r_cutoff = self.args.roi_diameter / 2

        # Set device
        self.device = setup_device(self.args)

        # Set log directory
        if self.args.name is None:
            self.args.name = self.args.start_time
        self.log_dir = os.path.join(self.args.log_dir, self.args.name)

        # Create configuration object
        self.config = args_to_config(self.args)

        # Setup logging and save directories
        if self.args.save_data:
            os.makedirs(self.log_dir)
            log_file_path = Path(self.log_dir) / "training.log"
            setup_logging(
                level=self.args.log_level, log_file=log_file_path, show_time=True, show_level=True
            )
            logger.info(f"Logging to: {self.log_dir}/training.log")
            self.writer = SummaryWriter(self.log_dir)
            save_config(self.config, os.path.join(self.log_dir, "config.yaml"))
            logger.info(f"Configuration saved to: {self.log_dir}/config.yaml")

    def load_image_and_pattern(self) -> None:
        """Load input image and generate sampling pattern."""
        # Calculate pixel size
        self.args.dx = (
            self.args.wavelength * self.args.obj_distance / (self.args.dxf * self.args.image_size)
        )

        # Set object size
        if self.args.obj_size is None:
            self.args.obj_size = int(self.args.obj_diameter / self.args.dx)

        print(f"Object size: {self.args.obj_size} pixels")  # noqa: T201
        print(f"Sample diameter: {self.args.sample_diameter} pixels")  # noqa: T201

        # Load or generate image
        if self.args.is_point_source:
            self.image = generate_point_sources(
                image_size=self.args.image_size,
                number_of_sources=self.args.point_source_number,
                sample_diameter=self.args.point_source_diameter,
                spacing=self.args.point_source_spacing,
            )
            self.image_gt = self.image.sum(0)
        else:
            self.image = load_image(
                self.args.input,
                size=self.args.obj_size,
                padded_size=self.args.obj_size if self.args.crop_obj else self.args.image_size,
                invert=self.args.invert_image,
            )
            self.image_gt = self.image

        self.image = self.image.to(self.device)
        self.image_gt = self.image_gt.to(self.device)

        # Ensure 4D format [B, C, H, W]
        if self.image.ndim == 2:
            self.image = self.image.unsqueeze(0).unsqueeze(0)
        elif self.image.ndim == 3:
            self.image = self.image.unsqueeze(0)

        if self.image_gt.ndim == 2:
            self.image_gt = self.image_gt.unsqueeze(0).unsqueeze(0)
        elif self.image_gt.ndim == 3:
            self.image_gt = self.image_gt.unsqueeze(0)

        # Generate sampling pattern
        self.pattern_spec = (
            self.args.pattern_fn
            if hasattr(self.args, "pattern_fn") and self.args.pattern_fn
            else self.config.telescope.pattern_fn
        )
        if self.pattern_spec is None:
            self.pattern_spec = "builtin:random"

        # Preview mode
        if hasattr(self.args, "preview_pattern") and self.args.preview_pattern:
            logger.info(f"Previewing pattern: {self.pattern_spec}")
            preview_dir = self.log_dir if (self.args.save_data and self.log_dir) else "."
            preview_save_path = Path(preview_dir) / "pattern_preview.png"
            preview_result = preview_pattern(
                self.pattern_spec, self.config.telescope, save_path=preview_save_path
            )
            logger.info("\nPattern Statistics:")
            for key, value in preview_result["statistics"].items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.2f}")
                else:
                    logger.info(f"  {key}: {value}")
            logger.info(f"\nPreview saved to: {preview_save_path}")
            sys.exit(0)

        # Generate pattern
        logger.info(f"Generating pattern: {self.pattern_spec}")
        self.sample_centers, self.pattern_metadata = load_and_generate_pattern(
            self.pattern_spec, self.config.telescope
        )

        # Update sample count
        self.args.original_n_samples = self.args.n_samples
        self.args.n_samples = len(self.sample_centers)

        logger.info(f"Generated {self.sample_centers.shape[0]} sampling positions")
        if self.pattern_metadata["docstring"]:
            pattern_desc = self.pattern_metadata["docstring"].split("\n")[0].strip()
            logger.info(f"Pattern: {pattern_desc}")

        # Determine pattern name for logging
        if self.pattern_spec.startswith("builtin:"):
            sampling_pattern = self.pattern_spec.split(":", 1)[1].capitalize()
        elif self.pattern_metadata.get("source_path"):
            sampling_pattern = f"Custom ({os.path.basename(self.pattern_metadata['source_path'])})"
        else:
            sampling_pattern = "Custom"

        logger.info(
            f"Object: {self.args.obj_name}, Size: {self.args.obj_size}px, "
            f"Sample diameter: {self.args.sample_diameter}px"
        )
        logger.info(f"Sampling: {sampling_pattern} - {self.args.n_samples} positions")
        if self.args.snr is not None:
            logger.info(f"SNR: {self.args.snr:.1f} dB")
        logger.info(
            f"Training: lr={self.args.lr}, loss_th={self.args.loss_th}, "
            f"max_epochs={self.args.max_epochs}"
        )

        # Save sample points
        if self.args.save_data:
            assert self.log_dir is not None, "Log directory must be set when save_data is True"
            torch.save(
                {"centers": self.sample_centers, "diameter": self.args.sample_diameter},
                os.path.join(self.log_dir, "sample_points.pt"),
            )

    def create_model_and_telescope(self) -> None:
        """Initialize model and telescope components."""
        # Configure line sampling
        samples_per_line_meas, dsnr = configure_line_sampling(
            self.args, self.args.sample_length, self.args.sample_diameter
        )
        self.args.samples_per_line_meas = samples_per_line_meas

        # Select propagator based on config
        selected_propagator = None
        if self.config.telescope.propagator_method is not None:
            from prism.config.constants import fresnel_number as calculate_fresnel_number
            from prism.core.propagators import select_propagator

            # Calculate aperture size (detector plane extent)
            # Fresnel number uses the aperture size at the detector/observation plane
            # NOT the projected FOV at the object plane (which would be image_size * dx)
            aperture_size = self.args.image_size * self.args.dxf

            # Calculate Fresnel number for logging
            fresnel_num = calculate_fresnel_number(
                aperture_size, self.args.obj_distance, self.args.wavelength
            )

            # Get propagator method
            method = self.config.telescope.propagator_method

            # Select propagator based on config
            selected_propagator = select_propagator(
                wavelength=self.args.wavelength,
                obj_distance=self.args.obj_distance,
                fov=aperture_size,
                method=method,
                image_size=self.args.image_size,
                dx=self.args.dx,
                dxf=self.args.dxf,
            )
            logger.info(
                f"Using propagator from config: {self.config.telescope.propagator_method} "
                f"(Fresnel number F={fresnel_num:.2e}, aperture={aperture_size:.4e} m, "
                f"wavelength={self.args.wavelength:.2e} m, distance={self.args.obj_distance:.2e} m)"
            )
        else:
            # Default: Auto-select propagator based on Fresnel number
            from prism.config.constants import fresnel_number as calculate_fresnel_number
            from prism.core.propagators import select_propagator

            # Calculate aperture size (detector plane extent)
            # Fresnel number uses the aperture size at the detector/observation plane
            # NOT the projected FOV at the object plane (which would be image_size * dx)
            aperture_size = self.args.image_size * self.args.dxf

            # Calculate Fresnel number for logging
            fresnel_num = calculate_fresnel_number(
                aperture_size, self.args.obj_distance, self.args.wavelength
            )

            # Auto-select propagator based on physics
            selected_propagator = select_propagator(
                wavelength=self.args.wavelength,
                obj_distance=self.args.obj_distance,
                fov=aperture_size,
                method="auto",
                image_size=self.args.image_size,
                dx=self.args.dx,
                dxf=self.args.dxf,
            )
            logger.info(
                f"Auto-selected propagator (Fresnel number F={fresnel_num:.2e}, "
                f"aperture={aperture_size:.4e} m, wavelength={self.args.wavelength:.2e} m, "
                f"distance={self.args.obj_distance:.2e} m)"
            )

        # Create telescope using unified API (Phase 3)
        telescope_config = TelescopeConfig(
            n_pixels=self.args.image_size,
            aperture_radius_pixels=self.args.sample_diameter / 2,
            snr=self.args.snr + dsnr if self.args.snr is not None else None,
            wavelength=self.args.wavelength,
            pixel_size=self.args.dxf,
        )
        assert self.device is not None, "Device must be set before creating telescope"
        self.telescope = Telescope(telescope_config).to(self.device)

        # Inject the selected propagator into the telescope
        if selected_propagator is not None:
            self.telescope._propagator = selected_propagator

        # Create line acquisition module if using line mode
        line_acquisition = None
        if self.args.sample_length > 0:
            from prism.core.line_acquisition import IncoherentLineAcquisition, LineAcquisitionConfig

            # Get mode from args, with fallback to 'fast' for backward compatibility
            line_mode = getattr(self.args, "line_mode", "fast")
            samples_per_pixel = getattr(self.args, "samples_per_pixel", 1.0)

            # Validate line_mode is one of the expected values
            if line_mode not in ("accurate", "fast"):
                line_mode = "fast"

            line_config = LineAcquisitionConfig(
                mode=line_mode,  # type: ignore[arg-type]
                samples_per_pixel=samples_per_pixel,
                min_samples=2,
                batch_size=64,
            )
            line_acquisition = IncoherentLineAcquisition(line_config, self.telescope)
            logger.info(
                f"Line acquisition enabled: mode={line_mode}, samples_per_pixel={samples_per_pixel}"
            )

        # Create measurement system wrapping the telescope
        assert self.device is not None, "Device must be set before creating measurement system"
        self.measurement_system = MeasurementSystem(
            self.telescope, line_acquisition=line_acquisition
        ).to(self.device)

        # Create model
        self.model = ProgressiveDecoder(
            input_size=self.args.image_size,
            use_bn=self.args.use_bn,
            output_activation=self.args.output_activation,
            use_leaky=self.args.use_leaky,
            middle_activation=self.args.middle_activation,
            complex_data=self.args.complex_data,
            output_size=self.args.obj_size,
            use_amp=getattr(self.args, "use_mixed_precision", False),
        ).to(self.device)

        # Create optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.args.lr, amsgrad=self.args.use_amsgrad
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10)

        logger.info("System prepared, starting training...")

    def load_checkpoint_if_needed(self) -> bool:
        """Load checkpoint if specified. Returns True if checkpoint loaded."""
        if self.args.checkpoint is None:
            self.args.n_samples_0 = 0
            return False

        assert self.model is not None, "Model must be created before loading checkpoint"

        checkpoint = torch.load(
            os.path.join(self.args.log_dir, self.args.checkpoint, "checkpoint.pt"),
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint["model"])
        sample_centers_ckpt = checkpoint["sample_centers"]
        self.sample_centers = sample_centers_ckpt
        self.args.n_samples_0 = 0
        del checkpoint
        logger.info(f"Loaded checkpoint from {self.args.checkpoint}")
        return True

    def create_trainer(self) -> None:
        """Create trainer instance."""
        assert self.model is not None, "Model must be created before trainer"
        assert self.optimizer is not None, "Optimizer must be created before trainer"
        assert self.scheduler is not None, "Scheduler must be created before trainer"
        assert self.measurement_system is not None, (
            "MeasurementSystem must be created before trainer"
        )
        assert self.device is not None, "Device must be set before trainer"

        # Create profiler if requested
        callbacks = []
        if getattr(self.args, "profile", False):
            from prism.profiling import ProfilerConfig, TrainingProfiler

            config = ProfilerConfig(enabled=True)
            self.profiler = TrainingProfiler(config)
            callbacks = [self.profiler.callback]
            logger.info("Profiling enabled")

        self.trainer = PRISMTrainer(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            measurement_system=self.measurement_system,
            args=self.args,
            device=self.device,
            log_dir=self.log_dir,
            writer=self.writer,
            use_amp=getattr(self.args, "use_mixed_precision", False),
            callbacks=callbacks,
        )

    def run_initialization(self) -> Any:
        """Run initialization phase and return figure handle."""
        assert self.sample_centers is not None, "Sample centers must be generated"
        assert self.telescope is not None, "Telescope must be created"
        assert self.image is not None, "Image must be loaded"
        assert self.image_gt is not None, "Ground truth image must be loaded"
        assert self.trainer is not None, "Trainer must be created"

        # Choose first sample center
        center, center_rec = patterns.create_patterns(
            self.sample_centers[0],
            self.args.samples_per_line_meas,
            self.args.samples_per_line_rec,
        )

        # Create initialization target
        assert self.measurement_system is not None, "MeasurementSystem must be created"
        with torch.no_grad():
            if self.args.initialization_target == "measurement":
                # Get measurement through new unified API
                # Convert center to list format expected by get_measurements
                center_list: list[list[float]] = [
                    list(center.tolist())
                    if hasattr(center, "tolist")
                    else [float(x) for x in center]
                ]
                measurement = self.measurement_system.get_measurements(self.image, center_list)
            elif self.args.initialization_target == "circle":
                # Generate circular aperture mask
                measurement = (
                    self.telescope.generate_aperture_mask(
                        center=[0.0, 0.0],
                        radius=self.args.obj_size / 2,
                    )
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            elif self.args.initialization_target == "synthetic_aperture":
                # Compute synthetic aperture from ALL sample positions
                # This is an advanced feature - compute by averaging multiple measurements
                import time

                logger.info(
                    f"Computing synthetic aperture preview from {len(self.sample_centers)} positions..."
                )

                start_time = time.time()
                # Compute by averaging measurements from all positions
                all_measurements = []
                batch_size = 100 if len(self.sample_centers) > 1000 else len(self.sample_centers)
                for i in range(0, len(self.sample_centers), batch_size):
                    batch_centers = self.sample_centers[i : i + batch_size]
                    for center_pos in batch_centers:
                        # Convert center to list format expected by get_measurements
                        center_coords: list[float] = (
                            list(center_pos.tolist())
                            if hasattr(center_pos, "tolist")
                            else [float(x) for x in center_pos]
                        )
                        meas = self.measurement_system.get_measurements(
                            self.image, [center_coords], add_noise=False
                        )
                        all_measurements.append(meas)
                # Average all measurements
                measurement = torch.stack(all_measurements).mean(dim=0)

                compute_time = time.time() - start_time
                logger.info(
                    f"Synthetic aperture computed in {compute_time:.2f}s "
                    f"(shape: {measurement.shape}, "
                    f"range: [{measurement.min():.3e}, {measurement.max():.3e}])"
                )
            else:
                raise ValueError(
                    f"Unknown initialization_target: {self.args.initialization_target}"
                )

            measurement = torch.cat([measurement, measurement], dim=0).detach().clone()
            if measurement.dtype == torch.bool:
                measurement = measurement.float()
            # Ensure measurement is on the correct device
            measurement = measurement.to(self.device)

        # Run initialization
        current_rec, figure = self.trainer.run_initialization(
            measurement,
            center,
            self.image_gt,
            figure=None,
            telescope=self.telescope,
            sample_centers=self.sample_centers,
        )
        return figure

    def run_training(self, figure: Any = None) -> dict[str, Any]:
        """Run progressive training and return results."""
        assert self.trainer is not None, "Trainer must be created"
        assert self.sample_centers is not None, "Sample centers must be generated"
        assert self.image is not None, "Image must be loaded"
        assert self.image_gt is not None, "Ground truth image must be loaded"

        results = self.trainer.run_progressive_training(
            sample_centers=self.sample_centers,
            image=self.image,
            image_gt=self.image_gt,
            samples_per_line_meas=self.args.samples_per_line_meas,
            figure=figure,
            pattern_metadata=self.pattern_metadata,
            pattern_spec=self.pattern_spec,
        )
        return results

    def save_final_checkpoint(self) -> None:
        """Save final checkpoint for single-sample runs."""
        if self.args.n_samples > 1 or not self.args.save_data:
            return

        import time

        assert self.log_dir is not None, "Log directory must be set"
        assert self.model is not None, "Model must be created"
        assert self.measurement_system is not None, "MeasurementSystem must be created"
        assert self.optimizer is not None, "Optimizer must be created"
        assert self.trainer is not None, "Trainer must be created"

        save_args(self.args, self.log_dir)
        save_checkpoint(
            {
                "model": self.model.state_dict(),
                "sample_centers": self.sample_centers,
                "last_center_idx": torch.tensor(0),
                "measurement_system": self.measurement_system.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "losses": torch.tensor(self.trainer.losses),
                "ssims": torch.tensor(self.trainer.ssims),
                "rmses": torch.tensor(self.trainer.rmses),
                "psnrs": torch.tensor(self.trainer.psnrs),
                "current_rec": self.trainer.current_reconstruction,
                "failed_samples": torch.tensor([]),
                "wall_time_seconds": time.time() - self.trainer.training_start_time,
                "args_dict": vars(self.args),
                "sample_times": (
                    torch.tensor(self.trainer.sample_times)
                    if self.trainer.sample_times
                    else torch.tensor([])
                ),
                "lr_history": (
                    torch.tensor(self.trainer.lr_history)
                    if self.trainer.lr_history
                    else torch.tensor([])
                ),
                "pattern_metadata": self.pattern_metadata,
                "pattern_spec": self.pattern_spec,
            },
            os.path.join(self.log_dir, "checkpoint.pt"),
        )

    def save_profile_if_needed(self) -> None:
        """Save profiling data if profiling was enabled."""
        if self.profiler is None:
            return

        # Determine output path
        if hasattr(self.args, "profile_output") and self.args.profile_output is not None:
            output_path = Path(self.args.profile_output)
        else:
            assert self.log_dir is not None, "Log directory must be set when profiling"
            output_path = Path(self.log_dir) / "profile.pt"

        # Save profile
        self.profiler.save(output_path)
        logger.info(f"Profile saved to {output_path}")

    def save_final_figures(self) -> None:
        """Generate and save final visualization figures."""
        if not self.args.save_data:
            return

        assert self.log_dir is not None, "Log directory must be set"
        assert self.telescope is not None, "Telescope must be created"
        assert self.image_gt is not None, "Ground truth image must be loaded"
        assert self.measurement_system is not None, "MeasurementSystem must be created"
        assert self.trainer is not None, "Trainer must be created"
        assert self.trainer.current_reconstruction is not None, "Reconstruction must exist"

        # Create static measurement for comparison
        with torch.no_grad():
            static_measurement = self.measurement_system.get_measurements(
                self.image_gt, [[0.0, 0.0]], add_noise=False
            )

        # Save figures using new visualization API
        with ReconstructionComparisonPlotter(PUBLICATION) as plotter:
            plotter.plot(
                ground_truth=self.image_gt,
                reconstruction=self.trainer.current_reconstruction,
                static_measurement=static_measurement,
                obj_size=self.args.obj_size,
            )
            plotter.save(os.path.join(self.log_dir, "final_reconstruction.png"))

        with SyntheticAperturePlotter(PUBLICATION) as plotter:
            plotter.plot(
                tensor=self.image_gt,
                telescope_agg=self.measurement_system,  # MeasurementSystem is compatible
                roi_diameter=self.args.roi_diameter,
            )
            plotter.save(os.path.join(self.log_dir, "synthetic_aperture.png"))

        with LearningCurvesPlotter(PUBLICATION) as plotter:
            plotter.plot(
                losses=self.trainer.losses,
                ssims=self.trainer.ssims,
                psnrs=self.trainer.psnrs,
            )
            plotter.save(os.path.join(self.log_dir, "learning_curves.png"))

        logger.info(f"Final figures saved to: {self.log_dir}")

    def start_dashboard_if_requested(self) -> None:
        """Launch dashboard if --dashboard flag is set."""
        if not hasattr(self.args, "dashboard") or not self.args.dashboard:
            return

        try:
            from prism.web.launcher import DashboardLauncher

            # Determine runs directory (parent of current experiment)
            runs_dir = Path(self.args.log_dir)

            # Get port
            port = self.args.dashboard_port if hasattr(self.args, "dashboard_port") else 8050

            # Create and start launcher
            self.dashboard_launcher = DashboardLauncher(runs_dir=runs_dir, port=port)

            if self.dashboard_launcher.start():
                logger.success("Dashboard integration enabled")
            else:
                logger.warning("Dashboard launch failed, continuing without dashboard")
                self.dashboard_launcher = None

        except ImportError as e:
            logger.error(
                f"Dashboard dependencies not available: {e}\n"
                "Install with: uv add dash dash-bootstrap-components"
            )
            self.dashboard_launcher = None
        except Exception as e:  # noqa: BLE001 - Dashboard start failure is non-fatal
            logger.error(f"Error starting dashboard: {e}")
            self.dashboard_launcher = None

    def stop_dashboard(self) -> None:
        """Stop dashboard if it was started."""
        if self.dashboard_launcher is not None:
            try:
                self.dashboard_launcher.stop()
                self.dashboard_launcher = None
            except Exception as e:  # noqa: BLE001 - Cleanup must not raise
                logger.error(f"Error stopping dashboard: {e}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        # Stop dashboard first
        self.stop_dashboard()

        if self.writer is not None:
            self.writer.close()

        # Log final summary
        if self.args.n_samples > 1 and self.trainer and self.trainer.failed_samples:
            logger.info(
                f"Training completed: {len(self.trainer.losses)} samples processed, "
                f"{len(self.trainer.failed_samples)} failed, "
                f"Final SSIM: {self.trainer.ssims[-1]:.3f}, "
                f"PSNR: {self.trainer.psnrs[-1]:.1f} dB"
            )
        else:
            logger.info("Training completed successfully")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~ Finished training ~~~~~~~~~~~~~~~~~~~~~~~~~")  # noqa: T201

    def run(self) -> None:
        """Run complete SPIDS experiment."""
        try:
            self.setup()

            # Start dashboard if requested (after setup, so log_dir is available)
            self.start_dashboard_if_requested()

            self.load_image_and_pattern()
            self.create_model_and_telescope()

            # Load checkpoint if specified
            checkpoint_loaded = self.load_checkpoint_if_needed()

            if not checkpoint_loaded:
                self.create_trainer()
                figure = self.run_initialization()
                self.run_training(figure=figure)
                self.save_final_checkpoint()
            else:
                # Checkpoint loaded - skip initialization
                self.create_trainer()
                self.run_training(figure=None)

            # Save profiling data if enabled
            self.save_profile_if_needed()

            self.save_final_figures()
        finally:
            # Ensure cleanup happens even if training fails
            self.cleanup()
