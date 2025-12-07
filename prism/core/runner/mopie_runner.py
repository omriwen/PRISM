"""
Experiment runner for Mo-PIE algorithm.

This module orchestrates the complete Mo-PIE experimental workflow including
setup, initialization, epochal training, and result generation.
"""

from __future__ import annotations

import datetime
import time
from typing import Any

import torch
from loguru import logger

from prism.core.algorithms import MoPIE
from prism.core.instruments import Telescope, TelescopeConfig
from prism.core.measurement_system import MeasurementSystem
from prism.core.runner.base import AbstractRunner, ExperimentResult
from prism.core.runner.mixins import DataLoadingMixin, LineSamplingMixin, SetupMixin
from prism.core.trainers.epochal import EpochalTrainer, EpochalTrainerConfig
from prism.utils.io import save_args, save_checkpoint
from prism.visualization import (
    PUBLICATION,
    LearningCurvesPlotter,
    ReconstructionComparisonPlotter,
    SyntheticAperturePlotter,
)


class MoPIERunner(AbstractRunner, SetupMixin, DataLoadingMixin, LineSamplingMixin):
    """
    Orchestrates Mo-PIE experiment from setup to completion.

    Handles all phases of a Mo-PIE experiment:
    - Argument validation and setup
    - Image loading and preprocessing
    - Pattern generation
    - MoPIE model initialization
    - Epochal training (all samples per epoch)
    - Checkpoint and figure generation

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Examples
    --------
    >>> from prism.cli.parser import create_mopie_parser
    >>> parser = create_mopie_parser()
    >>> args = parser.parse_args(['--obj_name', 'europa', '--name', 'test'])
    >>> runner = MoPIERunner(args)
    >>> runner.run()
    """

    def __init__(self, args: Any) -> None:
        super().__init__(args)

        # Components
        self.model: MoPIE | None = None
        self.trainer: EpochalTrainer | None = None
        self.telescope: Telescope | None = None
        self.measurement_system: MeasurementSystem | None = None

        # Data
        self.image: torch.Tensor | None = None
        self.image_gt: torch.Tensor | None = None
        self.sample_centers: torch.Tensor | None = None
        self.pattern_metadata: dict[str, Any] | None = None
        self.pattern_spec: str | None = None

        # Line sampling parameters
        self.samples_per_line_meas: int = 1
        self.dsnr: float = 0.0

    def setup(self) -> None:
        """Setup experiment: device, logging, directories, and configuration."""
        # Initialize timing
        self._setup_timing()
        self._print_run_header()

        # Set object parameters
        self._setup_object_params()

        # Set device
        self._setup_device()

        # Set log directory
        self._setup_log_directory()

        # Create configuration
        self._setup_config()

        # Setup logging and TensorBoard
        self._setup_logging_and_writer()

    def load_data(self) -> None:
        """Load input image and generate sampling pattern."""
        # Calculate pixel size and object size
        self._calculate_pixel_size()
        self._calculate_object_size()

        # Load image
        self._load_image()

        # Get pattern specification
        pattern_spec = self._get_pattern_spec()

        # Handle pattern preview mode
        if hasattr(self.args, "preview_pattern") and self.args.preview_pattern:
            self._handle_pattern_preview(pattern_spec)

        # Generate pattern
        self._generate_pattern(pattern_spec)

        # Log experiment info
        self._log_mopie_info()

        # Configure line sampling
        self.samples_per_line_meas, self.dsnr = self._configure_line_sampling()

        # Save sample points
        self._save_sample_points()

    def _log_mopie_info(self) -> None:
        """Log Mo-PIE specific experiment info."""
        if self.args.snr is not None:
            logger.info(f"SNR: {self.args.snr:.1f} dB")
        logger.info(
            f"Training: lr_obj={self.args.lr_obj}, lr_probe={self.args.lr_probe}, "
            f"n_epochs={self.args.n_epochs}"
        )

    def create_components(self) -> None:
        """Initialize Mo-PIE model, telescope, and trainer components."""
        assert self.device is not None, "Device must be set"
        assert self.image is not None, "Image must be loaded"

        # Calculate effective SNR
        effective_snr = (self.args.snr + self.dsnr) if self.args.snr is not None else 100.0

        # Create Mo-PIE model
        self.model = MoPIE(
            n=self.args.image_size,
            r=self.args.sample_diameter / 2,
            cropping=self.args.crop_obj,
            obj_size=self.args.obj_size,
            snr=effective_snr,
            ground_truth=self.image,
            fix_probe=getattr(self.args, "fix_probe", True),
            lr_obj=self.args.lr_obj,
            lr_probe=getattr(self.args, "lr_probe", 1.0),
            complex_data=getattr(self.args, "complex_data", False),
            parallel_update=getattr(self.args, "parallel_update", True),
            single_sample=getattr(self.args, "single_sample", False),
            blur_image=self.args.blur_image,
        )
        self.model = self.model.to(self.device)

        # Create telescope for visualization using unified API
        telescope_config = TelescopeConfig(
            n_pixels=self.args.image_size,
            aperture_radius_pixels=self.args.sample_diameter / 2,
            snr=self.args.snr + self.dsnr if self.args.snr is not None else None,
        )
        self.telescope = Telescope(telescope_config).to(self.device)

        # Create MeasurementSystem for visualization
        self.measurement_system = MeasurementSystem(self.telescope).to(self.device)

        # Create trainer
        trainer_config = EpochalTrainerConfig(
            n_epochs=self.args.n_epochs,
            rand_perm=getattr(self.args, "rand_perm", True),
            n_samples=self.args.n_samples,
            samples_per_line_meas=self.samples_per_line_meas,
            samples_per_line_rec=self.args.samples_per_line_rec,
        )

        self.trainer = EpochalTrainer(
            model=self.model,
            device=self.device,
            config=trainer_config,
            sample_centers=self.sample_centers,
            samples_per_line_meas=self.samples_per_line_meas,
            samples_per_line_rec=self.args.samples_per_line_rec,
            writer=self.writer,
            log_dir=str(self.log_dir) if self.log_dir else None,
            n_samples_0=getattr(self.args, "n_samples_0", 0),
            rand_perm=getattr(self.args, "rand_perm", True),
        )

        logger.info("Mo-PIE system prepared, starting training...")

    def run_experiment(self) -> ExperimentResult:
        """Run Mo-PIE epochal training and return results."""
        assert self.trainer is not None, "Trainer must be created"
        assert self.sample_centers is not None, "Sample centers must be generated"

        # Track start time
        training_start_time = time.time()

        # Run epochal training
        training_result = self.trainer.train(
            sample_centers=self.sample_centers,
            n_epochs=self.args.n_epochs,
        )

        # Calculate elapsed time
        elapsed_time = time.time() - training_start_time

        # Create result
        result = ExperimentResult(
            ssims=training_result.ssims,
            psnrs=training_result.psnrs,
            rmses=training_result.rmses,
            final_reconstruction=training_result.final_reconstruction,
            log_dir=self.log_dir,
            elapsed_time=elapsed_time,
            failed_samples=[],  # Mo-PIE doesn't track failed samples
        )

        return result

    def save_results(self, result: ExperimentResult) -> None:
        """Save final checkpoint and visualization figures."""
        if not self.args.save_data:
            return

        assert self.log_dir is not None, "Log directory must be set"
        assert self.model is not None, "Model must be created"
        assert self.trainer is not None, "Trainer must be created"
        assert self.telescope is not None, "Telescope must be created"
        assert self.measurement_system is not None, "MeasurementSystem must be created"
        assert self.image_gt is not None, "Ground truth image must be loaded"
        assert self.sample_centers is not None, "Sample centers must be generated"

        # Set end time
        self.args.end_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save args
        save_args(self.args, str(self.log_dir))

        # Save checkpoint
        save_checkpoint(
            {
                "object": self.model.Og.detach(),
                "probe": self.model.Pg.detach(),
                "sample_centers": self.sample_centers,
                "ssims": torch.tensor(self.trainer.ssims),
                "rmses": torch.tensor(self.trainer.rmses),
                "psnrs": torch.tensor(self.trainer.psnrs),
                "epoch_times": torch.tensor(self.trainer.epoch_times),
                "args_dict": vars(self.args),
                "pattern_metadata": self.pattern_metadata,
                "pattern_spec": self.pattern_spec,
            },
            str(self.log_dir / "checkpoint.pt"),
        )

        # Get current reconstruction
        current_reconstruction = self.model.Og.abs()

        # Create static measurement for comparison
        with torch.no_grad():
            static_measurement = self.telescope.forward(self.image_gt, aperture_center=[0.0, 0.0])

        # Populate cumulative mask for visualization
        # sample_centers has shape (n_samples, n_points, 2), squeeze to (n_samples, 2)
        self.measurement_system.add_mask(self.sample_centers.squeeze(1).tolist())

        # Save figures using visualization API
        with ReconstructionComparisonPlotter(PUBLICATION) as plotter:
            plotter.plot(
                ground_truth=self.image_gt,
                reconstruction=current_reconstruction,
                static_measurement=static_measurement,
                obj_size=self.args.obj_size,
            )
            plotter.save(str(self.log_dir / "final_reconstruction.png"))

        with SyntheticAperturePlotter(PUBLICATION) as plotter:
            plotter.plot(
                tensor=self.image_gt,
                telescope_agg=self.measurement_system,
                roi_diameter=self.args.roi_diameter,
            )
            plotter.save(str(self.log_dir / "synthetic_aperture.png"))

        with LearningCurvesPlotter(PUBLICATION) as plotter:
            plotter.plot(
                losses=[],  # No per-sample losses in Mo-PIE
                ssims=self.trainer.ssims,
                psnrs=self.trainer.psnrs,
            )
            plotter.save(str(self.log_dir / "learning_curves.png"))

        logger.info(f"Final figures saved to: {self.log_dir}")

    def cleanup(self) -> None:
        """Clean up resources after experiment completion."""
        # Close TensorBoard writer
        super().cleanup()

        # Log final summary
        if self.trainer and self.trainer.ssims:
            logger.info(
                f"Training completed: {self.args.n_epochs} epochs, "
                f"Final SSIM: {self.trainer.ssims[-1]:.3f}, "
                f"PSNR: {self.trainer.psnrs[-1]:.1f} dB"
            )
        else:
            logger.info("Training completed")

        print("~~~~~~~~~~~~~~~~~~~~~~~~~ Finished training ~~~~~~~~~~~~~~~~~~~~~~~~~")  # noqa: T201
