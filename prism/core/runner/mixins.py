"""
Common setup logic for experiment runners.

This module provides mixin classes that encapsulate reusable setup logic
shared between PRISMRunner and MoPIERunner.
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from prism.config import args_to_config, save_config
from prism.config.objects import get_obj_params
from prism.core.pattern_loader import load_and_generate_pattern
from prism.core.pattern_preview import preview_pattern
from prism.utils.image import generate_point_sources, load_image
from prism.utils.logging_config import setup_logging
from prism.utils.training_helpers import configure_line_sampling, setup_device


if TYPE_CHECKING:
    from prism.config.experiment import PRISMConfig


class SetupMixin:
    """Common setup logic for all runners.

    Provides methods for device setup, logging configuration, and
    directory creation that are shared between different runner types.

    Attributes Expected by Mixin
    ---------------------------
    args : Any
        Parsed command-line arguments
    device : torch.device | None
        Training device (set by mixin)
    log_dir : Path | None
        Log directory (set by mixin)
    writer : SummaryWriter | None
        TensorBoard writer (set by mixin)
    config : PRISMConfig | None
        Experiment configuration (set by mixin)
    """

    # Type hints for attributes that should exist on the class using this mixin
    args: Any
    device: Optional[torch.device]
    log_dir: Optional[Path]
    writer: Optional[SummaryWriter]
    config: Optional["PRISMConfig"]

    def _setup_timing(self) -> str:
        """Initialize timing and return start time string.

        Returns
        -------
        str
            Formatted start time string
        """
        start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.args.start_time = start_time
        return start_time

    def _setup_device(self) -> torch.device:
        """Set up computation device (CPU/GPU).

        Returns
        -------
        torch.device
            Configured device for computation
        """
        device = setup_device(self.args)
        self.device = device
        return device

    def _setup_object_params(self) -> None:
        """Set object parameters from predefined object database.

        Updates args with physical parameters (wavelength, distance, etc.)
        based on selected object name.
        """
        self.args = get_obj_params(self.args)

        # Set defaults for ROI and cutoff if not specified
        if self.args.roi_diameter is None:
            self.args.roi_diameter = self.args.image_size
        if self.args.samples_r_cutoff is None:
            self.args.samples_r_cutoff = self.args.roi_diameter / 2

    def _setup_log_directory(self) -> Path:
        """Set up logging directory.

        Creates the log directory if save_data is True.

        Returns
        -------
        Path
            Path to log directory
        """
        if self.args.name is None:
            self.args.name = self.args.start_time

        log_dir = Path(os.path.join(self.args.log_dir, self.args.name))
        self.log_dir = log_dir
        return log_dir

    def _setup_config(self) -> "PRISMConfig":
        """Create configuration object from args.

        Returns
        -------
        PRISMConfig
            Experiment configuration
        """
        config = args_to_config(self.args)
        self.config = config
        return config

    def _setup_logging_and_writer(self) -> Optional[SummaryWriter]:
        """Set up file logging and TensorBoard writer.

        Only creates directory and writer if save_data is True.

        Returns
        -------
        SummaryWriter | None
            TensorBoard writer if save_data is True, else None
        """
        if not self.args.save_data:
            self.writer = None
            return None

        assert self.log_dir is not None, "Log directory must be set before logging setup"
        os.makedirs(self.log_dir)

        log_file_path = self.log_dir / "training.log"
        setup_logging(
            level=self.args.log_level,
            log_file=log_file_path,
            show_time=True,
            show_level=True,
        )
        logger.info(f"Logging to: {self.log_dir}/training.log")

        writer = SummaryWriter(str(self.log_dir))
        self.writer = writer

        # Save configuration
        if self.config is not None:
            save_config(self.config, str(self.log_dir / "config.yaml"))
            logger.info(f"Configuration saved to: {self.log_dir}/config.yaml")

        return writer

    def _print_run_header(self) -> None:
        """Print training run header."""
        print("~~~~~~~~~~~~~~~~~~~~~~~~~ Starting training ~~~~~~~~~~~~~~~~~~~~~~~~~")  # noqa: T201
        print(f"Run name: {self.args.name}")  # noqa: T201
        print(f"Start time: {self.args.start_time}")  # noqa: T201


class DataLoadingMixin:
    """Common data loading logic for runners.

    Provides methods for loading images and generating sampling patterns
    that are shared between different runner types.

    Attributes Expected by Mixin
    ---------------------------
    args : Any
        Parsed command-line arguments
    config : PRISMConfig | None
        Experiment configuration
    device : torch.device | None
        Training device
    log_dir : Path | None
        Log directory
    image : torch.Tensor | None
        Loaded input image (set by mixin)
    image_gt : torch.Tensor | None
        Ground truth image (set by mixin)
    sample_centers : torch.Tensor | None
        Sample center positions (set by mixin)
    pattern_metadata : dict | None
        Pattern generation metadata (set by mixin)
    pattern_spec : str | None
        Pattern specification string (set by mixin)
    """

    # Type hints for attributes that should exist on the class using this mixin
    args: Any
    config: Optional["PRISMConfig"]
    device: Optional[torch.device]
    log_dir: Optional[Path]
    image: Optional[torch.Tensor]
    image_gt: Optional[torch.Tensor]
    sample_centers: Optional[torch.Tensor]
    pattern_metadata: Optional[dict[str, Any]]
    pattern_spec: Optional[str]

    def _calculate_pixel_size(self) -> float:
        """Calculate pixel size from physical parameters.

        Returns
        -------
        float
            Pixel size dx
        """
        dx = self.args.wavelength * self.args.obj_distance / (self.args.dxf * self.args.image_size)
        self.args.dx = dx
        return dx

    def _calculate_object_size(self) -> int:
        """Calculate object size in pixels.

        Returns
        -------
        int
            Object size in pixels
        """
        if self.args.obj_size is None:
            self.args.obj_size = int(self.args.obj_diameter / self.args.dx)

        print(f"Object size: {self.args.obj_size} pixels")  # noqa: T201
        print(f"Sample diameter: {self.args.sample_diameter} pixels")  # noqa: T201

        return self.args.obj_size

    def _load_image(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load or generate input image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Input image and ground truth image
        """
        if self.args.is_point_source:
            image = generate_point_sources(
                image_size=self.args.image_size,
                number_of_sources=self.args.point_source_number,
                sample_diameter=self.args.point_source_diameter,
                spacing=self.args.point_source_spacing,
            )
            image_gt = image.sum(0)
        else:
            image = load_image(
                self.args.input,
                size=self.args.obj_size,
                padded_size=self.args.obj_size if self.args.crop_obj else self.args.image_size,
                invert=self.args.invert_image,
            )
            image_gt = image

        # Move to device
        assert self.device is not None, "Device must be set before loading image"
        image = image.to(self.device)
        image_gt = image_gt.to(self.device)

        # Ensure 4D format [B, C, H, W]
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 3:
            image = image.unsqueeze(0)

        if image_gt.ndim == 2:
            image_gt = image_gt.unsqueeze(0).unsqueeze(0)
        elif image_gt.ndim == 3:
            image_gt = image_gt.unsqueeze(0)

        self.image = image
        self.image_gt = image_gt

        return image, image_gt

    def _get_pattern_spec(self) -> str:
        """Get pattern specification from args or config.

        Returns
        -------
        str
            Pattern specification string
        """
        pattern_spec = (
            self.args.pattern_fn
            if hasattr(self.args, "pattern_fn") and self.args.pattern_fn
            else self.config.telescope.pattern_fn
            if self.config
            else None
        )
        if pattern_spec is None:
            pattern_spec = "builtin:random"

        self.pattern_spec = pattern_spec
        return pattern_spec

    def _handle_pattern_preview(self, pattern_spec: str) -> bool:
        """Handle pattern preview mode if requested.

        Parameters
        ----------
        pattern_spec : str
            Pattern specification string

        Returns
        -------
        bool
            True if preview was requested (should exit), False otherwise
        """
        if not (hasattr(self.args, "preview_pattern") and self.args.preview_pattern):
            return False

        logger.info(f"Previewing pattern: {pattern_spec}")
        preview_dir = self.log_dir if (self.args.save_data and self.log_dir) else Path(".")
        preview_save_path = preview_dir / "pattern_preview.png"

        assert self.config is not None, "Config must be set for pattern preview"
        preview_result = preview_pattern(
            pattern_spec, self.config.telescope, save_path=preview_save_path
        )
        logger.info("\nPattern Statistics:")
        for key, value in preview_result["statistics"].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
        logger.info(f"\nPreview saved to: {preview_save_path}")
        sys.exit(0)

        return True  # Never reached due to sys.exit, but for type safety

    def _generate_pattern(self, pattern_spec: str) -> Tuple[torch.Tensor, dict[str, Any]]:
        """Generate sampling pattern.

        Parameters
        ----------
        pattern_spec : str
            Pattern specification string

        Returns
        -------
        tuple[torch.Tensor, dict]
            Sample centers and pattern metadata
        """
        logger.info(f"Generating pattern: {pattern_spec}")

        assert self.config is not None, "Config must be set for pattern generation"
        sample_centers, pattern_metadata = load_and_generate_pattern(
            pattern_spec, self.config.telescope
        )

        # Update sample count
        self.args.original_n_samples = self.args.n_samples
        self.args.n_samples = len(sample_centers)

        logger.info(f"Generated {sample_centers.shape[0]} sampling positions")
        if pattern_metadata["docstring"]:
            pattern_desc = pattern_metadata["docstring"].split("\n")[0].strip()
            logger.info(f"Pattern: {pattern_desc}")

        self.sample_centers = sample_centers
        self.pattern_metadata = pattern_metadata

        return sample_centers, pattern_metadata

    def _log_experiment_info(self, pattern_spec: str) -> None:
        """Log experiment configuration info.

        Parameters
        ----------
        pattern_spec : str
            Pattern specification string
        """
        # Determine pattern name for logging
        if pattern_spec.startswith("builtin:"):
            sampling_pattern = pattern_spec.split(":", 1)[1].capitalize()
        elif self.pattern_metadata and self.pattern_metadata.get("source_path"):
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

    def _save_sample_points(self) -> None:
        """Save sample points to disk if save_data is True."""
        if not self.args.save_data:
            return

        assert self.log_dir is not None, "Log directory must be set when save_data is True"
        assert self.sample_centers is not None, "Sample centers must be generated"

        torch.save(
            {"centers": self.sample_centers, "diameter": self.args.sample_diameter},
            str(self.log_dir / "sample_points.pt"),
        )


class LineSamplingMixin:
    """Mixin for line sampling configuration.

    Provides method to configure line sampling parameters.

    Attributes Expected by Mixin
    ---------------------------
    args : Any
        Parsed command-line arguments
    """

    args: Any

    def _configure_line_sampling(self) -> Tuple[int, float]:
        """Configure line sampling parameters.

        Returns
        -------
        tuple[int, float]
            samples_per_line_meas and dsnr adjustment
        """
        samples_per_line_meas, dsnr = configure_line_sampling(
            self.args, self.args.sample_length, self.args.sample_diameter
        )
        self.args.samples_per_line_meas = samples_per_line_meas
        return samples_per_line_meas, dsnr
