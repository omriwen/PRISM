"""Utility functions for SPIDS."""

# Image operations

from __future__ import annotations

from .image import (
    crop_image,
    crop_pad,
    generate_point_sources,
    get_image_size,
    load_image,
    pad_image,
)

# I/O operations
from .io import (
    export_results,
    load_checkpoint,
    save_args,
    save_checkpoint,
)

# Image quality metrics
from .metrics import (
    compute_rmse,
    compute_ssim,
    psnr,
)

# Progress tracking
from .progress import (
    ETACalculator,
    TrainingProgress,
)

# Parallel sampling utilities
from .sampling import (
    compute_energies_parallel,
    generate_samples_parallel,
    get_optimal_worker_count,
    parallel_telescope_measurements,
    sort_samples_by_energy_parallel,
)

# Training helpers
from .training_helpers import (
    configure_line_sampling,
    load_config_with_checkpoint_fallback,
    setup_device,
    validate_training_args,
)

# Fourier transforms
from .transforms import (
    create_mask,
    fft,
    ifft,
)

# Validation metrics for optical system validation
from .validation_metrics import (
    compare_to_theoretical,
    compute_l2_error,
    compute_mtf50,
    compute_mtf_from_esf,
    compute_peak_position_error,
    detect_resolved_elements,
    generate_validation_report,
    measure_element_contrast,
)


__all__ = [
    # Image operations
    "load_image",
    "generate_point_sources",
    "get_image_size",
    "pad_image",
    "crop_image",
    "crop_pad",
    # Fourier transforms
    "fft",
    "ifft",
    "create_mask",
    # Metrics
    "compute_ssim",
    "compute_rmse",
    "psnr",
    # Validation metrics
    "compute_mtf50",
    "compute_mtf_from_esf",
    "compute_l2_error",
    "compute_peak_position_error",
    "compare_to_theoretical",
    "detect_resolved_elements",
    "measure_element_contrast",
    "generate_validation_report",
    # I/O
    "save_args",
    "save_checkpoint",
    "load_checkpoint",
    "export_results",
    # Parallel sampling
    "generate_samples_parallel",
    "compute_energies_parallel",
    "sort_samples_by_energy_parallel",
    "parallel_telescope_measurements",
    "get_optimal_worker_count",
    # Progress tracking
    "TrainingProgress",
    "ETACalculator",
    # Training helpers
    "load_config_with_checkpoint_fallback",
    "setup_device",
    "configure_line_sampling",
    "validate_training_args",
]
