"""Optical elements and forward models for SPIDS instruments.

This module provides optical components for simulating microscope imaging
systems, including thin lens elements, forward models with automatic
regime selection, realistic detector noise modeling, and unified input
field handling.

Classes
-------
ThinLens
    Thin lens optical element with quadratic phase transformation.
MicroscopeForwardModel
    Unified microscope forward model with automatic regime selection.
FourFForwardModel
    Unified 4f optical system forward model with FFT padding support.
DetectorNoiseModel
    Realistic detector noise model (shot noise, read noise, dark current).
InputFieldHandler
    Unified input field validation and conversion for 4f optical systems.
    Supports multiple input modes (intensity/amplitude/complex), dimension
    standardization, and pixel size validation.

Functions
---------
compute_defocus_parameter
    Compute normalized defocus parameter δ.
select_forward_regime
    Select appropriate forward model regime based on defocus.
detect_input_mode
    Auto-detect input mode from tensor properties.
convert_to_complex_field
    Convert input to complex field for wave propagation.
prepare_field
    Complete input preparation pipeline (legacy, use InputFieldHandler for new code).
validate_fov_consistency
    Warn if input field's pixel size doesn't match grid.

Enums
-----
ForwardModelRegime
    Forward model regime selection (SIMPLIFIED, FULL, AUTO).
InputMode
    Input field representation mode (INTENSITY, AMPLITUDE, COMPLEX, AUTO).

Examples
--------
Basic usage with a thin lens:

>>> from prism.core.grid import Grid
>>> from prism.core.optics import ThinLens
>>> grid = Grid(nx=128, dx=1e-6, wavelength=532e-9)
>>> lens = ThinLens(focal_length=0.01, grid=grid)  # 10mm focal length

Using the microscope forward model:

>>> from prism.core.optics import MicroscopeForwardModel, ForwardModelRegime
>>> model = MicroscopeForwardModel(
...     grid=grid,
...     objective_focal=0.005,
...     tube_lens_focal=0.2,
...     working_distance=0.005,
...     na=1.4,
...     medium_index=1.515,
... )
>>> model.selected_regime  # Auto-selects based on defocus
ForwardModelRegime.SIMPLIFIED

Using the 4f forward model:

>>> from prism.core.optics import FourFForwardModel
>>> model = FourFForwardModel(grid, padding_factor=2.0)
>>> field = torch.randn(128, 128, dtype=torch.complex64)
>>> intensity = model(field)

Using the detector noise model:

>>> from prism.core.optics import DetectorNoiseModel
>>> import torch
>>> noise_model = DetectorNoiseModel(snr_db=40.0)
>>> clean_image = torch.rand(1, 1, 256, 256)
>>> noisy_image = noise_model(clean_image, add_noise=True)

Using the input field handler:

>>> from prism.core.optics import InputFieldHandler
>>> from prism.core.grid import Grid
>>> grid = Grid(nx=256, dx=10e-6, wavelength=520e-9)
>>> handler = InputFieldHandler(grid)
>>> # Auto-detect intensity input
>>> intensity = torch.rand(256, 256)
>>> field = handler.validate_and_convert(intensity, input_mode='auto')
>>> # Explicit amplitude input with batches
>>> amplitude = torch.rand(4, 1, 256, 256)
>>> field = handler.validate_and_convert(amplitude, input_mode='amplitude')

See Also
--------
spids.core.propagators : Wave propagation methods (ASM, Fresnel, Fraunhofer)
spids.core.instruments.microscope : Microscope instrument class
"""

from __future__ import annotations

from .aperture_masks import ApertureMaskGenerator
from .detector_noise import DetectorNoiseModel
from .four_f_forward import FourFForwardModel
from .fourier_utils import (
    aperture_center_equivalence,
    compute_k_space_coverage,
    compute_na_from_k_shift,
    illum_angle_to_k_shift,
    illum_position_to_k_shift,
    k_shift_to_illum_angle,
    k_shift_to_illum_position,
    k_shift_to_pixel,
    pixel_to_k_shift,
    validate_k_shift_within_na,
)
from .illumination import (
    IlluminationSource,
    IlluminationSourceType,
    create_illumination_envelope,
    create_illumination_field,
    create_phase_tilt,
    illumination_angle_to_k_center,
    k_center_to_illumination_angle,
)
from .input_handling import (
    InputFieldHandler,
    InputMode,
    convert_to_complex_field,
    detect_input_mode,
    prepare_field,
    validate_fov_consistency,
)
from .microscope_forward import (
    ForwardModelRegime,
    MicroscopeForwardModel,
    compute_defocus_parameter,
    select_forward_regime,
)
from .thin_lens import ThinLens


__all__ = [
    # Core optical elements
    "ThinLens",
    "MicroscopeForwardModel",
    "FourFForwardModel",
    "ApertureMaskGenerator",
    "ForwardModelRegime",
    "compute_defocus_parameter",
    "select_forward_regime",
    "DetectorNoiseModel",
    # Input handling
    "InputFieldHandler",
    "InputMode",
    "detect_input_mode",
    "convert_to_complex_field",
    "prepare_field",
    "validate_fov_consistency",
    # Illumination sources (scanning illumination mode)
    "IlluminationSource",
    "IlluminationSourceType",
    "create_illumination_envelope",
    "create_illumination_field",
    "create_phase_tilt",
    "illumination_angle_to_k_center",
    "k_center_to_illumination_angle",
    # Fourier utilities (k-space operations)
    "illum_angle_to_k_shift",
    "k_shift_to_illum_angle",
    "illum_position_to_k_shift",
    "k_shift_to_illum_position",
    "validate_k_shift_within_na",
    "compute_na_from_k_shift",
    "pixel_to_k_shift",
    "k_shift_to_pixel",
    "compute_k_space_coverage",
    "aperture_center_equivalence",
]
