"""Microscope forward model with automatic regime selection.

This module provides a unified forward model for microscope imaging that
automatically selects between a simplified FFT-based model (when object is
at the focal plane) and a full propagation model with lens phases (when
object is defocused).

The regime selection follows the same pattern as ``select_propagator()`` in
:mod:`spids.core.propagators`, auto-selecting based on physical parameters.

Model Regimes
-------------
SIMPLIFIED
    FFT-based forward model assuming object at the front focal plane.
    Uses direct Fourier transform relationship: a single FFT takes the
    object field to the back focal plane where pupils are applied.
    Fast and accurate when defocus parameter δ < 1%.

FULL
    Complete 4f propagation chain with explicit lens phases:

    1. Propagate from object plane to front focal plane (if defocused)
    2. Apply objective lens quadratic phase
    3. FFT to back focal plane
    4. Apply illumination and detection pupils
    5. IFFT to intermediate image plane
    6. Apply tube lens quadratic phase

    Required when object is significantly defocused from focal plane.

Physical Background
-------------------
The defocus parameter δ measures the fractional deviation of the object
from the front focal plane of the objective lens:

    δ = |d - f| / f

where d is the object-to-lens distance (working distance) and f is the
objective focal length. When δ ≈ 0, the Fraunhofer approximation is valid
at the back focal plane, and a single FFT accurately models the system.

See Also
--------
ThinLens : Thin lens optical element used in FULL model
spids.core.propagators.select_propagator : Similar auto-selection pattern
spids.core.instruments.microscope.Microscope : Main microscope instrument

References
----------
.. [1] Goodman, J. W. "Introduction to Fourier Optics", 4th ed., Chapter 6.
.. [2] Mertz, J. "Introduction to Optical Microscopy", Chapter 3.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import torch
from loguru import logger
from torch import Tensor, nn


if TYPE_CHECKING:
    from prism.core.grid import Grid
    from prism.core.optics.thin_lens import ThinLens
    from prism.core.propagators import AngularSpectrumPropagator


class ForwardModelRegime(Enum):
    """Forward model regime selection.

    Attributes
    ----------
    SIMPLIFIED : str
        FFT-based model assuming object at focal plane. Fast and accurate
        when the defocus parameter δ < threshold.
    FULL : str
        Propagation chain with explicit lens phases. Required when object
        is significantly defocused from the focal plane.
    AUTO : str
        Automatically select regime based on defocus parameter.
    """

    SIMPLIFIED = "simplified"  # FFT-based (Fraunhofer at focal plane)
    FULL = "full"  # Propagation chain with lens phases
    AUTO = "auto"  # Auto-select based on defocus parameter


def compute_defocus_parameter(
    object_distance: float,
    focal_length: float,
) -> float:
    """Compute normalized defocus parameter.

    The defocus parameter δ measures deviation from the focal configuration:

        δ = |d - f| / f

    where d is the object distance and f is the focal length.

    Parameters
    ----------
    object_distance : float
        Distance from object to lens front principal plane (meters).
    focal_length : float
        Lens focal length (meters).

    Returns
    -------
    float
        Defocus parameter (dimensionless). δ = 0 means object at focal plane.

    Raises
    ------
    ValueError
        If focal_length is zero.

    Examples
    --------
    >>> compute_defocus_parameter(0.01, 0.01)  # At focal plane
    0.0
    >>> compute_defocus_parameter(0.0101, 0.01)  # 1% defocus
    0.01
    >>> compute_defocus_parameter(0.015, 0.01)  # 50% defocus
    0.5
    """
    if focal_length == 0:
        raise ValueError("Focal length cannot be zero")

    return abs(object_distance - focal_length) / abs(focal_length)


def select_forward_regime(
    object_distance: float,
    focal_length: float,
    threshold: float = 0.01,
    method: ForwardModelRegime = ForwardModelRegime.AUTO,
) -> ForwardModelRegime:
    """Select appropriate forward model regime.

    Similar to select_propagator() in spids.core.propagators, this function
    auto-selects the optimal forward model based on physical parameters.

    The selection criterion is the defocus parameter:

        δ = |d - f| / f

    where d is the object-to-lens distance and f is the focal length.

    Parameters
    ----------
    object_distance : float
        Object-to-lens distance (meters).
    focal_length : float
        Lens focal length (meters).
    threshold : float, optional
        Defocus threshold for regime selection. Default 0.01 (1%).
        - δ < threshold: SIMPLIFIED model (single FFT)
        - δ >= threshold: FULL model (propagation + lens phases)
    method : ForwardModelRegime, optional
        Manual override. If not AUTO, returns this regime directly.

    Returns
    -------
    ForwardModelRegime
        Selected regime (SIMPLIFIED or FULL). Never returns AUTO.

    Examples
    --------
    >>> select_forward_regime(0.01, 0.01)  # At focal plane
    ForwardModelRegime.SIMPLIFIED
    >>> select_forward_regime(0.02, 0.01)  # 100% defocus
    ForwardModelRegime.FULL
    >>> select_forward_regime(0.02, 0.01, method=ForwardModelRegime.SIMPLIFIED)
    ForwardModelRegime.SIMPLIFIED  # Manual override
    """
    if method != ForwardModelRegime.AUTO:
        logger.info(f"Forward model regime: {method.value} (manual override)")
        return method

    delta = compute_defocus_parameter(object_distance, focal_length)

    if delta < threshold:
        logger.info(
            f"Auto-selected SIMPLIFIED forward model "
            f"(defocus δ={delta:.4f} < threshold={threshold})"
        )
        return ForwardModelRegime.SIMPLIFIED
    else:
        logger.info(
            f"Auto-selected FULL forward model (defocus δ={delta:.4f} >= threshold={threshold})"
        )
        return ForwardModelRegime.FULL


class MicroscopeForwardModel(nn.Module):
    """Unified microscope forward model with automatic regime selection.

    This forward model transparently selects between:

    - **SIMPLIFIED**: FFT-based model (current implementation). Fast and accurate
      when object is at or near the focal plane.

    - **FULL**: Propagation chain with explicit lens phases. Required when
      object is defocused.

    The selection is automatic based on the defocus parameter:

        δ = |working_distance - f_objective| / f_objective

    Parameters
    ----------
    grid : Grid
        Spatial grid in object space.
    objective_focal : float
        Objective lens focal length (meters). Computed as f_tube / magnification.
    tube_lens_focal : float
        Tube lens focal length (meters). Typically 0.2 (200mm).
    working_distance : float
        Object-to-objective distance (meters).
    na : float
        Numerical aperture of the objective.
    medium_index : float
        Refractive index of immersion medium.
    regime : ForwardModelRegime, optional
        Model selection: AUTO (default), SIMPLIFIED, or FULL.
    defocus_threshold : float, optional
        Threshold for regime selection. Default 0.01.
    padding_factor : float, optional
        Factor by which to pad the grid for FFT operations. Must be >= 1.0.
        Default 2.0. Padding helps reduce FFT artifacts at boundaries.
        The padded size is rounded to the next power of 2 for FFT efficiency.

    Attributes
    ----------
    selected_regime : ForwardModelRegime
        The regime that was selected (SIMPLIFIED or FULL).
    defocus_parameter : float
        Computed defocus parameter δ.
    padding_factor : float
        The padding factor used for FFT operations.
    original_size : tuple[int, int]
        Original grid size (nx, ny) before padding.
    padded_size : tuple[int, int]
        Padded grid size for FFT operations, rounded to power of 2.

    Examples
    --------
    >>> from prism.core.grid import Grid
    >>> from prism.core.optics import MicroscopeForwardModel, ForwardModelRegime
    >>> grid = Grid(nx=128, dx=1e-6, wavelength=532e-9)
    >>> model = MicroscopeForwardModel(
    ...     grid=grid,
    ...     objective_focal=0.005,  # 5mm
    ...     tube_lens_focal=0.2,    # 200mm
    ...     working_distance=0.005, # At focal plane
    ...     na=1.4,
    ...     medium_index=1.515,
    ... )
    >>> model.selected_regime
    ForwardModelRegime.SIMPLIFIED

    Notes
    -----
    The forward model is lazily initialized: FULL model components (lenses,
    propagator) are only created if FULL regime is selected, minimizing
    memory usage when SIMPLIFIED is sufficient.

    The model automatically logs its configuration at initialization,
    including the computed defocus parameter and selected regime.

    See Also
    --------
    ThinLens : Optical element used in FULL model
    compute_defocus_parameter : Defocus calculation
    select_forward_regime : Regime selection logic
    """

    def __init__(
        self,
        grid: "Grid",
        objective_focal: float,
        tube_lens_focal: float,
        working_distance: float,
        na: float,
        medium_index: float,
        regime: ForwardModelRegime = ForwardModelRegime.AUTO,
        defocus_threshold: float = 0.01,
        padding_factor: float = 2.0,
    ) -> None:
        super().__init__()

        self.grid = grid
        self.objective_focal = objective_focal
        self.tube_lens_focal = tube_lens_focal
        self.working_distance = working_distance
        self.na = na
        self.medium_index = medium_index
        self.defocus_threshold = defocus_threshold

        # Validate and store padding factor
        if padding_factor < 1.0:
            raise ValueError(f"padding_factor must be >= 1.0, got {padding_factor}")
        self.padding_factor = padding_factor
        self.original_size = (grid.nx, grid.ny)

        # Compute padded size (round to power of 2 for FFT efficiency)
        self.padded_grid: Optional["Grid"] = None
        if padding_factor > 1.0:
            padded_n = int(grid.nx * padding_factor)
            # Round up to next power of 2 for FFT efficiency
            padded_n = 2 ** int(torch.tensor(padded_n).float().log2().ceil())
            self.padded_size = (padded_n, padded_n)
            # Create padded grid with same pixel size but larger extent
            from prism.core.grid import Grid as GridClass

            self.padded_grid = GridClass(
                nx=padded_n,
                ny=padded_n,
                dx=grid.dx,
                dy=grid.dy,
                wavelength=grid.wl,
                device=grid.device,
            )
        else:
            self.padded_size = self.original_size

        # Compute defocus parameter
        self.defocus_parameter = compute_defocus_parameter(working_distance, objective_focal)

        # Select regime
        self.selected_regime = select_forward_regime(
            object_distance=working_distance,
            focal_length=objective_focal,
            threshold=defocus_threshold,
            method=regime,
        )

        # Log configuration
        self._log_configuration()

        # Initialize components based on regime
        # Placeholders for optional components (needed for module serialization)
        self.objective_lens: Optional["ThinLens"] = None
        self.tube_lens: Optional["ThinLens"] = None
        self.defocus_propagator: Optional["AngularSpectrumPropagator"] = None

        if self.selected_regime == ForwardModelRegime.FULL:
            self._init_full_model()

    def _log_configuration(self) -> None:
        """Log forward model configuration."""
        logger.info("=" * 60)
        logger.info("Microscope Forward Model Configuration")
        logger.info("=" * 60)
        logger.info(f"  Objective focal length: {self.objective_focal * 1e3:.2f} mm")
        logger.info(f"  Tube lens focal length: {self.tube_lens_focal * 1e3:.2f} mm")
        logger.info(f"  Working distance: {self.working_distance * 1e3:.3f} mm")
        logger.info(f"  Numerical aperture: {self.na}")
        logger.info(f"  Medium index: {self.medium_index}")
        logger.info(f"  Defocus parameter δ: {self.defocus_parameter:.6f}")
        logger.info(f"  Selected regime: {self.selected_regime.value}")
        logger.info("=" * 60)

    def _init_full_model(self) -> None:
        """Initialize components for full propagation model."""
        # Import here to avoid circular imports
        from prism.core.optics.thin_lens import ThinLens
        from prism.core.propagators import AngularSpectrumPropagator

        # Use padded grid if padding is enabled, otherwise use original grid
        grid_for_model = self.padded_grid if self.padded_grid is not None else self.grid

        # Objective lens aperture from NA: NA = n * sin(θ) ≈ n * D/(2f) for small angles
        # => D = 2 * NA * f / n
        aperture_diameter = 2 * self.na * self.objective_focal / self.medium_index

        # Create objective lens
        self.objective_lens = ThinLens(
            focal_length=self.objective_focal,
            grid=grid_for_model,
            aperture_diameter=aperture_diameter,
        )

        # Create tube lens (no aperture clipping, larger than beam)
        self.tube_lens = ThinLens(
            focal_length=self.tube_lens_focal,
            grid=grid_for_model,
            aperture_diameter=None,
        )

        # Propagator for defocus (object to focal plane)
        defocus_distance = self.working_distance - self.objective_focal
        if abs(defocus_distance) > 1e-12:  # Only if actually defocused
            self.defocus_propagator = AngularSpectrumPropagator(
                grid_for_model, distance=defocus_distance
            )
        else:
            self.defocus_propagator = None

    def _pad_field(self, field: Tensor) -> Tensor:
        """Zero-pad field for FFT anti-aliasing.

        Parameters
        ----------
        field : Tensor
            Input field of shape (..., H, W).

        Returns
        -------
        Tensor
            Padded field with size based on padding_factor.
        """
        if self.padding_factor <= 1.0:
            return field

        h, w = field.shape[-2:]
        target_h, target_w = self.padded_size

        # Calculate padding for each side (center the original)
        pad_h = target_h - h
        pad_w = target_w - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # F.pad uses (left, right, top, bottom) order
        return torch.nn.functional.pad(
            field,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

    def _crop_field(self, field: Tensor) -> Tensor:
        """Crop padded field back to original size.

        Parameters
        ----------
        field : Tensor
            Padded field.

        Returns
        -------
        Tensor
            Cropped field with original shape.
        """
        if self.padding_factor <= 1.0:
            return field

        h, w = field.shape[-2:]
        target_h, target_w = self.original_size

        # Calculate crop indices (center crop)
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2

        return field[..., start_h : start_h + target_h, start_w : start_w + target_w]

    def _pad_pupil(self, pupil: Tensor) -> Tensor:
        """Pad pupil function to match padded FFT size.

        Pupils are frequency-domain functions. Zero-padding embeds
        the original frequency response in a larger array, maintaining
        the same frequency cutoff.

        Parameters
        ----------
        pupil : Tensor
            Original pupil function.

        Returns
        -------
        Tensor
            Padded pupil function.
        """
        if self.padding_factor <= 1.0:
            return pupil

        # Use same padding logic as _pad_field
        return self._pad_field(pupil)

    def forward(
        self,
        field: Tensor,
        illum_pupil: Tensor,
        detect_pupil: Tensor,
    ) -> Tensor:
        """Forward propagation through microscope.

        Parameters
        ----------
        field : Tensor
            Complex field at object plane.
        illum_pupil : Tensor
            Illumination pupil function in Fourier domain.
        detect_pupil : Tensor
            Detection pupil function in Fourier domain.

        Returns
        -------
        Tensor
            Field at image plane (before intensity conversion).
        """
        if self.selected_regime == ForwardModelRegime.SIMPLIFIED:
            return self._forward_simplified(field, illum_pupil, detect_pupil)
        else:
            return self._forward_full(field, illum_pupil, detect_pupil)

    def _forward_simplified(
        self,
        field: Tensor,
        illum_pupil: Tensor,
        detect_pupil: Tensor,
    ) -> Tensor:
        """Simplified forward model (FFT-based, object at focal plane).

        Implements pad -> FFT -> pupils -> IFFT -> crop workflow
        when padding_factor > 1.0 to prevent wraparound artifacts.

        Parameters
        ----------
        field : Tensor
            Input complex field at object plane.
        illum_pupil : Tensor
            Illumination pupil function.
        detect_pupil : Tensor
            Detection pupil function.

        Returns
        -------
        Tensor
            Output field at image plane.
        """
        # Step 1: Pad input field
        field_padded = self._pad_field(field)

        # Step 2: Pad pupils to match padded size
        illum_padded = self._pad_pupil(illum_pupil)
        detect_padded = self._pad_pupil(detect_pupil)

        # Step 3: Object to back focal plane (Fourier transform)
        field_bfp = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field_padded)))

        # Step 4: Apply illumination and detection pupils
        field_illum = field_bfp * illum_padded
        field_filtered = field_illum * detect_padded

        # Step 5: Back focal plane to image plane (inverse Fourier transform)
        field_image = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(field_filtered)))

        # Step 6: Crop back to original size
        return self._crop_field(field_image)

    def _forward_full(
        self,
        field: Tensor,
        illum_pupil: Tensor,
        detect_pupil: Tensor,
    ) -> Tensor:
        """Full propagation model with lens phases.

        Implements pad -> propagate -> lens -> FFT -> pupils -> IFFT -> lens -> crop
        workflow when padding_factor > 1.0.

        Parameters
        ----------
        field : Tensor
            Input complex field at object plane.
        illum_pupil : Tensor
            Illumination pupil function.
        detect_pupil : Tensor
            Detection pupil function.

        Returns
        -------
        Tensor
            Output field at image plane.
        """
        # Step 1: Pad input field
        field_padded = self._pad_field(field)

        # Step 2: Propagate defocus distance (if any)
        if self.defocus_propagator is not None:
            field_padded = self.defocus_propagator(field_padded)

        # Step 3: Apply objective lens phase
        assert self.objective_lens is not None
        field_padded = self.objective_lens(field_padded)

        # Step 4: Pad pupils and FFT to back focal plane
        illum_padded = self._pad_pupil(illum_pupil)
        detect_padded = self._pad_pupil(detect_pupil)

        field_bfp = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field_padded)))

        # Step 5: Apply pupils
        field_illum = field_bfp * illum_padded
        field_filtered = field_illum * detect_padded

        # Step 6: IFFT to intermediate image plane
        field_intermediate = torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.ifftshift(field_filtered))
        )

        # Step 7: Apply tube lens phase
        assert self.tube_lens is not None
        field_image = self.tube_lens(field_intermediate)

        # Step 8: Crop back to original size
        return self._crop_field(field_image)

    def get_info(self) -> dict[str, Any]:
        """Get forward model information for logging/debugging.

        Returns
        -------
        dict[str, Any]
            Dictionary containing model configuration and state.
        """
        return {
            "regime": self.selected_regime.value,
            "defocus_parameter": self.defocus_parameter,
            "defocus_threshold": self.defocus_threshold,
            "working_distance_mm": self.working_distance * 1e3,
            "objective_focal_mm": self.objective_focal * 1e3,
            "tube_lens_focal_mm": self.tube_lens_focal * 1e3,
            "na": self.na,
            "medium_index": self.medium_index,
        }
