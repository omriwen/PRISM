"""Microscope implementation for SPIDS.

This module provides microscope simulation with support for various illumination
modes including brightfield, darkfield, phase contrast, and DIC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from torch import Tensor

from ..grid import Grid
from ..optics.detector_noise import DetectorNoiseModel
from ..optics.illumination import (
    IlluminationSource,
    IlluminationSourceType,
    create_illumination_field,
)
from ..propagators.angular_spectrum import AngularSpectrumPropagator
from ..propagators.base import CoherenceMode
from .base import InstrumentConfig
from .four_f_base import FourFSystem


if TYPE_CHECKING:
    from prism.core.optics import MicroscopeForwardModel


@dataclass
class MicroscopeConfig(InstrumentConfig):
    """Configuration for microscope systems.

    Attributes:
        numerical_aperture: Objective NA
        magnification: Total magnification
        medium_index: Immersion medium refractive index (1.0 for air, 1.33 for water, 1.515 for oil)
        tube_lens_focal: Tube lens focal length in meters
        working_distance: Working distance (object-to-objective) in meters.
            If None, defaults to f_objective (object at focal plane).
        default_illumination_na: Default NA for illumination (if None, uses detection NA)
        forward_model_regime: Forward model selection ('simplified', 'full', 'auto').
            'auto' (default): Auto-selects based on defocus parameter and threshold.
            'simplified': FFT-based 4f model, correct for in-focus imaging.
            'full': Propagation with lens phases, needed for defocus/z-stacks.
        defocus_threshold: Threshold for auto regime selection. When defocus parameter
            δ = |working_distance - f_obj| / f_obj exceeds this threshold, FULL model
            is selected. Default 0.01 (1% deviation).
        padding_factor: FFT padding factor to prevent wraparound artifacts.
            Default 2.0 (correct by default). Set to 1.0 for no padding.
    """

    numerical_aperture: float = 0.9
    magnification: float = 40.0
    medium_index: float = 1.0  # Air by default
    tube_lens_focal: float = 0.2  # 200mm typical
    working_distance: Optional[float] = None
    default_illumination_na: Optional[float] = None  # For Köhler illumination
    forward_model_regime: str = "auto"  # 'auto' (default), 'simplified', or 'full'
    defocus_threshold: float = 0.01  # 1% deviation threshold
    padding_factor: float = 2.0  # FFT padding factor (1.0 = no padding, 2.0 = recommended)

    def __post_init__(self) -> None:
        """Validate microscope-specific parameters."""
        self.validate()

        # Validate NA vs medium index
        if self.numerical_aperture > self.medium_index:
            raise ValueError(
                f"NA ({self.numerical_aperture}) cannot exceed medium index ({self.medium_index})"
            )

        # Check Nyquist sampling
        min_sampling = self.wavelength / (4 * self.numerical_aperture)
        object_pixel_size = self.pixel_size / self.magnification
        if object_pixel_size > min_sampling:
            raise ValueError(
                f"Undersampling detected: pixel size {object_pixel_size:.2e}m "
                f"exceeds Nyquist limit {min_sampling:.2e}m"
            )

        # Set default illumination NA if not specified
        if self.default_illumination_na is None:
            # Common practice: illumination NA = 0.7-0.9 * detection NA for brightfield
            self.default_illumination_na = 0.8 * self.numerical_aperture

        # Validate padding factor
        if self.padding_factor < 1.0:
            raise ValueError(f"padding_factor must be >= 1.0, got {self.padding_factor}")


class Microscope(FourFSystem):
    """Microscope implementation for near-field imaging.

    Inherits from FourFSystem, providing unified 4f forward model with
    microscope-specific illumination modes and pupil functions.

    Supports various illumination modes:
    - Brightfield: Direct transmitted/reflected light
    - Darkfield: Only scattered light (direct light blocked)
    - Phase contrast: Phase shifts converted to intensity
    - DIC: Differential interference contrast
    - Custom: User-defined illumination/detection pupils
    """

    def __init__(self, config: MicroscopeConfig):
        """Initialize microscope with configuration.

        Args:
            config: Microscope configuration
        """
        # Store microscope-specific parameters before calling super().__init__
        self.na = config.numerical_aperture
        self.magnification = config.magnification
        self.tube_lens_focal = config.tube_lens_focal
        self.default_illumination_na = config.default_illumination_na

        # Compute objective focal length from magnification
        # In a 4f system: M = f_tube / f_obj => f_obj = f_tube / M
        self.objective_focal = self.tube_lens_focal / self.magnification

        # Set working distance (default to focal plane if not specified)
        self.working_distance: float = (
            config.working_distance if config.working_distance is not None else self.objective_focal
        )

        # Forward model regime configuration
        self._forward_model_regime = config.forward_model_regime
        self._defocus_threshold = config.defocus_threshold

        # Initialize FourFSystem base class
        # Note: medium_index passed to FourFSystem for aperture mask generation
        super().__init__(
            config=config,
            padding_factor=config.padding_factor,
            aperture_cutoff_type="na",
            medium_index=config.medium_index,
            noise_model=None,  # Microscope has its own noise handling
        )

        # Set default aperture radius for SPIDS sub-aperture mode
        self._default_aperture_radius = self.pupil_radius_pixels

        # Lazy initialization of forward model (MicroscopeForwardModel for FULL regime)
        self._forward_model: Optional[MicroscopeForwardModel] = None

        # Coordinate grids for SPIDS aperture generation (lazy initialization)
        self._x: Optional[Tensor] = None
        self._y: Optional[Tensor] = None
        self._device: torch.device = torch.device("cpu")

        # Noise model (lazy initialization) - microscope-specific
        self._noise_model: Optional["DetectorNoiseModel"] = None

    @property
    def forward_model(self) -> "MicroscopeForwardModel":
        """Get forward model (lazy initialization).

        The forward model is created on first access to avoid circular imports
        and to ensure the grid is available.

        Returns:
            MicroscopeForwardModel: The configured forward model instance.
        """
        if self._forward_model is None:
            from prism.core.optics import ForwardModelRegime, MicroscopeForwardModel

            regime_map = {
                "auto": ForwardModelRegime.AUTO,
                "simplified": ForwardModelRegime.SIMPLIFIED,
                "full": ForwardModelRegime.FULL,
            }

            self._forward_model = MicroscopeForwardModel(
                grid=self.grid,
                objective_focal=self.objective_focal,
                tube_lens_focal=self.tube_lens_focal,
                working_distance=self.working_distance,
                na=self.na,
                medium_index=self.medium_index,
                regime=regime_map.get(self._forward_model_regime, ForwardModelRegime.AUTO),
                defocus_threshold=self._defocus_threshold,
                padding_factor=cast(MicroscopeConfig, self.config).padding_factor,
            )
        return self._forward_model

    def _create_grid(self) -> Grid:
        """Create high-resolution spatial grid for microscopy.

        Returns:
            Grid configured for object space sampling
        """
        # Object space pixel size (before magnification)
        object_pixel_size = self.config.pixel_size / self.magnification

        return Grid(
            nx=self.config.n_pixels, dx=object_pixel_size, wavelength=self.config.wavelength
        )

    def _select_propagator(self) -> AngularSpectrumPropagator:
        """Select propagator for microscope (always angular spectrum).

        Returns:
            AngularSpectrumPropagator for accurate near-field propagation
        """
        return AngularSpectrumPropagator(self.grid)

    def _create_pupil_function(
        self,
        na: float,
        pupil_type: str = "circular",
        annular_ratio: float = 0.0,
        phase_shift: float = 0.0,
    ) -> torch.Tensor:
        """Create pupil function for given NA and type.

        Args:
            na: Numerical aperture
            pupil_type: Type of pupil ('circular', 'annular', 'phase_ring')
            annular_ratio: Inner/outer radius ratio for annular pupils (0-1)
            phase_shift: Phase shift for phase contrast (radians)

        Returns:
            Complex pupil function
        """
        # Get frequency coordinates
        fx = self.grid.kx
        fy = self.grid.ky

        # NA cutoff in frequency space
        cutoff_freq = na / (self.medium_index * self.config.wavelength)

        # Normalized frequency radius
        r_freq = torch.sqrt(fx**2 + fy**2)
        r_norm = r_freq / cutoff_freq

        if pupil_type == "circular":
            # Simple circular pupil
            pupil = (r_norm <= 1.0).float()

        elif pupil_type == "annular":
            # Annular pupil (for darkfield)
            pupil = ((r_norm > annular_ratio) & (r_norm <= 1.0)).float()

        elif pupil_type == "phase_ring":
            # Phase ring for phase contrast
            ring_inner = 0.6
            ring_outer = 0.8
            pupil = torch.ones_like(r_norm, dtype=torch.complex64)
            ring_mask = (r_norm > ring_inner) & (r_norm < ring_outer)
            # Convert phase_shift to tensor for torch.exp
            phase_shift_tensor = torch.tensor(phase_shift, dtype=r_norm.dtype, device=r_norm.device)
            pupil[ring_mask] *= torch.exp(1j * phase_shift_tensor)
            pupil[r_norm > 1.0] = 0

        else:
            raise ValueError(f"Unknown pupil type: {pupil_type}")

        # Convert to complex if needed
        if not torch.is_complex(pupil):
            pupil = pupil.to(torch.complex64)

        return pupil

    def _create_pupils(
        self,
        illumination_mode: Optional[str] = None,
        illumination_params: Optional[dict] = None,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Create illumination and detection pupil functions.

        This implements the abstract method from FourFSystem, providing
        microscope-specific pupil configurations for various illumination modes.

        Args:
            illumination_mode: Illumination type ('brightfield', 'darkfield', 'phase',
                'dic', 'custom'). Defaults to 'brightfield' if None.
            illumination_params: Additional parameters for illumination mode

        Returns:
            Tuple of (illumination_pupil, detection_pupil) as complex tensors
        """
        # Default to brightfield
        if illumination_mode is None:
            illumination_mode = "brightfield"

        params = illumination_params or {}

        if illumination_mode == "brightfield":
            # Köhler illumination with partial coherence
            default_na = self.default_illumination_na or (0.8 * self.na)
            illum_na = params.get("illumination_na", default_na)
            illum_pupil = self._create_pupil_function(illum_na, "circular")
            detect_pupil = self._create_pupil_function(self.na, "circular")

        elif illumination_mode == "darkfield":
            # Annular illumination, central detection
            annular_ratio = params.get("annular_ratio", 0.8)
            illum_na = params.get("illumination_na", self.na)
            illum_pupil = self._create_pupil_function(illum_na, "annular", annular_ratio)
            detect_pupil = self._create_pupil_function(self.na, "circular")

        elif illumination_mode == "phase":
            # Phase contrast with phase ring
            phase_shift = params.get("phase_shift", np.pi / 2)
            illum_na = self.default_illumination_na or (0.8 * self.na)
            illum_pupil = self._create_pupil_function(illum_na, "circular")
            detect_pupil = self._create_pupil_function(
                self.na, "phase_ring", phase_shift=phase_shift
            )

        elif illumination_mode == "dic":
            # Differential interference contrast (simplified)
            shear = params.get("shear", 1.0)  # Shear amount in pixels
            illum_pupil = self._create_dic_pupils(shear)
            detect_pupil = self._create_pupil_function(self.na, "circular")

        elif illumination_mode == "custom":
            # User-provided pupils
            custom_illum = params.get("illumination_pupil")
            custom_detect = params.get("detection_pupil")
            if custom_illum is None or custom_detect is None:
                raise ValueError("Custom mode requires both illumination_pupil and detection_pupil")
            # After None check, assign to illum_pupil and detect_pupil
            illum_pupil = custom_illum
            detect_pupil = custom_detect

        else:
            # Default to brightfield
            illum_na = self.default_illumination_na or (0.8 * self.na)
            illum_pupil = self._create_pupil_function(illum_na, "circular")
            detect_pupil = self._create_pupil_function(self.na, "circular")

        return illum_pupil, detect_pupil

    def _create_dic_pupils(self, shear: float) -> torch.Tensor:
        """Create DIC illumination pupils (two displaced pupils).

        Args:
            shear: Lateral shear in pixels

        Returns:
            DIC illumination pupil
        """
        fx = self.grid.kx
        fy = self.grid.ky

        # Create two displaced pupils
        illum_na = self.default_illumination_na or (0.8 * self.na)
        cutoff_freq = illum_na / (self.medium_index * self.config.wavelength)

        # Shift amount in frequency space
        shift_freq = shear / (self.config.n_pixels * self.grid.dx)

        # Two circular pupils displaced along x
        r1 = torch.sqrt((fx - shift_freq) ** 2 + fy**2)
        r2 = torch.sqrt((fx + shift_freq) ** 2 + fy**2)

        pupil1 = (r1 <= cutoff_freq).float()
        pupil2 = (r2 <= cutoff_freq).float()

        # Combine with phase shift for interference
        phase_shift = torch.tensor(np.pi / 2, dtype=fx.dtype, device=fx.device)
        dic_pupil = pupil1.to(torch.complex64) + pupil2.to(torch.complex64) * torch.exp(
            1j * phase_shift
        )

        return dic_pupil / 2.0  # Normalize

    def compute_psf(
        self,
        z_slices: Optional[int] = None,
        illumination_mode: str = "brightfield",
        illumination_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute 2D or 3D PSF for microscope.

        Args:
            z_slices: Number of z-slices for 3D PSF (None for 2D)
            illumination_mode: Type of illumination
            illumination_params: Additional illumination parameters

        Returns:
            2D or 3D PSF tensor, normalized to max=1
        """
        # Get illumination and detection pupils
        illum_pupil, detect_pupil = self._create_pupils(illumination_mode, illumination_params)

        # Effective pupil is product for incoherent imaging
        # For coherent/partially coherent, this would be different
        effective_pupil = illum_pupil * detect_pupil  # type: ignore[operator]

        if z_slices is None:
            # 2D PSF at focus
            psf_field = torch.fft.ifft2(torch.fft.ifftshift(effective_pupil))
            psf = torch.abs(psf_field) ** 2
        else:
            # 3D PSF stack
            z_range = self._calculate_depth_of_field()
            z_positions = torch.linspace(-z_range / 2, z_range / 2, z_slices)
            psf_stack = []

            for z in z_positions:
                # Add defocus phase
                defocus_phase = self._defocus_phase(z.item())
                defocused_pupil = effective_pupil * defocus_phase

                # Compute PSF at this z
                psf_field = torch.fft.ifft2(torch.fft.ifftshift(defocused_pupil))
                psf_stack.append(torch.abs(psf_field) ** 2)

            psf = torch.stack(psf_stack, dim=0)

        # Normalize
        return psf / psf.max()

    def _defocus_phase(self, z: float) -> torch.Tensor:
        """Calculate defocus phase factor.

        Args:
            z: Defocus distance in meters

        Returns:
            Complex phase factor
        """
        fx = self.grid.kx
        fy = self.grid.ky
        k = 2 * np.pi * self.medium_index / self.config.wavelength

        # Normalized frequency
        f_norm = torch.sqrt(fx**2 + fy**2) * self.config.wavelength / self.medium_index

        # Phase factor for defocus (valid only within NA)
        kz = k * torch.sqrt(torch.clamp(1 - f_norm**2, min=0))
        phase = torch.exp(1j * kz * z)

        return phase

    def _calculate_depth_of_field(self) -> float:
        """Calculate depth of field for microscope.

        Returns:
            Depth of field in meters
        """
        # Axial resolution (depth of field)
        return 2 * self.config.wavelength * self.medium_index / (self.na**2)

    @property
    def resolution_limit(self) -> float:
        """Abbe diffraction limit.

        Returns:
            Lateral resolution limit in meters
        """
        return 0.61 * self.config.wavelength / self.na

    def forward(  # type: ignore[override]
        self,
        field: torch.Tensor,
        illumination_mode: Optional[str] = None,
        illumination_params: Optional[dict] = None,
        add_noise: bool = False,
        use_unified_model: bool = True,
        input_mode: str = "auto",
        input_pixel_size: Optional[float] = None,
        aperture_center: Optional[List[float]] = None,
        aperture_radius: Optional[float] = None,
        illumination_center: Optional[List[float]] = None,
        illumination_radius: Optional[float] = None,
        illumination_source_type: IlluminationSourceType = IlluminationSourceType.POINT,
        coherence_mode: CoherenceMode = CoherenceMode.COHERENT,
        source_intensity: Optional[torch.Tensor] = None,
        n_source_points: int = 100,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward imaging through microscope.

        This method supports both SIMPLIFIED (4f FFT) and FULL (propagation chain)
        forward models, selected based on the configured regime. It also handles
        SPIDS sub-aperture mode when aperture_center is specified, and scanning
        illumination mode when illumination_center is specified.

        Parameters
        ----------
        field : torch.Tensor
            Input field at object plane. Shape: (B, C, H, W) or (H, W).
        illumination_mode : str, optional
            Illumination mode (default: "brightfield").
            Options: 'brightfield', 'darkfield', 'phase', 'dic', 'custom'
        illumination_params : dict, optional
            Parameters for illumination mode.
        add_noise : bool
            Whether to add detector noise (default: False).
        use_unified_model : bool
            Whether to use MicroscopeForwardModel (default: True).
            If False, uses legacy FFT-only model.
        input_mode : str
            How to interpret the input field:
            - 'intensity': Field is I = |E|^2, converted via sqrt(I)
            - 'amplitude': Field is |E|, used directly
            - 'complex': Field is already complex E = |E|*exp(i*phi)
            - 'auto' (default): Auto-detect from dtype (warns if ambiguous)
        input_pixel_size : float, optional
            Physical size of input pixels (meters). Used for FOV validation.
            Recommended: pass target.config.pixel_size to enable validation.
        aperture_center : List[float], optional
            Center position [y, x] for sub-aperture in k-space (pixels from DC).
            If provided, uses a sub-aperture at this position instead of full pupil.
            Used for SPIDS synthetic aperture measurements.
            **Mutually exclusive** with illumination_center.
        aperture_radius : float, optional
            Radius of sub-aperture in pixels. Only used when aperture_center is set.
            Defaults to pupil_radius_pixels if not specified.
        illumination_center : List[float], optional
            k-space center [ky, kx] in 1/meters for scanning illumination.
            If provided, uses scanning illumination mode instead of scanning aperture.
            This is equivalent to aperture_center for point sources by reciprocity.
            **Mutually exclusive** with aperture_center.
        illumination_radius : float, optional
            For finite-size sources (GAUSSIAN, CIRCULAR), this is the k-space
            width parameter (sigma for Gaussian, radius for circular) in 1/m.
            Only used when illumination_center is set.
        illumination_source_type : IlluminationSourceType
            Type of illumination source for scanning illumination mode:
            - POINT (default): Tilted plane wave, equivalent to scanning aperture
            - GAUSSIAN: Gaussian intensity profile (partial coherence)
            - CIRCULAR: Circular (top-hat) profile
        coherence_mode : CoherenceMode, default=CoherenceMode.COHERENT
            Illumination coherence mode:
            - COHERENT: Standard coherent amplitude transfer (laser illumination)
            - INCOHERENT: OTF-based propagation for fluorescence imaging
            - PARTIALLY_COHERENT: Extended source integration for LED brightfield
        source_intensity : torch.Tensor, optional
            Source intensity distribution for PARTIALLY_COHERENT mode.
            Required when coherence_mode=PARTIALLY_COHERENT.
        n_source_points : int, default=100
            Number of source points for PARTIALLY_COHERENT mode sampling.

        Returns
        -------
        torch.Tensor
            Simulated measurement intensity.

        Raises
        ------
        ValueError
            If both aperture_center and illumination_center are specified.

        Notes
        -----
        Most targets generate intensity patterns (transmission 0-1).
        For physically correct simulation, use input_mode='intensity'
        to automatically convert via sqrt().

        **SPIDS Scanning Modes**:

        - **Scanning aperture** (aperture_center): Traditional synthetic aperture.
          Sub-aperture scans k-space, object illumination is uniform.
        - **Scanning illumination** (illumination_center): FPM-style synthetic aperture.
          Tilted illumination shifts object spectrum, detection at DC.

        For point sources, both modes are mathematically equivalent by the
        Fourier shift theorem. Finite sources (GAUSSIAN, CIRCULAR) introduce
        partial coherence effects only available in scanning illumination mode.

        The forward model regime (SIMPLIFIED vs FULL) is selected during
        initialization based on the defocus parameter. SIMPLIFIED uses the
        FourFSystem base class implementation (FFT-based), while FULL uses
        MicroscopeForwardModel with explicit lens phases.

        For incoherent and partially coherent modes, the base class
        FourFSystem implementation is used.
        """
        # For non-coherent modes, delegate to base class FourFSystem
        # (which handles OTF and extended source propagation)
        if coherence_mode != CoherenceMode.COHERENT:
            return super().forward(
                field,
                illumination_mode=illumination_mode,
                illumination_params=illumination_params,
                add_noise=add_noise,
                input_mode=input_mode,
                input_pixel_size=input_pixel_size,
                coherence_mode=coherence_mode,
                source_intensity=source_intensity,
                n_source_points=n_source_points,
                **kwargs,
            )

        # Check for mutually exclusive parameters
        if aperture_center is not None and illumination_center is not None:
            raise ValueError(
                "aperture_center and illumination_center are mutually exclusive. "
                "Use aperture_center for scanning aperture mode (k-space position in pixels), "
                "or illumination_center for scanning illumination mode (k-space position in 1/m)."
            )

        # Validate input with input_mode support
        field = self.validate_field(
            field,
            input_mode=input_mode,
            input_pixel_size=input_pixel_size,
        )

        # SPIDS scanning illumination mode: tilted illumination, detect at DC
        if illumination_center is not None:
            return self._forward_scanning_illumination(
                field=field,
                illum_center=illumination_center,
                illum_radius=illumination_radius,
                source_type=illumination_source_type,
                add_noise=add_noise,
            )

        # SPIDS scanning aperture mode: uniform illumination, sub-aperture in k-space
        if aperture_center is not None:
            return self._forward_prism_aperture(field, aperture_center, aperture_radius, add_noise)

        # Get illumination and detection pupils
        illum_pupil, detect_pupil = self._create_pupils(illumination_mode, illumination_params)

        # Ensure pupils are on the same device as the input field
        if illum_pupil is not None:
            illum_pupil = illum_pupil.to(field.device)
        if detect_pupil is not None:
            detect_pupil = detect_pupil.to(field.device)

        # Forward through optical system
        if use_unified_model:
            # Use MicroscopeForwardModel (handles both SIMPLIFIED and FULL regimes)
            field_image = self.forward_model(field, illum_pupil, detect_pupil)
        else:
            # Legacy FFT-only implementation (for backward compatibility)
            # This is equivalent to FourFSystem.forward() for SIMPLIFIED regime
            assert illum_pupil is not None, "illum_pupil required for legacy forward"
            assert detect_pupil is not None, "detect_pupil required for legacy forward"
            field_image = self._forward_legacy(field, illum_pupil, detect_pupil)

        # Convert to intensity
        if illumination_mode == "phase" or illumination_mode == "dic":
            # For phase/DIC, we need to preserve phase information in intensity
            # This is a simplified model
            intensity = torch.abs(field_image) ** 2
            # Add phase contribution for visualization
            phase_contrast = torch.real(field_image) * 0.5 + 0.5
            image = intensity * phase_contrast
        else:
            # Standard intensity detection
            image = torch.abs(field_image) ** 2

        # Add noise if requested
        if add_noise:
            image = self._add_detector_noise(image)

        return image

    def _forward_legacy(
        self,
        field: torch.Tensor,
        illum_pupil: torch.Tensor,
        detect_pupil: torch.Tensor,
    ) -> torch.Tensor:
        """Legacy forward model (FFT-only, for backward compatibility).

        This is the original implementation that assumes the object is exactly
        at the front focal plane of the objective. Use for backward compatibility
        or when you need the simpler model regardless of defocus.

        Args:
            field: Complex field at object plane
            illum_pupil: Illumination pupil function in Fourier domain
            detect_pupil: Detection pupil function in Fourier domain

        Returns:
            Complex field at image plane (before intensity conversion)
        """
        # Object to back focal plane (Fourier transform)
        # Center the spectrum to match pupil coordinates (DC at [N/2,N/2])
        field_bfp = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field)))

        # Apply illumination transfer function
        field_illum = field_bfp * illum_pupil

        # Apply detection transfer function (objective pupil)
        field_filtered = field_illum * detect_pupil

        # Back focal plane to image plane (inverse Fourier transform)
        # Uncenter spectrum before inverse FFT, then center output
        field_image = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(field_filtered)))

        return cast(Tensor, field_image)

    def _add_detector_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add realistic detector noise.

        Args:
            image: Clean image

        Returns:
            Noisy image
        """
        # Shot noise (Poisson approximation using Gaussian)
        if image.max() > 0:
            # Scale to photon counts (arbitrary scaling)
            photon_scale = 1000
            image_photons = image * photon_scale

            # Poisson noise approximated by Gaussian with variance = mean
            shot_noise = torch.randn_like(image) * torch.sqrt(image_photons + 1) / photon_scale
        else:
            shot_noise = torch.zeros_like(image)

        # Read noise (Gaussian)
        read_noise = torch.randn_like(image) * 0.01 * image.max()

        # Dark current (small constant + Gaussian)
        dark_current = torch.randn_like(image) * 0.002 * image.max()

        # Combine noises
        noisy_image = image + shot_noise + read_noise + dark_current

        # Ensure non-negative
        return torch.clamp(noisy_image, min=0)

    def _forward_prism_aperture(
        self,
        field: torch.Tensor,
        aperture_center: List[float],
        aperture_radius: Optional[float],
        add_noise: bool,
    ) -> torch.Tensor:
        """Forward pass through a sub-aperture for SPIDS measurements.

        This method implements the forward model for synthetic aperture imaging,
        where we sample different positions in k-space (pupil plane) with
        sub-apertures.

        Args:
            field: Input field (already validated)
            aperture_center: [y, x] center position in k-space (pixels from DC)
            aperture_radius: Sub-aperture radius in pixels (None uses pupil_radius_pixels)
            add_noise: Whether to add detector noise

        Returns:
            Intensity measurement through the sub-aperture
        """
        # Handle input dimensions
        squeeze_output = False
        if field.ndim == 2:
            field = field.unsqueeze(0).unsqueeze(0)
            squeeze_output = True
        elif field.ndim == 3:
            field = field.unsqueeze(0)
            squeeze_output = True

        # Propagate to k-space
        field_kspace = self.propagate_to_kspace(field)

        # Generate sub-aperture mask at specified position
        # Ensure mask is on the same device as the field
        mask = self.generate_aperture_mask(aperture_center, aperture_radius)
        mask = mask.to(field_kspace.device)

        # Apply mask in k-space
        field_kspace_masked = field_kspace * mask

        # Propagate back to spatial domain
        image = self.propagate_to_spatial(field_kspace_masked)

        # Add noise if requested
        if add_noise:
            image = self._add_detector_noise(image)

        # Restore original dimensions if needed
        if squeeze_output:
            image = image.squeeze()

        return image

    def _forward_scanning_illumination(
        self,
        field: Tensor,
        illum_center: Union[List[float], Tuple[float, float]],
        illum_radius: Optional[float] = None,
        source_type: IlluminationSourceType = IlluminationSourceType.POINT,
        add_noise: bool = False,
    ) -> Tensor:
        """Forward pass using scanning illumination mode.

        This implements an alternative forward model for synthetic aperture
        imaging using tilted illumination instead of scanning apertures.
        The key difference is:

        - **Scanning aperture**: Object is illuminated uniformly, sub-aperture
          samples different k-space regions.
        - **Scanning illumination**: Object is illuminated with tilted plane wave
          (or finite source), detection aperture stays at DC.

        For point-like illumination (tilted plane waves), both approaches are
        mathematically equivalent by the Fourier shift theorem.

        Parameters
        ----------
        field : Tensor
            Input complex field at object plane. Shape: (H, W) or (B, C, H, W).
        illum_center : List[float] or Tuple[float, float]
            k-space center [ky, kx] in 1/meters for the illumination.
            This defines the tilt angle of the illumination.
        illum_radius : float, optional
            For finite-size sources (GAUSSIAN, CIRCULAR), this is the k-space
            width parameter (sigma for Gaussian, radius for circular) in 1/m.
            Ignored for POINT sources.
        source_type : IlluminationSourceType
            Type of illumination source:
            - POINT: Tilted plane wave (default, equivalent to scanning aperture)
            - GAUSSIAN: Gaussian intensity profile (partial coherence)
            - CIRCULAR: Circular (top-hat) profile
        add_noise : bool
            Whether to add detector noise (default: False).

        Returns
        -------
        Tensor
            Intensity measurement through the scanning illumination system.

        Notes
        -----
        The scanning illumination forward model:

        1. Create illumination field: E_illum = A(x,y) × exp(i·2π·(kx·x + ky·y))
           where A(x,y) is the spatial envelope (unity for point source).

        2. Illuminate object: E_obj_illum = E_obj × E_illum

        3. FFT to k-space: Ẽ = FFT(E_obj_illum)
           This shifts the object spectrum by (kx, ky).

        4. Apply detection aperture at DC: Ẽ_filtered = Ẽ × H_detect(k)
           The detection aperture is centered at DC.

        5. IFFT and take intensity: I = |IFFT(Ẽ_filtered)|²

        For POINT sources, this is equivalent to scanning aperture by reciprocity:
        - Aperture at position +k samples O(k)
        - Illumination at -k shifts spectrum so O(k) appears at DC

        Examples
        --------
        >>> # Point source illumination (equivalent to scanning aperture)
        >>> intensity = microscope._forward_scanning_illumination(
        ...     field=obj_field,
        ...     illum_center=[0.1e6, 0.05e6],  # k-space position
        ...     source_type=IlluminationSourceType.POINT,
        ... )

        >>> # Gaussian source (partial coherence)
        >>> intensity = microscope._forward_scanning_illumination(
        ...     field=obj_field,
        ...     illum_center=[0.1e6, 0.0],
        ...     illum_radius=0.02e6,  # k-space sigma
        ...     source_type=IlluminationSourceType.GAUSSIAN,
        ... )

        See Also
        --------
        _forward_prism_aperture : Scanning aperture forward model
        create_illumination_field : Create illumination field from source
        """
        # Handle input dimensions
        squeeze_output = False
        if field.ndim == 2:
            field = field.unsqueeze(0).unsqueeze(0)
            squeeze_output = True
        elif field.ndim == 3:
            field = field.unsqueeze(0)
            squeeze_output = True

        device = field.device

        # Create illumination source configuration
        source_kwargs: Dict[str, Any] = {
            "source_type": source_type,
            "k_center": list(illum_center),
        }

        # Add size parameters for finite sources
        if source_type == IlluminationSourceType.GAUSSIAN:
            if illum_radius is None:
                raise ValueError("GAUSSIAN source requires illum_radius (sigma)")
            source_kwargs["sigma"] = illum_radius
        elif source_type == IlluminationSourceType.CIRCULAR:
            if illum_radius is None:
                raise ValueError("CIRCULAR source requires illum_radius")
            source_kwargs["radius"] = illum_radius

        source = IlluminationSource(**source_kwargs)

        # Create illumination field (envelope × phase tilt)
        illum_field = create_illumination_field(self.grid, source, device=device)

        # Apply illumination to object field
        # For batched input: field is (B, C, H, W), illum_field is (H, W)
        field_illuminated = field * illum_field.unsqueeze(0).unsqueeze(0)

        # Propagate to k-space (FFT with proper centering)
        # The illumination phase tilt shifts the object spectrum
        field_kspace = self.propagate_to_kspace(field_illuminated)

        # Create detection aperture at DC
        # For scanning illumination, we detect at DC (center of k-space)
        detection_mask = self.generate_aperture_mask(
            center=[0.0, 0.0],  # Detection at DC
            radius=self.pupil_radius_pixels,
        )
        detection_mask = detection_mask.to(device)

        # Apply detection aperture
        field_kspace_filtered = field_kspace * detection_mask

        # Propagate back to spatial domain
        image = self.propagate_to_spatial(field_kspace_filtered)

        # Add noise if requested
        if add_noise:
            image = self._add_detector_noise(image)

        # Restore original dimensions
        if squeeze_output:
            image = image.squeeze()

        return image

    def get_info(self) -> dict:
        """Get microscope information summary.

        Returns:
            Dictionary with microscope parameters
        """
        info = super().get_info()
        info.update(
            {
                "numerical_aperture": self.na,
                "magnification": self.magnification,
                "medium_index": self.medium_index,
                "objective_focal_mm": self.objective_focal * 1e3,
                "tube_lens_focal_mm": self.tube_lens_focal * 1e3,
                "working_distance_mm": self.working_distance * 1e3,
                "depth_of_field": self._calculate_depth_of_field(),
                "object_pixel_size": self.config.pixel_size / self.magnification,
                "default_illumination_na": self.default_illumination_na,
                "forward_model_regime": self._forward_model_regime,
            }
        )
        return info

    # =========================================================================
    # SPIDS-Compatible Methods for MeasurementSystem Integration
    # =========================================================================
    #
    # These methods enable the Microscope to work with MeasurementSystem for
    # progressive synthetic aperture reconstruction. The k-space (pupil plane)
    # is sampled with sub-apertures at different positions.

    @property
    def x(self) -> Tensor:
        """Spatial x coordinates centered at image center (pixels).

        Used for aperture mask generation in k-space.

        Returns:
            Tensor of shape (1, n_pixels) with values from -n//2 to n//2-1
        """
        if self._x is None:
            n = self.config.n_pixels
            self._x = torch.arange(0, n, device=self._device).unsqueeze(0) - n // 2
        return self._x

    @property
    def y(self) -> Tensor:
        """Spatial y coordinates centered at image center (pixels).

        Used for aperture mask generation in k-space.

        Returns:
            Tensor of shape (n_pixels, 1) with values from -n//2 to n//2-1
        """
        if self._y is None:
            n = self.config.n_pixels
            self._y = torch.arange(0, n, device=self._device).unsqueeze(1) - n // 2
        return self._y

    @property
    def noise_model(self) -> Optional["DetectorNoiseModel"]:
        """Get noise model for SPIDS measurements.

        Returns:
            DetectorNoiseModel instance if configured, None otherwise.
        """
        return self._noise_model

    @noise_model.setter
    def noise_model(self, model: Optional["DetectorNoiseModel"]) -> None:
        """Set noise model for SPIDS measurements."""
        self._noise_model = model

    @property
    def pupil_radius_pixels(self) -> float:
        """Get the objective pupil radius in pixels.

        This is the radius of the full NA aperture in the pupil plane,
        expressed in pixel units. Used as the default sub-aperture radius
        for SPIDS sampling.

        Returns:
            Pupil radius in pixels
        """
        # The NA defines the cutoff frequency: f_cutoff = NA / (n * lambda)
        # In pixel units, this maps to a radius in the FFT grid
        cutoff_freq = self.na / (self.medium_index * self.config.wavelength)
        # Convert to pixels: freq * (N * dx) where dx is object pixel size
        object_pixel_size = self.config.pixel_size / self.magnification
        radius_pixels = cutoff_freq * (self.config.n_pixels * object_pixel_size)
        return radius_pixels

    # Note: propagate_to_kspace() and propagate_to_spatial() are inherited from FourFSystem
    # They provide the correct FFT workflow with proper fftshift operations

    def generate_aperture_mask(
        self,
        center: Optional[List[float]] = None,
        radius: Optional[float] = None,
    ) -> Tensor:
        """Generate a circular sub-aperture mask at specified k-space position.

        Creates a circular aperture in the pupil plane (k-space) at the
        specified center position. This enables SPIDS synthetic aperture
        construction by sampling different regions of k-space.

        Args:
            center: Center position [y, x] in pixels from DC (0,0).
                   Defaults to [0.0, 0.0] (centered on DC).
            radius: Sub-aperture radius in pixels.
                   Defaults to pupil_radius_pixels (full NA aperture).

        Returns:
            Boolean mask of shape (n_pixels, n_pixels)

        Example:
            >>> # Create centered aperture (conventional imaging)
            >>> mask = microscope.generate_aperture_mask([0, 0])
            >>>
            >>> # Create off-center sub-aperture for SPIDS
            >>> mask = microscope.generate_aperture_mask([10, 5], radius=15)
        """
        if center is None:
            center = [0.0, 0.0]

        if radius is None:
            radius = self.pupil_radius_pixels

        cy, cx = center

        # Distance from center in pixel coordinates
        dist = torch.sqrt((self.y - cy) ** 2 + (self.x - cx) ** 2)

        # Create circular mask
        mask = dist <= radius

        return mask.squeeze()

    def generate_aperture_masks(
        self,
        centers: Union[Tensor, List[List[float]]],
        radius: Optional[float] = None,
    ) -> Tensor:
        """Generate multiple aperture masks efficiently.

        Args:
            centers: List of [y, x] centers or Tensor of shape (N, 2)
            radius: Sub-aperture radius in pixels.
                   Defaults to pupil_radius_pixels.

        Returns:
            Boolean masks of shape (N, n_pixels, n_pixels)

        Example:
            >>> centers = [[0, 0], [10, 10], [-10, 10]]
            >>> masks = microscope.generate_aperture_masks(centers, radius=15)
            >>> print(masks.shape)  # (3, 512, 512)
        """
        if radius is None:
            radius = self.pupil_radius_pixels

        # Convert to tensor if list
        if isinstance(centers, list):
            centers_tensor = torch.tensor(centers, dtype=torch.float32, device=self._device)
        else:
            centers_tensor = centers.to(self._device)

        n_centers = centers_tensor.shape[0]

        # Broadcast computation for efficiency
        # centers_tensor: (N, 2) -> cy: (N, 1, 1), cx: (N, 1, 1)
        cy = centers_tensor[:, 0].view(n_centers, 1, 1)
        cx = centers_tensor[:, 1].view(n_centers, 1, 1)

        # x: (1, n_pixels), y: (n_pixels, 1) -> broadcast to (N, n_pixels, n_pixels)
        x_grid = self.x.unsqueeze(0)  # (1, 1, n_pixels)
        y_grid = self.y.unsqueeze(0)  # (1, n_pixels, 1)

        # Distance from each center
        dist = torch.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)

        # Create circular masks
        masks = dist <= radius

        return masks

    def generate_illumination_pattern(
        self,
        k_center: Union[List[float], Tuple[float, float]],
        source_type: IlluminationSourceType = IlluminationSourceType.POINT,
        k_width: Optional[float] = None,
    ) -> Tensor:
        """Generate illumination field for visualization and debugging.

        Creates the complex illumination field that would be used in scanning
        illumination mode. Useful for visualizing the illumination pattern
        and understanding how different source types affect the illumination.

        Parameters
        ----------
        k_center : List[float] or Tuple[float, float]
            k-space center [ky, kx] in 1/meters for the illumination.
            This defines the tilt angle of the illumination.
        source_type : IlluminationSourceType
            Type of illumination source:
            - POINT (default): Tilted plane wave (complex exponential)
            - GAUSSIAN: Gaussian envelope × phase tilt
            - CIRCULAR: Circular envelope × phase tilt
        k_width : float, optional
            For finite-size sources (GAUSSIAN, CIRCULAR), this is the k-space
            width parameter (sigma for Gaussian, radius for circular) in 1/m.
            Required for GAUSSIAN and CIRCULAR types.

        Returns
        -------
        Tensor
            Complex illumination field of shape (n_pixels, n_pixels).

        Examples
        --------
        >>> # Point source (tilted plane wave)
        >>> illum = microscope.generate_illumination_pattern(
        ...     k_center=[0.1e6, 0.0],
        ...     source_type=IlluminationSourceType.POINT,
        ... )
        >>> print(illum.shape)
        torch.Size([512, 512])

        >>> # Gaussian source (partial coherence)
        >>> illum = microscope.generate_illumination_pattern(
        ...     k_center=[0.1e6, 0.0],
        ...     source_type=IlluminationSourceType.GAUSSIAN,
        ...     k_width=0.02e6,  # k-space sigma
        ... )

        >>> # Visualize the illumination intensity
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(torch.abs(illum)**2)
        >>> plt.title("Illumination Intensity")
        >>> plt.show()

        Notes
        -----
        For POINT sources, the illumination field is a pure complex exponential:
        E(x,y) = exp(i * 2π * (kx*x + ky*y))

        For finite sources, the field is modulated by an envelope:
        E(x,y) = A(x,y) * exp(i * 2π * (kx*x + ky*y))

        where A(x,y) is a Gaussian or circular profile in spatial domain.
        """
        # Build source configuration
        source_kwargs: Dict[str, Any] = {
            "source_type": source_type,
            "k_center": list(k_center),
        }

        if source_type == IlluminationSourceType.GAUSSIAN:
            if k_width is None:
                raise ValueError("GAUSSIAN source requires k_width (sigma)")
            source_kwargs["sigma"] = k_width
        elif source_type == IlluminationSourceType.CIRCULAR:
            if k_width is None:
                raise ValueError("CIRCULAR source requires k_width (radius)")
            source_kwargs["radius"] = k_width

        source = IlluminationSource(**source_kwargs)

        # Create and return the illumination field
        return create_illumination_field(self.grid, source, device=self._device)

    def to(self, device: torch.device) -> "Microscope":
        """Move microscope to specified device.

        Args:
            device: Target device (e.g., torch.device("cuda"))

        Returns:
            Self for method chaining
        """
        self._device = device

        # Move coordinate grids if initialized
        if self._x is not None:
            self._x = self._x.to(device)
        if self._y is not None:
            self._y = self._y.to(device)

        # Move noise model if present
        if self._noise_model is not None and hasattr(self._noise_model, "to"):
            self._noise_model.to(device)

        return self
