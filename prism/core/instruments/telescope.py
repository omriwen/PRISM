"""Telescope instrument implementation for SPIDS.

This module provides a clean telescope implementation that follows the unified
Instrument interface without SPIDS-specific measurement logic or nn.Module
inheritance.

The Telescope class implements pure optical physics:
- Point spread function (PSF) computation
- Fraunhofer diffraction propagation
- Aperture mask generation
- Resolution limits

For SPIDS-specific progressive measurement accumulation, use MeasurementSystem
which wraps any Instrument instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union

import torch
from torch import Tensor

from prism.models.noise import ShotNoise
from prism.utils.transforms import fft

from ..apertures import Aperture, CircularAperture, create_aperture
from ..grid import Grid
from ..optics.detector_noise import DetectorNoiseModel
from ..propagators import FraunhoferPropagator
from ..propagators.base import CoherenceMode, Propagator
from .base import InstrumentConfig
from .four_f_base import FourFSystem


@dataclass
class TelescopeConfig(InstrumentConfig):
    """Configuration for telescope optical systems.

    Inherits from InstrumentConfig:
        wavelength: float = 550e-9 (meters)
        n_pixels: int = 1024
        pixel_size: float = 6.5e-6 (meters)
        grid_size: Optional[float] = None

    Telescope-specific attributes:
        aperture_diameter: Physical aperture diameter in meters (for resolution calculations)
        aperture_radius_pixels: Aperture mask radius in pixels (for mask generation)
        focal_length: Telescope focal length in meters (optional)
        snr: Signal-to-noise ratio in dB (None for noiseless)
        aperture_type: Type of aperture ('circular', 'hexagonal', 'obscured')
        aperture_kwargs: Additional aperture-specific parameters
    """

    aperture_diameter: Optional[float] = None
    aperture_radius_pixels: float = 10.0
    focal_length: Optional[float] = None
    snr: Optional[float] = None
    aperture_type: str = "circular"
    aperture_kwargs: Optional[dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate telescope configuration after initialization."""
        if self.aperture_kwargs is None:
            self.aperture_kwargs = {}
        self.validate()

    def validate(self) -> None:
        """Validate telescope-specific parameters."""
        super().validate()
        if self.aperture_radius_pixels <= 0:
            raise ValueError(f"Aperture radius must be positive, got {self.aperture_radius_pixels}")
        if self.aperture_diameter is not None and self.aperture_diameter <= 0:
            raise ValueError(f"Aperture diameter must be positive, got {self.aperture_diameter}")
        if self.focal_length is not None and self.focal_length <= 0:
            raise ValueError(f"Focal length must be positive, got {self.focal_length}")
        if self.snr is not None and self.snr <= 0:
            raise ValueError(f"SNR must be positive, got {self.snr}")
        if self.aperture_type not in ("circular", "hexagonal", "obscured"):
            raise ValueError(f"Unknown aperture type: {self.aperture_type}")


class Telescope(FourFSystem):
    """Telescope optical system - inherits from FourFSystem base.

    This class implements the optical physics of a telescope using the unified
    FourFSystem infrastructure. Telescopes use a single aperture as the detection
    pupil (no separate illumination pupil).

    Key characteristics:
    - Inherits from FourFSystem (unified 4f implementation)
    - Config-only initialization (no old-style parameters)
    - Standard forward() signature matching Instrument.forward()
    - Pure optical physics (PSF, propagation, aperture masks)

    The SPIDS-specific measurement accumulation logic is handled by
    MeasurementSystem, which wraps this (or any other) Instrument.

    Attributes:
        config: TelescopeConfig with all optical parameters
        aperture: Aperture instance for mask generation
        noise_model: Optional noise model (ShotNoise or None)

    Example:
        >>> config = TelescopeConfig(
        ...     n_pixels=512,
        ...     aperture_radius_pixels=25,
        ...     aperture_diameter=8.2,  # VLT 8.2m
        ...     wavelength=550e-9,
        ...     snr=40.0
        ... )
        >>> telescope = Telescope(config)
        >>> psf = telescope.compute_psf()
        >>> output = telescope.forward(input_field)
    """

    def __init__(self, config: TelescopeConfig) -> None:
        """Initialize telescope with configuration.

        Args:
            config: TelescopeConfig with all telescope parameters
        """
        # Initialize FourFSystem with telescope-specific settings
        # - padding_factor=2.0 for good anti-aliasing
        # - aperture_cutoff_type='pixels' since telescope uses pixel-based apertures
        # - medium_index=1.0 (vacuum/air)
        # - noise_model will be created below and passed to base
        noise_model = Telescope._create_noise_model_from_config(config)

        super().__init__(
            config,
            padding_factor=2.0,
            aperture_cutoff_type="pixels",
            medium_index=1.0,
            noise_model=noise_model,
        )

        self._telescope_config = config  # Type-safe reference

        # Set default aperture radius for base class generate_aperture_mask
        self._default_aperture_radius = config.aperture_radius_pixels

        # Create aperture using strategy pattern
        self.aperture = self._create_aperture()

        # Coordinate grids (lazy initialization) - needed for aperture generation
        self._x: Optional[Tensor] = None
        self._y: Optional[Tensor] = None
        self._device: torch.device = torch.device("cpu")

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> "Telescope":
        """Override _apply to track device changes for coordinate grids.

        This is called by .to(), .cuda(), .cpu() etc. We use it to update
        the _device attribute so that lazily-created coordinate grids
        will be on the correct device.
        """
        # Call parent _apply first
        result = super()._apply(fn)

        # Detect device change by applying fn to a dummy tensor
        dummy = torch.zeros(1)
        new_device = fn(dummy).device

        # Update device tracking and reset cached coordinates
        if self._device != new_device:
            self._device = new_device
            # Reset cached coordinates so they'll be recreated on correct device
            self._x = None
            self._y = None

        return result

    @staticmethod
    def _create_noise_model_from_config(config: TelescopeConfig) -> Optional[Any]:
        """Create noise model from config (static method for __init__).

        Args:
            config: TelescopeConfig with SNR setting

        Returns:
            ShotNoise instance if snr is set, None otherwise
        """
        if config.snr is not None:
            return ShotNoise(config.snr)
        return None

    def _create_pupils(
        self,
        illumination_mode: Optional[str] = None,
        illumination_params: Optional[dict] = None,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Create pupils for telescope (detection pupil only).

        For telescopes, there is no separate illumination system, so the
        illumination pupil is None. The detection pupil is the telescope
        aperture mask.

        Args:
            illumination_mode: Not used for telescopes (always coherent)
            illumination_params: Not used for telescopes

        Returns:
            (None, detection_pupil) where detection_pupil is the aperture mask
        """
        # Generate aperture mask as detection pupil
        # Use the base class aperture generator with 'pixels' cutoff type
        detection_pupil = self._aperture_generator_lazy.circular(
            radius=self._telescope_config.aperture_radius_pixels
        )

        # No illumination pupil for telescope (coherent illumination)
        return None, detection_pupil

    def _create_aperture(self) -> Aperture:
        """Create aperture using strategy pattern.

        Returns:
            Aperture instance based on config.aperture_type
        """
        config = self._telescope_config
        aperture_kwargs = config.aperture_kwargs or {}

        if config.aperture_type == "circular":
            if "radius" not in aperture_kwargs:
                aperture_kwargs["radius"] = config.aperture_radius_pixels
            return CircularAperture(**aperture_kwargs)
        elif config.aperture_type == "hexagonal":
            if "side_length" not in aperture_kwargs:
                aperture_kwargs["side_length"] = config.aperture_radius_pixels
            return create_aperture("hexagonal", **aperture_kwargs)
        elif config.aperture_type == "obscured":
            if "outer_radius" not in aperture_kwargs:
                aperture_kwargs["outer_radius"] = config.aperture_radius_pixels
            return create_aperture("obscured", **aperture_kwargs)
        else:
            raise ValueError(f"Unknown aperture type: {config.aperture_type}")

    def _create_grid(self) -> Grid:
        """Create computational grid for telescope imaging.

        Returns:
            Grid instance configured for telescope detector plane
        """
        return Grid(
            nx=self.config.n_pixels,
            dx=self.config.pixel_size,
            wavelength=self.config.wavelength,
        )

    def _select_propagator(self) -> Propagator:
        """Select appropriate propagator for telescope.

        Returns:
            FraunhoferPropagator for far-field imaging

        Notes:
            Telescopes use Fraunhofer propagation by default as they
            operate in the far-field regime (Fresnel number << 1).
        """
        return FraunhoferPropagator()

    def compute_psf(self, center: Optional[List[float]] = None, **kwargs: Any) -> Tensor:
        """Compute telescope point spread function.

        Args:
            center: Optional aperture center [y, x]. Defaults to [0, 0].
            **kwargs: Additional parameters (unused)

        Returns:
            2D PSF tensor normalized to peak value of 1

        Notes:
            Computes PSF via:
            1. Generate aperture mask at center
            2. FFT (Fraunhofer diffraction to focal plane)
            3. Intensity (|.|^2)
            4. Normalize to peak = 1
        """
        if center is None:
            center = [0.0, 0.0]

        # Generate aperture mask
        aperture_mask = self.generate_aperture_mask(center)

        # Convert to complex field
        aperture_field = aperture_mask.to(torch.complex64)

        # Propagate to focal plane (FFT)
        psf_field = fft(aperture_field)

        # Compute intensity
        psf = torch.abs(psf_field) ** 2

        # Normalize
        if psf.max() > 0:
            psf = psf / psf.max()

        return psf

    def forward(  # type: ignore[override]
        self,
        field: Tensor,
        illumination_mode: Optional[str] = None,
        illumination_params: Optional[dict] = None,
        aperture_center: Optional[List[float]] = None,
        add_noise: bool = False,
        coherence_mode: CoherenceMode = CoherenceMode.COHERENT,
        source_intensity: Optional[Tensor] = None,
        n_source_points: int = 100,
        **kwargs: Any,
    ) -> Tensor:
        """Forward propagation through telescope.

        Standard Instrument interface for optical propagation.

        Args:
            field: Input field at object plane [B, C, H, W] or [H, W]
            illumination_mode: Not used for telescope (always coherent)
            illumination_params: Not used for telescope
            aperture_center: Optional [y, x] center for aperture mask.
                            If None, uses full aperture via base class.
            add_noise: Whether to add shot noise
            coherence_mode: Illumination coherence mode:
                - COHERENT: Standard coherent amplitude transfer (default)
                - INCOHERENT: OTF-based propagation for extended objects
                - PARTIALLY_COHERENT: Extended source integration
            source_intensity: Source intensity distribution for PARTIALLY_COHERENT mode.
                Required when coherence_mode=PARTIALLY_COHERENT.
            n_source_points: Number of source points for PARTIALLY_COHERENT sampling.
            **kwargs: Additional parameters

        Returns:
            Intensity at focal plane

        Notes:
            This method maintains backward compatibility with the old telescope
            implementation while delegating to FourFSystem for the core 4f physics.

            If aperture_center is None: Uses base class FourFSystem.forward()
            If aperture_center is specified: Custom telescope logic for sub-aperture
        """
        # If no aperture center specified, use base class implementation
        if aperture_center is None:
            return super().forward(
                field,
                illumination_mode=illumination_mode,
                illumination_params=illumination_params,
                add_noise=add_noise,
                coherence_mode=coherence_mode,
                source_intensity=source_intensity,
                n_source_points=n_source_points,
                **kwargs,
            )

        # Custom telescope logic for sub-aperture imaging
        # (used by SPIDS for synthetic aperture construction)
        squeeze_output = False
        if field.ndim == 2:
            field = field.unsqueeze(0).unsqueeze(0)
            squeeze_output = True
        elif field.ndim == 3:
            field = field.unsqueeze(0)
            squeeze_output = True

        # Propagate to k-space
        field_kspace = self.propagate_to_kspace(field)

        # Apply aperture mask at specified center
        mask = self.generate_aperture_mask(aperture_center)
        field_kspace = field_kspace * mask

        # Propagate to detector and get intensity
        output = self.propagate_to_spatial(field_kspace)

        # Add noise if requested
        if add_noise and self._noise_model is not None:
            output = self._noise_model(output, add_noise=True)

        # Restore original dimensions if needed
        if squeeze_output:
            output = output.squeeze()

        return output

    @property
    def noise_model(self) -> Optional[DetectorNoiseModel]:
        """Access to noise model for backward compatibility.

        Returns:
            DetectorNoiseModel instance if configured, None otherwise.
            Currently returns ShotNoise, which is a subclass of DetectorNoiseModel.
        """
        return self._noise_model

    @property
    def resolution_limit(self) -> float:
        """Theoretical angular resolution limit (Rayleigh criterion).

        Returns:
            Angular resolution in radians: theta = 1.22 * lambda / D

        Notes:
            If aperture_diameter not set, estimates from pixel radius.
        """
        config = self._telescope_config

        if config.aperture_diameter is not None:
            diameter = config.aperture_diameter
        else:
            # Estimate from pixel radius and pixel size
            diameter = 2 * config.aperture_radius_pixels * config.pixel_size

        return 1.22 * config.wavelength / diameter

    @property
    def x(self) -> Tensor:
        """Spatial x coordinates centered at image center."""
        if self._x is None:
            n = self.config.n_pixels
            self._x = torch.arange(0, n, device=self._device).unsqueeze(0) - n // 2
        return self._x

    @property
    def y(self) -> Tensor:
        """Spatial y coordinates centered at image center."""
        if self._y is None:
            n = self.config.n_pixels
            self._y = torch.arange(0, n, device=self._device).unsqueeze(1) - n // 2
        return self._y

    def generate_aperture_mask(
        self,
        center: Optional[List[float]] = None,
        radius: Optional[float] = None,
    ) -> Tensor:
        """Generate aperture mask at specified center.

        This method uses the telescope's aperture strategy pattern (circular,
        hexagonal, obscured) for flexibility. This is kept for backward
        compatibility with existing code.

        Args:
            center: Center position [y, x]. Defaults to [0, 0].
            radius: Override aperture radius. Defaults to config value.

        Returns:
            Boolean mask of shape (n_pixels, n_pixels)
        """
        if center is None:
            center = [0.0, 0.0]

        # For backward compatibility with radius override
        if radius is not None and isinstance(self.aperture, CircularAperture):
            temp_aperture = CircularAperture(radius=radius)
            return temp_aperture.generate(self.x, self.y, center)

        return self.aperture.generate(self.x, self.y, center)

    def generate_aperture_masks(
        self,
        centers: Union[Tensor, List[List[float]]],
        radius: Optional[float] = None,
    ) -> Tensor:
        """Generate multiple aperture masks efficiently (GPU-native vectorized).

        When centers is a GPU Tensor, this method generates masks entirely on
        the GPU without any CPU operations or data transfers, providing
        significant speedup for batched line acquisition.

        Args:
            centers: List of [y, x] centers or Tensor of shape [N, 2]
            radius: Override aperture radius. Defaults to config value.

        Returns:
            Boolean masks of shape (N, n_pixels, n_pixels)

        Performance:
            For N=64 centers on 512×512 grid (GPU):
            - Previous (CPU conversion): ~70ms
            - GPU-native: ~2-3ms (10-30x faster)
        """
        # Pass centers directly to aperture - CircularAperture.generate_batch
        # now handles both Tensor and List inputs natively on GPU
        # (Previously converted to list here, forcing CPU operations)

        # For backward compatibility with radius override
        if radius is not None and isinstance(self.aperture, CircularAperture):
            temp_aperture = CircularAperture(radius=radius)
            return temp_aperture.generate_batch(self.x, self.y, centers)

        return self.aperture.generate_batch(self.x, self.y, centers)

    def propagate_to_kspace(self, field: Tensor) -> Tensor:
        """Propagate field to k-space (Fraunhofer diffraction).

        Args:
            field: Input field at object plane

        Returns:
            Complex-valued k-space representation
        """
        return fft(field).squeeze()

    def propagate_to_spatial(self, field_kspace: Tensor) -> Tensor:
        """Propagate k-space field to spatial domain and compute intensity.

        Args:
            field_kspace: Complex k-space field

        Returns:
            Real-valued intensity at detector
        """
        # Second FFT for Fraunhofer propagation to detector
        spatial = fft(field_kspace).abs()

        # Flip for coordinate convention
        return spatial.flip((-2, -1))

    def to(self, device: torch.device) -> "Telescope":
        """Move telescope to specified device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self._device = device
        if self._x is not None:
            self._x = self._x.to(device)
        if self._y is not None:
            self._y = self._y.to(device)
        return self
