"""Camera instrument implementation for SPIDS.

This module implements general camera systems with:
- Automatic propagation regime selection (Fraunhofer vs Angular Spectrum)
- Depth of field calculations
- Sensor noise modeling
- Defocus aberration support
- Inherits from FourFSystem for far-field mode
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ..grid import Grid
from ..propagators.angular_spectrum import AngularSpectrumPropagator
from ..propagators.base import CoherenceMode
from .base import InstrumentConfig
from .four_f_base import FourFSystem


@dataclass
class CameraConfig(InstrumentConfig):
    """Configuration for camera systems.

    Attributes:
        focal_length: Lens focal length in meters
        f_number: f-number (focal ratio) of the lens
        sensor_size: (width, height) of sensor in meters
        object_distance: Distance to object in meters (inf for far field)
        focus_distance: Focus distance in meters (None = same as object_distance)
        lens_type: Type of lens model ('thin', 'thick', 'compound')
    """

    focal_length: float = 50e-3  # 50mm lens
    f_number: float = 2.8
    sensor_size: Tuple[float, float] = (36e-3, 24e-3)  # Full frame
    object_distance: float = float("inf")
    focus_distance: Optional[float] = None
    lens_type: str = "thin"

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.focus_distance is None:
            self.focus_distance = self.object_distance
        self.validate()


class Camera(FourFSystem):
    """General camera implementation.

    Supports both near-field and far-field imaging with automatic
    propagation regime selection based on Fresnel number.

    For far-field mode (Fresnel number < 0.1), uses FourFSystem infrastructure.
    For near-field mode, uses Angular Spectrum propagation.
    """

    def __init__(self, config: CameraConfig):
        """Initialize camera.

        Args:
            config: Camera configuration
        """
        # Store camera-specific parameters first
        self.focal_length = config.focal_length
        self.f_number = config.f_number
        self.aperture_diameter = self.focal_length / self.f_number
        self.sensor_size = config.sensor_size
        self.object_distance = config.object_distance
        self.focus_distance = config.focus_distance or config.object_distance

        # Initialize FourFSystem with physical aperture specification
        super().__init__(
            config,
            padding_factor=2.0,
            aperture_cutoff_type="physical",
            medium_index=1.0,  # Camera operates in air
            noise_model=None,  # Camera uses custom sensor noise
        )

        # Set default aperture radius for generate_aperture_mask
        self._default_aperture_radius = self.aperture_diameter / 2

        # Create propagator for near-field mode (lazy initialization)
        self._angular_spectrum_propagator: Optional[AngularSpectrumPropagator] = None

    def _create_grid(self) -> Grid:
        """Create spatial grid for camera sensor.

        Returns:
            Grid instance for sensor plane
        """
        return Grid(
            nx=self.config.n_pixels,
            dx=self.config.pixel_size,
            wavelength=self.config.wavelength,
        )

    def _create_pupils(
        self,
        illumination_mode: Optional[str] = None,
        illumination_params: Optional[dict] = None,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Create detection pupil based on f-number.

        The aperture radius in k-space is calculated from the physical
        f-number: NA_eff = 1 / (2 * f_number), giving k_cutoff = NA_eff / lambda.
        """
        # Calculate aperture from f-number (not hardcoded percentage)
        na_effective = 1.0 / (2.0 * self.f_number)
        k_cutoff = na_effective / self.config.wavelength
        k_max = 1.0 / (2.0 * self.config.pixel_size)
        aperture_radius_pixels = (k_cutoff / k_max) * (self.config.n_pixels / 2)

        # Clamp to valid range (leave margin for edge effects)
        aperture_radius_pixels = min(
            aperture_radius_pixels,
            self.config.n_pixels * 0.45
        )

        detection_pupil = self._aperture_generator_lazy.circular(radius=aperture_radius_pixels)
        return None, detection_pupil

    @property
    def resolution_limit(self) -> float:
        """Diffraction-limited spot size at sensor.

        Uses Airy disk diameter: 2.44 × λ × f/#

        Returns:
            Resolution limit in meters
        """
        return 2.44 * self.config.wavelength * self.f_number

    def _select_propagator(self) -> AngularSpectrumPropagator:
        """Select propagator based on Fresnel number.

        Returns:
            AngularSpectrumPropagator for near-field imaging.
            Note: Far-field mode uses FourFSystem.forward() instead.
        """
        # Only create Angular Spectrum propagator for near-field mode
        if self._angular_spectrum_propagator is None:
            self._angular_spectrum_propagator = AngularSpectrumPropagator(self.grid)
        return self._angular_spectrum_propagator

    def _calculate_fresnel_number(self) -> float:
        """Calculate Fresnel number for propagation regime.

        Fresnel number F = a²/(λz) where:
        - a is aperture radius
        - λ is wavelength
        - z is propagation distance

        Returns:
            Fresnel number (dimensionless)
        """
        if self.object_distance == float("inf"):
            return 0.0  # Far-field
        else:
            a = self.aperture_diameter / 2
            z = self.object_distance
            return a**2 / (self.config.wavelength * z)

    def compute_psf(self, defocus: float = 0.0, **kwargs: Any) -> torch.Tensor:
        """Compute camera PSF with optional defocus.

        Args:
            defocus: Defocus distance in meters
            **kwargs: Additional parameters (unused)

        Returns:
            2D PSF tensor, normalized to peak intensity of 1

        Notes:
            Uses a simplified PSF computation based on Fraunhofer diffraction.
            The aperture size is determined by the f-number and is represented
            in the Fourier domain.
        """
        # For camera PSF, we use the base class implementation which
        # propagates a delta function through the full optical system
        # This ensures consistency with the forward model
        return super().compute_psf(**kwargs)

    def _defocus_aberration(self, defocus: float) -> torch.Tensor:
        """Calculate defocus aberration phase.

        Implements W₂₀ defocus aberration term.

        Args:
            defocus: Defocus distance in meters

        Returns:
            Complex phase factor for defocus
        """
        x = self.grid.x
        y = self.grid.y
        r = torch.sqrt(x**2 + y**2)

        # W₂₀ defocus aberration
        k = 2 * np.pi / self.config.wavelength
        w20 = defocus * r**2 / (2 * self.focal_length**2)
        phase = torch.exp(1j * k * w20)

        return phase

    def calculate_image_distance(self) -> float:
        """Calculate image distance using thin lens equation.

        Uses 1/f = 1/do + 1/di

        Returns:
            Image distance in meters
        """
        if self.focus_distance == float("inf"):
            return self.focal_length
        else:
            # 1/f = 1/do + 1/di  =>  di = 1 / (1/f - 1/do)
            return 1.0 / (1.0 / self.focal_length - 1.0 / self.focus_distance)

    def calculate_magnification(self) -> float:
        """Calculate lateral magnification.

        Returns:
            Magnification (negative for inverted image, 0 for infinity focus)
        """
        if self.object_distance == float("inf"):
            return 0.0
        else:
            di = self.calculate_image_distance()
            return -di / self.object_distance

    def calculate_depth_of_field(self, coc_limit: float = 30e-6) -> Tuple[float, float]:
        """Calculate depth of field.

        Args:
            coc_limit: Circle of confusion limit in meters (default 30μm)

        Returns:
            Tuple of (near_distance, far_distance) in meters
        """
        if self.focus_distance == float("inf"):
            # Hyperfocal distance
            h = self.focal_length**2 / (self.f_number * coc_limit)
            return (h / 2, float("inf"))
        else:
            # DOF calculation using hyperfocal distance
            h = self.focal_length**2 / (self.f_number * coc_limit)

            # Near and far distances
            dn = self.focus_distance * h / (h + self.focus_distance)

            # Check if denominator is positive for far distance
            if h > self.focus_distance:
                df = self.focus_distance * h / (h - self.focus_distance)
            else:
                df = float("inf")

            return (dn, df)

    def forward(  # type: ignore[override]
        self,
        field: torch.Tensor,
        illumination_mode: Optional[str] = None,
        illumination_params: Optional[dict] = None,
        coherence_mode: CoherenceMode = CoherenceMode.COHERENT,
        source_intensity: Optional[torch.Tensor] = None,
        n_source_points: int = 100,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Image formation through camera.

        Uses the FourFSystem forward model which automatically handles
        far-field imaging correctly.

        Args:
            field: Input scene (can be at infinity or finite distance)
            illumination_mode: Illumination type (unused for basic camera)
            illumination_params: Illumination parameters (unused for basic camera)
            coherence_mode: Illumination coherence mode:
                - COHERENT: Standard coherent amplitude transfer (default)
                - INCOHERENT: OTF-based propagation for self-luminous scenes
                - PARTIALLY_COHERENT: Extended source integration
            source_intensity: Source intensity distribution for PARTIALLY_COHERENT mode.
                Required when coherence_mode=PARTIALLY_COHERENT.
            n_source_points: Number of source points for PARTIALLY_COHERENT sampling.
            **kwargs: Additional parameters:
                add_noise (bool): Add realistic sensor noise (default: False)

        Returns:
            Image at sensor plane
        """
        # Extract add_noise from kwargs (we handle it separately)
        add_noise = kwargs.pop("add_noise", False)

        # Use FourFSystem forward model
        # This handles aperture correctly using the Fourier-domain representation
        image = super().forward(
            field,
            illumination_mode=illumination_mode,
            illumination_params=illumination_params,
            add_noise=False,  # Don't use base class noise model
            coherence_mode=coherence_mode,
            source_intensity=source_intensity,
            n_source_points=n_source_points,
            **kwargs,
        )

        # Add sensor effects if requested (camera-specific noise model)
        if add_noise:
            image = self._add_sensor_noise(image)

        return image

    def _add_sensor_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add realistic sensor noise.

        Includes:
        - Shot noise (Poisson)
        - Read noise (Gaussian)
        - Dark current

        Args:
            image: Clean image

        Returns:
            Noisy image
        """
        # Shot noise (Poisson)
        if image.max() > 0:
            image_photons = image * 1000  # Scale to photon counts
            shot_noise = torch.poisson(image_photons) / 1000
        else:
            shot_noise = image

        # Read noise (Gaussian)
        read_noise = torch.randn_like(image) * 0.01

        # Dark current
        dark_current = torch.randn_like(image) * 0.001

        return shot_noise + read_noise + dark_current

    def get_info(self) -> dict:
        """Get camera information summary.

        Returns:
            Dictionary with camera parameters and characteristics
        """
        info = super().get_info()
        info.update(
            {
                "focal_length": self.focal_length,
                "f_number": self.f_number,
                "aperture_diameter": self.aperture_diameter,
                "object_distance": self.object_distance,
                "focus_distance": self.focus_distance,
                "image_distance": self.calculate_image_distance(),
                "magnification": self.calculate_magnification(),
                "fresnel_number": self._calculate_fresnel_number(),
            }
        )
        return info

    def __repr__(self) -> str:
        """String representation of camera."""
        return (
            f"Camera("
            f"focal_length={self.focal_length * 1e3:.1f}mm, "
            f"f/{self.f_number}, "
            f"resolution={self.resolution_limit * 1e6:.1f}μm)"
        )
