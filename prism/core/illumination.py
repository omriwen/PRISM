"""
Illumination source models for microscopy simulation.

This module provides physically accurate illumination source generators for
Brightfield (BF) and Darkfield (DF) microscopy simulations. The source grids
can be used with ExtendedSourcePropagator for source integration loops.

Key concepts:
- Brightfield: NA_source <= NA_obj (unscattered light passes through)
- Darkfield: NA_source > NA_obj (only scattered light enters objective)

References:
    - Goodman, J.W. (2005). Introduction to Fourier Optics (3rd ed.)
    - Mertz, J. (2019). Introduction to Optical Microscopy (2nd ed.)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


class ContrastMode(str, Enum):
    """Illumination contrast modes."""

    BRIGHTFIELD = "brightfield"
    DARKFIELD = "darkfield"


@dataclass
class IlluminationConfig:
    """Configuration for illumination sources.

    Attributes:
        na_objective: Numerical aperture of the objective lens.
        na_condenser: Numerical aperture of the condenser (for LED/extended sources).
        wavelength: Illumination wavelength in meters.
        mode: Contrast mode ('brightfield' or 'darkfield').
        medium_index: Refractive index of the medium (default: 1.0 for air).

    Raises:
        ValueError: If NA values are invalid for the specified mode.

    Examples:
        >>> config = IlluminationConfig(
        ...     na_objective=0.9,
        ...     na_condenser=0.7,
        ...     wavelength=532e-9,
        ...     mode="brightfield"
        ... )
    """

    na_objective: float
    na_condenser: float
    wavelength: float
    mode: Literal["brightfield", "darkfield"] = "brightfield"
    medium_index: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.na_objective <= 0:
            raise ValueError(f"NA_objective must be positive, got {self.na_objective}")
        if self.na_condenser <= 0:
            raise ValueError(f"NA_condenser must be positive, got {self.na_condenser}")
        if self.wavelength <= 0:
            raise ValueError(f"Wavelength must be positive, got {self.wavelength}")
        if self.na_objective > self.medium_index:
            raise ValueError(
                f"NA_objective ({self.na_objective}) cannot exceed "
                f"medium_index ({self.medium_index})"
            )

        # Validate mode-specific constraints
        if self.mode == "brightfield":
            if self.na_condenser > self.na_objective:
                raise ValueError(
                    f"Brightfield requires NA_condenser <= NA_objective. "
                    f"Got NA_cond={self.na_condenser}, NA_obj={self.na_objective}"
                )
        elif self.mode == "darkfield":
            if self.na_condenser <= self.na_objective:
                raise ValueError(
                    f"Darkfield requires NA_condenser > NA_objective. "
                    f"Got NA_cond={self.na_condenser}, NA_obj={self.na_objective}"
                )

    @property
    def cutoff_frequency(self) -> float:
        """Objective pupil cutoff frequency in cycles/meter."""
        return self.na_objective / self.wavelength


class SourceGeometry:
    """Generate source point grids for illumination integration.

    This class provides static methods to create source point distributions
    for various illumination geometries. The generated grids can be used
    with ExtendedSourcePropagator for physically accurate simulations.

    Methods:
        disk_grid: Circular grid for brightfield illumination
        annular_grid: Ring grid for darkfield illumination
        single_point: Single point for coherent illumination
        oblique_point: Off-axis point for oblique darkfield
    """

    @staticmethod
    def disk_grid(
        na: float,
        wavelength: float,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate circular disk grid for brightfield illumination.

        Creates a uniform distribution of source points within a circular
        region defined by the specified NA.

        Args:
            na: Maximum numerical aperture (defines disk radius).
            wavelength: Illumination wavelength in meters.
            n_points: Approximate number of source points.
            device: PyTorch device for output tensors.

        Returns:
            positions: Source positions as angular frequencies (N, 2).
                Format: (theta_x, theta_y) where theta = sin(angle).
            weights: Uniform weights for each source point (N,).

        Examples:
            >>> positions, weights = SourceGeometry.disk_grid(
            ...     na=0.7, wavelength=532e-9, n_points=100
            ... )
        """
        device = device or torch.device("cpu")

        # Calculate grid dimensions for approximately n_points
        # Area of disk = pi * r^2, so for square grid, side = sqrt(n_points * pi / 4)
        n_side = max(3, int(np.sqrt(n_points * 4 / np.pi)))

        # Maximum angular frequency
        max_freq = na / wavelength

        # Create uniform grid in frequency space
        theta_x = torch.linspace(-max_freq, max_freq, n_side, device=device)
        theta_y = torch.linspace(-max_freq, max_freq, n_side, device=device)
        grid_x, grid_y = torch.meshgrid(theta_x, theta_y, indexing="ij")

        # Flatten and filter to disk
        positions_x = grid_x.flatten()
        positions_y = grid_y.flatten()
        radius = torch.sqrt(positions_x**2 + positions_y**2)
        mask = radius <= max_freq

        # Stack filtered positions
        positions = torch.stack([positions_x[mask], positions_y[mask]], dim=1)

        # Uniform weights (normalized to sum to 1)
        n_actual = positions.shape[0]
        weights = torch.ones(n_actual, device=device) / n_actual

        return positions, weights

    @staticmethod
    def annular_grid(
        na_inner: float,
        na_outer: float,
        wavelength: float,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate annular (ring) grid for darkfield illumination.

        Creates a distribution of source points within an annular region.
        The inner radius is typically just outside the objective NA to
        ensure unscattered light is blocked.

        Args:
            na_inner: Inner NA (should be > NA_objective for darkfield).
            na_outer: Outer NA (condenser NA limit).
            wavelength: Illumination wavelength in meters.
            n_points: Approximate number of source points.
            device: PyTorch device for output tensors.

        Returns:
            positions: Source positions as angular frequencies (N, 2).
            weights: Weights proportional to radius (for uniform area sampling).

        Raises:
            ValueError: If na_inner >= na_outer.

        Examples:
            >>> # Darkfield with NA_obj=0.9, NA_cond=1.2
            >>> positions, weights = SourceGeometry.annular_grid(
            ...     na_inner=0.95,  # Just outside objective
            ...     na_outer=1.2,   # Condenser limit
            ...     wavelength=532e-9,
            ...     n_points=100
            ... )
        """
        if na_inner >= na_outer:
            raise ValueError(
                f"na_inner must be less than na_outer. Got inner={na_inner}, outer={na_outer}"
            )

        device = device or torch.device("cpu")

        # Convert NA to angular frequency
        freq_inner = na_inner / wavelength
        freq_outer = na_outer / wavelength

        # Estimate number of radial and angular samples
        # Area of annulus = pi * (r_outer^2 - r_inner^2)
        area_ratio = (freq_outer**2 - freq_inner**2) / freq_outer**2
        n_radial = max(3, int(np.sqrt(n_points / (2 * np.pi) / area_ratio)))
        n_angular = max(8, int(2 * np.pi * n_radial))

        # Generate radial positions (uniform in r^2 for uniform area sampling)
        r_squared = torch.linspace(freq_inner**2, freq_outer**2, n_radial, device=device)
        radii = torch.sqrt(r_squared)

        # Generate angular positions
        angles = torch.linspace(0, 2 * np.pi, n_angular + 1, device=device)[:-1]

        # Create grid
        positions_list = []
        weights_list = []

        for r in radii:
            for theta in angles:
                x = r * torch.cos(theta)
                y = r * torch.sin(theta)
                positions_list.append(torch.stack([x, y]))
                # Weight proportional to radius for uniform area sampling
                weights_list.append(r)

        positions = torch.stack(positions_list, dim=0)
        weights = torch.stack(weights_list)
        weights = weights / weights.sum()  # Normalize

        return positions, weights

    @staticmethod
    def single_point(
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate single on-axis point for coherent brightfield.

        Returns a single source point at the origin (on-axis illumination),
        representing a coherent laser source in brightfield configuration.

        Args:
            device: PyTorch device for output tensors.

        Returns:
            positions: Single point at origin (1, 2).
            weights: Unit weight (1,).

        Examples:
            >>> positions, weights = SourceGeometry.single_point()
            >>> print(positions)  # tensor([[0., 0.]])
        """
        device = device or torch.device("cpu")
        positions = torch.zeros((1, 2), device=device)
        weights = torch.ones(1, device=device)
        return positions, weights

    @staticmethod
    def oblique_point(
        na: float,
        wavelength: float,
        azimuth: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate single off-axis point for oblique darkfield.

        Creates a single source point tilted beyond the objective NA,
        useful for oblique illumination darkfield with a laser source.

        Args:
            na: Numerical aperture defining the tilt angle.
            wavelength: Illumination wavelength in meters.
            azimuth: Azimuthal angle in radians (0 = positive x-axis).
            device: PyTorch device for output tensors.

        Returns:
            positions: Single off-axis point (1, 2).
            weights: Unit weight (1,).

        Examples:
            >>> # Oblique DF tilted along x-axis, NA=1.0
            >>> positions, weights = SourceGeometry.oblique_point(
            ...     na=1.0, wavelength=532e-9, azimuth=0.0
            ... )
        """
        device = device or torch.device("cpu")

        freq = na / wavelength
        x = freq * np.cos(azimuth)
        y = freq * np.sin(azimuth)

        positions = torch.tensor([[x, y]], device=device, dtype=torch.float32)
        weights = torch.ones(1, device=device)

        return positions, weights


class IlluminationSource:
    """Base class for illumination source models.

    Subclasses implement specific source types (LED, Laser, Solar) with
    their characteristic spatial coherence properties.

    Attributes:
        config: Illumination configuration.
    """

    def __init__(self, config: IlluminationConfig):
        """Initialize illumination source.

        Args:
            config: Illumination configuration with NA and wavelength.
        """
        self.config = config

    def get_source_grid(
        self,
        n_points: int = 100,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate source point grid for integration.

        Args:
            n_points: Number of source points.
            device: PyTorch device for output tensors.

        Returns:
            positions: Source positions (N, 2).
            weights: Source weights (N,).
        """
        raise NotImplementedError("Subclasses must implement get_source_grid")


class LEDSource(IlluminationSource):
    """Extended LED source for partially coherent illumination.

    Models a standard LED or lamp source with partial spatial coherence.
    The source is treated as an extended incoherent source, which produces
    reduced speckle/interference compared to laser illumination.

    For brightfield: Circular disk of source points within NA_condenser.
    For darkfield: Annular ring of source points outside NA_objective.

    Examples:
        >>> config = IlluminationConfig(
        ...     na_objective=0.9,
        ...     na_condenser=0.7,
        ...     wavelength=532e-9,
        ...     mode="brightfield"
        ... )
        >>> led = LEDSource(config)
        >>> positions, weights = led.get_source_grid(n_points=200)
    """

    def get_source_grid(
        self,
        n_points: int = 100,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate LED source grid.

        Args:
            n_points: Number of source points.
            device: PyTorch device for output tensors.

        Returns:
            positions: Source positions (N, 2).
            weights: Source weights (N,).
        """
        if self.config.mode == "brightfield":
            return SourceGeometry.disk_grid(
                na=self.config.na_condenser,
                wavelength=self.config.wavelength,
                n_points=n_points,
                device=device,
            )
        else:  # darkfield
            # Inner radius just outside objective NA
            epsilon = 0.05 * self.config.na_objective
            return SourceGeometry.annular_grid(
                na_inner=self.config.na_objective + epsilon,
                na_outer=self.config.na_condenser,
                wavelength=self.config.wavelength,
                n_points=n_points,
                device=device,
            )


class LaserSource(IlluminationSource):
    """Coherent laser source for high-contrast illumination.

    Models a laser source as a single coherent point. This produces
    maximum interference effects, including holographic-like fringes
    and speckle patterns.

    For brightfield: Single on-axis point (theta_x = theta_y = 0).
    For darkfield: Single off-axis point tilted beyond NA_objective
                   (oblique illumination darkfield).

    Attributes:
        azimuth: Azimuthal angle for oblique illumination (radians).

    Examples:
        >>> config = IlluminationConfig(
        ...     na_objective=0.9,
        ...     na_condenser=1.1,
        ...     wavelength=532e-9,
        ...     mode="darkfield"
        ... )
        >>> laser = LaserSource(config, azimuth=0.0)  # Tilted along x
        >>> positions, weights = laser.get_source_grid()
    """

    def __init__(
        self,
        config: IlluminationConfig,
        azimuth: float = 0.0,
    ):
        """Initialize laser source.

        Args:
            config: Illumination configuration.
            azimuth: Azimuthal angle for oblique darkfield (radians).
        """
        super().__init__(config)
        self.azimuth = azimuth

    def get_source_grid(
        self,
        n_points: int = 1,  # Ignored for laser (always single point)
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate laser source grid (single point).

        Args:
            n_points: Ignored (laser is always single point).
            device: PyTorch device for output tensors.

        Returns:
            positions: Single source point (1, 2).
            weights: Unit weight (1,).
        """
        if self.config.mode == "brightfield":
            return SourceGeometry.single_point(device=device)
        else:  # darkfield
            # Tilt to just outside objective NA
            tilt_na = self.config.na_objective * 1.1
            tilt_na = min(tilt_na, self.config.na_condenser)
            return SourceGeometry.oblique_point(
                na=tilt_na,
                wavelength=self.config.wavelength,
                azimuth=self.azimuth,
                device=device,
            )


class SolarSource(IlluminationSource):
    """Solar illumination source for space/astronomy applications.

    Models solar illumination in two configurations:
    - Direct (collimated): Coherent-like, single point source
    - Diffused (ambient/cloud): Extended source, acts like LED

    Note: Darkfield with direct solar is not physically meaningful
    unless the sun is blocked or side-scattering is used.

    Attributes:
        diffused: Whether the solar light is diffused (ambient).

    Examples:
        >>> config = IlluminationConfig(
        ...     na_objective=0.4,
        ...     na_condenser=0.3,
        ...     wavelength=550e-9,
        ...     mode="brightfield"
        ... )
        >>> solar = SolarSource(config, diffused=True)
        >>> positions, weights = solar.get_source_grid(n_points=100)
    """

    def __init__(
        self,
        config: IlluminationConfig,
        diffused: bool = False,
    ):
        """Initialize solar source.

        Args:
            config: Illumination configuration.
            diffused: True for diffused (ambient) light, False for direct.
        """
        super().__init__(config)
        self.diffused = diffused

    def get_source_grid(
        self,
        n_points: int = 100,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate solar source grid.

        Args:
            n_points: Number of source points (for diffused mode).
            device: PyTorch device for output tensors.

        Returns:
            positions: Source positions (N, 2).
            weights: Source weights (N,).

        Raises:
            ValueError: If darkfield mode with direct solar is requested.
        """
        if not self.diffused:
            # Direct solar - single point (or small 3x3 grid)
            if self.config.mode == "darkfield":
                raise ValueError(
                    "Direct solar illumination is not compatible with darkfield. "
                    "Use diffused=True or side-scattering geometry."
                )
            return SourceGeometry.single_point(device=device)
        else:
            # Diffused solar - acts like LED
            if self.config.mode == "brightfield":
                return SourceGeometry.disk_grid(
                    na=self.config.na_condenser,
                    wavelength=self.config.wavelength,
                    n_points=n_points,
                    device=device,
                )
            else:  # darkfield with diffused light
                epsilon = 0.05 * self.config.na_objective
                return SourceGeometry.annular_grid(
                    na_inner=self.config.na_objective + epsilon,
                    na_outer=self.config.na_condenser,
                    wavelength=self.config.wavelength,
                    n_points=n_points,
                    device=device,
                )


def create_illumination_source(
    source_type: str,
    config: IlluminationConfig,
    **kwargs: float | bool,
) -> IlluminationSource:
    """Factory function to create illumination sources.

    Args:
        source_type: Type of source ('led', 'laser', 'solar'). Case-insensitive.
        config: Illumination configuration.
        **kwargs: Additional arguments for specific source types:
            - laser: azimuth (float)
            - solar: diffused (bool)

    Returns:
        IlluminationSource instance.

    Raises:
        ValueError: If source_type is unknown.

    Examples:
        >>> config = IlluminationConfig(
        ...     na_objective=0.9,
        ...     na_condenser=0.7,
        ...     wavelength=532e-9,
        ...     mode="brightfield"
        ... )
        >>> source = create_illumination_source("led", config)
    """
    source_type_lower = source_type.lower()

    if source_type_lower == "led":
        return LEDSource(config)
    elif source_type_lower == "laser":
        azimuth = float(kwargs.get("azimuth", 0.0))
        return LaserSource(config, azimuth=azimuth)
    elif source_type_lower == "solar":
        diffused = bool(kwargs.get("diffused", False))
        return SolarSource(config, diffused=diffused)
    else:
        raise ValueError(
            f"Unknown source type: {source_type}. Valid types: 'led', 'laser', 'solar'"
        )


def validate_bf_df_configuration(
    na_objective: float,
    na_condenser: float,
    mode: str,
) -> bool:
    """Validate brightfield/darkfield configuration.

    Checks that NA values are consistent with the specified mode:
    - Brightfield: NA_condenser <= NA_objective
    - Darkfield: NA_condenser > NA_objective

    Args:
        na_objective: Objective numerical aperture.
        na_condenser: Condenser numerical aperture.
        mode: Contrast mode ('brightfield' or 'darkfield').

    Returns:
        True if configuration is valid.

    Raises:
        ValueError: If configuration is invalid with descriptive message.
    """
    if mode.lower() == "brightfield":
        if na_condenser > na_objective:
            raise ValueError(
                f"Brightfield requires NA_condenser <= NA_objective. "
                f"Got NA_cond={na_condenser}, NA_obj={na_objective}. "
                f"Either reduce NA_condenser or switch to darkfield mode."
            )
    elif mode.lower() == "darkfield":
        if na_condenser <= na_objective:
            raise ValueError(
                f"Darkfield requires NA_condenser > NA_objective. "
                f"Got NA_cond={na_condenser}, NA_obj={na_objective}. "
                f"Either increase NA_condenser or switch to brightfield mode."
            )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'brightfield' or 'darkfield'.")

    return True
