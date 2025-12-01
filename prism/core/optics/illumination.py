"""Illumination source models for scanning illumination forward model.

This module provides illumination source modeling for Fourier Ptychographic
Microscopy (FPM) style reconstruction. It supports various source geometries
including point sources (tilted plane waves), Gaussian sources, circular sources,
and custom LED array patterns.

The illumination model applies both a spatial envelope (for finite-size sources)
and a phase tilt (for off-axis illumination angle) to create the complete
illumination field.

Classes
-------
IlluminationSourceType
    Enum specifying the type of illumination source geometry.
IlluminationSource
    Dataclass containing illumination source parameters.

Functions
---------
create_illumination_envelope
    Generate spatial envelope for finite-size illumination sources.
create_phase_tilt
    Generate phase tilt for tilted plane wave illumination.
create_illumination_field
    Create complete illumination field (envelope × phase tilt).

Examples
--------
Create a point source (tilted plane wave):

>>> from prism.core.optics.illumination import (
...     IlluminationSource, IlluminationSourceType, create_illumination_field
... )
>>> from prism.core.grid import Grid
>>> grid = Grid(nx=256, dx=1e-6, wavelength=520e-9)
>>> source = IlluminationSource(
...     source_type=IlluminationSourceType.POINT,
...     k_center=[0.1e6, 0.0],  # k-space position
... )
>>> field = create_illumination_field(grid, source)

Create a Gaussian source with finite extent:

>>> source = IlluminationSource(
...     source_type=IlluminationSourceType.GAUSSIAN,
...     k_center=[0.1e6, 0.05e6],
...     sigma=0.01e6,  # k-space width
... )
>>> field = create_illumination_field(grid, source)

See Also
--------
spids.core.optics.fourier_utils : k-space shift calculations
spids.core.instruments.microscope : Microscope with scanning illumination support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..grid import Grid


class IlluminationSourceType(Enum):
    """Illumination source geometry types.

    Attributes
    ----------
    POINT : auto
        Point source producing a tilted plane wave. This is mathematically
        equivalent to scanning aperture for ideal coherent imaging.
    GAUSSIAN : auto
        Gaussian intensity profile source. Produces "soft" k-space sampling
        with partial coherence effects.
    CIRCULAR : auto
        Circular (top-hat) intensity profile source. Models finite-size LEDs
        or fiber bundle sources.
    CUSTOM : auto
        User-defined source profile. Allows arbitrary illumination patterns.
    """

    POINT = auto()
    GAUSSIAN = auto()
    CIRCULAR = auto()
    CUSTOM = auto()


@dataclass
class IlluminationSource:
    """Illumination source configuration.

    Parameters
    ----------
    source_type : IlluminationSourceType
        Type of illumination source geometry.
    k_center : List[float]
        Center position in k-space [ky, kx] in 1/meters.
        For point sources, this defines the tilt angle.
        For finite sources, this is the center of the source profile.
    sigma : float, optional
        Width parameter for Gaussian sources (standard deviation in k-space).
        Units: 1/meters. Only used for GAUSSIAN type.
    radius : float, optional
        Radius for circular sources in k-space.
        Units: 1/meters. Only used for CIRCULAR type.
    intensity : float, default=1.0
        Relative intensity of the source.
    custom_profile : Tensor, optional
        Custom source profile for CUSTOM type. Should be a 2D tensor
        representing the source intensity distribution in k-space.

    Examples
    --------
    Point source at 10° tilt angle (approximate):

    >>> # For wavelength=520nm, angle=10°, k = sin(theta)/lambda
    >>> k_tilt = np.sin(np.radians(10)) / 520e-9  # ~3.3e5 1/m
    >>> source = IlluminationSource(
    ...     source_type=IlluminationSourceType.POINT,
    ...     k_center=[k_tilt, 0],
    ... )

    Gaussian LED source:

    >>> source = IlluminationSource(
    ...     source_type=IlluminationSourceType.GAUSSIAN,
    ...     k_center=[0.1e6, 0.0],
    ...     sigma=0.02e6,
    ... )
    """

    source_type: IlluminationSourceType
    k_center: List[float] = field(default_factory=lambda: [0.0, 0.0])
    sigma: Optional[float] = None
    radius: Optional[float] = None
    intensity: float = 1.0
    custom_profile: Optional[Tensor] = None

    def __post_init__(self) -> None:
        """Validate source parameters."""
        if self.source_type == IlluminationSourceType.GAUSSIAN and self.sigma is None:
            raise ValueError("GAUSSIAN source type requires sigma parameter")
        if self.source_type == IlluminationSourceType.CIRCULAR and self.radius is None:
            raise ValueError("CIRCULAR source type requires radius parameter")
        if self.source_type == IlluminationSourceType.CUSTOM and self.custom_profile is None:
            raise ValueError("CUSTOM source type requires custom_profile tensor")
        if len(self.k_center) != 2:
            raise ValueError(f"k_center must have 2 elements [ky, kx], got {len(self.k_center)}")


def create_phase_tilt(
    grid: Grid,
    k_center: Union[List[float], Tuple[float, float]],
    device: Optional[torch.device] = None,
) -> Tensor:
    """Generate phase tilt for tilted plane wave illumination.

    Creates a complex exponential representing a tilted plane wave:
    exp(i * 2π * (kx*x + ky*y))

    This phase tilt, when multiplied with the object field in spatial domain,
    shifts the object spectrum in k-space by (kx, ky).

    Parameters
    ----------
    grid : Grid
        Spatial grid defining the coordinate system.
    k_center : List[float] or Tuple[float, float]
        k-space center [ky, kx] in 1/meters. Defines the illumination angle:
        sin(theta_x) = kx * wavelength
        sin(theta_y) = ky * wavelength
    device : torch.device, optional
        Device for computation. Defaults to grid device.

    Returns
    -------
    Tensor
        Complex phase tilt of shape (grid.ny, grid.nx).

    Notes
    -----
    The phase tilt implements the Fourier shift theorem:
    F{f(x) * exp(i*2π*k₀*x)} = F̃(k - k₀)

    This means multiplying the object by exp(i*2π*k₀*x) in spatial domain
    shifts its spectrum by k₀ in frequency domain.

    Examples
    --------
    >>> grid = Grid(nx=256, dx=1e-6, wavelength=520e-9)
    >>> phase = create_phase_tilt(grid, [0.1e6, 0.0])
    >>> print(phase.shape)
    torch.Size([256, 256])
    """
    if device is None:
        device = torch.device(grid.device)

    ky, kx = k_center

    # Get spatial coordinates
    x = grid.x.to(device)  # Shape: (1, nx)
    y = grid.y.to(device)  # Shape: (ny, 1)

    # Phase = 2π * (kx*x + ky*y)
    phase = 2 * np.pi * (kx * x + ky * y)

    # Return complex exponential
    return torch.exp(1j * phase).to(torch.complex64)


def create_illumination_envelope(
    grid: Grid,
    source: IlluminationSource,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Generate spatial envelope for finite-size illumination sources.

    Creates the intensity envelope for the illumination source. For point
    sources, returns unity (no envelope). For finite sources, the envelope
    represents the source's spatial intensity distribution.

    Parameters
    ----------
    grid : Grid
        Spatial grid defining the coordinate system.
    source : IlluminationSource
        Illumination source configuration.
    device : torch.device, optional
        Device for computation. Defaults to grid device.

    Returns
    -------
    Tensor
        Real-valued envelope of shape (grid.ny, grid.nx), normalized to max=1.

    Notes
    -----
    The envelope is defined in **spatial** domain, not k-space. For a source
    with k-space width σ_k, the spatial envelope width is approximately:
    σ_x ≈ 1 / (2π * σ_k)

    This envelope, when applied to the illumination field, produces partial
    coherence effects in the resulting measurement.

    Examples
    --------
    >>> grid = Grid(nx=256, dx=1e-6, wavelength=520e-9)
    >>> source = IlluminationSource(
    ...     source_type=IlluminationSourceType.GAUSSIAN,
    ...     k_center=[0, 0],
    ...     sigma=0.1e6,
    ... )
    >>> envelope = create_illumination_envelope(grid, source)
    """
    if device is None:
        device = torch.device(grid.device)

    x = grid.x.to(device)  # Shape: (1, nx)
    y = grid.y.to(device)  # Shape: (ny, 1)

    if source.source_type == IlluminationSourceType.POINT:
        # Point source: unity envelope (infinite coherence)
        return torch.ones(grid.ny, grid.nx, dtype=torch.float32, device=device)

    elif source.source_type == IlluminationSourceType.GAUSSIAN:
        # Gaussian envelope
        # k-space width σ_k corresponds to spatial width σ_x = 1/(2π*σ_k)
        sigma_k = source.sigma
        if sigma_k is None or sigma_k <= 0:
            raise ValueError("Gaussian source requires positive sigma")

        # Spatial width from k-space width
        sigma_x = 1.0 / (2 * np.pi * sigma_k)

        # Compute Gaussian envelope
        r_sq = x**2 + y**2
        envelope = torch.exp(-r_sq / (2 * sigma_x**2))

        return envelope.float()

    elif source.source_type == IlluminationSourceType.CIRCULAR:
        # Circular (top-hat) envelope
        # k-space radius corresponds to spatial radius via Fourier relationship
        radius_k = source.radius
        if radius_k is None or radius_k <= 0:
            raise ValueError("Circular source requires positive radius")

        # Spatial radius (first zero of Bessel function approximation)
        # For top-hat in k-space, spatial profile is jinc function
        # We approximate as circular region with radius ~ 1/(2π*radius_k)
        radius_x = 1.0 / (2 * np.pi * radius_k)

        # Compute circular envelope
        r = torch.sqrt(x**2 + y**2)
        envelope = (r <= radius_x).float()

        return envelope

    elif source.source_type == IlluminationSourceType.CUSTOM:
        # Custom profile provided by user
        if source.custom_profile is None:
            raise ValueError("Custom source requires custom_profile tensor")

        profile = source.custom_profile.to(device)

        # Ensure correct shape
        if profile.shape != (grid.ny, grid.nx):
            raise ValueError(
                f"Custom profile shape {profile.shape} doesn't match grid ({grid.ny}, {grid.nx})"
            )

        # Normalize to max=1
        profile = profile / profile.max()

        return profile.float()

    else:
        raise ValueError(f"Unknown source type: {source.source_type}")


def create_illumination_field(
    grid: Grid,
    source: IlluminationSource,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Create complete illumination field (envelope × phase tilt).

    Combines the spatial envelope (for finite-size sources) with the phase
    tilt (for off-axis illumination) to create the complete complex
    illumination field.

    Parameters
    ----------
    grid : Grid
        Spatial grid defining the coordinate system.
    source : IlluminationSource
        Illumination source configuration.
    device : torch.device, optional
        Device for computation. Defaults to grid device.

    Returns
    -------
    Tensor
        Complex illumination field of shape (grid.ny, grid.nx).

    Notes
    -----
    The illumination field E_illum is:
    E_illum(x,y) = A(x,y) * exp(i*2π*(kx*x + ky*y))

    where A(x,y) is the envelope and (kx, ky) is the k-space center.

    For point sources (A=1), this is a pure tilted plane wave.
    For finite sources, A introduces partial coherence effects.

    Examples
    --------
    >>> grid = Grid(nx=256, dx=1e-6, wavelength=520e-9)
    >>> source = IlluminationSource(
    ...     source_type=IlluminationSourceType.POINT,
    ...     k_center=[0.1e6, 0.0],
    ... )
    >>> field = create_illumination_field(grid, source)
    >>> print(field.dtype)
    torch.complex64
    """
    if device is None:
        device = torch.device(grid.device)

    # Create phase tilt
    phase_tilt = create_phase_tilt(grid, source.k_center, device)

    # Create envelope
    envelope = create_illumination_envelope(grid, source, device)

    # Combine: field = intensity * envelope * phase_tilt
    field = source.intensity * envelope * phase_tilt

    return field.to(torch.complex64)


def illumination_angle_to_k_center(
    theta_x: float,
    theta_y: float,
    wavelength: float,
) -> Tuple[float, float]:
    """Convert illumination angles to k-space center position.

    Parameters
    ----------
    theta_x : float
        Illumination angle in x-direction (radians).
    theta_y : float
        Illumination angle in y-direction (radians).
    wavelength : float
        Optical wavelength in meters.

    Returns
    -------
    Tuple[float, float]
        k-space center [ky, kx] in 1/meters.

    Notes
    -----
    The relationship between angle and k-space position is:
    kx = sin(theta_x) / wavelength
    ky = sin(theta_y) / wavelength

    Examples
    --------
    >>> # 10 degree tilt in x-direction at 520nm wavelength
    >>> ky, kx = illumination_angle_to_k_center(
    ...     theta_x=np.radians(10),
    ...     theta_y=0,
    ...     wavelength=520e-9,
    ... )
    >>> print(f"kx = {kx:.2e} 1/m")
    kx = 3.34e+05 1/m
    """
    kx = np.sin(theta_x) / wavelength
    ky = np.sin(theta_y) / wavelength
    return (ky, kx)


def k_center_to_illumination_angle(
    k_center: Union[List[float], Tuple[float, float]],
    wavelength: float,
) -> Tuple[float, float]:
    """Convert k-space center position to illumination angles.

    Parameters
    ----------
    k_center : List[float] or Tuple[float, float]
        k-space center [ky, kx] in 1/meters.
    wavelength : float
        Optical wavelength in meters.

    Returns
    -------
    Tuple[float, float]
        Illumination angles (theta_y, theta_x) in radians.

    Raises
    ------
    ValueError
        If the k-space position corresponds to an evanescent wave
        (|k| > 1/wavelength).

    Examples
    --------
    >>> theta_y, theta_x = k_center_to_illumination_angle(
    ...     k_center=[0, 3.34e5],
    ...     wavelength=520e-9,
    ... )
    >>> print(f"theta_x = {np.degrees(theta_x):.1f} degrees")
    theta_x = 10.0 degrees
    """
    ky, kx = k_center

    # Check for evanescent waves
    sin_theta_x = kx * wavelength
    sin_theta_y = ky * wavelength

    if abs(sin_theta_x) > 1.0:
        raise ValueError(
            f"k_x={kx:.2e} exceeds propagating wave limit for wavelength={wavelength:.2e}. "
            f"|sin(theta_x)| = {abs(sin_theta_x):.2f} > 1"
        )
    if abs(sin_theta_y) > 1.0:
        raise ValueError(
            f"k_y={ky:.2e} exceeds propagating wave limit for wavelength={wavelength:.2e}. "
            f"|sin(theta_y)| = {abs(sin_theta_y):.2f} > 1"
        )

    theta_x = np.arcsin(sin_theta_x)
    theta_y = np.arcsin(sin_theta_y)

    return (theta_y, theta_x)
