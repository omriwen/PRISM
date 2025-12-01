"""Fourier transform utilities for k-space operations.

This module provides utility functions for k-space shift calculations and
conversions between illumination parameters and k-space positions. These
utilities support the scanning illumination forward model and enable
validation of k-space sampling within optical system constraints.

Functions
---------
illum_position_to_k_shift
    Convert spatial illumination position to k-space shift.
k_shift_to_illum_position
    Convert k-space shift to spatial illumination position.
k_shift_to_illum_angle
    Convert k-shift to physical illumination angle.
illum_angle_to_k_shift
    Convert illumination angle to k-space shift.
validate_k_shift_within_na
    Check if k-shift is within objective NA.
pixel_to_k_shift
    Convert pixel-space shift to k-space shift.
k_shift_to_pixel
    Convert k-space shift to pixel coordinates.
pixel_to_spatial
    Convert pixel shift to spatial position in meters.
spatial_to_pixel
    Convert spatial position to pixel shift.
spatial_position_to_effective_k
    Calculate effective k-space shift from spatial source position.

Examples
--------
Convert illumination angle to k-space position:

>>> from prism.core.optics.fourier_utils import illum_angle_to_k_shift
>>> import numpy as np
>>> k_shift = illum_angle_to_k_shift(
...     theta=np.radians(10),  # 10 degree tilt
...     wavelength=520e-9,
... )
>>> print(f"k_shift = {k_shift:.2e} 1/m")

Validate k-shift is within objective NA:

>>> from prism.core.optics.fourier_utils import validate_k_shift_within_na
>>> is_valid = validate_k_shift_within_na(
...     k_shift=[0.5e6, 0.5e6],
...     na=1.4,
...     wavelength=520e-9,
...     medium_index=1.515,
... )

See Also
--------
spids.core.optics.illumination : Illumination source models
spids.core.instruments.microscope : Microscope with scanning illumination
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..grid import Grid


def illum_angle_to_k_shift(
    theta: float,
    wavelength: float,
    medium_index: float = 1.0,
) -> float:
    """Convert illumination angle to k-space shift magnitude.

    For a tilted plane wave at angle theta from the optical axis,
    the k-space shift is: k = n * sin(theta) / lambda

    Parameters
    ----------
    theta : float
        Illumination angle from optical axis in radians.
    wavelength : float
        Optical wavelength in meters.
    medium_index : float, default=1.0
        Refractive index of the medium (1.0 for air, 1.33 for water, 1.515 for oil).

    Returns
    -------
    float
        k-space shift magnitude in 1/meters.

    Examples
    --------
    >>> # 10 degree tilt in air at 520nm
    >>> k = illum_angle_to_k_shift(np.radians(10), 520e-9)
    >>> print(f"k = {k:.2e} 1/m")
    k = 3.34e+05 1/m

    >>> # Same angle in oil immersion
    >>> k_oil = illum_angle_to_k_shift(np.radians(10), 520e-9, medium_index=1.515)
    >>> print(f"k_oil = {k_oil:.2e} 1/m")
    k_oil = 5.06e+05 1/m
    """
    return float(medium_index * np.sin(theta) / wavelength)


def k_shift_to_illum_angle(
    k_shift: float,
    wavelength: float,
    medium_index: float = 1.0,
) -> float:
    """Convert k-space shift magnitude to illumination angle.

    Inverse of illum_angle_to_k_shift. Returns the illumination angle
    that produces the given k-space shift.

    Parameters
    ----------
    k_shift : float
        k-space shift magnitude in 1/meters.
    wavelength : float
        Optical wavelength in meters.
    medium_index : float, default=1.0
        Refractive index of the medium.

    Returns
    -------
    float
        Illumination angle in radians.

    Raises
    ------
    ValueError
        If k_shift corresponds to an evanescent wave (|sin(theta)| > 1).

    Examples
    --------
    >>> theta = k_shift_to_illum_angle(3.34e5, 520e-9)
    >>> print(f"theta = {np.degrees(theta):.1f} degrees")
    theta = 10.0 degrees
    """
    sin_theta = k_shift * wavelength / medium_index

    if abs(sin_theta) > 1.0:
        raise ValueError(
            f"k_shift={k_shift:.2e} exceeds propagating wave limit. "
            f"|sin(theta)| = {abs(sin_theta):.2f} > 1. "
            f"This corresponds to an evanescent wave."
        )

    return float(np.arcsin(sin_theta))


def illum_position_to_k_shift(
    position: Union[List[float], Tuple[float, float]],
    focal_length: float,
    wavelength: float,
) -> Tuple[float, float]:
    """Convert spatial illumination position to k-space shift.

    For an illumination source at position (x, y) in the back focal plane
    of a lens, the resulting k-space shift is:
    (ky, kx) = (y, x) / (wavelength * focal_length)

    This is used when the illumination is created by placing a source
    (e.g., LED) at a specific position in the pupil plane.

    Parameters
    ----------
    position : List[float] or Tuple[float, float]
        Spatial position [y, x] of illumination source in meters.
    focal_length : float
        Focal length of the condenser/objective lens in meters.
    wavelength : float
        Optical wavelength in meters.

    Returns
    -------
    Tuple[float, float]
        k-space shift [ky, kx] in 1/meters.

    Examples
    --------
    >>> # LED at 1mm off-axis in pupil plane
    >>> ky, kx = illum_position_to_k_shift(
    ...     position=[0.001, 0],  # 1mm in y
    ...     focal_length=0.1,     # 100mm focal length
    ...     wavelength=520e-9,
    ... )
    >>> print(f"ky = {ky:.2e} 1/m")
    ky = 1.92e+04 1/m
    """
    y, x = position
    kx = x / (wavelength * focal_length)
    ky = y / (wavelength * focal_length)
    return (ky, kx)


def k_shift_to_illum_position(
    k_shift: Union[List[float], Tuple[float, float]],
    focal_length: float,
    wavelength: float,
) -> Tuple[float, float]:
    """Convert k-space shift to spatial illumination position.

    Inverse of illum_position_to_k_shift. Returns the pupil plane
    position that produces the given k-space shift.

    Parameters
    ----------
    k_shift : List[float] or Tuple[float, float]
        k-space shift [ky, kx] in 1/meters.
    focal_length : float
        Focal length of the condenser/objective lens in meters.
    wavelength : float
        Optical wavelength in meters.

    Returns
    -------
    Tuple[float, float]
        Spatial position [y, x] in meters.

    Examples
    --------
    >>> y, x = k_shift_to_illum_position(
    ...     k_shift=[1.92e4, 0],
    ...     focal_length=0.1,
    ...     wavelength=520e-9,
    ... )
    >>> print(f"y = {y*1000:.2f} mm")
    y = 1.00 mm
    """
    ky, kx = k_shift
    x = kx * wavelength * focal_length
    y = ky * wavelength * focal_length
    return (y, x)


def validate_k_shift_within_na(
    k_shift: Union[List[float], Tuple[float, float], float],
    na: float,
    wavelength: float,
    medium_index: float = 1.0,
    tolerance: float = 1e-6,
) -> bool:
    """Check if k-shift is within objective numerical aperture.

    The NA defines the maximum collectible k-space radius:
    k_max = NA / (medium_index * wavelength)

    This function checks whether the given k-shift falls within this limit.

    Parameters
    ----------
    k_shift : List[float], Tuple[float, float], or float
        k-space shift. If 2-element list/tuple, interpreted as [ky, kx].
        If scalar, interpreted as magnitude.
    na : float
        Numerical aperture of the objective.
    wavelength : float
        Optical wavelength in meters.
    medium_index : float, default=1.0
        Refractive index of the medium.
    tolerance : float, default=1e-6
        Relative tolerance for boundary comparison.

    Returns
    -------
    bool
        True if k-shift is within NA, False otherwise.

    Examples
    --------
    >>> # Check if 10 degree tilt is within NA=0.3
    >>> k = illum_angle_to_k_shift(np.radians(10), 520e-9)
    >>> is_valid = validate_k_shift_within_na(k, na=0.3, wavelength=520e-9)
    >>> print(is_valid)
    True

    >>> # Check 2D k-shift
    >>> is_valid = validate_k_shift_within_na(
    ...     [0.5e6, 0.5e6],
    ...     na=1.4,
    ...     wavelength=520e-9,
    ...     medium_index=1.515,
    ... )
    """
    # Maximum k within NA
    k_max = na / (medium_index * wavelength)

    # Compute k-shift magnitude
    if isinstance(k_shift, (list, tuple)):
        ky, kx = k_shift
        k_magnitude = np.sqrt(ky**2 + kx**2)
    else:
        k_magnitude = abs(k_shift)

    # Compare with tolerance
    return bool(k_magnitude <= k_max * (1 + tolerance))


def compute_na_from_k_shift(
    k_shift: Union[List[float], Tuple[float, float], float],
    wavelength: float,
    medium_index: float = 1.0,
) -> float:
    """Compute the effective NA corresponding to a k-space shift.

    This is the minimum NA required to collect light at the given k-shift.

    Parameters
    ----------
    k_shift : List[float], Tuple[float, float], or float
        k-space shift. If 2-element list/tuple, interpreted as [ky, kx].
        If scalar, interpreted as magnitude.
    wavelength : float
        Optical wavelength in meters.
    medium_index : float, default=1.0
        Refractive index of the medium.

    Returns
    -------
    float
        Effective NA corresponding to the k-shift.

    Examples
    --------
    >>> na_eff = compute_na_from_k_shift(3.34e5, 520e-9)
    >>> print(f"Effective NA = {na_eff:.3f}")
    Effective NA = 0.174
    """
    # Compute k-shift magnitude
    if isinstance(k_shift, (list, tuple)):
        ky, kx = k_shift
        k_magnitude = np.sqrt(ky**2 + kx**2)
    else:
        k_magnitude = abs(k_shift)

    # NA = k * medium_index * wavelength
    return float(k_magnitude * medium_index * wavelength)


def pixel_to_k_shift(
    pixel_shift: Union[List[float], Tuple[float, float]],
    grid: Grid,
) -> Tuple[float, float]:
    """Convert pixel-space shift to k-space shift.

    Converts a shift specified in pixel units (relative to DC) to
    k-space coordinates in 1/meters.

    Parameters
    ----------
    pixel_shift : List[float] or Tuple[float, float]
        Shift in pixel units [py, px] from DC.
    grid : Grid
        Spatial grid defining the coordinate system.

    Returns
    -------
    Tuple[float, float]
        k-space shift [ky, kx] in 1/meters.

    Examples
    --------
    >>> grid = Grid(nx=256, dx=1e-6, wavelength=520e-9)
    >>> ky, kx = pixel_to_k_shift([10, 5], grid)
    >>> print(f"ky = {ky:.2e}, kx = {kx:.2e}")
    """
    py, px = pixel_shift

    # k-space resolution: dk = 1 / (n * dx) = 1 / FOV
    dkx = 1.0 / (grid.nx * grid.dx)
    dky = 1.0 / (grid.ny * grid.dy)

    kx = px * dkx
    ky = py * dky

    return (ky, kx)


def k_shift_to_pixel(
    k_shift: Union[List[float], Tuple[float, float]],
    grid: Grid,
) -> Tuple[float, float]:
    """Convert k-space shift to pixel coordinates.

    Converts a k-space shift in 1/meters to pixel units relative to DC.

    Parameters
    ----------
    k_shift : List[float] or Tuple[float, float]
        k-space shift [ky, kx] in 1/meters.
    grid : Grid
        Spatial grid defining the coordinate system.

    Returns
    -------
    Tuple[float, float]
        Pixel shift [py, px] from DC.

    Examples
    --------
    >>> grid = Grid(nx=256, dx=1e-6, wavelength=520e-9)
    >>> py, px = k_shift_to_pixel([1e4, 5e3], grid)
    >>> print(f"py = {py:.1f}, px = {px:.1f}")
    """
    ky, kx = k_shift

    # k-space resolution: dk = 1 / (n * dx) = 1 / FOV
    dkx = 1.0 / (grid.nx * grid.dx)
    dky = 1.0 / (grid.ny * grid.dy)

    px = kx / dkx
    py = ky / dky

    return (py, px)


def compute_k_space_coverage(
    centers: Union[List[List[float]], Tensor],
    radius: float,
    grid: Grid,
) -> Tensor:
    """Compute cumulative k-space coverage from multiple apertures.

    Creates a mask showing which regions of k-space are covered by
    a set of circular apertures at the given centers.

    Parameters
    ----------
    centers : List[List[float]] or Tensor
        List of k-space centers [[ky1, kx1], [ky2, kx2], ...] in 1/meters,
        or Tensor of shape (N, 2).
    radius : float
        Radius of each aperture in k-space (1/meters).
    grid : Grid
        Spatial grid defining the k-space coordinate system.

    Returns
    -------
    Tensor
        Boolean mask of shape (grid.ny, grid.nx) showing covered regions.

    Examples
    --------
    >>> grid = Grid(nx=256, dx=1e-6, wavelength=520e-9)
    >>> centers = [[0, 0], [0.1e6, 0], [-0.1e6, 0]]
    >>> coverage = compute_k_space_coverage(centers, radius=0.05e6, grid=grid)
    >>> coverage_fraction = coverage.sum() / coverage.numel()
    """
    # Convert centers to tensor if needed
    if isinstance(centers, list):
        centers_tensor = torch.tensor(centers, dtype=torch.float32)
    else:
        centers_tensor = centers.float()

    # Get k-space coordinates
    kx = grid.kx  # Shape: (1, nx)
    ky = grid.ky  # Shape: (ny, 1)

    # Initialize coverage mask
    coverage = torch.zeros(grid.ny, grid.nx, dtype=torch.bool, device=kx.device)

    # Add each aperture to coverage
    for i in range(centers_tensor.shape[0]):
        ky_c, kx_c = centers_tensor[i]

        # Distance from center
        dist = torch.sqrt((kx - kx_c) ** 2 + (ky - ky_c) ** 2)

        # Add to coverage
        coverage = coverage | (dist <= radius)

    return coverage


def aperture_center_equivalence(
    aperture_center: Union[List[float], Tuple[float, float]],
    grid: Grid,
) -> Tuple[float, float]:
    """Convert scanning aperture center to equivalent illumination k-shift.

    For the scanning illumination model, an aperture at position (ky, kx)
    is equivalent to illumination with k-shift (-ky, -kx) and detection
    at DC. This function computes that equivalence.

    Parameters
    ----------
    aperture_center : List[float] or Tuple[float, float]
        Aperture center [py, px] in pixel units from DC.
    grid : Grid
        Spatial grid defining the coordinate system.

    Returns
    -------
    Tuple[float, float]
        Equivalent illumination k-shift [ky, kx] in 1/meters.

    Notes
    -----
    By the Fourier shift theorem:
    - Scanning aperture at +k: samples O(k)
    - Scanning illumination at -k: shifts O to O(k-(-k))=O(k) at DC

    The sign convention ensures that both methods sample the same
    k-space region.

    Examples
    --------
    >>> grid = Grid(nx=256, dx=1e-6, wavelength=520e-9)
    >>> # Aperture at pixel position [10, 5]
    >>> ky, kx = aperture_center_equivalence([10, 5], grid)
    >>> # This gives the illumination k-shift that produces equivalent sampling
    """
    py, px = aperture_center

    # Convert to k-space
    ky, kx = pixel_to_k_shift([py, px], grid)

    # Negate for equivalence (aperture at +k ↔ illumination at -k)
    return (-ky, -kx)


def pixel_to_spatial(
    pixel_shift: Union[List[float], Tuple[float, float]],
    grid: Grid,
) -> Tuple[float, float]:
    """Convert pixel shift to spatial position in meters.

    Parameters
    ----------
    pixel_shift : List[float] or Tuple[float, float]
        Pixel shift [py, px] from center (DC).
    grid : Grid
        Spatial grid with dx, dy pixel sizes.

    Returns
    -------
    Tuple[float, float]
        Spatial position [y, x] in meters.
    """
    py, px = pixel_shift
    y = py * grid.dy
    x = px * grid.dx
    return (y, x)


def spatial_to_pixel(
    spatial_pos: Union[List[float], Tuple[float, float]],
    grid: Grid,
) -> Tuple[float, float]:
    """Convert spatial position (meters) to pixel shift.

    Parameters
    ----------
    spatial_pos : List[float] or Tuple[float, float]
        Spatial position [y, x] in meters.
    grid : Grid
        Spatial grid with dx, dy pixel sizes.

    Returns
    -------
    Tuple[float, float]
        Pixel shift [py, px] from center.
    """
    y, x = spatial_pos
    py = y / grid.dy
    px = x / grid.dx
    return (py, px)


def spatial_position_to_effective_k(
    spatial_pos: Union[List[float], Tuple[float, float]],
    source_distance: float,
    wavelength: float,
) -> Tuple[float, float]:
    """Calculate effective k-space shift from spatial source position.

    For a source at position (x₀, y₀, -z), the effective k-shift at
    the object center (0, 0, 0) is approximately:
    kx ≈ x₀ / (λ * z), ky ≈ y₀ / (λ * z)

    This is the paraxial approximation for small angles.

    Parameters
    ----------
    spatial_pos : List[float] or Tuple[float, float]
        Source position [y, x] in meters (in object plane coordinates).
    source_distance : float
        Source-to-object distance (z) in meters.
    wavelength : float
        Optical wavelength in meters.

    Returns
    -------
    Tuple[float, float]
        Effective k-shift [ky, kx] in 1/meters.
    """
    y0, x0 = spatial_pos
    kx = x0 / (wavelength * source_distance)
    ky = y0 / (wavelength * source_distance)
    return (ky, kx)
