"""
K-space sampling pattern generation for sparse aperture telescope measurements.

This module implements various sampling strategies for sparse aperture synthesis.
It generates k-space sampling patterns including:
- Fermat spiral: Optimized Fibonacci-based spiral for efficient coverage
- Star pattern: Radial lines at multiple angles
- Uniform random: Circle or square distributed samples
- Line sampling: Extended apertures along specified directions

The Fermat spiral is recommended for most applications as it provides
optimal k-space coverage with minimal samples. Each sampling function
returns coordinates in pixel space centered at origin.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from prism.types import TelescopeProtocol


# Type aliases for sampling
SamplingShape = Literal["circle", "square"]


# ============================================================================
# Random Sampling Utilities
# ============================================================================


def random_centered(*args: Any, **kwargs: Any) -> Tensor:
    """
    Create random tensor uniformly distributed between -1 and 1.

    Args:
        *args: Positional arguments passed to torch.rand
        **kwargs: Keyword arguments passed to torch.rand

    Returns:
        Tensor: Random values in range [-1, 1]

    Example:
        >>> points = random_centered(100, 2)  # 100 points in 2D, range [-1, 1]
        >>> assert points.min() >= -1 and points.max() <= 1
    """
    return 2 * torch.rand(*args, **kwargs) - 1


def rand_radius(*args: Any, **kwargs: Any) -> Tensor:
    """
    Create random tensor with values between 0 and 1 for uniform circle sampling.

    This function applies square root transformation to ensure that points (x, y)
    within a circle of radius 1 are uniformly distributed spatially. Without this
    correction, points would cluster near the center.

    Args:
        *args: Positional arguments passed to torch.rand
        **kwargs: Keyword arguments passed to torch.rand

    Returns:
        Tensor: Random radii with sqrt transform for uniform spatial distribution

    Example:
        >>> r = rand_radius(1000)
        >>> # Points will be uniformly distributed in circle, not clustered at center
        >>> theta = rand_angle(1000)
        >>> x, y = r * torch.cos(theta), r * torch.sin(theta)

    Notes:
        The sqrt transformation compensates for the fact that equal angular bins
        have larger area at larger radii (area scales with r²).
    """
    return torch.sqrt(torch.rand(*args, **kwargs))


def rand_angle(*args: Any, **kwargs: Any) -> Tensor:
    """
    Create random tensor uniformly distributed between 0 and 2π.

    Args:
        *args: Positional arguments passed to torch.rand
        **kwargs: Keyword arguments passed to torch.rand

    Returns:
        Tensor: Random angles in range [0, 2π]

    Example:
        >>> angles = rand_angle(100)  # 100 random angles
        >>> assert angles.min() >= 0 and angles.max() <= 2 * torch.pi
    """
    return 2 * torch.pi * torch.rand(*args, **kwargs)


# ============================================================================
# Geometry Utilities
# ============================================================================


def position_to_pixel(points: Tensor) -> Tensor:
    """
    Convert continuous positions to discrete pixel coordinates.

    Args:
        points (Tensor): Continuous coordinates, shape [..., 2]

    Returns:
        Tensor: Rounded pixel coordinates, same shape as input

    Example:
        >>> positions = torch.tensor([[1.7, 2.3], [-0.4, 3.8]])
        >>> pixels = position_to_pixel(positions)
        >>> # Result: [[2., 2.], [0., 4.]]
    """
    return torch.round(points)


def points_dist(p1: Tensor, p2: Tensor | int = 0) -> Tensor:
    """
    Calculate Euclidean distance between points.

    Args:
        p1 (Tensor): Points with shape (n_points, 2) or (n_points, 3)
        p2 (Tensor or float): Second set of points or origin (default: 0)

    Returns:
        Tensor: Distances with shape (n_points,)

    Example:
        >>> p1 = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
        >>> distances = points_dist(p1)  # Distance from origin
        >>> # Result: [5.0, 5.0]
        >>> p2 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        >>> distances = points_dist(p1, p2)  # Distance between point pairs

    Notes:
        If p2 is not provided, distances are computed from the origin.
        Works for both 2D and 3D points.
    """
    result: Tensor = (p1 - p2).norm(dim=-1)  # type: ignore[assignment]
    return result


def sort_by_radius(points: Tensor) -> Tensor:
    """
    Sort points by their distance from the origin.

    Useful for sequential sampling strategies where we want to start from
    the center and move outward (low to high spatial frequencies).

    Args:
        points (Tensor): Points with shape (n_points, 2) or (n_points, 3)

    Returns:
        Tensor: Sorted points with same shape, ordered by increasing distance

    Example:
        >>> points = torch.tensor([[5.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
        >>> sorted_pts = sort_by_radius(points)
        >>> # Result: [[1., 1.], [3., 0.], [5., 0.]]  (distances: 1.41, 3, 5)

    Notes:
        This is commonly used for progressive sampling schemes where
        low spatial frequencies (near center) are measured first.
    """
    return points[torch.argsort(points_dist(points))]


def points_energy(obj: Tensor, points: Tensor, telescope: TelescopeProtocol) -> Tensor:
    """
    Calculate measurement energy for each sampling point.

    This function computes the signal energy (L2 norm) that would be measured
    at each point in k-space. Higher energy points provide more information
    and can be prioritized in adaptive sampling schemes.

    Args:
        obj (Tensor): The object to be measured, shape [C, H, W]
        points (Tensor): K-space sampling centers, shape [N, 2]
        telescope: Telescope instance used to measure the object

    Returns:
        Tensor: Energy for each point, shape [N]

    Example:
        >>> from prism.core.instruments import Telescope, TelescopeConfig
        >>> config = TelescopeConfig(n_pixels=256, aperture_radius_pixels=10)
        >>> telescope = Telescope(config)
        >>> obj = torch.randn(1, 256, 256)
        >>> points = torch.randn(50, 2)
        >>> energies = points_energy(obj, points, telescope)
        >>> # energies.shape: [50]

    Notes:
        - Vectorized implementation: ~5-10x faster than loop for N > 50
        - Energy computed as L2 norm of measurement
        - Useful for energy-based adaptive sampling strategies
    """
    # Convert points to list of lists format expected by telescope
    if points.dim() == 2:
        centers_list = points.tolist()
    else:
        centers_list = [point.tolist() for point in points]

    # Batch process all measurements at once
    # telescope() returns (N, 1, H, W) when given multiple centers
    measurements = telescope(obj, centers=centers_list)  # (N, 1, H, W)

    # Compute norm for each measurement
    # Flatten spatial dimensions and compute L2 norm
    energies: Tensor = measurements.view(measurements.shape[0], -1).norm(dim=1)  # type: ignore[assignment]  # (N,)

    return energies


def sort_by_energy(obj: Tensor, points: Tensor, telescope: TelescopeProtocol) -> Tensor:
    """
    Sort sampling points by their measurement energy (descending).

    This function prioritizes high-energy measurements which typically provide
    more information content. Useful for adaptive sampling strategies.

    Args:
        obj (Tensor): The object to be measured, shape [C, H, W]
        points (Tensor): K-space sampling centers, shape [N, D, 2] or [N, 2]
        telescope: Telescope instance used to measure the object

    Returns:
        Tensor: Points sorted by energy (highest first), same shape as input

    Example:
        >>> from prism.core.instruments import Telescope, TelescopeConfig
        >>> config = TelescopeConfig(n_pixels=256, aperture_radius_pixels=10)
        >>> telescope = Telescope(config)
        >>> obj = torch.randn(1, 256, 256)
        >>> points = torch.randn(50, 1, 2)
        >>> sorted_pts = sort_by_energy(obj, points, telescope)
        >>> # Most informative sampling points are first

    Notes:
        Sorts in descending order (highest energy first).
        Preserves the device of input points.
    """
    energies = points_energy(obj, points.squeeze(), telescope)
    sorted_indices = torch.argsort(energies, descending=True).to(points.device)
    return points[sorted_indices]


# ============================================================================
# Uniform Sampling Patterns
# ============================================================================


def uniform_circle(n_points: int, diameter: float = 1.0) -> Tensor:
    """
    Generate uniformly distributed points within a circle.

    Args:
        n_points (int): Number of points to generate
        diameter (float): Circle diameter in pixels (default: 1.0)

    Returns:
        Tensor: Points with shape (n_points, 2), sorted by radius

    Example:
        >>> points = uniform_circle(100, diameter=50.0)
        >>> # 100 points uniformly distributed in circle of radius 25 pixels

    Notes:
        - First point is always at the origin (center)
        - Points sorted by increasing distance from center
        - Uses sqrt transform for true uniform spatial distribution
    """
    r = 0.5 * diameter * rand_radius(n_points)

    # Ensure first point is at center
    idx = torch.argsort(r, descending=True)
    r[idx[0]] = 0

    theta = rand_angle(n_points)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    points = torch.stack((y, x), dim=-1)

    return sort_by_radius(points)


def uniform_square(n_points: int, side: float = 1.0) -> Tensor:
    """
    Generate uniformly distributed points within a square.

    Args:
        n_points (int): Number of points to generate
        side (float): Square side length in pixels (default: 1.0)

    Returns:
        Tensor: Points with shape (n_points, 2), sorted by radius

    Example:
        >>> points = uniform_square(100, side=50.0)
        >>> # 100 points uniformly distributed in 50x50 square

    Notes:
        Points are sorted by distance from center for consistency
        with circular sampling.
    """
    points = 0.5 * side * random_centered(n_points, 2)
    return sort_by_radius(points)


def generate_sample_centers(
    n_points: int, size: float = 1.0, shape: SamplingShape = "circle"
) -> Tensor:
    """
    Generate uniformly distributed sampling centers within a given shape.

    Args:
        n_points (int): Number of sample centers to generate
        size (float): Size of sampling region (diameter for circle, side for square)
        shape (SamplingShape): Shape of sampling region ("circle" or "square")

    Returns:
        Tensor: Sample centers with shape (n_points, 1, 2)

    Raises:
        ValueError: If shape is not "circle" or "square"

    Example:
        >>> centers = generate_sample_centers(50, size=100, shape="circle")
        >>> centers.shape
        torch.Size([50, 1, 2])

    Notes:
        Returns shape (n_points, 1, 2) to be compatible with line sampling
        functions that need a center dimension.
    """
    if shape == "circle":
        samples = uniform_circle(n_points, size)
    elif shape == "square":
        samples = uniform_square(n_points, size)
    else:
        msg = "Shape must be either 'circle' or 'square'"
        raise ValueError(msg)
    return samples.unsqueeze(1)


# ============================================================================
# Line Sampling Patterns
# ============================================================================


def centers_to_random_ends(
    centers: Tensor, length: float = 10, angle: Optional[float] = None
) -> Tensor:
    """
    Generate line endpoints from center points.

    Creates lines of specified length centered at given points. Line orientations
    are either random or fixed to a specific angle.

    Args:
        centers (Tensor): Line centers with shape (n_points, 1, 2)
        length (float): Line length in pixels (default: 10)
        angle (float or None): Fixed line angle in radians, or None for random angles

    Returns:
        Tensor: Line endpoints with shape (n_points, 2, 2) where [:, 0, :] is start
                and [:, 1, :] is end point

    Example:
        >>> centers = torch.zeros(10, 1, 2)  # 10 lines centered at origin
        >>> ends = centers_to_random_ends(centers, length=20.0)
        >>> ends.shape
        torch.Size([10, 2, 2])
        >>> # Can specify fixed angle
        >>> ends_h = centers_to_random_ends(centers, length=20.0, angle=0.0)

    Notes:
        When angle is None, each line gets a random orientation.
        Lines are symmetric about their centers.
    """
    n_points = centers.shape[0]
    line_angles = rand_angle(n_points)

    # Override with fixed angle if provided
    if angle is not None:
        line_angles = torch.full_like(line_angles, angle)

    # Compute half-length displacement vectors
    # dr is shape (n_points, 1, 2) with [dy, dx] components
    dr = (
        0.5
        * length
        * torch.stack((torch.sin(line_angles), torch.cos(line_angles)), dim=-1).unsqueeze(1)
    )

    # Return [start, end] endpoints
    return centers + torch.cat([-dr, dr], dim=1)


def generate_line_ends(
    n_points: int, length: float = 10, size: float = 10, shape: SamplingShape = "circle"
) -> Tensor:
    """
    Generate line sampling endpoints uniformly distributed within a shape.

    Args:
        n_points (int): Number of lines to generate
        length (float): Line length in pixels
        size (float): Size of region for line centers (adjusted to keep lines in bounds)
        shape (SamplingShape): Shape of sampling region ("circle" or "square")

    Returns:
        Tensor: Line endpoints with shape (n_points, 2, 2)

    Example:
        >>> line_ends = generate_line_ends(50, length=20, size=100, shape="circle")
        >>> # 50 random lines of length 20, centered within circle of diameter 100

    Notes:
        - First line is always centered at origin (0, 0)
        - Line centers are distributed within size - length/2 to keep endpoints in bounds
        - Line orientations are random
    """
    # Generate centers with reduced size to keep line ends within bounds
    line_centers = generate_sample_centers(n_points - 1, size=size - length / 2, shape=shape)

    # Prepend origin as first line center
    line_centers = torch.cat([torch.zeros(size=(1, 1, 2)), line_centers], dim=0)

    return centers_to_random_ends(line_centers, length)


def generate_samples(
    n_points: int, length: float = 10, size: float = 10, shape: SamplingShape = "circle"
) -> Tensor:
    """
    Generate sampling pattern (points or lines) uniformly distributed within a shape.

    This is a convenience function that handles both point sampling (length=0)
    and line sampling (length>0).

    Args:
        n_points (int): Number of samples to generate
        length (float): Sample length in pixels (0 for point sampling, >0 for lines)
        size (float): Size of sampling region
        shape (SamplingShape): Shape of sampling region ("circle" or "square")

    Returns:
        Tensor: Sample positions
            - If length=0: shape (n_points, 1, 2) for point sampling
            - If length>0: shape (n_points, 2, 2) for line endpoints

    Raises:
        ValueError: If length is negative

    Example:
        >>> # Point sampling
        >>> points = generate_samples(100, length=0, size=50, shape="circle")
        >>> points.shape
        torch.Size([100, 1, 2])
        >>> # Line sampling
        >>> lines = generate_samples(100, length=20, size=50, shape="circle")
        >>> lines.shape
        torch.Size([100, 2, 2])

    Notes:
        This is the main entry point for generating uniform random samples.
        Use Fermat spiral (generate_fermat_spiral) for better k-space coverage.
    """
    if length == 0:
        return generate_sample_centers(n_points, size, shape)
    elif length > 0:
        return generate_line_ends(n_points, length, size, shape)
    else:
        msg = "Length must be non-negative"
        raise ValueError(msg)


# ============================================================================
# Pattern Creation (Line Discretization)
# ============================================================================


def create_pattern(sample_ends: Tensor, length: int = 10) -> Tensor:
    """
    Convert line endpoints to discrete sample points.

    Takes a pair of endpoints and generates evenly-spaced points along the line.

    Args:
        sample_ends (Tensor): Line endpoints with shape (2, 2) [[y1, x1], [y2, x2]]
        length (int): Number of discrete points to generate along the line

    Returns:
        Tensor: Discrete points along line, shape (length, 2)

    Example:
        >>> ends = torch.tensor([[0.0, 0.0], [10.0, 10.0]])  # Diagonal line
        >>> points = create_pattern(ends, length=11)
        >>> points.shape
        torch.Size([11, 2])
        >>> # Points evenly spaced from (0,0) to (10,10)

    Notes:
        If length=0, returns the endpoints unchanged.
        Points are evenly distributed using linspace.
    """
    if length == 0:
        return sample_ends
    else:
        ys = torch.linspace(sample_ends[0, 0], sample_ends[1, 0], int(length)).view(-1)
        xs = torch.linspace(sample_ends[0, 1], sample_ends[1, 1], int(length)).view(-1)
        return torch.stack((ys, xs), dim=-1)


def create_patterns(
    sample_ends: Tensor, length: int = 10, length_rec: Optional[int] = None
) -> Tensor | Tuple[Tensor, Tensor]:
    """
    Create discrete patterns with optional dual sampling density.

    Allows creating two versions of the same line with different numbers of
    sample points (e.g., different resolution for measurement vs reconstruction).

    Args:
        sample_ends (Tensor): Line endpoints with shape (2, 2)
        length (int): Primary number of samples per line
        length_rec (int or None): Secondary number of samples (optional)

    Returns:
        Tensor or Tuple[Tensor, Tensor]:
            - If length_rec is None: single pattern
            - If length_rec is provided: tuple of (pattern1, pattern2)

    Example:
        >>> ends = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
        >>> # Single pattern
        >>> pattern = create_patterns(ends, length=10)
        >>> # Dual patterns (e.g., measurement vs reconstruction)
        >>> meas_pattern, rec_pattern = create_patterns(ends, length=5, length_rec=20)
    """
    if length_rec is None:
        return create_pattern(sample_ends, length)
    else:
        return create_pattern(sample_ends, length), create_pattern(sample_ends, length_rec)


# ============================================================================
# Advanced Sampling Patterns
# ============================================================================


def generate_star_pattern(
    n_angles: int = 4, length: float = 10, size: float = 10, shape: SamplingShape = "circle"
) -> Tensor:
    """
    Generate star-shaped radial sampling pattern.

    Creates radial lines emanating from center at evenly-spaced angles. Lines
    are distributed at multiple radii to achieve dense coverage.

    Args:
        n_angles (int): Number of radial directions (default: 4)
        length (float): Line length in pixels
        size (float): Maximum radius for line centers
        shape (SamplingShape): Unused, kept for API compatibility

    Returns:
        Tensor: Line endpoints with shape (n_lines, 2, 2)

    Example:
        >>> # 4-pointed star pattern
        >>> star = generate_star_pattern(n_angles=4, length=20, size=100)
        >>> # Creates lines at 0°, 45°, 90°, 135°, etc.

    Notes:
        - Lines are placed at radii from 0 to size/2 in steps of length
        - Angular spacing is π/n_angles (twice as dense as 2π/n_angles)
        - First line always passes through center
        - Useful for testing rotational symmetry in reconstructions
    """
    # Radial positions for line centers
    line_centers_r = torch.arange(0, size / 2 - length / 2, length)

    # Angular positions (π/n_angles spacing)
    line_angles = torch.arange(0, 2 * torch.pi, torch.pi / n_angles)

    lines_ends = torch.tensor([])

    for r in line_centers_r:
        for ang in line_angles:
            # Skip duplicate at origin (only include first angle)
            if r > 0 or ang < torch.pi:
                # Unit vector in angular direction
                angs = torch.tensor([torch.cos(ang), torch.sin(ang)]).unsqueeze(0).unsqueeze(0)

                # Line center at radius r
                line_center = r * angs

                # Half-length displacement
                dr = 0.5 * length * angs

                # Line endpoints
                line_ends = line_center + torch.cat([-dr, dr], dim=1)

                lines_ends = torch.cat([lines_ends, line_ends], dim=0)

    return lines_ends


def generate_fermat_spiral(
    n_points: int,
    r_max: float,
    length: float = 0,
    r_cutoff: Optional[float] = None,
    line_angle: Optional[float] = None,
) -> Tensor:
    """
    Generate Fermat (Fibonacci) spiral sampling pattern.

    The Fermat spiral provides optimal k-space coverage with minimal samples by
    using the golden angle (≈137.508°) between successive points. This is the
    recommended sampling pattern for most SPIDS applications.

    Args:
        n_points (int): Number of spiral points to generate
        r_max (float): Maximum radius of spiral (typically image_size / 2)
        length (float): Line length for extended apertures (0 for point sampling)
        r_cutoff (float or None): Optional maximum radius cutoff (filters outer points)
        line_angle (float or None): Fixed angle for lines (None for random, 0 for horizontal)

    Returns:
        Tensor: Sampling positions
            - If length=0: shape (n_points, 1, 2) for point samples
            - If length>0: shape (n_points, 2, 2) for line endpoints

    Example:
        >>> # Point sampling with Fermat spiral
        >>> points = generate_fermat_spiral(n_points=100, r_max=128, length=0)
        >>> points.shape
        torch.Size([100, 1, 2])
        >>> # Line sampling with horizontal apertures
        >>> lines = generate_fermat_spiral(
        ...     n_points=100, r_max=128, length=20, line_angle=0.0
        ... )
        >>> lines.shape
        torch.Size([100, 2, 2])

    Notes:
        Golden angle (φ ≈ 137.508°) ensures optimal coverage:
        - Avoids radial line artifacts from regular angular sampling
        - Provides approximately uniform density at all radii
        - Based on Fibonacci sequence (related to φ = (1+√5)/2)

        Radius scaling: r = c√n ensures uniform density (compensates for
        larger circumference at larger radii).

        Reference: Vogel, H. (1979). "A better way to construct the sunflower head".
        Mathematical Biosciences, 44(3-4), 179-189.
    """
    # Golden angle in radians (360° / φ where φ = golden ratio)
    phi0 = 137.508 * torch.pi / 180

    # Scaling constant for uniform density
    c = r_max / np.sqrt(n_points)

    # Generate spiral points
    n = torch.arange(n_points)
    r = c * torch.sqrt(n)

    # Apply radius cutoff if specified
    if r_cutoff is not None:
        r = r[r <= r_cutoff]
        n = torch.arange(len(r))

    # Angular positions using golden angle
    phi = phi0 * n

    # Convert polar to Cartesian coordinates
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    centers = torch.stack((y, x), dim=-1).unsqueeze(1)

    # Generate line endpoints if length > 0
    if length > 0:
        return centers_to_random_ends(centers, length, line_angle)
    else:
        return centers
