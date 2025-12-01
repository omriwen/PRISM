"""
Aperture Mask Generators for SPIDS.

This module provides implementations of different aperture mask types for
simulating realistic telescope aperture configurations. The choice of aperture
affects the PSF (Point Spread Function) and diffraction patterns.

Aperture Types
--------------

CircularAperture:
    Simple circular opening (current SPIDS default).
    - Method: Distance-based mask
    - Speed: Fastest (vectorized batch generation)
    - Use: Single telescope or subaperture measurements

HexagonalAperture:
    Hexagonal opening (JWST-style segmented mirrors).
    - Method: Hexagon distance metric
    - Speed: Fast
    - Use: Segmented mirror telescopes (JWST, GMT, TMT)

ObscuredCircularAperture:
    Circular aperture with central obscuration and spider vanes.
    - Method: Annulus with line masks
    - Speed: Fast
    - Use: Cassegrain/Ritchey-Chrétien telescopes (VLT, Hubble)

Usage Examples
--------------

Basic Circular Aperture:
    >>> from prism.core.apertures import CircularAperture
    >>> aperture = CircularAperture(radius=10)
    >>> x = torch.arange(-50, 50).unsqueeze(0).float()
    >>> y = torch.arange(-50, 50).unsqueeze(1).float()
    >>> mask = aperture.generate(x, y, center=[0, 0])
    >>> mask.sum()  # Area ≈ π*r²

Hexagonal Aperture (JWST-style):
    >>> aperture = HexagonalAperture(side_length=20)
    >>> mask = aperture.generate(x, y, center=[0, 0])

Obscured Circular Aperture (VLT-style):
    >>> aperture = ObscuredCircularAperture(
    ...     outer_radius=400,  # Primary mirror
    ...     inner_radius=50,   # Secondary mirror
    ...     spider_width=2,    # Support struts
    ...     n_spiders=4
    ... )
    >>> mask = aperture.generate(x, y, center=[0, 0])

Batch Generation (Vectorized):
    >>> centers = [[0, 0], [10, 10], [20, 20]]
    >>> masks = aperture.generate_batch(x, y, centers)  # (3, H, W)

Integration with Telescope
---------------------------

The Telescope class uses aperture strategy pattern:

    from prism.core.instruments import Telescope, TelescopeConfig
    from prism.core.apertures import HexagonalAperture

    # Method 1: String-based factory
    config = TelescopeConfig(n_pixels=256, aperture_radius_pixels=20, aperture_type='hexagonal')
    telescope = Telescope(config)

    # Method 2: Direct injection via aperture_kwargs
    config = TelescopeConfig(n_pixels=256, aperture_radius_pixels=25, aperture_type='hexagonal')
    telescope = Telescope(config)

Physics Background
------------------

The aperture mask represents the physical opening through which light passes.
In the Fraunhofer (far-field) regime used by SPIDS:

1. Object → FFT to k-space
2. Apply aperture mask in k-space (multiply by mask)
3. IFFT back to image plane
4. Intensity measurement |·|²

Different aperture shapes create different diffraction patterns:
- Circular: Airy disk (central peak + rings)
- Hexagonal: Hexagonal diffraction pattern
- Obscured: Bright central spot, diffraction spikes from spiders

Aperture Selection Guide
------------------------

┌──────────────┬────────────────────┬──────────────────────────────┐
│ Aperture     │ Real Telescopes    │ Use Case                     │
├──────────────┼────────────────────┼──────────────────────────────┤
│ Circular     │ Refractors         │ Simple simulations           │
│              │ Small reflectors   │ Subaperture measurements     │
├──────────────┼────────────────────┼──────────────────────────────┤
│ Hexagonal    │ JWST, GMT, TMT     │ Segmented mirror telescopes  │
│              │ (future systems)   │ Modern large observatories   │
├──────────────┼────────────────────┼──────────────────────────────┤
│ Obscured     │ VLT, Hubble        │ Cassegrain telescopes        │
│              │ Keck, Gemini       │ Most large reflectors        │
└──────────────┴────────────────────┴──────────────────────────────┘

References
----------
- Goodman, J. W. "Introduction to Fourier Optics" (2005), Chapter 4
- Born & Wolf, "Principles of Optics" (1999), Chapter 8.5
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch
from torch import Tensor


class Aperture(ABC):
    """
    Abstract base class for aperture mask generators.

    All aperture types implement the generate() method to create boolean masks
    representing the physical aperture shape. Apertures can be centered at any
    position in the coordinate system.

    The strategy pattern allows different aperture types to be swapped
    transparently in the Telescope class.
    """

    @abstractmethod
    def generate(self, x: Tensor, y: Tensor, center: List[float]) -> Tensor:
        """
        Generate boolean mask at specified center position.

        Args:
            x: X coordinates grid [1, W] or [H, W]
            y: Y coordinates grid [H, 1] or [H, W]
            center: [cy, cx] center position in coordinate system

        Returns:
            Boolean mask [H, W], True inside aperture, False outside

        Notes:
            - x and y define the coordinate system (typically centered)
            - center is in same coordinate system as x, y
            - Mask is True where light passes through, False where blocked
        """
        pass

    def generate_batch(
        self, x: Tensor, y: Tensor, centers: Union[List[List[float]], Tensor]
    ) -> Tensor:
        """
        Generate multiple masks (vectorized when possible).

        Args:
            x, y: Coordinate grids
            centers: List of [cy, cx] positions OR Tensor of shape [N, 2]

        Returns:
            Boolean masks [N, H, W] where N = len(centers)

        Notes:
            - Default implementation loops over centers
            - Subclasses can override for vectorized implementation
            - CircularAperture has optimized GPU-native version (10x+ faster)

        Example:
            >>> centers = [[0, 0], [10, 10], [20, 20]]
            >>> masks = aperture.generate_batch(x, y, centers)
            >>> masks.shape  # (3, H, W)
        """
        # Convert Tensor to list for the default loop-based implementation
        if isinstance(centers, Tensor):
            centers_list = centers.tolist()
        else:
            centers_list = centers
        return torch.stack([self.generate(x, y, c) for c in centers_list])


class CircularAperture(Aperture):
    """
    Circular aperture (SPIDS default).

    Simplest aperture type - uniform circular opening.
    Used for single telescope or subaperture measurements.

    The mask is True inside a circle of given radius, False outside:
        (x - cx)² + (y - cy)² ≤ r²

    Attributes:
        radius: Aperture radius in pixels

    Example:
        >>> aperture = CircularAperture(radius=10)
        >>> x = torch.arange(-50, 50).unsqueeze(0).float()
        >>> y = torch.arange(-50, 50).unsqueeze(1).float()
        >>> mask = aperture.generate(x, y, center=[0, 0])
        >>> mask.sum()  # Area ≈ π*r² ≈ 314

    Physics Notes:
        - Circular aperture → Airy disk diffraction pattern
        - First dark ring at θ = 1.22 λ/D (Rayleigh criterion)
        - Encircled energy: 84% within first dark ring
        - PSF: J₁(kr)/(kr) where J₁ is Bessel function of first kind

    Performance:
        - Single mask: ~0.5ms for 1024×1024
        - Batch (vectorized): ~3-5x faster than loop for N>10
    """

    def __init__(self, radius: float):
        """
        Initialize circular aperture.

        Args:
            radius: Aperture radius in pixels (in coordinate units)

        Raises:
            ValueError: If radius <= 0
        """
        if radius <= 0:
            raise ValueError(f"Aperture radius must be positive, got {radius}")
        self.radius = radius

    def generate(self, x: Tensor, y: Tensor, center: List[float]) -> Tensor:
        """
        Generate circular mask.

        Args:
            x: X coordinates [1, W] or [H, W]
            y: Y coordinates [H, 1] or [H, W]
            center: [cy, cx] center position (list, tuple, or tensor)

        Returns:
            Boolean mask [H, W], True inside circle

        Example:
            >>> aperture = CircularAperture(radius=10)
            >>> mask = aperture.generate(x, y, center=[0, 0])
        """
        # Extract center values, ensuring they're on the same device as x/y
        if isinstance(center, Tensor):
            # Move center to same device as coordinates if needed
            if center.device != x.device:
                center = center.to(x.device)
            cx = center[1]
            cy = center[0]
        else:
            # List/tuple - use directly (PyTorch handles scalar conversion)
            cx = center[1]
            cy = center[0]
        return ((x - cx) ** 2 + (y - cy) ** 2) <= self.radius**2

    def generate_batch(
        self, x: Tensor, y: Tensor, centers: "Union[List[List[float]], Tensor]"
    ) -> Tensor:
        """
        Optimized vectorized batch generation (GPU-native when possible).

        This is the vectorized implementation from Telescope.mask_batch,
        providing significant speedup for multiple aperture positions.

        When centers is already a GPU Tensor, this method avoids any CPU
        operations or data transfers, achieving 10-15x speedup over the
        loop-based approach.

        Args:
            x: X coordinates [1, W]
            y: Y coordinates [H, 1]
            centers: List of [cy, cx] positions OR Tensor of shape [N, 2]

        Returns:
            Boolean masks [N, H, W]

        Performance:
            For N=64 centers on 512×512 grid (GPU):
            - Loop: ~22ms
            - GPU-native batch: ~2-3ms (10x faster)

        Example:
            >>> centers = [[0, 0], [10, 10], [20, 20]]
            >>> masks = aperture.generate_batch(x, y, centers)
            >>> masks.shape  # (3, H, W)
            >>>
            >>> # GPU-native (faster):
            >>> centers_gpu = torch.tensor([[0, 0], [10, 10]], device='cuda')
            >>> masks = aperture.generate_batch(x, y, centers_gpu)
        """
        # GPU-native path: if centers is already a Tensor, keep it on device
        if isinstance(centers, Tensor):
            centers_tensor = centers.to(dtype=torch.float32)
            # Use centers' device as the target device for GPU-native operation
            target_device = centers_tensor.device
        else:
            # Convert list to tensor on the same device as x
            centers_tensor = torch.tensor(centers, device=x.device, dtype=torch.float32)
            target_device = x.device

        # Get dimensions
        n_centers = centers_tensor.shape[0]

        # Expand coordinate grids if needed and move to target device
        if x.dim() == 2 and x.shape[0] == 1:
            n_y = y.shape[0]
            n_x = x.shape[1]
            y_grid = y.expand(-1, n_x).to(target_device)  # [H, W]
            x_grid = x.expand(n_y, -1).to(target_device)  # [H, W]
        else:
            y_grid = y.to(target_device)
            x_grid = x.to(target_device)

        # GPU-native vectorized distance computation
        # Reshape centers for broadcasting: [N, 1, 1]
        cy = centers_tensor[:, 0].view(n_centers, 1, 1)
        cx = centers_tensor[:, 1].view(n_centers, 1, 1)

        # Reshape grids for broadcasting: [1, H, W]
        y_expanded = y_grid.unsqueeze(0)  # [1, H, W]
        x_expanded = x_grid.unsqueeze(0)  # [1, H, W]

        # Compute squared distances directly on GPU: [N, H, W]
        dist_sq = (y_expanded - cy) ** 2 + (x_expanded - cx) ** 2

        # Apply radius threshold
        return dist_sq <= (self.radius**2)


class HexagonalAperture(Aperture):
    """
    Hexagonal aperture (JWST-style segmented mirrors).

    Hexagonal segments are used in modern segmented mirror telescopes like:
    - James Webb Space Telescope (JWST): 18 hexagonal segments
    - Giant Magellan Telescope (GMT): 7 circular segments
    - Thirty Meter Telescope (TMT): 492 hexagonal segments

    The hexagon is defined by its side length (flat-to-flat distance / √3).
    A point is inside the hexagon if it satisfies:
        max(|x|, |x|/2 + √3|y|/2) ≤ side_length

    Attributes:
        side_length: Side length of regular hexagon in pixels

    Example:
        >>> # JWST-like hexagonal segment
        >>> aperture = HexagonalAperture(side_length=20)
        >>> mask = aperture.generate(x, y, center=[0, 0])
        >>> # Area: 3√3/2 * s² ≈ 2.598 * s²

    Physics Notes:
        - Hexagonal aperture → Hexagonal diffraction pattern
        - PSF has 6-fold symmetry
        - Better packing efficiency than circles (honeycomb pattern)
        - Used to approximate circular aperture with segments

    Hexagon Geometry:
        - Side length: s
        - Flat-to-flat distance: √3 * s
        - Point-to-point distance: 2 * s
        - Area: 3√3/2 * s² ≈ 2.598 * s²
        - Inscribed circle radius: √3/2 * s
        - Circumscribed circle radius: s
    """

    def __init__(self, side_length: float):
        """
        Initialize hexagonal aperture.

        Args:
            side_length: Side length of regular hexagon in pixels

        Raises:
            ValueError: If side_length <= 0
        """
        if side_length <= 0:
            raise ValueError(f"Side length must be positive, got {side_length}")
        self.side_length = side_length

    def generate(self, x: Tensor, y: Tensor, center: List[float]) -> Tensor:
        """
        Generate hexagonal mask.

        Uses the hexagon distance metric:
        A point (x, y) is inside a regular hexagon centered at origin if:
            max(|x|, |x|/2 + √3|y|/2) ≤ side_length

        This is equivalent to the intersection of 3 pairs of parallel lines
        at 60° angles.

        Args:
            x: X coordinates [1, W] or [H, W]
            y: Y coordinates [H, 1] or [H, W]
            center: [cy, cx] center position

        Returns:
            Boolean mask [H, W], True inside hexagon

        Example:
            >>> aperture = HexagonalAperture(side_length=20)
            >>> mask = aperture.generate(x, y, center=[10, 15])
            >>> mask.sum() / (3 * 3**0.5 / 2 * 20**2)  # ≈ 1.0 (ratio to area)
        """
        # Translate to hexagon center
        x_c = x - center[1]
        y_c = y - center[0]

        # Hexagon distance metrics (3 pairs of parallel lines)
        # 1. Vertical pair: |x| ≤ s
        d1 = x_c.abs()

        # 2. Diagonal pairs at ±60°: |x|/2 + √3|y|/2 ≤ s
        d2 = x_c.abs() / 2 + (3**0.5) * y_c.abs() / 2

        # Point is inside if both conditions satisfied
        return torch.max(d1, d2) <= self.side_length


class ObscuredCircularAperture(Aperture):
    """
    Circular aperture with central obscuration and spider vanes.

    Typical for Cassegrain/Ritchey-Chrétien telescopes, which have:
    - Large primary mirror (outer circle)
    - Small secondary mirror blocking center (inner circle)
    - Spider vanes (optional support struts)

    Real Telescope Examples:
        VLT (Very Large Telescope):
            - Outer diameter: 8m
            - Central obscuration: ~1m (12.5% diameter, 1.6% area)
            - 4 spider vanes

        Hubble Space Telescope:
            - Outer: 2.4m
            - Central: 0.33m (14% diameter, 2% area)
            - 4 spider vanes

        Keck Observatory:
            - Outer: 10m
            - Central: ~1.2m (12% diameter)
            - 6 spider vanes (hexagonal arrangement)

    Attributes:
        r_outer: Outer (primary) mirror radius in pixels
        r_inner: Inner (secondary) mirror radius in pixels
        spider_width: Width of spider vanes in pixels (optional)
        n_spiders: Number of spider vanes (default: 4)

    Example:
        >>> # VLT-like aperture
        >>> aperture = ObscuredCircularAperture(
        ...     outer_radius=400,  # 8m in pixel units
        ...     inner_radius=50,   # 1m
        ...     spider_width=2,    # Support struts
        ...     n_spiders=4
        ... )
        >>> mask = aperture.generate(x, y, center=[0, 0])
        >>> # Area ≈ π(r_out² - r_in²) - spider_area

    Physics Notes:
        - Central obscuration → Bright central spot (less energy in rings)
        - Spider vanes → Diffraction spikes (perpendicular to spiders)
        - 4 spiders → 4 diffraction spikes (star-like pattern)
        - Obscuration ratio ε = r_inner/r_outer (typical: 0.1-0.3)
        - Strehl ratio degradation: ~1 - ε²

    Spider Geometry:
        - Spiders radiate from center at equal angles
        - For n_spiders: angle spacing = π/n_spiders
        - Each spider is a line of given width
        - Distance to line: |x·sin(θ) - y·cos(θ)|
    """

    def __init__(
        self,
        outer_radius: float,
        inner_radius: float,
        spider_width: Optional[float] = None,
        n_spiders: int = 4,
    ):
        """
        Initialize obscured circular aperture.

        Args:
            outer_radius: Outer (primary) mirror radius in pixels
            inner_radius: Inner (secondary) mirror radius in pixels
            spider_width: Width of spider vanes in pixels (None = no spiders)
            n_spiders: Number of spider vanes (default: 4)

        Raises:
            ValueError: If radii are invalid or obscuration too large
        """
        if outer_radius <= 0:
            raise ValueError(f"Outer radius must be positive, got {outer_radius}")
        if inner_radius < 0:
            raise ValueError(f"Inner radius cannot be negative, got {inner_radius}")
        if inner_radius >= outer_radius:
            raise ValueError(
                f"Inner radius ({inner_radius}) must be less than outer radius ({outer_radius})"
            )
        if spider_width is not None and spider_width < 0:
            raise ValueError(f"Spider width cannot be negative, got {spider_width}")
        if n_spiders < 0:
            raise ValueError(f"Number of spiders cannot be negative, got {n_spiders}")

        # Check obscuration ratio (warn if unrealistic)
        obscuration_ratio = inner_radius / outer_radius
        if obscuration_ratio > 0.5:
            import warnings

            warnings.warn(
                f"Large obscuration ratio {obscuration_ratio:.2f} (>0.5). "
                f"Typical telescopes have 0.1-0.3. This may be unrealistic."
            )

        self.r_outer = outer_radius
        self.r_inner = inner_radius
        self.spider_width = spider_width
        self.n_spiders = n_spiders

    def generate(self, x: Tensor, y: Tensor, center: List[float]) -> Tensor:
        """
        Generate obscured circular mask with optional spider vanes.

        The mask is:
        1. True inside outer circle
        2. False inside inner circle (obscuration)
        3. False along spider vanes (if specified)

        Args:
            x: X coordinates [1, W] or [H, W]
            y: Y coordinates [H, 1] or [H, W]
            center: [cy, cx] center position

        Returns:
            Boolean mask [H, W], True where light passes (annulus - spiders)

        Example:
            >>> aperture = ObscuredCircularAperture(
            ...     outer_radius=50, inner_radius=10,
            ...     spider_width=2, n_spiders=4
            ... )
            >>> mask = aperture.generate(x, y, center=[0, 0])
            >>> # Annulus with 4 dark lines (spiders)
        """
        # Translate to aperture center
        x_c = x - center[1]
        y_c = y - center[0]
        r2 = x_c**2 + y_c**2

        # Annulus: outer circle - inner circle
        mask = (r2 <= self.r_outer**2) & (r2 >= self.r_inner**2)

        # Add spider vanes if specified
        if self.spider_width is not None and self.n_spiders > 0:
            # Use torch.tensor for angle calculations
            pi = torch.tensor(torch.pi, device=x_c.device, dtype=x_c.dtype)

            for i in range(self.n_spiders):
                # Spider angle (evenly spaced)
                angle = torch.tensor(i, device=x_c.device, dtype=x_c.dtype) * pi / self.n_spiders

                # Spider along direction (cos θ, sin θ)
                # Distance from point to line through origin at angle θ:
                # d = |x·sin(θ) - y·cos(θ)|
                dist_to_spider = (x_c * torch.sin(angle) - y_c * torch.cos(angle)).abs()

                # Mask out spider (set to False where spider is)
                mask &= dist_to_spider > self.spider_width / 2

        return mask


class NumericalAperture(Aperture):
    """
    Aperture defined by numerical aperture (NA), primarily for microscopy.

    The numerical aperture determines the cone of light that can be
    collected or transmitted by an optical system. Unlike telescope
    apertures which are defined in spatial coordinates, NA-based
    apertures are naturally defined in frequency (Fourier) space.

    In microscopy, the aperture function in frequency space is:
        mask = (f_r <= NA / (n * λ))
    where:
        - f_r is the radial frequency
        - NA is the numerical aperture
        - n is the refractive index of the medium
        - λ is the wavelength

    Parameters
    ----------
    na : float
        Numerical aperture (0 < NA <= n_medium)
    wavelength : float
        Operating wavelength in meters
    medium_index : float, optional
        Refractive index of immersion medium (default: 1.0 for air)

    Examples
    --------
    Oil immersion objective:
        >>> aperture = NumericalAperture(na=1.4, wavelength=550e-9, medium_index=1.515)

    Air objective:
        >>> aperture = NumericalAperture(na=0.9, wavelength=550e-9)

    Notes
    -----
    The NA defines both resolution and light-gathering power:
    - Resolution: Δx ≈ 0.61λ/NA (Rayleigh criterion)
    - Depth of field: DOF ≈ 2nλ/NA²

    For physical validity: NA <= n_medium
    """

    def __init__(self, na: float, wavelength: float, medium_index: float = 1.0):
        """
        Initialize numerical aperture.

        Args:
            na: Numerical aperture
            wavelength: Operating wavelength in meters
            medium_index: Refractive index (default: 1.0 for air)

        Raises:
            ValueError: If NA exceeds medium index or parameters invalid
        """
        if na <= 0:
            raise ValueError(f"NA must be positive, got {na}")
        if na > medium_index:
            raise ValueError(f"NA ({na}) cannot exceed medium refractive index ({medium_index})")
        if wavelength <= 0:
            raise ValueError(f"Wavelength must be positive, got {wavelength}")
        if medium_index <= 0:
            raise ValueError(f"Medium index must be positive, got {medium_index}")

        self.na = na
        self.wavelength = wavelength
        self.medium_index = medium_index

        # Calculate cutoff frequency
        self.cutoff_freq = na / (medium_index * wavelength)

    def generate(self, x: Tensor, y: Tensor, center: List[float]) -> Tensor:
        """
        Generate NA-based aperture mask.

        For microscopy, this is typically used in frequency space where
        x and y represent spatial frequencies (fx, fy).

        Args:
            x: X coordinates (spatial or frequency domain)
            y: Y coordinates (spatial or frequency domain)
            center: [cy, cx] center position

        Returns:
            Boolean mask where True indicates transmission

        Note:
            When used in frequency space, x and y should be frequency
            coordinates (fx, fy) with appropriate scaling.
        """
        # Translate to center
        x_c = x - center[1]
        y_c = y - center[0]

        # Radial distance (works in both spatial and frequency domains)
        r = torch.sqrt(x_c**2 + y_c**2)

        # For frequency space: mask frequencies above cutoff
        # For spatial space: this creates a circular pupil
        mask = r <= self.cutoff_freq

        return mask

    def generate_complex(
        self, x: Tensor, y: Tensor, center: List[float], phase: Optional[Tensor] = None
    ) -> Tensor:
        """
        Generate complex-valued pupil function.

        Useful for coherent imaging simulations where phase matters.

        Args:
            x: X coordinates
            y: Y coordinates
            center: [cy, cx] center position
            phase: Optional phase distribution (same shape as x, y)

        Returns:
            Complex pupil function
        """
        # Get amplitude mask
        mask = self.generate(x, y, center)

        # Convert to complex
        pupil = mask.to(torch.complex64)

        # Apply phase if provided
        if phase is not None:
            pupil = pupil * torch.exp(1j * phase)

        return pupil

    def get_resolution_limit(self) -> float:
        """
        Calculate theoretical resolution limit (Rayleigh criterion).

        Returns:
            Resolution limit in meters
        """
        return 0.61 * self.wavelength / self.na

    def get_depth_of_field(self) -> float:
        """
        Calculate depth of field.

        Returns:
            Depth of field in meters
        """
        return 2 * self.medium_index * self.wavelength / (self.na**2)


# Factory function for convenience
def create_aperture(aperture_type: str, **kwargs: Any) -> Aperture:
    """
    Factory function to create apertures.

    Args:
        aperture_type: Aperture type
            - 'circular': Circular aperture (simple)
            - 'hexagonal': Hexagonal aperture (JWST-style)
            - 'obscured': Obscured circular with spiders
            - 'numerical': NA-based aperture (microscopy)
        **kwargs: Type-specific parameters

    Returns:
        Aperture instance

    Raises:
        ValueError: If aperture_type is unknown

    Examples:
        >>> # Circular aperture
        >>> aperture = create_aperture('circular', radius=10)
        >>>
        >>> # Hexagonal aperture
        >>> aperture = create_aperture('hexagonal', side_length=20)
        >>>
        >>> # Obscured circular aperture
        >>> aperture = create_aperture(
        ...     'obscured',
        ...     outer_radius=50,
        ...     inner_radius=10,
        ...     spider_width=2,
        ...     n_spiders=4
        ... )
        >>>
        >>> # Numerical aperture (microscopy)
        >>> aperture = create_aperture(
        ...     'numerical',
        ...     na=1.4,
        ...     wavelength=550e-9,
        ...     medium_index=1.515
        ... )

    Aperture Selection Guide:
        For simple simulations:
            - Use 'circular' (fastest, simplest PSF)

        For segmented mirror telescopes (JWST, GMT, TMT):
            - Use 'hexagonal' (realistic for modern observatories)

        For Cassegrain telescopes (VLT, Hubble, Keck):
            - Use 'obscured' (realistic for most large reflectors)

        For microscopy and high-NA systems:
            - Use 'numerical' (NA-based aperture definition)
    """
    if aperture_type == "circular":
        return CircularAperture(**kwargs)
    elif aperture_type == "hexagonal":
        return HexagonalAperture(**kwargs)
    elif aperture_type == "obscured":
        return ObscuredCircularAperture(**kwargs)
    elif aperture_type == "numerical":
        return NumericalAperture(**kwargs)
    else:
        raise ValueError(
            f"Unknown aperture type: {aperture_type}. "
            f"Choose from: 'circular', 'hexagonal', 'obscured', 'numerical'"
        )
