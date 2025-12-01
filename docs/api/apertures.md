# prism.core.apertures

Aperture Mask Generators for PRISM.

This module provides implementations of different aperture mask types for
simulating realistic telescope aperture configurations. The choice of aperture
affects the PSF (Point Spread Function) and diffraction patterns.

Aperture Types
--------------

CircularAperture:
    Simple circular opening (current PRISM default).
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

    from prism.core.telescope import Telescope
    from prism.core.apertures import HexagonalAperture

    # Method 1: String-based factory
    telescope = Telescope(n=256, r=20, aperture_type='hexagonal')

    # Method 2: Direct injection
    custom_aperture = HexagonalAperture(side_length=25)
    telescope = Telescope(n=256, aperture=custom_aperture)

Physics Background
------------------

The aperture mask represents the physical opening through which light passes.
In the Fraunhofer (far-field) regime used by PRISM:

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

## Classes

### Aperture

```python
Aperture(/, *args, **kwargs)
```

Abstract base class for aperture mask generators.

All aperture types implement the generate() method to create boolean masks
representing the physical aperture shape. Apertures can be centered at any
position in the coordinate system.

The strategy pattern allows different aperture types to be swapped
transparently in the Telescope class.

#### Methods

##### `generate`

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

##### `generate_batch`

Generate multiple masks (vectorized when possible).

Args:
    x, y: Coordinate grids
    centers: List of [cy, cx] positions

Returns:
    Boolean masks [N, H, W] where N = len(centers)

Notes:
    - Default implementation loops over centers
    - Subclasses can override for vectorized implementation
    - CircularAperture has optimized vectorized version (3-5x faster)

Example:
    >>> centers = [[0, 0], [10, 10], [20, 20]]
    >>> masks = aperture.generate_batch(x, y, centers)
    >>> masks.shape  # (3, H, W)

### CircularAperture

```python
CircularAperture(radius: float)
```

Circular aperture (PRISM default).

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

#### Methods

##### `__init__`

Initialize circular aperture.

Args:
    radius: Aperture radius in pixels (in coordinate units)

Raises:
    ValueError: If radius <= 0

##### `generate`

Generate circular mask.

Args:
    x: X coordinates [1, W] or [H, W]
    y: Y coordinates [H, 1] or [H, W]
    center: [cy, cx] center position

Returns:
    Boolean mask [H, W], True inside circle

Example:
    >>> aperture = CircularAperture(radius=10)
    >>> mask = aperture.generate(x, y, center=[0, 0])

##### `generate_batch`

Optimized vectorized batch generation (3-5x faster).

This is the vectorized implementation from Telescope.mask_batch,
providing significant speedup for multiple aperture positions.

Args:
    x: X coordinates [1, W]
    y: Y coordinates [H, 1]
    centers: List of [cy, cx] positions

Returns:
    Boolean masks [N, H, W]

Performance:
    For N=20 centers on 1024×1024 grid:
    - Loop: ~10ms
    - Vectorized: ~2ms (5x faster)

Example:
    >>> centers = [[0, 0], [10, 10], [20, 20]]
    >>> masks = aperture.generate_batch(x, y, centers)
    >>> masks.shape  # (3, H, W)

### HexagonalAperture

```python
HexagonalAperture(side_length: float)
```

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

#### Methods

##### `__init__`

Initialize hexagonal aperture.

Args:
    side_length: Side length of regular hexagon in pixels

Raises:
    ValueError: If side_length <= 0

##### `generate`

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

##### `generate_batch`

Generate multiple masks (vectorized when possible).

Args:
    x, y: Coordinate grids
    centers: List of [cy, cx] positions

Returns:
    Boolean masks [N, H, W] where N = len(centers)

Notes:
    - Default implementation loops over centers
    - Subclasses can override for vectorized implementation
    - CircularAperture has optimized vectorized version (3-5x faster)

Example:
    >>> centers = [[0, 0], [10, 10], [20, 20]]
    >>> masks = aperture.generate_batch(x, y, centers)
    >>> masks.shape  # (3, H, W)

### NumericalAperture

```python
NumericalAperture(na: float, wavelength: float, medium_index: float = 1.0)
```

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

#### Methods

##### `__init__`

Initialize numerical aperture.

Args:
    na: Numerical aperture
    wavelength: Operating wavelength in meters
    medium_index: Refractive index (default: 1.0 for air)

Raises:
    ValueError: If NA exceeds medium index or parameters invalid

##### `generate`

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

##### `generate_batch`

Generate multiple masks (vectorized when possible).

Args:
    x, y: Coordinate grids
    centers: List of [cy, cx] positions

Returns:
    Boolean masks [N, H, W] where N = len(centers)

Notes:
    - Default implementation loops over centers
    - Subclasses can override for vectorized implementation
    - CircularAperture has optimized vectorized version (3-5x faster)

Example:
    >>> centers = [[0, 0], [10, 10], [20, 20]]
    >>> masks = aperture.generate_batch(x, y, centers)
    >>> masks.shape  # (3, H, W)

##### `generate_complex`

Generate complex-valued pupil function.

Useful for coherent imaging simulations where phase matters.

Args:
    x: X coordinates
    y: Y coordinates
    center: [cy, cx] center position
    phase: Optional phase distribution (same shape as x, y)

Returns:
    Complex pupil function

##### `get_depth_of_field`

Calculate depth of field.

Returns:
    Depth of field in meters

##### `get_resolution_limit`

Calculate theoretical resolution limit (Rayleigh criterion).

Returns:
    Resolution limit in meters

### ObscuredCircularAperture

```python
ObscuredCircularAperture(outer_radius: float, inner_radius: float, spider_width: Optional[float] = None, n_spiders: int = 4)
```

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

#### Methods

##### `__init__`

Initialize obscured circular aperture.

Args:
    outer_radius: Outer (primary) mirror radius in pixels
    inner_radius: Inner (secondary) mirror radius in pixels
    spider_width: Width of spider vanes in pixels (None = no spiders)
    n_spiders: Number of spider vanes (default: 4)

Raises:
    ValueError: If radii are invalid or obscuration too large

##### `generate`

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

##### `generate_batch`

Generate multiple masks (vectorized when possible).

Args:
    x, y: Coordinate grids
    centers: List of [cy, cx] positions

Returns:
    Boolean masks [N, H, W] where N = len(centers)

Notes:
    - Default implementation loops over centers
    - Subclasses can override for vectorized implementation
    - CircularAperture has optimized vectorized version (3-5x faster)

Example:
    >>> centers = [[0, 0], [10, 10], [20, 20]]
    >>> masks = aperture.generate_batch(x, y, centers)
    >>> masks.shape  # (3, H, W)

## Functions

### create_aperture

```python
create_aperture(aperture_type: str, **kwargs: Any) -> prism.core.apertures.Aperture
```

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
