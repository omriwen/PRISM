# prism.core.optics.aperture_masks

Unified aperture mask generator for all instruments.

This module provides a unified interface for generating aperture and pupil masks across different optical instruments (microscopes, telescopes, cameras). It supports various mask geometries and can work with both numerical aperture (NA) specifications and physical/pixel radius specifications.

## Overview

The `ApertureMaskGenerator` class consolidates all aperture/pupil mask generation functionality that was previously duplicated across instrument classes. It provides:

- Multiple mask geometries (circular, annular, hexagonal, phase rings, etc.)
- Flexible radius specifications (NA, physical, pixels)
- Automatic coordinate transformations
- Device-aware operations (CPU/GPU)

**Supported Mask Types**:
- Circular pupils (NA-limited for microscopes, physical for telescopes)
- Annular pupils (darkfield microscopy)
- Phase rings (phase contrast microscopy)
- Sub-apertures (PRISM synthetic aperture)
- Hexagonal apertures (segmented telescopes)
- Obscured apertures (central obstruction)

## Classes

### ApertureMaskGenerator

```python
ApertureMaskGenerator(
    grid: Grid,
    cutoff_type: str = 'na',
    wavelength: Optional[float] = None,
    medium_index: float = 1.0
)
```

Unified aperture mask generator for all instruments.

This class provides methods to generate various types of aperture and pupil masks for optical systems. It handles coordinate transformations and supports both physical (NA-based) and geometric (pixel/radius-based) specifications.

#### Parameters

- **grid** : `Grid`

  Spatial/frequency grid defining the coordinate system.

- **cutoff_type** : `str`, default='na'

  How to interpret radius specifications:
  - `'na'`: Use numerical aperture (NA) and wavelength to compute frequency cutoff
  - `'physical'`: Use physical radius in meters
  - `'pixels'`: Use radius directly in pixel units

- **wavelength** : `float`, optional

  Wavelength in meters. Required when cutoff_type='na'. If not provided, uses grid.wl.

- **medium_index** : `float`, default=1.0

  Refractive index of the medium (1.0 for air, 1.33 for water, 1.515 for oil).

#### Attributes

- **grid** : `Grid`

  The spatial/frequency grid.

- **cutoff_type** : `str`

  Type of radius specification being used.

- **wavelength** : `float`

  Wavelength in meters.

- **medium_index** : `float`

  Refractive index of the medium.

#### Methods

##### `__init__`

Initialize aperture mask generator.

```python
__init__(
    grid: Grid,
    cutoff_type: str = 'na',
    wavelength: Optional[float] = None,
    medium_index: float = 1.0
) -> None
```

**Parameters**:
- **grid** : Spatial/frequency grid defining the coordinate system.
- **cutoff_type** : How to interpret radius specifications ('na', 'physical', or 'pixels').
- **wavelength** : Wavelength in meters. Required when cutoff_type='na'.
- **medium_index** : Refractive index of the medium.

**Raises**:
- **ValueError** : If cutoff_type='na' but wavelength is not provided. If cutoff_type is not one of 'na', 'physical', 'pixels'.

##### `circular`

Generate circular aperture/pupil mask.

```python
circular(
    radius: Optional[float] = None,
    na: Optional[float] = None,
    center: Optional[List[float]] = None
) -> Tensor
```

Creates a binary circular mask in the frequency domain (k-space). For microscopes, this represents the objective NA. For telescopes, this represents the aperture diameter.

**Parameters**:
- **radius** : Radius in units specified by cutoff_type.
- **na** : Numerical aperture (only for cutoff_type='na').
- **center** : Center position [y, x] in pixels from DC. Defaults to [0, 0].

**Returns**:
- **Tensor** : Binary mask of shape (n_pixels, n_pixels), dtype float32.

**Examples**:
```python
>>> # Microscope: NA-based circular pupil
>>> mask = generator.circular(na=1.4)
>>>
>>> # Telescope: pixel-based circular aperture
>>> mask = generator.circular(radius=50)
>>>
>>> # Off-center sub-aperture
>>> mask = generator.circular(radius=25, center=[10, 5])
```

##### `annular`

Generate annular aperture/pupil mask.

```python
annular(
    inner_radius: Optional[float] = None,
    outer_radius: Optional[float] = None,
    inner_na: Optional[float] = None,
    outer_na: Optional[float] = None,
    center: Optional[List[float]] = None
) -> Tensor
```

Creates a ring-shaped mask (outer circle minus inner circle). Used for darkfield microscopy where the direct beam is blocked.

**Parameters**:
- **inner_radius** : Inner radius in units specified by cutoff_type.
- **outer_radius** : Outer radius in units specified by cutoff_type.
- **inner_na** : Inner numerical aperture (only for cutoff_type='na').
- **outer_na** : Outer numerical aperture (only for cutoff_type='na').
- **center** : Center position [y, x] in pixels from DC. Defaults to [0, 0].

**Returns**:
- **Tensor** : Binary mask of shape (n_pixels, n_pixels), dtype float32.

**Examples**:
```python
>>> # Darkfield microscopy: annular illumination
>>> mask = generator.annular(inner_na=0.8, outer_na=1.3)
>>>
>>> # Pixel-based annular mask
>>> mask = generator.annular(inner_radius=20, outer_radius=50)
```

##### `phase_ring`

Generate phase contrast ring mask.

```python
phase_ring(
    radius: Optional[float] = None,
    ring_inner: float = 0.6,
    ring_outer: float = 0.8,
    phase_shift: float = np.pi / 2,
    na: Optional[float] = None
) -> Tensor
```

Creates a mask with a phase-shifted ring region for phase contrast microscopy. The ring typically corresponds to the direct (unscattered) light, while the rest of the pupil transmits scattered light without phase shift.

**Parameters**:
- **radius** : Overall pupil radius in units specified by cutoff_type.
- **ring_inner** : Inner ring radius (normalized to radius, 0-1 range).
- **ring_outer** : Outer ring radius (normalized to radius, 0-1 range).
- **phase_shift** : Phase shift applied to the ring region (radians). Default π/2.
- **na** : Numerical aperture for overall pupil (only for cutoff_type='na').

**Returns**:
- **Tensor** : Complex-valued mask of shape (n_pixels, n_pixels), dtype complex64.

**Examples**:
```python
>>> # Phase contrast microscopy with π/2 phase ring
>>> mask = generator.phase_ring(na=1.4, ring_inner=0.6, ring_outer=0.8)
>>>
>>> # Custom phase shift
>>> mask = generator.phase_ring(
...     radius=50, ring_inner=0.5, ring_outer=0.7,
...     phase_shift=np.pi/4
... )
```

##### `sub_aperture`

Generate sub-aperture mask for PRISM synthetic aperture.

```python
sub_aperture(
    center: List[float],
    radius: Optional[float] = None,
    na: Optional[float] = None
) -> Tensor
```

Creates a circular aperture at a specified k-space position. This is used for PRISM progressive synthetic aperture reconstruction, where different regions of k-space are sampled sequentially.

**Parameters**:
- **center** : Center position [y, x] in pixels from DC.
- **radius** : Sub-aperture radius in units specified by cutoff_type.
- **na** : Numerical aperture for sub-aperture (only for cutoff_type='na').

**Returns**:
- **Tensor** : Binary mask of shape (n_pixels, n_pixels), dtype float32.

**Examples**:
```python
>>> # PRISM sub-aperture at offset position
>>> mask = generator.sub_aperture(center=[10, 5], radius=15)
>>>
>>> # NA-based sub-aperture
>>> mask = generator.sub_aperture(center=[0, 20], na=0.3)
```

##### `hexagonal`

Generate hexagonal aperture mask.

```python
hexagonal(
    radius: Optional[float] = None,
    na: Optional[float] = None,
    center: Optional[List[float]] = None
) -> Tensor
```

Creates a regular hexagon aperture, useful for segmented telescope mirrors (e.g., James Webb Space Telescope) or hexagonal pixel arrays.

**Parameters**:
- **radius** : Circumradius (center to vertex) in units specified by cutoff_type.
- **na** : Numerical aperture for circumscribed circle (only for cutoff_type='na').
- **center** : Center position [y, x] in pixels from DC. Defaults to [0, 0].

**Returns**:
- **Tensor** : Binary mask of shape (n_pixels, n_pixels), dtype float32.

**Examples**:
```python
>>> # Hexagonal telescope aperture
>>> mask = generator.hexagonal(radius=50)
>>>
>>> # Hexagonal microscope pupil
>>> mask = generator.hexagonal(na=1.2)
```

**Notes**:

The hexagon is oriented with a flat edge at the top (vertex pointing up).

##### `obscured`

Generate obscured aperture mask (central obstruction).

```python
obscured(
    outer_radius: Optional[float] = None,
    inner_radius: Optional[float] = None,
    outer_na: Optional[float] = None,
    inner_na: Optional[float] = None,
    center: Optional[List[float]] = None
) -> Tensor
```

Creates a circular aperture with a central circular obstruction. Common in reflecting telescopes where the secondary mirror blocks the center of the primary mirror.

**Parameters**:
- **outer_radius** : Outer aperture radius in units specified by cutoff_type.
- **inner_radius** : Inner obstruction radius in units specified by cutoff_type.
- **outer_na** : Outer numerical aperture (only for cutoff_type='na').
- **inner_na** : Inner numerical aperture (only for cutoff_type='na').
- **center** : Center position [y, x] in pixels from DC. Defaults to [0, 0].

**Returns**:
- **Tensor** : Binary mask of shape (n_pixels, n_pixels), dtype float32.

**Examples**:
```python
>>> # Cassegrain telescope with 30% central obstruction
>>> mask = generator.obscured(outer_radius=50, inner_radius=15)
>>>
>>> # NA-based obscured pupil
>>> mask = generator.obscured(outer_na=1.4, inner_na=0.4)
```

**Notes**:

This is equivalent to `annular()` but with clearer naming for telescope applications.

##### `to`

Move generator to specified device.

```python
to(device: torch.device) -> ApertureMaskGenerator
```

**Parameters**:
- **device** : Target device (e.g., torch.device("cuda")).

**Returns**:
- **ApertureMaskGenerator** : Self for method chaining.

## Usage Examples

### Microscope: NA-Based Masks

```python
from prism.core.grid import Grid
from prism.core.optics.aperture_masks import ApertureMaskGenerator

# Create grid
grid = Grid(nx=512, dx=10e-6, wavelength=550e-9)

# Create generator for microscope (NA-based)
generator = ApertureMaskGenerator(
    grid,
    cutoff_type='na',
    wavelength=550e-9,
    medium_index=1.0
)

# Detection pupil (NA = 1.4)
detect_pupil = generator.circular(na=1.4)

# Illumination pupil (NA = 1.0)
illum_pupil = generator.circular(na=1.0)

# Darkfield: annular illumination
darkfield_illum = generator.annular(inner_na=0.8, outer_na=1.3)

# Phase contrast
phase_pupil = generator.phase_ring(
    na=1.4,
    ring_inner=0.6,
    ring_outer=0.8,
    phase_shift=np.pi/2
)
```

### Telescope: Pixel-Based Masks

```python
# Create generator for telescope (pixel-based)
generator = ApertureMaskGenerator(
    grid,
    cutoff_type='pixels'
)

# Circular aperture (50 pixel radius)
circular_aperture = generator.circular(radius=50)

# Hexagonal aperture (segmented mirror)
hexagonal_aperture = generator.hexagonal(radius=50)

# Obscured aperture (Cassegrain telescope)
# Outer radius: 50 pixels, Inner obstruction: 15 pixels (30%)
obscured_aperture = generator.obscured(
    outer_radius=50,
    inner_radius=15
)
```

### PRISM: Sub-Aperture Scanning

```python
# Create generator for PRISM
generator = ApertureMaskGenerator(
    grid,
    cutoff_type='pixels'
)

# Define sub-aperture radius
sub_aperture_radius = 15

# Scan k-space with overlapping sub-apertures
centers = [
    [0, 0],    # Center
    [10, 0],   # Right
    [0, 10],   # Up
    [-10, 0],  # Left
    [0, -10],  # Down
]

sub_apertures = []
for center in centers:
    mask = generator.sub_aperture(
        center=center,
        radius=sub_aperture_radius
    )
    sub_apertures.append(mask)
```

### Physical Radius Specification

```python
# Create generator with physical radius (meters)
generator = ApertureMaskGenerator(
    grid,
    cutoff_type='physical'
)

# Aperture with 5mm radius
aperture_5mm = generator.circular(radius=5e-3)

# Annular aperture: 2mm to 5mm
annular_aperture = generator.annular(
    inner_radius=2e-3,
    outer_radius=5e-3
)
```

### GPU Acceleration

```python
import torch

# Create generator on GPU
device = torch.device('cuda')
grid_gpu = grid.to(device)
generator_gpu = ApertureMaskGenerator(grid_gpu, cutoff_type='na', wavelength=550e-9)
generator_gpu = generator_gpu.to(device)

# Generate masks on GPU
pupil_gpu = generator_gpu.circular(na=1.4)
print(pupil_gpu.device)  # cuda:0
```

### Off-Center Apertures

```python
# Create off-center aperture for oblique illumination
oblique_aperture = generator.circular(
    radius=30,
    center=[20, 10]  # Offset from DC
)

# Multiple off-center apertures for structured illumination
angles = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
offset_distance = 25

apertures = []
for angle in angles:
    center_y = offset_distance * np.sin(angle)
    center_x = offset_distance * np.cos(angle)
    mask = generator.circular(
        radius=15,
        center=[center_y, center_x]
    )
    apertures.append(mask)
```

## Coordinate Convention

All mask generation methods use the following coordinate convention:

- **DC (zero frequency)** is at the center of the grid: `[N//2, N//2]`
- **Center parameter** `[y, x]` is offset from DC in pixel units
- **Positive y** points down (following matrix indexing)
- **Positive x** points right

This matches the output of `torch.fft.fftshift()` and ensures consistency with k-space representations.

## Frequency Cutoff Conversion

The generator automatically converts between different radius specifications:

### NA-Based (Microscopes)

```
f_cutoff = NA / (n * λ)
```

where:
- `NA`: Numerical aperture
- `n`: Medium refractive index
- `λ`: Wavelength

### Physical Radius (Telescopes)

```
f_cutoff ≈ radius / FOV
```

where:
- `radius`: Physical aperture radius (meters)
- `FOV`: Field of view (meters)

### Pixel Radius (Generic)

```
f_cutoff = radius * df
```

where:
- `radius`: Radius in pixels
- `df = 1 / FOV`: Frequency resolution

## See Also

- `prism.core.instruments.four_f_base.FourFSystem` : Uses ApertureMaskGenerator for pupil creation
- `prism.core.optics.four_f_forward.FourFForwardModel` : Applies pupils generated by this class
- `prism.core.grid.Grid` : Spatial/frequency grid used for coordinate systems
- `prism.core.instruments.microscope.Microscope` : Uses NA-based apertures
- `prism.core.instruments.telescope.Telescope` : Uses pixel/physical-based apertures
