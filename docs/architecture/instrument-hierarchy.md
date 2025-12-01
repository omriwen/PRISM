# Instrument Class Hierarchy

**Last Updated**: 2025-11-28
**Related**: [Forward Model Physics](forward-model-physics.md)

This document describes the class hierarchy for optical instruments in PRISM, focusing on the Four-F System consolidation that provides a unified base class for microscopes, telescopes, and cameras.

---

## Table of Contents

1. [Overview](#overview)
2. [Class Hierarchy Diagram](#class-hierarchy-diagram)
3. [Base Classes](#base-classes)
4. [FourFSystem Architecture](#fourfsystem-architecture)
5. [Concrete Instruments](#concrete-instruments)
6. [Component Classes](#component-classes)
7. [Extension Guide](#extension-guide)
8. [Code Examples](#code-examples)

---

## Overview

The PRISM instrument architecture follows a two-level abstraction:

1. **Instrument ABC**: Defines the interface all optical instruments must implement
2. **FourFSystem ABC**: Provides unified implementation for all 4f-based instruments

This consolidation eliminates ~500 lines of duplicated code by extracting common 4f optical system behavior into reusable components.

### Design Principles

- **Composition over Inheritance**: Core functionality (forward model, aperture masks, noise) is provided by composable components
- **Single Responsibility**: Each component class has one clear purpose
- **Extension Points**: Subclasses customize behavior through well-defined abstract methods
- **Type Safety**: Abstract methods enforce required implementations

---

## Class Hierarchy Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Instrument (ABC)                             │
│  prism/core/instruments/base.py                                  │
├─────────────────────────────────────────────────────────────────┤
│  Abstract Methods:                                               │
│    - forward()                : Forward propagation              │
│    - compute_psf()            : Point spread function            │
│    - resolution_limit         : Theoretical resolution           │
│    - _create_grid()           : Grid initialization              │
│    - _select_propagator()     : Propagator selection             │
│                                                                   │
│  Provided Methods:                                               │
│    - validate_field()         : Input validation & conversion    │
│    - get_info()               : Instrument information           │
│    - get_instrument_type()    : Type identifier                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
           ┌───────────┴────────────┐
           │                        │
           ▼                        ▼
┌─────────────────────┐   ┌──────────────────────────┐
│  FourFSystem (ABC)  │   │  [Other Instruments]     │
│  prism/core/        │   │  (Future: holography,    │
│  instruments/       │   │   light-field, etc.)     │
│  four_f_base.py     │   │                          │
├─────────────────────┤   └──────────────────────────┘
│  Abstract Methods:  │
│    - _create_pupils()         : Instrument-specific pupils       │
│    - resolution_limit         : Resolution calculation           │
│                                                                   │
│  Provided Methods:                                               │
│    - forward()                : Unified 4f forward model         │
│    - propagate_to_kspace()    : Object → Fourier plane          │
│    - propagate_to_spatial()   : Fourier → image plane           │
│    - generate_aperture_mask() : Sub-aperture masks              │
│    - compute_psf()            : Default PSF computation          │
│                                                                   │
│  Components:                                                     │
│    - _forward_model           : FourFForwardModel                │
│    - _aperture_generator      : ApertureMaskGenerator            │
│    - _noise_model             : DetectorNoiseModel (optional)    │
└──────────────────┬──────────────────────────────────────────────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
┌─────────────┐ ┌──────────┐ ┌────────┐
│ Microscope  │ │Telescope │ │ Camera │
│ prism/core/ │ │prism/core│ │prism/  │
│ instruments/│ │/instr.../│ │core/   │
│ microscope  │ │telescope │ │instr../│
│ .py         │ │.py       │ │camera  │
│             │ │          │ │.py     │
├─────────────┤ ├──────────┤ ├────────┤
│Implements:  │ │Implements│ │Implem. │
│ _create_    │ │_create_  │ │_create_│
│  pupils()   │ │ pupils() │ │pupils()│
│             │ │          │ │        │
│ resolution_ │ │resolut._ │ │resol._ │
│  limit      │ │ limit    │ │limit   │
│             │ │          │ │        │
│Custom:      │ │Custom:   │ │Custom: │
│ Illumin.    │ │ Aperture │ │ Focus  │
│  modes      │ │  shapes  │ │  ctrl  │
│ (bright/    │ │ (circ/   │ │ Depth  │
│  dark/DIC)  │ │  hex)    │ │  of    │
│             │ │          │ │  field │
└─────────────┘ └──────────┘ └────────┘
```

### Component Composition Diagram

```
┌────────────────────────────────────────────────────────────┐
│                     FourFSystem                             │
│                    (Instrument)                             │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         FourFForwardModel (Component)               │   │
│  │  prism/core/optics/four_f_forward.py                │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  • Pad input field (anti-aliasing)                  │   │
│  │  • FFT to Fourier plane                             │   │
│  │  • Apply illumination pupil                         │   │
│  │  • Apply detection pupil                            │   │
│  │  • IFFT to image plane                              │   │
│  │  • Crop to original size                            │   │
│  │  • Compute intensity |E|²                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │      ApertureMaskGenerator (Component)              │   │
│  │  prism/core/optics/aperture_masks.py                │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  • circular()      : NA-limited or pixel-based      │   │
│  │  • annular()       : Darkfield illumination         │   │
│  │  • phase_ring()    : Phase contrast                 │   │
│  │  • sub_aperture()  : PRISM k-space sampling         │   │
│  │  • hexagonal()     : Segmented apertures            │   │
│  │  • obscured()      : Central obstruction            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │      DetectorNoiseModel (Optional Component)        │   │
│  │  prism/core/optics/detector_noise.py                │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  • Shot noise (Poisson statistics)                  │   │
│  │  • Read noise (Gaussian, additive)                  │   │
│  │  • Dark current (thermal electrons)                 │   │
│  │  • SNR-based or component-based mode                │   │
│  │  • Enable/disable control                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Base Classes

### Instrument (ABC)

**File**: `prism/core/instruments/base.py`

The root abstract base class defining the interface for all optical instruments.

#### Abstract Methods (Must Implement)

```python
def _create_grid(self) -> Grid:
    """Create computational grid for instrument."""

def _select_propagator(self) -> Propagator:
    """Select appropriate propagator for instrument."""

def compute_psf(self, **kwargs) -> torch.Tensor:
    """Compute point spread function."""

def forward(self, field, illumination_mode=None, ...) -> torch.Tensor:
    """Forward propagation through instrument."""

@property
def resolution_limit(self) -> float:
    """Theoretical resolution limit."""
```

#### Provided Methods (Inherited)

```python
def validate_field(self, field, input_mode='auto', ...) -> torch.Tensor:
    """Validate and convert input to complex field."""

def get_info(self) -> dict:
    """Get instrument information summary."""

def get_instrument_type(self) -> str:
    """Return instrument type identifier."""
```

#### Key Attributes

- `config: InstrumentConfig` - Configuration parameters (wavelength, n_pixels, pixel_size)
- `grid: Grid` - Computational grid (lazy initialization)
- `propagator: Propagator` - Wave propagator (lazy initialization)

---

### FourFSystem (ABC)

**File**: `prism/core/instruments/four_f_base.py`

Abstract base class for all instruments based on the four-focal-length (4f) optical system. Provides unified implementation of the simplified 4f forward model.

#### Physical Model

The 4f system implements:

```
I(x,y) = |IFFT{ P_det(u,v) · P_illum(u,v) · FFT{E_object(x,y)} }|²
```

Where:
- `E_object`: Complex field at object plane
- `FFT/IFFT`: Fourier transforms
- `P_illum`: Illumination pupil function (instrument-specific)
- `P_det`: Detection pupil function (instrument-specific)
- `I`: Measured intensity

#### Abstract Methods (Must Implement)

```python
def _create_pupils(
    self,
    illumination_mode: Optional[str] = None,
    illumination_params: Optional[dict] = None,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Create illumination and detection pupil functions.

    Returns
    -------
    tuple[Tensor or None, Tensor or None]
        (illumination_pupil, detection_pupil) in k-space
    """

@property
def resolution_limit(self) -> float:
    """Theoretical resolution limit (meters or radians)."""
```

#### Provided Methods (Inherited by Subclasses)

```python
def forward(self, field, illumination_mode=None, add_noise=False, ...) -> Tensor:
    """Unified forward propagation through 4f system."""

def propagate_to_kspace(self, field: Tensor) -> Tensor:
    """FFT from object plane to pupil plane."""

def propagate_to_spatial(self, field_kspace: Tensor) -> Tensor:
    """IFFT from pupil plane to image plane + intensity."""

def generate_aperture_mask(self, center=None, radius=None) -> Tensor:
    """Generate circular sub-aperture mask for PRISM."""

def compute_psf(self, **kwargs) -> Tensor:
    """Default PSF computation via delta function."""
```

#### Component Instances (Composition)

```python
self._forward_model: FourFForwardModel
    # Core 4f propagation (pad → FFT → pupils → IFFT → crop)

self._aperture_generator: ApertureMaskGenerator
    # Unified aperture/pupil mask generation

self._noise_model: Optional[DetectorNoiseModel]
    # Optional realistic detector noise
```

#### Constructor Parameters

```python
def __init__(
    self,
    config: InstrumentConfig,
    padding_factor: float = 2.0,          # FFT padding for anti-aliasing
    aperture_cutoff_type: str = 'na',     # 'na', 'physical', 'pixels'
    medium_index: float = 1.0,            # Refractive index
    noise_model: Optional[DetectorNoiseModel] = None,
)
```

---

## FourFSystem Architecture

### Design Rationale

The `FourFSystem` base class consolidates three types of duplication:

1. **Forward model logic** - All instruments use the same FFT-based 4f propagation
2. **K-space utilities** - Shared methods for object ↔ Fourier plane transformations
3. **Aperture generation** - Common interface for mask creation

By extracting this into a base class with composable components, we achieve:

- **Code reduction**: ~500 lines of duplication eliminated
- **Maintainability**: Single source of truth for 4f physics
- **Extensibility**: New instruments inherit complete functionality
- **Testability**: Components can be tested independently

### Instrument Customization

Subclasses customize behavior by:

1. **Implementing `_create_pupils()`** - Define instrument-specific pupil functions
2. **Setting `aperture_cutoff_type`** - Choose NA-based, physical, or pixel-based apertures
3. **Providing `resolution_limit`** - Calculate theoretical resolution
4. **Optionally overriding methods** - Custom PSF, grid, or propagator selection

---

## Concrete Instruments

### Microscope

**File**: `prism/core/instruments/microscope.py`

**Optical Configuration**: 4f system with NA-limited pupils

**Key Features**:
- Multiple illumination modes (brightfield, darkfield, phase contrast, DIC)
- NA-based aperture specifications
- Configurable illumination/detection NA
- Support for water/oil immersion (medium_index)

**Pupil Implementation**:

```python
def _create_pupils(self, illumination_mode=None, illumination_params=None):
    """Create microscope-specific pupils based on illumination mode."""
    params = illumination_params or {}

    if illumination_mode == 'brightfield':
        illum_na = params.get('illumination_na', 0.8 * self.na)
        illum = self._aperture_generator_lazy.circular(na=illum_na)
        detect = self._aperture_generator_lazy.circular(na=self.na)

    elif illumination_mode == 'darkfield':
        annular_ratio = params.get('annular_ratio', 0.8)
        illum = self._aperture_generator_lazy.annular(
            inner_na=annular_ratio*self.na, outer_na=self.na)
        detect = self._aperture_generator_lazy.circular(na=self.na)

    elif illumination_mode == 'phase':
        illum = self._aperture_generator_lazy.circular(na=0.8*self.na)
        detect = self._aperture_generator_lazy.phase_ring(
            na=self.na, ring_inner=0.6, ring_outer=0.8)

    # ... other modes

    return illum, detect
```

**Resolution Limit**: Abbe diffraction limit

```python
@property
def resolution_limit(self) -> float:
    return 0.61 * self.config.wavelength / self.na
```

---

### Telescope

**File**: `prism/core/instruments/telescope.py`

**Optical Configuration**: Far-field Fraunhofer diffraction

**Key Features**:
- Circular and hexagonal apertures
- Central obstruction support (secondary mirror)
- Pixel-based aperture specifications
- Astronomical imaging (object at infinity)

**Pupil Implementation**:

```python
def _create_pupils(self, illumination_mode=None, illumination_params=None):
    """Create telescope aperture mask."""
    params = illumination_params or {}

    aperture_shape = params.get('aperture_shape', 'circular')

    if aperture_shape == 'circular':
        aperture = self._aperture_generator_lazy.circular(
            radius=self.aperture_radius_pixels)
    elif aperture_shape == 'hexagonal':
        aperture = self._aperture_generator_lazy.hexagonal(
            radius=self.aperture_radius_pixels)
    elif aperture_shape == 'obscured':
        aperture = self._aperture_generator_lazy.obscured(
            outer_radius=self.aperture_radius_pixels,
            inner_radius=params.get('obstruction_ratio', 0.3) * self.aperture_radius_pixels)

    # Telescope uses single aperture (no separate illumination)
    return None, aperture
```

**Resolution Limit**: Rayleigh criterion

```python
@property
def resolution_limit(self) -> float:
    """Angular resolution in radians."""
    return 1.22 * self.config.wavelength / self.aperture_diameter
```

---

### Camera

**File**: `prism/core/instruments/camera.py`

**Optical Configuration**: Lens-based imaging with automatic regime selection

**Key Features**:
- Automatic Fresnel/Fraunhofer regime selection
- Focus control (object distance, focal length)
- Depth of field calculation
- F-number based aperture

**Pupil Implementation**:

```python
def _create_pupils(self, illumination_mode=None, illumination_params=None):
    """Create camera aperture (circular, defined by f-number)."""
    params = illumination_params or {}

    # Aperture diameter from f-number: D = f / N
    aperture_diameter = self.focal_length / self.f_number

    # Convert to pixel radius
    aperture_radius_pixels = aperture_diameter / (2 * self.grid.dx)

    aperture = self._aperture_generator_lazy.circular(
        radius=aperture_radius_pixels)

    # Camera uses single aperture
    return None, aperture
```

**Resolution Limit**: Airy disk size

```python
@property
def resolution_limit(self) -> float:
    """Airy disk diameter in meters."""
    return 2.44 * self.config.wavelength * self.f_number
```

---

## Component Classes

### FourFForwardModel

**File**: `prism/core/optics/four_f_forward.py`

**Purpose**: Core 4f optical system forward propagation

**Key Responsibilities**:
- Pad input field to prevent FFT wraparound artifacts
- Transform to Fourier plane (pupil plane)
- Apply illumination and detection pupils
- Transform back to image plane
- Crop to original size
- Compute intensity or return complex field

**Algorithm**:

```
Input: field [H, W], illumination_pupil, detection_pupil

1. Pad field to padded_size (power of 2, padding_factor × original)
2. field_bfp = fftshift(fft2(fftshift(field_padded)))
3. field_illum = field_bfp * illumination_pupil
4. field_filtered = field_illum * detection_pupil
5. field_image = ifftshift(ifft2(ifftshift(field_filtered)))
6. field_cropped = crop(field_image, original_size)
7. intensity = |field_cropped|² (if return_complex=False)

Output: intensity [H, W] or complex field [H, W]
```

**Configuration**:
- `padding_factor` (default 2.0): FFT padding to prevent aliasing
- `normalize_output` (default True): Normalize intensity to [0, 1]

---

### ApertureMaskGenerator

**File**: `prism/core/optics/aperture_masks.py`

**Purpose**: Unified aperture and pupil mask generation

**Supported Geometries**:

| Method | Use Case | Parameters |
|--------|----------|------------|
| `circular()` | Standard pupils/apertures | `na` or `radius`, `center` |
| `annular()` | Darkfield microscopy | `inner_na/radius`, `outer_na/radius` |
| `phase_ring()` | Phase contrast | `na/radius`, `ring_inner`, `ring_outer`, `phase_shift` |
| `sub_aperture()` | PRISM k-space sampling | `center`, `radius/na` |
| `hexagonal()` | Segmented mirrors | `na/radius`, `center` |
| `obscured()` | Central obstruction | `outer_na/radius`, `inner_na/radius` |

**Cutoff Type Modes**:

1. **'na'** (Numerical Aperture) - For microscopes
   - Frequency cutoff: `k_cutoff = NA / (n × λ)`
   - Example: `circular(na=1.4)` for high-NA objective

2. **'physical'** (Physical Radius) - For telescopes
   - Radius in meters
   - Example: `circular(radius=2.0)` for 4m diameter telescope

3. **'pixels'** (Pixel Units) - Generic
   - Radius in pixel count
   - Example: `circular(radius=50)` for 50-pixel radius

**Coordinate System**:
- DC (zero frequency) at center of grid
- Matches `fftshift` convention
- Center parameter `[y, x]` in pixels from DC

---

### DetectorNoiseModel

**File**: `prism/core/optics/detector_noise.py`

**Purpose**: Realistic detector noise simulation

**Noise Components**:

1. **Shot Noise** (Poisson statistics)
   - Variance proportional to signal: `σ² ∝ I`
   - Approximated as Gaussian for high photon counts
   - Controlled by `photon_scale`

2. **Read Noise** (Gaussian, additive)
   - Electronic noise from detector readout
   - Constant variance, signal-independent
   - Controlled by `read_noise_fraction`

3. **Dark Current** (Gaussian, additive)
   - Thermal electrons accumulated during exposure
   - Controlled by `dark_current_fraction`

**Operating Modes**:

1. **SNR-based Mode**:
   ```python
   noise_model = DetectorNoiseModel(snr_db=40.0)
   ```
   - Automatically scales noise to achieve target SNR
   - SNR definition: `SNR(dB) = 20 × log₁₀(signal_max / noise_std)`

2. **Component-based Mode**:
   ```python
   noise_model = DetectorNoiseModel(
       photon_scale=1000.0,
       read_noise_fraction=0.01,
       dark_current_fraction=0.002,
   )
   ```
   - Explicit control of each noise source
   - Physically motivated parameters

**Control Methods**:
- `enable()` / `disable()`: Toggle noise on/off
- `set_snr(snr_db)`: Update SNR level (switches to SNR mode)

---

## Extension Guide

### Creating a New 4f-Based Instrument

To create a new instrument that uses the 4f optical model:

#### Step 1: Inherit from FourFSystem

```python
from prism.core.instruments.four_f_base import FourFSystem
from prism.core.instruments.base import InstrumentConfig

class MyInstrument(FourFSystem):
    """Custom 4f-based instrument."""

    def __init__(
        self,
        config: InstrumentConfig,
        custom_param: float,
    ) -> None:
        # Store custom parameters
        self.custom_param = custom_param

        # Initialize FourFSystem
        super().__init__(
            config,
            padding_factor=2.0,
            aperture_cutoff_type='na',  # or 'physical', 'pixels'
            medium_index=1.0,
        )
```

#### Step 2: Implement _create_pupils()

Define how illumination and detection pupils are created:

```python
def _create_pupils(
    self,
    illumination_mode: Optional[str] = None,
    illumination_params: Optional[dict] = None,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Create instrument-specific pupil functions."""

    # Example: Create custom pupils based on instrument parameters
    params = illumination_params or {}

    # Illumination pupil (can be None for identity)
    illum_pupil = self._aperture_generator_lazy.circular(
        na=params.get('illum_na', 0.5)
    )

    # Detection pupil (can be None for identity)
    detect_pupil = self._aperture_generator_lazy.circular(
        na=params.get('detect_na', 1.0)
    )

    return illum_pupil, detect_pupil
```

#### Step 3: Implement resolution_limit Property

```python
@property
def resolution_limit(self) -> float:
    """Calculate theoretical resolution limit.

    Returns
    -------
    float
        Resolution limit in appropriate units (meters or radians)
    """
    # Example: Abbe limit
    return 0.61 * self.config.wavelength / self.custom_param
```

#### Step 4: (Optional) Override Default Methods

Customize grid creation, PSF computation, or other behaviors:

```python
def _create_grid(self) -> Grid:
    """Custom grid creation."""
    from prism.core.grid import Grid

    return Grid(
        nx=self.config.n_pixels,
        dx=self.config.pixel_size,
        wavelength=self.config.wavelength,
        # Custom parameters...
    )

def compute_psf(self, **kwargs) -> Tensor:
    """Custom PSF computation."""
    # Example: 3D PSF stack for volumetric imaging
    psf_stack = []
    for z in z_positions:
        psf_slice = self.forward(delta_function, defocus=z, **kwargs)
        psf_stack.append(psf_slice)
    return torch.stack(psf_stack, dim=0)
```

---

### Example: Confocal Microscope

Here's a complete example of extending FourFSystem:

```python
from prism.core.instruments.four_f_base import FourFSystem
from prism.core.instruments.base import InstrumentConfig
import torch
from torch import Tensor
from typing import Optional

class ConfocalMicroscope(FourFSystem):
    """Confocal scanning microscope with pinhole detection.

    The confocal configuration uses identical illumination and detection
    pupils, with a small pinhole at the image plane for optical sectioning.

    Parameters
    ----------
    config : InstrumentConfig
        Standard instrument configuration
    na : float
        Numerical aperture of objective
    pinhole_radius_au : float, default=1.0
        Pinhole radius in Airy units (AU). 1.0 AU = first Airy minimum.
    """

    def __init__(
        self,
        config: InstrumentConfig,
        na: float = 1.4,
        pinhole_radius_au: float = 1.0,
    ) -> None:
        self.na = na
        self.pinhole_radius_au = pinhole_radius_au

        # Initialize with NA-based apertures
        super().__init__(
            config,
            padding_factor=2.0,
            aperture_cutoff_type='na',
            medium_index=1.0,  # Can be changed for oil/water
        )

    def _create_pupils(
        self,
        illumination_mode: Optional[str] = None,
        illumination_params: Optional[dict] = None,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Create confocal pupils (identical for illumination and detection)."""

        # Confocal uses full NA for both illumination and detection
        pupil = self._aperture_generator_lazy.circular(na=self.na)

        return pupil, pupil

    @property
    def resolution_limit(self) -> float:
        """Confocal lateral resolution (improved by √2 vs widefield)."""
        # Confocal improves lateral resolution by factor of √2
        return (0.61 * self.config.wavelength / self.na) / torch.sqrt(torch.tensor(2.0))

    def apply_pinhole(self, field: Tensor) -> Tensor:
        """Apply pinhole mask at image plane.

        Parameters
        ----------
        field : Tensor
            Complex field at image plane

        Returns
        -------
        Tensor
            Field after pinhole filtering
        """
        # Calculate pinhole radius in pixels
        # 1 AU = 1.22 * λ / NA (first Airy minimum)
        airy_radius_meters = 1.22 * self.config.wavelength / self.na
        pinhole_radius_meters = self.pinhole_radius_au * airy_radius_meters
        pinhole_radius_pixels = pinhole_radius_meters / self.grid.dx

        # Create pinhole mask in spatial domain
        x = self.grid.x
        y = self.grid.y
        r = torch.sqrt(x**2 + y**2)
        pinhole_mask = (r <= pinhole_radius_meters).float()

        # Apply pinhole
        return field * pinhole_mask

    def forward(
        self,
        field: Tensor,
        apply_pinhole: bool = True,
        **kwargs,
    ) -> Tensor:
        """Forward propagation through confocal system.

        Parameters
        ----------
        field : Tensor
            Input field at object plane
        apply_pinhole : bool, default=True
            Whether to apply pinhole filtering
        **kwargs
            Additional arguments passed to parent forward()

        Returns
        -------
        Tensor
            Detected intensity after confocal filtering
        """
        # Get complex field output from 4f system
        kwargs['return_complex'] = True  # Force complex output
        field_image = super().forward(field, **kwargs)

        # Apply pinhole if requested
        if apply_pinhole:
            field_image = self.apply_pinhole(field_image)

        # Convert to intensity
        intensity = torch.abs(field_image) ** 2

        # Normalize if needed
        if intensity.max() > 0:
            intensity = intensity / intensity.max()

        return intensity
```

**Usage**:

```python
# Create confocal microscope
config = InstrumentConfig(
    wavelength=550e-9,
    n_pixels=512,
    pixel_size=10e-6,
)
confocal = ConfocalMicroscope(config, na=1.4, pinhole_radius_au=1.0)

# Forward propagation
field = torch.randn(512, 512, dtype=torch.complex64)
intensity = confocal.forward(field)

# Compute PSF
psf = confocal.compute_psf()
```

---

## Code Examples

### Example 1: Basic Usage - Microscope

```python
from prism.core.instruments import Microscope
from prism.core.instruments.base import InstrumentConfig
import torch

# Configure microscope
config = InstrumentConfig(
    wavelength=550e-9,      # 550 nm (green)
    n_pixels=512,           # 512×512 grid
    pixel_size=10e-6,       # 10 μm pixels
)

microscope = Microscope(
    config=config,
    na=1.4,                 # High NA objective
    medium_index=1.515,     # Oil immersion
)

# Create input field
field = torch.randn(512, 512, dtype=torch.complex64)

# Brightfield imaging
brightfield = microscope.forward(
    field,
    illumination_mode='brightfield',
)

# Darkfield imaging
darkfield = microscope.forward(
    field,
    illumination_mode='darkfield',
    illumination_params={'annular_ratio': 0.8},
)

# Phase contrast
phase = microscope.forward(
    field,
    illumination_mode='phase',
)
```

---

### Example 2: PRISM Sub-Aperture Imaging

```python
from prism.core.instruments import Microscope
import torch

# Create microscope
microscope = Microscope(config, na=1.4)

# Get object k-space representation
field = torch.randn(512, 512, dtype=torch.complex64)
field_kspace = microscope.propagate_to_kspace(field)

# Generate sub-aperture masks for synthetic aperture
sub_aperture_centers = [
    [0, 0],      # Center
    [20, 0],     # Right
    [0, 20],     # Up
    [-20, 0],    # Left
    [0, -20],    # Down
]

sub_aperture_images = []
for center in sub_aperture_centers:
    # Create sub-aperture mask
    mask = microscope.generate_aperture_mask(
        center=center,
        radius=15,  # pixels
    )

    # Apply mask and propagate to spatial domain
    masked_kspace = field_kspace * mask
    sub_image = microscope.propagate_to_spatial(masked_kspace)

    sub_aperture_images.append(sub_image)

# Combine sub-apertures for synthetic aperture reconstruction
synthetic_aperture = torch.stack(sub_aperture_images).mean(dim=0)
```

---

### Example 3: Adding Noise

```python
from prism.core.instruments import Microscope
from prism.core.optics import DetectorNoiseModel

# Create noise model
noise_model = DetectorNoiseModel(
    snr_db=40.0,  # 40 dB SNR
)

# Create microscope with noise
microscope = Microscope(
    config,
    na=1.4,
    noise_model=noise_model,
)

# Forward with noise
noisy_image = microscope.forward(
    field,
    illumination_mode='brightfield',
    add_noise=True,  # Enable noise
)

# Adjust SNR
microscope._noise_model.set_snr(50.0)  # Less noise
cleaner_image = microscope.forward(field, add_noise=True)

# Disable noise temporarily
microscope._noise_model.disable()
clean_image = microscope.forward(field, add_noise=True)  # No noise added
microscope._noise_model.enable()
```

---

### Example 4: Custom Aperture Shapes (Telescope)

```python
from prism.core.instruments import Telescope
from prism.core.instruments.base import InstrumentConfig

config = InstrumentConfig(
    wavelength=550e-9,
    n_pixels=512,
    pixel_size=1e-6,
)

telescope = Telescope(
    config,
    aperture_diameter=2.0,  # 2m diameter
)

# Circular aperture (default)
image_circular = telescope.forward(
    field,
    illumination_params={'aperture_shape': 'circular'},
)

# Hexagonal aperture (James Webb style)
image_hex = telescope.forward(
    field,
    illumination_params={'aperture_shape': 'hexagonal'},
)

# Obscured aperture (Cassegrain secondary mirror)
image_obscured = telescope.forward(
    field,
    illumination_params={
        'aperture_shape': 'obscured',
        'obstruction_ratio': 0.3,  # 30% central obstruction
    },
)

# Compute PSF for each aperture
psf_circular = telescope.compute_psf(
    illumination_params={'aperture_shape': 'circular'}
)
psf_hex = telescope.compute_psf(
    illumination_params={'aperture_shape': 'hexagonal'}
)
```

---

### Example 5: Camera with Focus Control

```python
from prism.core.instruments import Camera
from prism.core.instruments.base import InstrumentConfig

config = InstrumentConfig(
    wavelength=550e-9,
    n_pixels=512,
    pixel_size=5e-6,
)

camera = Camera(
    config,
    focal_length=50e-3,     # 50mm lens
    f_number=2.8,           # f/2.8
    object_distance=1.0,    # 1m to subject
)

# In-focus imaging
in_focus = camera.forward(field)

# Defocused imaging
camera.object_distance = 0.8  # Move object closer
defocused = camera.forward(field)

# Check depth of field
dof_near, dof_far = camera.depth_of_field()
print(f"Depth of field: {dof_near:.3f}m to {dof_far:.3f}m")

# Compute resolution
print(f"Airy disk size: {camera.resolution_limit*1e6:.2f} μm")
```

---

## Summary

The FourFSystem architecture provides:

1. **Unified Implementation**: Single source of truth for 4f optical physics
2. **Component Composition**: Reusable FourFForwardModel, ApertureMaskGenerator, DetectorNoiseModel
3. **Clear Extension Points**: Abstract methods define customization interface
4. **Reduced Duplication**: ~500 lines of code consolidation
5. **Instrument Diversity**: Supports microscopes, telescopes, cameras, and custom instruments

**Key Design Principles**:
- Composition over inheritance for flexibility
- Abstract methods enforce implementation contracts
- Default implementations reduce boilerplate
- Components enable independent testing and reuse

**For More Information**:
- [Forward Model Physics](forward-model-physics.md) - Mathematical equations and regimes
- [Code Reference](../../prism/core/instruments/) - Source code implementations
- [Implementation Plan](../plans/four-f-system-consolidation.md) - Consolidation roadmap
