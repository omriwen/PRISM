# Scanning Modes Technical Reference

**Quick-lookup reference for scanning aperture vs scanning illumination modes in PRISM synthetic aperture imaging.**

## Quick Navigation

- [Overview](#overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Scanning Aperture Mode](#scanning-aperture-mode)
- [Scanning Illumination Mode](#scanning-illumination-mode)
- [Spatial Illumination Scanning](#spatial-illumination-scanning)
- [Equivalence and Differences](#equivalence-and-differences)
- [Partial Coherence Effects](#partial-coherence-effects)
- [API Reference](#api-reference)
- [Physical Interpretation](#physical-interpretation)

---

## Overview

PRISM supports two scanning modes for synthetic aperture imaging:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Scanning Aperture** | Fixed illumination, moving detection aperture in k-space | Traditional PRISM, Synthetic Aperture Imaging |
| **Scanning Illumination** | Moving illumination angle, fixed detection at DC | Fourier Ptychographic Microscopy (FPM) |

Both modes achieve synthetic aperture reconstruction by sampling different regions of k-space sequentially.

### When to Use Each Mode

**Scanning Aperture (Default)**:
- Simulating traditional PRISM hardware
- Direct k-space sampling interpretation
- Coherent imaging systems

**Scanning Illumination**:
- Fourier Ptychographic Microscopy (FPM) workflows
- LED array microscopy simulation
- Finite source/partial coherence modeling
- Physical angle-varied illumination systems

---

## Mathematical Foundations

### Fourier Shift Theorem

The equivalence between scanning modes is based on the Fourier shift theorem:

$$\mathcal{F}\{f(x) \cdot e^{i 2\pi k_0 x}\} = \tilde{F}(k - k_0)$$

Multiplying by a phase tilt in spatial domain shifts the spectrum in k-space.

### Object Spectrum

For an object with complex transmission $O(x)$, its Fourier transform is:

$$\tilde{O}(k) = \mathcal{F}\{O(x)\}$$

---

## Scanning Aperture Mode

### Forward Model

```
Object O → FFT → Õ(k) → Mask at (kx, ky) → IFFT → |·|² → Intensity
```

**Mathematical formulation**:

$$I_{aperture}(x) = \left| \mathcal{F}^{-1}\left\{ \tilde{O}(k) \cdot A(k - k_{scan}) \right\} \right|^2$$

Where:
- $\tilde{O}(k)$ = Object spectrum
- $A(k)$ = Detection aperture (typically circular)
- $k_{scan}$ = Scan position in k-space
- $I_{aperture}$ = Measured intensity

### k-Space Interpretation

At each scan position, the sub-aperture samples a circular region of k-space centered at $k_{scan}$:

```
Full k-space          Sub-aperture sample
    ┌─────────┐          ┌─────────┐
    │  ┌───┐  │          │         │
    │  │ ○ │  │    →     │    ●    │
    │  └───┘  │          │         │
    └─────────┘          └─────────┘
  DC at center        Sampling at k_scan
```

### Implementation

```python
from prism.core.instruments import Microscope, MicroscopeConfig

config = MicroscopeConfig(
    n_pixels=512,
    pixel_size=5e-6,
    wavelength=520e-9,
    numerical_aperture=0.9,
    magnification=40,
)
microscope = Microscope(config)

# Forward pass with scanning aperture
measurement = microscope.forward(
    field=object_field,
    aperture_center=[10, 5],  # Pixel offset from DC
    aperture_radius=15,       # Pixels
)
```

---

## Scanning Illumination Mode

### Forward Model

```
Object O × exp(i·k_illum·x) → FFT → Õ(k-k_illum) → Mask at DC → IFFT → |·|² → Intensity
```

**Mathematical formulation**:

$$I_{illum}(x) = \left| \mathcal{F}^{-1}\left\{ \tilde{O}(k - k_{illum}) \cdot A(k) \right\} \right|^2$$

Where:
- $k_{illum}$ = Illumination tilt (k-space shift)
- $A(k)$ = Detection aperture centered at DC

### Physical Interpretation

Tilted illumination shifts the object spectrum in k-space:

```
Plane wave illumination        Tilted illumination
        │                           ╲
        │                            ╲
        ▼                             ▼
    ┌───────┐                     ┌───────┐
    │   O   │                     │   O   │
    └───────┘                     └───────┘

Spectrum at DC               Spectrum shifted by k_illum
    ┌─────────┐                  ┌─────────┐
    │    ●    │                  │  ●      │
    │   DC    │                  │         │
    └─────────┘                  └─────────┘
```

### Implementation (Planned)

```python
# After Task 2.1-2.2 implementation
measurement = microscope.forward(
    field=object_field,
    illumination_center=[10, 5],   # k-space position (pixels from DC)
    illumination_radius=15,        # Sub-aperture radius
    illumination_source_type='point',  # or 'gaussian', 'circular'
)
```

### Source Types

| Type | Description | Coherence |
|------|-------------|-----------|
| `POINT` | Tilted plane wave | Fully coherent |
| `GAUSSIAN` | Gaussian intensity profile | Partially coherent |
| `CIRCULAR` | Top-hat intensity profile | Partially coherent |
| `CUSTOM` | User-defined profile | Depends on profile |

---

## Spatial Illumination Scanning

### Overview

Spatial illumination mode (`IlluminationScanMethod.SPATIAL`) models a physically-shifted
illumination source at finite distance from the object. Unlike angular illumination
(tilted plane wave from source at infinity), this creates position-dependent
illumination angles.

### Physical Model

For a point source at position (x₀, y₀, -z):
- Illumination at object point (x, y) arrives at angle θ ≈ atan2(x-x₀, z)
- Phase varies spatially: φ(x,y) ≈ k·[(x-x₀)² + (y-y₀)²] / (2z)

This creates a spherical/quadratic wavefront centered at the source position.

### Usage

```python
from prism.core.measurement_system import (
    MeasurementSystem,
    MeasurementSystemConfig,
    ScanningMode,
    IlluminationScanMethod,
)

config = MeasurementSystemConfig(
    scanning_mode=ScanningMode.ILLUMINATION,
    illumination_scan_method=IlluminationScanMethod.SPATIAL,
    illumination_source_type="GAUSSIAN",
    illumination_radius=5e-6,  # Physical sigma in meters
    illumination_source_distance=10e-3,  # 10mm source distance
)
ms = MeasurementSystem(microscope, config=config)

# Centers in pixel units (converted to meters internally)
meas = ms.get_measurements(obj, [[0, 0], [10, 5]])
```

### Comparison with Angular Mode

| Aspect | ANGULAR | SPATIAL |
|--------|---------|---------|
| Source location | At infinity | Finite distance |
| Phase profile | Uniform tilt | Quadratic/spherical |
| illumination_radius units | k-space (1/m) | Physical (m) |
| Requires distance | No | Yes |

---

## Equivalence and Differences

### Perfect Equivalence (Point Source)

For **point-like illumination** (ideal tilted plane wave), the two modes are mathematically equivalent:

$$I_{aperture}(x; k_{scan}) = I_{illum}(x; k_{illum} = -k_{scan})$$

**Proof**:
1. Scanning aperture at $+k$: samples $\tilde{O}(k)$ around $k$
2. Tilted illumination at $-k$: shifts spectrum by $-k$, then samples around DC
3. Both sample the same k-space region: $\tilde{O}(k)$

### Numerical Verification

```python
# Equivalence test (should give same result)
I_aperture = microscope.forward(field, aperture_center=[10, 0])
I_illum = microscope.forward(field, illumination_center=[-10, 0])

# These should match within numerical precision
assert torch.allclose(I_aperture, I_illum, rtol=1e-6)
```

### When They Differ

For **finite-size illumination sources**, the modes are NOT exactly equivalent:

| Aspect | Scanning Aperture | Scanning Illumination |
|--------|-------------------|----------------------|
| k-space sampling | Hard circular mask | Convolution with source FT |
| Edge behavior | Sharp cutoff | Soft/smooth edge |
| Coherence | Fully coherent | Partially coherent |
| Physical model | Detection aperture | Source size |

---

## Partial Coherence Effects

### Source Size and Coherence

For finite-size sources, the spatial coherence width is:

$$\sigma_x \approx \frac{1}{2\pi \sigma_k}$$

Where $\sigma_k$ is the k-space width of the source.

### Effect on k-Space Sampling

```
Point source (coherent)          Finite source (partial coherence)
     ┌─────────┐                      ┌─────────┐
     │    │    │                      │  ░░░░   │
     │────●────│                      │ ░░●░░░  │
     │    │    │                      │  ░░░░   │
     └─────────┘                      └─────────┘
   Sharp boundary                  Smooth/soft boundary
```

### Gaussian Source Example

For a Gaussian source with k-space standard deviation $\sigma_k$:

- **k-space transfer function**: $H(k) = \exp(-k^2 / 2\sigma_k^2)$
- **Spatial coherence length**: $l_c \approx 1 / (2\pi \sigma_k)$
- **Effective NA reduction**: Source size reduces effective resolution

### Implementation

```python
from prism.core.optics.illumination import (
    IlluminationSource,
    IlluminationSourceType,
)

# Gaussian source with finite extent
source = IlluminationSource(
    source_type=IlluminationSourceType.GAUSSIAN,
    k_center=[0.1e6, 0],  # 1/meters
    sigma=0.02e6,         # k-space width
)
```

---

## API Reference

### Illumination Module

```python
from prism.core.optics.illumination import (
    IlluminationSourceType,  # Enum: POINT, GAUSSIAN, CIRCULAR, CUSTOM
    IlluminationSource,      # Dataclass for source configuration
    create_phase_tilt,       # Generate tilted plane wave
    create_illumination_envelope,  # Generate source envelope
    create_illumination_field,     # Combined illumination field
)
```

### Fourier Utilities

```python
from prism.core.optics.fourier_utils import (
    illum_angle_to_k_shift,     # Angle → k-space
    k_shift_to_illum_angle,     # k-space → angle
    illum_position_to_k_shift,  # Position → k-space
    k_shift_to_illum_position,  # k-space → position
    validate_k_shift_within_na, # Check NA limits
    pixel_to_k_shift,           # Pixel → k-space
    k_shift_to_pixel,           # k-space → pixel
)
```

### Conversion Examples

```python
import numpy as np
from prism.core.optics.fourier_utils import (
    illum_angle_to_k_shift,
    k_shift_to_illum_angle,
    validate_k_shift_within_na,
)

# Convert 10° illumination angle to k-space
k = illum_angle_to_k_shift(
    theta=np.radians(10),
    wavelength=520e-9,
    medium_index=1.0,
)
print(f"k = {k:.2e} 1/m")  # k = 3.34e+05 1/m

# Check if within NA=0.5
is_valid = validate_k_shift_within_na(
    k_shift=k,
    na=0.5,
    wavelength=520e-9,
)
print(f"Within NA: {is_valid}")  # True
```

---

## Physical Interpretation

### Scanning Aperture

- **Physical setup**: Single illumination, scanning detector or mask
- **Analogy**: Radio telescope with steerable dish
- **k-space view**: Direct sampling of object spectrum

### Scanning Illumination

- **Physical setup**: LED array or tilted illumination, fixed detector
- **Analogy**: Flashlight illuminating from different angles
- **k-space view**: Shift-then-sample paradigm

### Synthetic Aperture Reconstruction

Both modes enable super-resolution through synthetic aperture:

```
Individual measurements         Synthetic aperture
   ○ ○ ○                           ┌─────────┐
  ○ ● ○ ○        Combine →         │  ●●●●●  │
   ○ ○ ○                           │  ●●●●●  │
                                   │  ●●●●●  │
                                   └─────────┘
Low-NA images              Extended k-space coverage
```

The final resolution is determined by the total k-space coverage, not the individual aperture size.

---

## Quick Reference Tables

### Angle-to-k Conversion (λ = 520nm, air)

| Angle (°) | k (1/m) | Effective NA |
|-----------|---------|--------------|
| 1 | 3.36×10⁴ | 0.017 |
| 5 | 1.68×10⁵ | 0.087 |
| 10 | 3.34×10⁵ | 0.174 |
| 20 | 6.58×10⁵ | 0.342 |
| 30 | 9.62×10⁵ | 0.500 |
| 45 | 1.36×10⁶ | 0.707 |
| 60 | 1.67×10⁶ | 0.866 |

### Mode Comparison Summary

| Feature | Scanning Aperture | Scanning Illumination |
|---------|-------------------|----------------------|
| Forward model | FFT → Mask → IFFT | Tilt × Object → FFT → Mask → IFFT |
| k-space sampling | Direct | Shift + sample at DC |
| Coherence model | Fully coherent | Coherent or partial |
| Source modeling | No | Yes (POINT, GAUSSIAN, etc.) |
| Physical analog | Scanning detector | Tilted/LED illumination |
| PRISM mode | Default | Alternative (FPM-style) |

---

## Source Files

This reference is based on:
- [prism/core/optics/illumination.py](../../prism/core/optics/illumination.py)
- [prism/core/optics/fourier_utils.py](../../prism/core/optics/fourier_utils.py)
- [prism/core/instruments/microscope.py](../../prism/core/instruments/microscope.py)

## Related References

- [Fresnel Propagation Regimes](fresnel_propagation_regimes.md) - Diffraction regime selection
- [Optical Resolution Limits](optical_resolution_limits.md) - Rayleigh criterion
- [Microscopy Parameters](microscopy_parameters.md) - Microscope specifications

## Related Documentation

- [Scanning Illumination Implementation Plan](../plans/scanning-illumination-forward-model.md)
- [Spatial Illumination Scanning Implementation Plan](../plans/spatial-illumination-scanning.md)

---

**Last Updated**: 2025-12-01
**Status**: Phase 5 implementation - Spatial illumination mode documented
