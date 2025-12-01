# Propagator Selection Guide

## Overview

PRISM supports multiple light propagation methods optimized for different physical scenarios:

- **Fraunhofer**: Far-field approximation (astronomy, distant objects)
- **Fresnel**: Near-field propagation (lab setups, intermediate distances)
- **Angular Spectrum**: General-purpose method (works for all cases but slower)

The system can automatically select the appropriate propagator based on the physics of your observation, or you can manually specify one.

---

## Automatic Selection

PRISM can automatically select the appropriate propagator based on physical parameters. The selection is based on the Fresnel number, which characterizes whether the observation is in the far-field or near-field regime.

### Using Config Files

Add `propagator_method: auto` to your YAML configuration:

```yaml
# config.yaml
telescope:
  propagator_method: auto

physics:
  wavelength: 550e-9      # 550 nm (visible light)
  obj_distance: 1e6       # 1 km
  obj_diameter: 1.0       # 1 meter object

image:
  image_size: 1024        # 1024x1024 pixels
```

When `method: auto`, PRISM calculates the Fresnel number and selects:
- **Fraunhofer** if Fresnel number << 1 (far-field)
- **Angular Spectrum** if Fresnel number ~ 1 or > 1 (near-field)

### Using CLI

```bash
python -m prism.cli --propagator-method auto --config my_config.yaml
```

Or with inline parameters:

```bash
python main.py \
  --propagator-method auto \
  --obj_name europa \
  --wavelength 550e-9 \
  --obj_distance 6.28e11 \
  --name europa_auto
```

---

## Manual Override

You can manually specify a propagator if you know which method is appropriate for your scenario.

### Via Config File

```yaml
telescope:
  propagator_method: fraunhofer  # or fresnel, angular_spectrum
```

### Via CLI

```bash
# Use Fraunhofer propagator explicitly
python main.py --propagator-method fraunhofer --config my_config.yaml

# Use Fresnel propagator
python main.py --propagator-method fresnel --config my_config.yaml

# Use Angular Spectrum propagator
python main.py --propagator-method angular_spectrum --config my_config.yaml
```

---

## When to Use Each Propagator

### Fraunhofer Propagator

**Best for:** Far-field observations (astronomy, distant objects)

**Characteristics:**
- Fastest computation
- Valid when Fresnel number << 1
- Typical for: planetary observation, stellar imaging, long-distance scenarios

**Example scenarios:**
- Europa observation from Earth (~6.28×10¹¹ m)
- Titan observation (~1.2×10¹² m)
- Betelgeuse imaging (~6×10¹⁸ m)

**Physics:**
- Fresnel number: N_F = a²/(λz) << 1
- Far-field criterion: z >> a²/λ

```yaml
# Example: Europa observation
physics:
  wavelength: 550e-9        # 550 nm
  obj_distance: 6.28e11     # ~628 million km
  obj_diameter: 3.1e6       # Europa diameter

telescope:
  propagator_method: fraunhofer  # or auto (will select Fraunhofer)
```

---

### Fresnel Propagator (1-Step Impulse Response)

**Best for:** Medium-distance propagation (0.1 < F < 10)

**Characteristics:**
- Single FFT algorithm (~2x faster than old 2-step)
- Output grid scales with distance: dx_out = λz/(N·dx_in)
- Valid for z > z_crit = N·dx²/λ
- More accurate than old 2-step method (reversibility: <1% error)
- Balanced between speed and accuracy

**Example scenarios:**
- Laboratory optical setups (10 cm - 10 m)
- Intermediate distance imaging
- When Angular Spectrum is too slow but Fraunhofer too approximate

**Physics:**
- Fresnel number: 0.1 < F < 10
- Uses impulse response formulation
- Single quadratic phase in spatial domain before FFT
- Single quadratic phase in output domain after FFT

**Output Grid Scaling:**
The 1-step method changes pixel size at the output. This is fundamental physics -
the observation grid "zooms out" as the beam diffracts.

```yaml
# Example: Lab setup with 1-step Fresnel
physics:
  wavelength: 633e-9        # HeNe laser (633 nm)
  obj_distance: 1.0         # 1 meter
  fov: 0.01                 # 1 cm field of view

telescope:
  propagator_method: fresnel
  image_size: 256
  dx: 39e-6                 # Computed from FOV/image_size

# Output grid will have different pixel size!
# dx_out = (633e-9 * 1.0) / (256 * 39e-6) ≈ 6.4e-5 m
```

**Accessing Output Grid:**
```python
from prism.core.grid import Grid
from prism.core.propagators import FresnelPropagator

grid = Grid(nx=256, dx=39e-6, wavelength=633e-9)
prop = FresnelPropagator(grid, distance=1.0)
output = prop(input_field)

print(f"Input  pixel size: {grid.dx:.2e} m")
print(f"Output pixel size: {prop.output_grid.dx:.2e} m")
# Input:  3.90e-05 m
# Output: 6.40e-05 m (scaled by λz/(N·dx))
```

---

### Angular Spectrum Propagator

**Best for:** General-purpose, all scenarios (especially near-field)

**Characteristics:**
- Most accurate for all propagation distances
- Higher computational cost
- Works well for near-field (Fresnel number ≥ 1)
- Safe choice when unsure

**Example scenarios:**
- Near-field imaging (< 1 m)
- Complex wavefront propagation
- When accuracy is paramount

**Physics:**
- Valid for all Fresnel numbers
- Uses Fourier-based propagation

```yaml
# Example: Near-field scenario
physics:
  wavelength: 550e-9        # 550 nm
  obj_distance: 0.1         # 10 cm
  obj_diameter: 0.02        # 2 cm object

telescope:
  propagator_method: angular_spectrum
```

---

## Decision Tree

```
┌─────────────────────────────────────┐
│ What is your observation scenario? │
└─────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
    Far-field          Intermediate/Near-field
  (astronomy)         (lab setup, F > 0.1)
  F << 0.1                    │
        │              ┌──────┴──────┐
        ▼              ▼             ▼
   Fraunhofer    Fresnel (1-step)   Angular Spectrum
   (fastest)     (balanced)          (most accurate)
        │              │              │
        └──────────────┴──────────────┘
                  ▼
           Not sure? Use auto!
```

---

## Configuration Examples

### Example 1: Europa Observation (Auto-Selection)

```yaml
name: europa_observation
comment: Europa observation with automatic propagator selection

physics:
  wavelength: 550e-9
  obj_distance: 6.28e11    # ~628 million km
  obj_diameter: 3.1e6      # Europa diameter ~3100 km
  obj_name: europa

telescope:
  propagator_method: auto   # Will select Fraunhofer
  n_samples: 200
  sample_diameter: 50

image:
  image_size: 1024
```

**Expected behavior:** Auto-selects Fraunhofer (Fresnel number ~ 10⁻¹³)

---

### Example 2: Lab Experiment (Manual Fresnel)

```yaml
name: lab_diffraction
comment: Laboratory diffraction experiment

physics:
  wavelength: 633e-9       # HeNe laser
  obj_distance: 2.0        # 2 meters
  obj_diameter: 0.05       # 5 cm aperture

telescope:
  propagator_method: fresnel
  n_samples: 150
  sample_diameter: 30

image:
  image_size: 512
```

---

### Example 3: Near-Field Imaging (Angular Spectrum)

```yaml
name: near_field_imaging
comment: Near-field microscopy setup

physics:
  wavelength: 550e-9
  obj_distance: 0.05       # 5 cm
  obj_diameter: 0.01       # 1 cm

telescope:
  propagator_method: angular_spectrum
  n_samples: 100
  sample_diameter: 20

image:
  image_size: 256
```

---

## Performance Considerations

| Propagator        | Relative Speed | Memory Usage | Accuracy        |
|-------------------|----------------|--------------|-----------------|
| Fraunhofer        | 1.0× (fastest) | Low          | Excellent (far) |
| Fresnel           | ~1.2×          | Medium       | Good (near)     |
| Angular Spectrum  | ~1.5-2.0×      | High         | Excellent (all) |

**Recommendation:** Use `auto` selection for optimal performance while maintaining accuracy.

---

## Validation and Debugging

### Check Selected Propagator

When using `auto`, PRISM logs which propagator was selected:

```
[INFO] Using propagator from config: auto (FOV=1.0000e-03 m, wavelength=5.50e-07 m, distance=6.28e+11 m)
[INFO] Auto-selected: FraunhoferPropagator (Fresnel number: 1.23e-13)
```

### Common Issues

**Issue:** Propagator selection seems wrong
- **Solution:** Check physics parameters (wavelength, distance, FOV)
- **Debug:** Look at logged Fresnel number

**Issue:** Slower than expected
- **Solution:** Use `auto` or manually select Fraunhofer for far-field cases
- **Debug:** Check if Angular Spectrum is being used unnecessarily

**Issue:** Results look incorrect
- **Solution:** Verify physics parameters match your scenario
- **Debug:** Try `angular_spectrum` (most robust) to establish baseline

---

## Backward Compatibility

**Default behavior:** If `propagator_method` is not specified (or set to `None`), PRISM uses the original `FreeSpacePropagator` for backward compatibility with existing configs and scripts.

```yaml
# Old configs without propagator_method still work
telescope:
  n_samples: 200
  sample_diameter: 50
  # No propagator_method specified → uses FreeSpacePropagator
```

---

## Advanced: Physics Background

### Fresnel Number

The Fresnel number characterizes the propagation regime:

```
N_F = a² / (λ z)

where:
  a = characteristic aperture size (or FOV/2)
  λ = wavelength
  z = propagation distance
```

**Regimes:**
- N_F << 1: Far-field (Fraunhofer approximation valid)
- N_F ~ 1: Fresnel regime (intermediate)
- N_F >> 1: Near-field (geometric optics approximation)

PRISM auto-selection uses:
- N_F < 0.1 → Fraunhofer
- N_F ≥ 0.1 → Angular Spectrum

---

## References

- [Propagator Implementation Guide](../implementation_guides/foundational_revision/02_propagators.md)
- [Performance Optimization Guide](../PERFORMANCE_OPTIMIZATION_GUIDE.md)
- [Migration Guide](../MIGRATION_GUIDE.md)

---

## Incoherent Illumination

PRISM supports incoherent and partially coherent illumination for simulating extended astronomical sources where light from different source points has no fixed phase relationship.

### When to Use Incoherent Propagation

Use incoherent propagation when:

- **Extended astronomical sources**: Planets, nebulae, galaxies with no coherent illumination
- **Thermal/broadband sources**: Sources with short coherence length
- **Ground-based seeing-limited imaging**: Atmospheric turbulence destroys coherence
- **Intensity-based simulations**: When only concerned with power distribution

### Available Incoherent Propagators

#### OTFPropagator (Fully Incoherent)

Best for most incoherent imaging scenarios. Uses the Optical Transfer Function (autocorrelation of the pupil function) to propagate intensity distributions.

```python
from prism.core.propagators import OTFPropagator
from prism.core.grid import Grid

# Create grid and aperture
grid = Grid(nx=256, dx=1e-5, wavelength=550e-9)
aperture = create_circular_aperture(grid, radius=0.01)

# Create propagator
propagator = OTFPropagator(aperture, grid)

# Propagate intensity (real-valued input)
intensity_out = propagator(intensity_in)
```

**Characteristics:**
- Operates on real intensity tensors (not complex fields)
- Uses precomputed OTF (autocorrelation of pupil)
- Fast: single FFT-based convolution
- Energy conserving
- No speckle or interference artifacts

**When to use:**
- Fully incoherent imaging (isoplanatic)
- Extended sources much larger than coherence length
- Speed is critical and source is uniform

#### ExtendedSourcePropagator (Partially Coherent)

For complex extended sources or non-isoplanatic systems. Models spatially extended incoherent sources by decomposing them into independent coherent point sources.

```python
from prism.core.propagators import (
    ExtendedSourcePropagator,
    FraunhoferPropagator,
    create_stellar_disk,
)

# Create propagator with coherent backbone
coherent_prop = FraunhoferPropagator()
propagator = ExtendedSourcePropagator(
    coherent_propagator=coherent_prop,
    grid=grid,
    n_source_points=500,
    sampling_method="adaptive",  # or "grid", "monte_carlo"
    batch_size=64,
)

# Create stellar disk with limb darkening
stellar_disk = create_stellar_disk(
    grid,
    angular_diameter=0.01,
    limb_darkening=0.6,  # Sun-like limb darkening
)

# Propagate
intensity_out = propagator(stellar_disk, aperture=aperture)
```

**Characteristics:**
- Decomposes source into coherent point sources
- Supports three sampling methods: "grid", "monte_carlo", "adaptive"
- Handles non-isoplanatic systems (spatially-varying PSF)
- More computationally intensive than OTF

**When to use:**
- Partially coherent illumination (finite-size sources)
- Per-source-point control (e.g., limb darkening)
- PSF varies across field of view
- Modeling specific source geometries (binary stars, disks)

### Source Geometry Helpers

PRISM provides helper functions for creating common source geometries:

```python
from prism.core.propagators import (
    create_stellar_disk,
    create_gaussian_source,
    create_binary_source,
    create_ring_source,
    estimate_required_samples,
)

# Stellar disk with limb darkening (sun-like)
sun_disk = create_stellar_disk(grid, angular_diameter=0.01, limb_darkening=0.6)

# Gaussian source (e.g., unresolved star with seeing)
gaussian = create_gaussian_source(grid, sigma=5e-5)

# Binary star system
binary = create_binary_source(
    grid,
    separation=20 * grid.dx,
    flux_ratio=0.5,  # secondary is half brightness
    position_angle=0.0,
)

# Ring/annulus (planetary rings, circumstellar disk)
ring = create_ring_source(grid, inner_radius=0.005, outer_radius=0.01)

# Estimate number of samples needed for accuracy
n_samples = estimate_required_samples(stellar_disk, grid, target_snr=100.0)
```

### Using Factory Functions

```python
from prism.core.propagators import create_propagator, select_propagator

# Create OTF propagator via factory
propagator = create_propagator(
    "otf",
    aperture=aperture,
    grid=grid,
)

# Create Extended Source propagator via factory
propagator = create_propagator(
    "extended_source",
    grid=grid,
    n_source_points=500,
    sampling_method="adaptive",
)

# Automatic incoherent selection
propagator = select_propagator(
    wavelength=550e-9,
    obj_distance=6.28e11,
    fov=0.01,
    illumination="incoherent",  # Key parameter
    aperture=aperture,
    grid=grid,
)

# Partially coherent selection
propagator = select_propagator(
    wavelength=550e-9,
    obj_distance=6.28e11,
    fov=0.01,
    illumination="partially_coherent",
    grid=grid,
    n_source_points=500,
)
```

### Telescope Integration

The `Telescope` class supports incoherent propagation directly:

```python
from prism.core.telescope import Telescope

telescope = Telescope(grid=grid, aperture=aperture)

# Incoherent propagation
output = telescope.propagate_incoherent(intensity_in)

# Extended source propagation
output = telescope.propagate_extended_source(
    source_intensity,
    n_source_points=500,
)

# General-purpose with illumination mode
output = telescope.propagate_with_illumination(
    tensor,
    illumination="incoherent",  # or "partially_coherent", "coherent"
)
```

### Coherent vs Incoherent Comparison

| Property | Coherent | Incoherent |
|----------|----------|------------|
| Input | Complex field | Real intensity |
| Transfer function | CTF (pupil) | OTF (autocorr of pupil) |
| Interference | Yes | No |
| Speckle | Yes | No |
| Contrast | Higher | Lower |
| Typical sources | Lasers, point stars | Extended objects |
| PRISM propagators | Fraunhofer, AngularSpectrum | OTF, ExtendedSource |

### Mathematical Background

**Optical Transfer Function (OTF)**:
```
OTF(fx, fy) = ∫∫ P(x, y) · P*(x - λz·fx, y - λz·fy) dx dy
            = Autocorrelation{P(x, y)}
```

**Incoherent imaging equation**:
```
I_out(x, y) = IFFT[FFT[I_in(x, y)] × OTF(fx, fy)]
```

**Van Cittert-Zernike Theorem** (for partially coherent):
```
Coherence length: L_c ≈ λz / θ_source
```
Larger source angular diameter → shorter coherence length.

### Configuration Examples

#### OTF for Extended Nebula

```yaml
physics:
  wavelength: 656.3e-9    # H-alpha
  obj_distance: 4.1e19    # Orion Nebula (~1300 pc)

telescope:
  propagator_method: otf
  illumination: incoherent
  aperture_diameter: 10.0  # 10m telescope
```

#### Extended Source for Stellar Disk

```yaml
physics:
  wavelength: 550e-9
  obj_distance: 6e18      # Betelgeuse (~650 ly)

telescope:
  propagator_method: extended_source
  illumination: partially_coherent
  n_source_points: 500
  sampling_method: adaptive
  limb_darkening: 0.3     # Red giant
```

---

## Summary

**Quick Start:**
1. Add `propagator_method: auto` to your config (recommended)
2. Or use `--propagator-method auto` on the command line
3. PRISM will select the optimal propagator based on physics

**Manual Control:**
- Far-field (astronomy): Use `fraunhofer`
- Near-field (lab): Use `fresnel` or `angular_spectrum`
- Extended sources (incoherent): Use `otf`
- Stellar disks (partially coherent): Use `extended_source`
- Unsure: Use `auto`

**Illumination Modes:**
- `coherent`: Point sources, lasers, unresolved stars (default)
- `incoherent`: Extended objects, thermal sources, seeing-limited
- `partially_coherent`: Finite-size sources with partial interference

**Best Practice:** Start with `auto`, switch to manual override only if you have specific requirements.
