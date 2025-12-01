# Coherence Modes in FourFSystem

This guide demonstrates the three illumination coherence modes available in PRISM instruments: **coherent**, **incoherent**, and **partially coherent**.

## Overview

Real-world imaging scenarios require different physical models depending on the light source:

| Mode | Physical Scenario | Example Applications |
|------|-------------------|---------------------|
| **Coherent** | Laser illumination | Holography, coherent microscopy |
| **Incoherent** | Self-luminous objects | Fluorescence microscopy, astronomy |
| **Partially Coherent** | Extended sources | LED brightfield, Köhler illumination |

## Physics Background

### Coherent Illumination

Standard amplitude transfer where complex field propagates through the system:

```
I(x,y) = |IFFT{ P_det · P_illum · FFT{E_object} }|²
```

Phase information is preserved through the optical system.

### Incoherent Illumination

For self-luminous objects (fluorescence), each point emits independently:

```
I_out = IFFT{ OTF · FFT{I_in} }
where OTF = Autocorr(P_det)
```

**Key Point**: The illumination pupil is **ignored** because the object itself is the light source (emission, not transmission).

### Partially Coherent Illumination

Extended source illumination integrates over source points:

```
I = Σ_i w_i · |coherent_propagate(E, phase_i)|²
```

Each source point illuminates coherently; different source points are mutually incoherent.

---

## Example 1: Fluorescence Microscopy (Incoherent)

Fluorescence imaging is the canonical example of incoherent imaging. The sample emits light (self-luminous), so only the detection optics matter.

```python
import torch
from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.propagators import CoherenceMode

# Create microscope
config = MicroscopeConfig(
    wavelength=525e-9,  # GFP emission wavelength
    numerical_aperture=1.4,
    n_pixels=512,
    pixel_size=6.5e-6,
    magnification=60,
)
microscope = Microscope(config)

# Create fluorescent sample (emission intensity pattern)
# In fluorescence, the input represents emission intensity, not transmission
n = 512
sample = torch.zeros(n, n)
# Add some "fluorescent beads"
for cx, cy in [(200, 200), (256, 256), (300, 320)]:
    y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
    sample += torch.exp(-((x - cx)**2 + (y - cy)**2) / (10**2))

# Forward model with incoherent illumination
image = microscope.forward(
    sample,
    coherence_mode=CoherenceMode.INCOHERENT,
    illumination_mode='brightfield',  # Detection pupil only
)

print(f"Input shape: {sample.shape}")
print(f"Output shape: {image.shape}")
print(f"Output range: [{image.min():.4f}, {image.max():.4f}]")
```

### Key Points for Incoherent Mode

1. **Input is intensity**: The input represents emission intensity (real, non-negative)
2. **Complex inputs are auto-converted**: `|E|²` is computed automatically
3. **Only detection pupil matters**: Illumination pupil is ignored
4. **No phase contrast**: Phase information is lost

### Illumination Mode Warnings

Using phase-sensitive illumination modes with incoherent imaging triggers a warning:

```python
# This will warn - darkfield requires coherent scattering
image = microscope.forward(
    sample,
    coherence_mode=CoherenceMode.INCOHERENT,
    illumination_mode='darkfield',  # Warning issued
)
```

---

## Example 2: LED Brightfield (Partially Coherent)

LED microscopy uses an extended source, creating partially coherent illumination. This is modeled by integrating over source points.

```python
import torch
from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.propagators import CoherenceMode

# Create microscope
config = MicroscopeConfig(
    wavelength=550e-9,  # Green LED
    numerical_aperture=0.95,
    n_pixels=512,
    pixel_size=6.5e-6,
    magnification=40,
)
microscope = Microscope(config)

# Create transmission sample (complex field)
n = 512
y, x = torch.meshgrid(
    torch.linspace(-1, 1, n),
    torch.linspace(-1, 1, n),
    indexing='ij'
)

# Phase object (e.g., cell with varying optical path length)
phase = 0.5 * torch.exp(-(x**2 + y**2) / 0.1)
sample = torch.exp(1j * phase * 2 * torch.pi)

# Create extended source (Gaussian distribution)
# Represents the condenser aperture / LED pattern
sigma = 0.1  # Source extent (larger = more extended)
source_intensity = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
source_intensity = source_intensity / source_intensity.sum()  # Normalize

# Forward model with partially coherent illumination
image = microscope.forward(
    sample,
    coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
    source_intensity=source_intensity,
    n_source_points=200,  # More points = more accurate, slower
    illumination_mode='brightfield',
)

print(f"Input (complex): {sample.shape}, dtype: {sample.dtype}")
print(f"Source: {source_intensity.shape}")
print(f"Output: {image.shape}")
```

### Tuning Partially Coherent Parameters

**n_source_points**: Controls accuracy vs. speed trade-off

```python
# Fast but less accurate
image_fast = microscope.forward(
    sample,
    coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
    source_intensity=source,
    n_source_points=50,
)

# Slower but more accurate
image_accurate = microscope.forward(
    sample,
    coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
    source_intensity=source,
    n_source_points=500,
)
```

**Source extent (sigma)**: Affects spatial coherence

- **Small sigma** → More coherent (approaches fully coherent limit)
- **Large sigma** → Less coherent (more blurring, less speckle)

---

## Example 3: PSF Comparison

Compare PSF characteristics across coherence modes:

```python
import torch
import matplotlib.pyplot as plt
from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.propagators import CoherenceMode

# Create microscope
config = MicroscopeConfig(
    wavelength=550e-9,
    numerical_aperture=1.2,
    n_pixels=256,
    pixel_size=6.5e-6,
    magnification=60,
)
microscope = Microscope(config)

# Create point source
n = 256
delta = torch.zeros(n, n, dtype=torch.complex64)
delta[n // 2, n // 2] = 1.0

# Compute PSF for coherent mode
psf_coherent = microscope.forward(
    delta,
    coherence_mode=CoherenceMode.COHERENT,
)

# Compute PSF for incoherent mode
# (For incoherent, input is intensity, so use |delta|² = delta)
psf_incoherent = microscope.forward(
    delta.abs(),  # Real-valued intensity
    coherence_mode=CoherenceMode.INCOHERENT,
)

# Compute PSF for partially coherent mode
source = torch.zeros(n, n)
source[n//2, n//2] = 1.0  # Point source → approaches coherent
psf_partial_point = microscope.forward(
    delta,
    coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
    source_intensity=source,
    n_source_points=50,
)

# Extended source for comparison
y, x = torch.meshgrid(
    torch.linspace(-1, 1, n),
    torch.linspace(-1, 1, n),
    indexing='ij'
)
source_extended = torch.exp(-(x**2 + y**2) / 0.02)
psf_partial_extended = microscope.forward(
    delta,
    coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
    source_intensity=source_extended,
    n_source_points=100,
)

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(psf_coherent.numpy(), cmap='hot')
axes[0, 0].set_title('Coherent PSF')

axes[0, 1].imshow(psf_incoherent.numpy(), cmap='hot')
axes[0, 1].set_title('Incoherent PSF (OTF-based)')

axes[1, 0].imshow(psf_partial_point.numpy(), cmap='hot')
axes[1, 0].set_title('Partially Coherent (Point Source)')

axes[1, 1].imshow(psf_partial_extended.numpy(), cmap='hot')
axes[1, 1].set_title('Partially Coherent (Extended Source)')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('psf_comparison.png', dpi=150)
plt.show()
```

### Expected Results

1. **Coherent PSF**: Sharp Airy pattern with distinct rings
2. **Incoherent PSF**: Broader, smoother (OTF bandwidth is 2× coherent)
3. **Partially coherent (point)**: Should match coherent closely
4. **Partially coherent (extended)**: Intermediate between coherent and incoherent

---

## Example 4: Using with Different Instruments

All FourFSystem-based instruments support coherence modes:

### Telescope (Astronomy)

```python
from prism.core.instruments import Telescope, TelescopeConfig
from prism.core.propagators import CoherenceMode

config = TelescopeConfig(
    wavelength=550e-9,
    aperture_diameter=2.4,  # meters (HST-like)
    n_pixels=512,
    pixel_size=15e-6,
    focal_length=57.6,
)
telescope = Telescope(config)

# Stars are incoherent point sources
star_field = create_star_field(512)
image = telescope.forward(
    star_field,
    coherence_mode=CoherenceMode.INCOHERENT,
)
```

### Camera

```python
from prism.core.instruments import Camera, CameraConfig
from prism.core.propagators import CoherenceMode

config = CameraConfig(
    wavelength=550e-9,
    f_number=2.8,
    n_pixels=1024,
    pixel_size=4.5e-6,
    focal_length=50e-3,
)
camera = Camera(config)

# Natural scenes are typically incoherent
scene = load_scene_image()
image = camera.forward(
    scene,
    coherence_mode=CoherenceMode.INCOHERENT,
)
```

---

## API Reference

### CoherenceMode Enum

```python
from prism.core.propagators import CoherenceMode

CoherenceMode.COHERENT           # Laser illumination
CoherenceMode.INCOHERENT         # Self-luminous (fluorescence)
CoherenceMode.PARTIALLY_COHERENT # Extended source (LED)
```

### Forward Method Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coherence_mode` | `CoherenceMode` | `COHERENT` | Illumination coherence mode |
| `source_intensity` | `Tensor` | `None` | Source distribution (required for `PARTIALLY_COHERENT`) |
| `n_source_points` | `int` | `100` | Number of source samples for partial coherence |

---

## Best Practices

### When to Use Each Mode

| Use Case | Recommended Mode |
|----------|-----------------|
| Laser microscopy | `COHERENT` |
| Holography | `COHERENT` |
| Fluorescence microscopy | `INCOHERENT` |
| Stellar imaging | `INCOHERENT` |
| LED brightfield | `PARTIALLY_COHERENT` |
| Köhler illumination | `PARTIALLY_COHERENT` |

### Performance Tips

1. **Start with fewer source points** (50-100) during development
2. **Increase n_source_points** (200-500) for final results
3. **Use GPU** for partially coherent mode (many forward passes)
4. **Cache source_intensity** if using the same source repeatedly

### Common Pitfalls

1. **Forgetting source_intensity** for `PARTIALLY_COHERENT` mode → ValueError
2. **Using phase-sensitive modes with `INCOHERENT`** → UserWarning (may not be physically meaningful)
3. **Expecting phase contrast from incoherent mode** → Phase is lost in self-luminous imaging

---

## See Also

- [FourFSystem API](../api/four_f_base.md) - Complete API reference
- [Microscope Documentation](../api/microscope.md) - Microscope-specific features
- [OTF Propagator](../api/incoherent.md) - Incoherent propagation details
