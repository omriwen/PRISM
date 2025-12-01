# Physical Constants Reference

**Quick-lookup for universal constants, length units, and optical criteria used in PRISM.**

## Universal Constants

| Constant | Symbol | Value | Units | PRISM Variable |
|----------|--------|-------|-------|----------------|
| Speed of light | c | 3×10⁸ | m/s | `c` |
| Solar radius | R☉ | 6.957×10⁸ | m | `solar_radius` |
| Pi | π | 3.14159... | - | `pi` (from numpy) |

## Length Units

### Standard SI Units

| Unit | Symbol | Value in Meters | PRISM Variable |
|------|--------|-----------------|----------------|
| Nanometer | nm | 1×10⁻⁹ | `nm` |
| Micrometer | µm | 1×10⁻⁶ | `um` |
| Millimeter | mm | 1×10⁻³ | `mm` |
| Centimeter | cm | 1×10⁻² | `cm` |
| Kilometer | km | 1×10³ | `km` |

### Astronomical Units

| Unit | Symbol | Value in Meters | PRISM Variable |
|------|--------|-----------------|----------------|
| Astronomical unit | AU | 1.496×10¹¹ | `au` |
| Light year | ly | 9.461×10¹⁵ | `ly` |
| Parsec | pc | 3.086×10¹⁶ | `pc` |

## Length Conversion Table

Quick conversions for common optical wavelengths and distances:

| From | To | Multiply By | Example |
|------|-----|-------------|---------|
| nm → m | meters | 1×10⁻⁹ | 550 nm = 550×10⁻⁹ m |
| µm → m | meters | 1×10⁻⁶ | 10 µm = 10×10⁻⁶ m |
| mm → m | meters | 1×10⁻³ | 50 mm = 0.05 m |
| cm → m | meters | 1×10⁻² | 6.5 cm = 0.065 m |
| m → AU | astronomical units | 1/1.496×10¹¹ | 1.496×10¹¹ m = 1 AU |
| m → ly | light years | 1/9.461×10¹⁵ | 9.461×10¹⁵ m = 1 ly |
| m → pc | parsecs | 1/3.086×10¹⁶ | 3.086×10¹⁶ m = 1 pc |

## Optical Wavelengths

Common wavelengths used in optical systems:

| Color/Application | Wavelength (nm) | Wavelength (m) | Use Case |
|-------------------|----------------|----------------|----------|
| UV (mercury lamp) | 365 | 3.65×10⁻⁷ | Fluorescence excitation |
| Violet | 405 | 4.05×10⁻⁷ | Confocal microscopy |
| Blue (argon laser) | 488 | 4.88×10⁻⁷ | GFP fluorescence |
| Green (typical) | 550 | 5.50×10⁻⁷ | Standard illumination |
| Red (HeNe laser) | 633 | 6.33×10⁻⁷ | Alignment, confocal |
| Near-IR | 850 | 8.50×10⁻⁷ | Deep tissue imaging |

## Fresnel Number

### Definition

The Fresnel number characterizes the diffraction regime:

$$F = \frac{a^2}{\lambda z}$$

Where:
- **a** = aperture radius or characteristic width (m)
- **λ** = wavelength (m)
- **z** = propagation distance (m)

### Regime Classification

| Fresnel Number | Regime | Propagation Method | Description |
|----------------|--------|-------------------|-------------|
| F > 10 | Near-field | Angular Spectrum | Geometric optics valid, diffraction minimal |
| 0.1 < F < 10 | Intermediate (Fresnel) | Angular Spectrum | Fresnel diffraction dominant |
| F < 0.1 | Far-field (Fraunhofer) | Fraunhofer (FFT) | Far-field diffraction, simple Fourier transform |

### Physical Interpretation

- **Near-field (F >> 1)**: Wave behaves like geometric rays, diffraction negligible
- **Fresnel zone (F ~ 1)**: Wave exhibits curved wavefront diffraction
- **Far-field (F << 1)**: Wave exhibits plane-wave diffraction, angular spectrum simplifies to Fourier transform

## Optical Criteria Functions

PRISM provides functions to evaluate optical regimes:

### Fresnel Number Calculation

```python
from prism.config.constants import fresnel_number

# Microscope: a=5mm, z=10cm, λ=550nm
F = fresnel_number(width=0.005, distance=0.1, wavelength=550e-9)
# Result: F ≈ 454 (near-field, use angular spectrum)

# Drone: a=25mm, z=50m, λ=550nm
F = fresnel_number(width=0.025, distance=50, wavelength=550e-9)
# Result: F ≈ 22.7 (Fresnel regime, use angular spectrum)

# Telescope: a=1m, z=10pc, λ=550nm
F = fresnel_number(width=1.0, distance=3.086e17, wavelength=550e-9)
# Result: F ≈ 5.9×10⁻¹² (far-field, use Fraunhofer/FFT)
```

### Fraunhofer Regime Check

```python
from prism.config.constants import is_fraunhofer

# Check if Fraunhofer approximation is valid (F < 0.1)
is_far_field = is_fraunhofer(width=1.0, distance=1e6, wavelength=550e-9)
```

### Fresnel Regime Check

```python
from prism.config.constants import is_fresnel, fresnel_number_critical

# Check paraxial approximation validity
F_par = fresnel_number_critical(width=0.05, distance=100, wavelength=550e-9)
is_fresnel_valid = is_fresnel(width=0.05, distance=100, wavelength=550e-9)
```

### Coherence Radius

The coherence radius determines spatial coherence:

```python
from prism.config.constants import r_coh

# Coherence radius for extended source
r_c = r_coh(source_width=1e-3, distance=1.0, wavelength=550e-9)
# Result: Coherence radius in meters
```

**Formula**:
$$r_{\text{coh}} = \frac{\lambda z}{\pi w}$$

Where:
- **w** = source width (m)
- **z** = distance from source (m)
- **λ** = wavelength (m)

## Usage Examples

### Convert Units

```python
from prism.config.constants import nm, um, mm, cm

# Define wavelength in nanometers
wavelength = 550 * nm  # = 550e-9 m

# Define aperture in millimeters
aperture = 25 * mm  # = 0.025 m

# Define distance in centimeters
distance = 10 * cm  # = 0.1 m
```

### Calculate Diffraction Limits

```python
from prism.config.constants import c, nm

# Abbe diffraction limit for microscopy
wavelength = 550 * nm
NA = 1.4
lateral_resolution = 0.61 * wavelength / NA
# Result: ~240 nm

# Rayleigh criterion for telescope
diameter = 1.0  # 1m telescope
angular_resolution = 1.22 * wavelength / diameter
# Result: ~6.7×10⁻⁷ radians ≈ 0.14 arcseconds
```

## Source Files

This reference is based on:
- [prism/config/constants.py](../../prism/config/constants.py) (lines 22-114)

## Related References

- [Optical Resolution Limits](optical_resolution_limits.md) - Resolution formulas and limits
- [Fresnel Propagation Regimes](fresnel_propagation_regimes.md) - Propagation regime lookup tables
- [Microscopy Parameters](microscopy_parameters.md) - Microscope-specific wavelengths and NA values
- [Drone Camera Parameters](drone_camera_parameters.md) - Drone-specific distances and apertures

---

**Last Updated**: 2025-01-26
**Accuracy**: All values verified against `prism/config/constants.py`
**Formulas**: Standard optical physics (Born & Wolf, Goodman)
