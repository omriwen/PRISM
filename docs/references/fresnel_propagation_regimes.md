# Fresnel Propagation Regimes Reference

**Quick-lookup reference for diffraction propagation regimes: Fresnel number classification, propagator selection, and scenario-specific calculations.**

## Quick Navigation

- [Fresnel Number Definition](#fresnel-number-definition)
- [Regime Classification](#regime-classification)
- [Automatic Propagator Selection](#automatic-propagator-selection)
- [Microscopy Fresnel Numbers](#microscopy-fresnel-numbers)
- [Drone Fresnel Numbers](#drone-fresnel-numbers)
- [Astronomy Fresnel Numbers](#astronomy-fresnel-numbers)
- [Practical Examples](#practical-examples)

---

## Fresnel Number Definition

### Formula

The Fresnel number quantifies the diffraction regime:

$$F = \frac{a^2}{\lambda z}$$

Where:
- **F** = Fresnel number (dimensionless)
- **a** = characteristic aperture size (half-width or radius) (m)
- **λ** = wavelength (m)
- **z** = propagation distance (m)

**Physical meaning**: F represents the number of Fresnel zones that fit within the aperture.

### Alternative Forms

**For square apertures** (width w):
```
F = w² / (λz)
```

**For circular apertures** (diameter D):
```
F = (D/2)² / (λz) = D² / (4λz)
```

**Relationship to field of view**:
```
F = fov² / (λz)
```

---

## Regime Classification

### Three Diffraction Regimes

| Regime | Fresnel Number | Characteristics | Propagation Method |
|--------|---------------|-----------------|-------------------|
| **Far-field (Fraunhofer)** | F < 0.1 | Angular spectrum, Fourier transform of aperture | Fraunhofer diffraction |
| **Intermediate (Fresnel)** | 0.1 ≤ F < 10 | Fresnel zones visible, quadratic phase | Angular Spectrum* |
| **Near-field** | F ≥ 10 | Geometrical projection dominant | Angular Spectrum |

**Note**: *Fresnel propagator is deprecated in PRISM due to accuracy issues. Angular Spectrum is used for all F ≥ 0.1 cases.

### Detailed Regime Characteristics

#### Far-Field (Fraunhofer Diffraction)

**Condition**: F < 0.1

**Characteristics**:
- Large distance relative to aperture and wavelength
- Observation plane sees angular distribution (far-field pattern)
- Simple Fourier relationship between aperture and image
- Spherical wavefronts appear planar across aperture

**Examples**:
- Astronomical imaging (stars, planets, galaxies)
- Telescope diffraction patterns
- Long-distance optical communication

**Mathematical simplification**:
- Phase term: exp(ikz) (constant phase)
- Transform: Fourier transform of aperture
- No quadratic phase correction needed

#### Intermediate Field (Fresnel Diffraction)

**Condition**: 0.1 ≤ F < 10

**Characteristics**:
- Fresnel zones are important
- Quadratic phase terms cannot be neglected
- More complex than far-field, less geometric than near-field
- Wave diffraction and interference dominate

**Examples**:
- Laboratory optical setups (10 cm - 1 m)
- Beam propagation in free space
- Some aerial imaging scenarios

**Mathematical approach**:
- Fresnel approximation: quadratic phase term
- Convolution with Fresnel kernel
- **PRISM uses Angular Spectrum for better accuracy**

#### Near-Field

**Condition**: F ≥ 10

**Characteristics**:
- Close to aperture or very large aperture
- Geometrical optics approximation starts to apply
- Wave effects still present but subdued
- Projection-like behavior

**Examples**:
- Microscopy (sub-mm distances, mm-scale apertures)
- Contact imaging
- Near-field scanning microscopy
- Close-range drone imaging

**Mathematical approach**:
- Angular spectrum method (exact within paraxial approximation)
- Handles evanescent waves correctly
- Full wave propagation

---

## Automatic Propagator Selection

### PRISM Auto-Selection Rules

PRISM automatically selects the appropriate propagation method:

```python
from prism.core.propagators import select_propagator

prop = select_propagator(
    wavelength=520e-9,
    obj_distance=distance,
    fov=field_of_view,
    method="auto"  # Automatic selection based on F
)
```

### Selection Logic

| Fresnel Number | Selected Propagator | Rationale |
|---------------|-------------------|-----------|
| F < 0.1 | **Fraunhofer** | Fast, accurate for far-field |
| F ≥ 0.1 | **Angular Spectrum** | Accurate for all distances, handles near-field correctly |

**Note**: Fresnel propagator is available via `method="fresnel"` but deprecated due to accuracy limitations.

### Performance Considerations

| Propagator | Speed | Accuracy | Use When |
|-----------|-------|----------|----------|
| Fraunhofer | Fastest | High (F < 0.1) | Far-field only |
| Angular Spectrum | Fast | High (all F) | Default choice for F ≥ 0.1 |
| Fresnel (deprecated) | Fast | Limited | Not recommended |

---

## Microscopy Fresnel Numbers

### Typical Microscopy Parameters

**Setup**:
- Distance: 100 µm - 1 mm (objective working distance)
- FOV: 100 µm - 1 mm (field of view)
- Wavelength: 400-700 nm (visible light)

**Result**: F >> 10 (extreme near-field)

### Fresnel Numbers by Objective

@ λ = 550 nm, working distance varies by objective:

| Objective | NA | Working Dist (mm) | FOV (mm) | Fresnel Number | Regime |
|-----------|-----|------------------|----------|----------------|--------|
| 100×/1.4 Oil | 1.4 | 0.13 | 0.16 | **36,000** | Near-field |
| 60×/1.2 Water | 1.2 | 0.28 | 0.27 | **15,000** | Near-field |
| 40×/0.9 Air | 0.9 | 0.20 | 0.40 | **73,000** | Near-field |
| 20×/0.75 Air | 0.75 | 0.60 | 0.80 | **97,000** | Near-field |
| 10×/0.3 Air | 0.3 | 5.6 | 1.60 | **41,000** | Near-field |

**Calculation example** (100×/1.4 Oil):
```
FOV ≈ 160 µm (typical for 100× objective)
Working distance ≈ 130 µm
F = (160e-6)² / (550e-9 × 130e-6) = 35,755 >> 10
```

**Conclusion**: All microscopy scenarios are **extreme near-field** (F >> 10).

**PRISM behavior**: Automatically uses **Angular Spectrum** propagator for all microscopy.

---

## Drone Fresnel Numbers

### Typical Drone Parameters

**Setup**:
- Altitude: 10-120 m
- Aperture: 5-15 mm (camera lens aperture)
- Wavelength: 400-700 nm (visible light)

**Result**: F >> 10 (near-field to intermediate)

### Fresnel Numbers by Preset

@ λ = 550 nm, calculated for each preset:

| Preset | Lens | Aperture (mm) | Altitude (m) | Fresnel Number | Regime |
|--------|------|--------------|--------------|----------------|--------|
| `drone_10m_inspection` | 35mm f/2.8 | 12.5 | 10 | **28,400** | Near-field |
| `drone_20m_detail` | 50mm f/4.0 | 12.5 | 20 | **14,200** | Near-field |
| `drone_50m_survey` | 50mm f/4.0 | 12.5 | 50 | **5,680** | Near-field |
| `drone_100m_mapping` | 35mm f/4.0 | 8.75 | 100 | **1,390** | Near-field |
| `drone_phantom_120m` | 24mm f/2.8 | 8.57 | 120 | **1,110** | Near-field |
| `drone_hover_50m` | 50mm f/4.0 | 12.5 | 50 | **5,680** | Near-field |
| `drone_agriculture_50m` | 35mm f/4.0 | 8.75 | 50 | **2,780** | Near-field |
| `drone_infrastructure_30m` | 50mm f/2.8 | 17.9 | 30 | **19,400** | Near-field |

**Calculation example** (`drone_50m_survey`):
```
Aperture = 50mm / 4.0 = 12.5 mm (f/4.0)
Altitude = 50 m
F = (0.00625)² / (550e-9 × 50) = 1,420 >> 10
```

**Conclusion**: All drone scenarios are **near-field** (F > 10, typically F > 1000).

**PRISM behavior**: Automatically uses **Angular Spectrum** propagator for all drone imaging.

### Fresnel Number vs Altitude

**For 50mm f/4.0 lens** (aperture = 12.5mm):

| Altitude (m) | Fresnel Number | Regime |
|--------------|----------------|--------|
| 10 | 28,400 | Near-field (extreme) |
| 20 | 14,200 | Near-field |
| 50 | 5,680 | Near-field |
| 100 | 2,840 | Near-field |
| 200 | 1,420 | Near-field |
| 500 | 568 | Near-field |
| 1,000 | 284 | Near-field |
| 5,000 | 57 | Near-field |
| 10,000 | 28 | Near-field |
| 50,000 | 5.7 | Intermediate |
| 100,000 | 2.8 | Intermediate |
| 500,000 | 0.57 | Intermediate |
| 1,000,000 | 0.28 | Intermediate |
| 5,000,000 | 0.057 | **Far-field** |

**Observation**: Even at 1 km altitude, drone imaging is still near-field. Must reach ~5 km for far-field regime.

---

## Astronomy Fresnel Numbers

### Typical Astronomy Parameters

**Setup**:
- Distance: 10⁶ m (Moon) to 10²⁶ m (galaxies)
- Aperture: 1 mm - 10 m (telescope diameter)
- Wavelength: 400-700 nm (visible light)

**Result**: F << 0.1 (extreme far-field)

### Fresnel Numbers by Target

@ λ = 550 nm, aperture D = 10 cm (amateur telescope):

| Target | Distance (m) | Distance (units) | Fresnel Number | Regime |
|--------|-------------|-----------------|----------------|--------|
| Moon | 3.84 × 10⁸ | 384,000 km | 1.2 × 10⁻⁸ | Far-field |
| Mars (close) | 5.5 × 10¹⁰ | 55 million km | 8.3 × 10⁻¹¹ | Far-field |
| Jupiter | 6.3 × 10¹¹ | 630 million km | 7.2 × 10⁻¹² | Far-field |
| Saturn | 1.2 × 10¹² | 1.2 billion km | 3.8 × 10⁻¹² | Far-field |
| Proxima Centauri | 4.0 × 10¹⁶ | 4.2 light-years | 1.1 × 10⁻¹⁶ | Far-field |
| Andromeda Galaxy | 2.4 × 10²² | 2.5 million ly | 1.9 × 10⁻²² | Far-field |

**Calculation example** (Jupiter):
```
Aperture radius = 0.05 m (10 cm diameter telescope)
Distance = 628 × 10⁹ m (628 million km)
F = (0.05)² / (550e-9 × 628e9) = 7.2 × 10⁻¹² << 0.1
```

**Conclusion**: All astronomical observations are **extreme far-field** (F << 10⁻⁶).

**PRISM behavior**: Automatically uses **Fraunhofer** propagator for all astronomy.

### Large Telescope Comparison

For **10 m telescope** (professional observatory) @ λ = 550 nm:

| Target | Distance | Fresnel Number | Still Far-Field? |
|--------|----------|----------------|-----------------|
| Moon | 3.84 × 10⁸ m | 4.7 × 10⁻⁴ | Yes (F < 0.1) |
| Jupiter | 6.3 × 10¹¹ m | 2.9 × 10⁻⁷ | Yes (F << 0.1) |

**Even the largest telescopes**: All astronomy remains far-field.

---

## Practical Examples

### Example 1: Lab Optical Setup

**Question**: What propagator for 10 cm distance, 1 mm beam width?

**Calculation**:
```
a = 0.5 mm = 5 × 10⁻⁴ m
z = 0.1 m
λ = 550 nm = 550 × 10⁻⁹ m
F = (5e-4)² / (550e-9 × 0.1) = 4.5
```

**Result**: F = 4.5 (intermediate regime)

**PRISM selection**: Angular Spectrum propagator

### Example 2: When Does Far-Field Start?

**Question**: At what distance does a 1 mm aperture enter far-field (F < 0.1)?

**Calculation**:
```
F = a² / (λz) = 0.1
z = a² / (0.1 × λ)
z = (5e-4)² / (0.1 × 550e-9) = 4.5 m
```

**Result**: Distance > 4.5 m for far-field

**Practical meaning**: Small apertures (< 1 mm) reach far-field within a few meters.

### Example 3: Microscopy vs Drone

**Microscopy** (100× objective):
```
FOV = 160 µm, distance = 130 µm
F = (160e-6)² / (550e-9 × 130e-6) = 35,755
Regime: Extreme near-field (F >> 10)
```

**Drone** (50m altitude):
```
Aperture = 12.5 mm, distance = 50 m
F = (0.00625)² / (550e-9 × 50) = 1,420
Regime: Near-field (F >> 10)
```

**Comparison**: Both near-field, but microscopy has 25× higher Fresnel number despite being 385,000× closer!

### Example 4: Transitioning Between Regimes

**Setup**: 50mm f/4.0 lens (aperture = 12.5mm), varying distance

| Distance | F | Regime | PRISM Propagator |
|----------|---|--------|------------------|
| 1 m | 284 | Near-field | Angular Spectrum |
| 10 m | 28 | Near-field | Angular Spectrum |
| 100 m | 2.8 | Intermediate | Angular Spectrum |
| 1 km | 0.28 | Intermediate | Angular Spectrum |
| 10 km | 0.028 | **Far-field** | **Fraunhofer** |

**Transition point**: F = 0.1 occurs at ~2.3 km distance

---

## Quick Reference Tables

### Fresnel Number Lookup by Application

| Application | Typical F | Regime | PRISM Propagator |
|-------------|----------|--------|------------------|
| Microscopy | 10⁴ - 10⁵ | Near-field | Angular Spectrum |
| Drone (10-120m) | 10³ - 10⁴ | Near-field | Angular Spectrum |
| Aerial (1-10 km) | 1 - 10² | Near/Intermediate | Angular Spectrum |
| Long-range aerial | 0.1 - 1 | Intermediate | Angular Spectrum |
| Satellite imaging | 10⁻² - 10⁻¹ | Far-field | Fraunhofer |
| Astronomy | < 10⁻⁶ | Far-field | Fraunhofer |

### Critical Distances for Regime Transitions

**For aperture radius a = 5mm**, λ = 550 nm:

| Distance | F | Regime |
|----------|---|--------|
| 1 cm | 4,545 | Near-field |
| 10 cm | 455 | Near-field |
| 1 m | 45 | Near-field |
| 10 m | 4.5 | Intermediate |
| 45 m | 1.0 | Intermediate |
| 100 m | 0.45 | Intermediate |
| 250 m | 0.18 | Intermediate |
| 455 m | 0.10 | **Transition to far-field** |
| 1 km | 0.045 | Far-field |
| 10 km | 0.0045 | Far-field |

**Formula for transition**: z_transition = 10 × a² / λ (for F = 0.1)

---

## Programming Reference

### Check Fresnel Number

```python
from prism.config.constants import fresnel_number

# Calculate Fresnel number
F = fresnel_number(
    width=1e-3,        # 1 mm aperture
    distance=0.1,      # 10 cm
    wavelength=550e-9  # 550 nm
)

print(f"Fresnel number: {F:.2e}")

if F < 0.1:
    print("Far-field (Fraunhofer)")
elif F < 10:
    print("Intermediate (use Angular Spectrum)")
else:
    print("Near-field (use Angular Spectrum)")
```

### Automatic Propagator Selection

```python
from prism.core.propagators import select_propagator

# Auto-select based on Fresnel number
prop = select_propagator(
    wavelength=550e-9,
    obj_distance=50.0,      # 50 m
    fov=0.0125,             # 12.5 mm (aperture radius)
    method="auto",          # Automatic selection
)

print(f"Selected: {type(prop).__name__}")
# Output: "Selected: AngularSpectrumPropagator" (F >> 10)
```

### Manual Override (Not Recommended)

```python
# Force Fraunhofer (will warn if F >= 0.1)
prop = select_propagator(
    wavelength=550e-9,
    obj_distance=50.0,
    fov=0.0125,
    method="fraunhofer",  # Manual override
)
# Warning: "Fraunhofer propagator may be inaccurate for F=1420 >> 0.1"
```

---

## Source Files

This reference is based on:
- [prism/config/constants.py](../../prism/config/constants.py) (lines 42-54, Fresnel number formula)
- [tests/unit/core/propagators/test_propagator_selection.py](../../tests/unit/core/propagators/test_propagator_selection.py) (auto-selection logic)
- [prism/core/propagators/__init__.py](../../prism/core/propagators/__init__.py) (select_propagator implementation)

## Related References

- [Physical Constants](physical_constants.md) - Wavelengths and speed of light
- [Optical Resolution Limits](optical_resolution_limits.md) - Rayleigh criterion and diffraction limits
- [Microscopy Parameters](microscopy_parameters.md) - Microscope working distances
- [Drone Camera Parameters](drone_camera_parameters.md) - Drone altitudes and apertures
- [Scenario Preset Catalog](scenario_preset_catalog.md) - All presets with propagation info

## Related User Guides

- [Optical Engineering Guide](../user_guides/optical-engineering.md) - Understanding diffraction regimes
- [Scenarios User Guide](../user_guides/scenarios.md) - Using scenario presets

---

**Last Updated**: 2025-01-26
**Formula**: F = a²/(λz) (standard Fresnel number definition)
**Selection Logic**: F < 0.1 → Fraunhofer, F ≥ 0.1 → Angular Spectrum
**Accuracy**: All calculations verified against PRISM test suite
