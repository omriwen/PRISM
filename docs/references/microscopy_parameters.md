# Microscopy Parameters Reference

**Complete reference for microscope objective specifications, resolution limits, and all 9 PRISM microscope presets.**

## Quick Navigation

- [Objective Specifications](#objective-specifications)
- [Resolution Formulas](#resolution-formulas)
- [Immersion Media](#immersion-media)
- [Illumination Modes](#illumination-modes)
- [Complete Preset Table](#complete-preset-table)
- [Usage Examples](#usage-examples)

---

## Objective Specifications

### Oil Immersion Objectives (n = 1.515)

Highest resolution, requires immersion oil between objective and coverslip.

| Magnification | NA | Working Distance | Resolution @ 550nm | Use Case |
|---------------|-----|------------------|-------------------|----------|
| 100× | 1.4 | ~0.13 mm | ~240 nm | Cellular detail, high-resolution imaging |
| 60× | 1.42 | ~0.15 mm | ~236 nm | High-resolution with wider FOV |

**Typical specs**: Plan Apochromat, Oil immersion (n=1.515), coverslip required (0.17mm #1.5)

### Water Immersion Objectives (n = 1.33)

For live tissue imaging, less aberration with aqueous samples.

| Magnification | NA | Working Distance | Resolution @ 550nm | Use Case |
|---------------|-----|------------------|-------------------|----------|
| 60× | 1.2 | ~0.28 mm | ~280 nm | Live tissue, aqueous samples |
| 40× | 1.15 | ~0.60 mm | ~290 nm | Thick specimens, longer working distance |

**Typical specs**: Plan Apochromat, Water immersion (n=1.33), correction collar for coverslip thickness

### Air Objectives (n = 1.0)

Convenient, no immersion medium required.

| Magnification | NA | Working Distance | Resolution @ 550nm | Use Case |
|---------------|-----|------------------|-------------------|----------|
| 40× | 0.95 | ~0.18 mm | ~353 nm | High-NA dry objective |
| 40× | 0.9 | ~0.20 mm | ~370 nm | Standard high-magnification |
| 20× | 0.75 | ~0.60 mm | ~450 nm | Tissue sections, medium detail |
| 10× | 0.45 | ~4.0 mm | ~750 nm | Sample navigation |
| 10× | 0.3 | ~5.6 mm | ~1117 nm | Low-mag overview |

**Typical specs**: Plan Achromat or Plan Fluorite, no coverslip correction needed for some models

---

## Resolution Formulas

### Lateral Resolution (Abbe Limit)

The minimum resolvable distance between two point sources:

$$\Delta x = \frac{0.61 \lambda}{\text{NA}}$$

Where:
- **Δx** = lateral resolution (m)
- **λ** = wavelength (m)
- **NA** = numerical aperture (dimensionless)

**Physical meaning**: Two points closer than Δx will appear as a single blur spot.

### Axial Resolution (Depth of Field)

The minimum resolvable distance along the optical axis:

$$\Delta z = \frac{2 n \lambda}{\text{NA}^2}$$

Where:
- **Δz** = axial resolution (m)
- **n** = refractive index of immersion medium
- **λ** = wavelength (m)
- **NA** = numerical aperture (dimensionless)

**Physical meaning**: Features separated by less than Δz in depth cannot be distinguished.

### Numerical Aperture (NA)

$$\text{NA} = n \sin(\alpha)$$

Where:
- **n** = refractive index of immersion medium (1.0 air, 1.33 water, 1.515 oil)
- **α** = half-angle of the light cone collected by the objective

**Maximum NA values**:
- Air: NA ≤ 1.0 (physically limited by n=1.0)
- Water: NA ≤ 1.33 (theoretically limited by n=1.33, practical max ~1.2)
- Oil: NA ≤ 1.515 (theoretically limited by n=1.515, practical max ~1.45)

### Resolution vs NA Lookup Table

At λ = 550 nm (green light):

| NA | Medium | Lateral Res (nm) | Axial Res (µm) | Example Objective |
|-----|--------|-----------------|----------------|-------------------|
| 1.4 | Oil | 240 | 1.2 | 100×/1.4 Oil |
| 1.2 | Water | 280 | 1.5 | 60×/1.2 Water |
| 0.95 | Air | 353 | 2.4 | 40×/0.95 Air |
| 0.9 | Air | 371 | 2.7 | 40×/0.9 Air |
| 0.75 | Air | 445 | 3.9 | 20×/0.75 Air |
| 0.45 | Air | 742 | 10.8 | 10×/0.45 Air |
| 0.3 | Air | 1113 | 24.4 | 10×/0.3 Air |

### Resolution vs Wavelength

For 100× 1.4NA oil objective:

| Wavelength | Color | Lateral Res (nm) | Application |
|------------|-------|------------------|-------------|
| 405 nm | Violet | 177 | Confocal, DAPI fluorescence |
| 488 nm | Blue-green | 213 | GFP fluorescence |
| 550 nm | Green | 240 | Standard brightfield |
| 633 nm | Red | 276 | Cy5 fluorescence |

---

## Immersion Media

### Refractive Index Table

| Medium | Refractive Index (n) | Max Theoretical NA | Max Practical NA | Notes |
|--------|---------------------|-------------------|------------------|-------|
| Air | 1.000 | 1.0 | ~0.95 | Convenient, no coupling required |
| Water | 1.333 | 1.33 | ~1.2 | Live cells, aqueous samples |
| Glycerol | 1.47 | 1.47 | ~1.35 | Deep tissue imaging |
| Oil (Type A) | 1.515 | 1.515 | ~1.45 | Highest resolution, coverslip |
| Oil (Type F) | 1.518 | 1.518 | ~1.45 | Fluorescence applications |

**Key points**:
- Higher n allows higher NA → better resolution
- Oil immersion eliminates air gap between objective and coverslip
- Water immersion reduces spherical aberration in aqueous samples
- Immersion medium must match objective design

---

## Illumination Modes

PRISM currently supports specification of illumination modes (implementation varies):

| Mode | Description | Contrast Mechanism | Best For |
|------|-------------|-------------------|----------|
| **Brightfield** | Direct transmission illumination | Absorption | Stained samples, high contrast |
| **Darkfield** | Oblique illumination, scatter detected | Scattering | Unstained cells, particles |
| **Phase contrast** | Converts phase shifts to intensity | Phase interference | Live cells, transparent samples |
| **DIC** | Differential interference contrast | Phase gradient | 3D-like relief, unstained samples |

**Note**: PRISM currently treats all modes as coherent illumination with varying pupil functions. Full incoherent illumination and advanced contrast modes are planned enhancements.

### Köhler Illumination

PRISM microscope models use Köhler illumination by default:
- Illumination NA = 0.8 × objective NA (standard setting)
- Provides even illumination across field of view
- Allows adjustment of resolution vs contrast tradeoff

---

## Complete Preset Table

All 9 microscope presets with computed parameters @ λ = 550 nm:

| Preset Name | Mag | NA | Medium | λ (nm) | Lateral Res (nm) | Axial Res (µm) | Mode | Use Case |
|-------------|-----|----|---------|----|-----------------|----------------|------|----------|
| `microscope_100x_oil` | 100× | 1.4 | Oil | 550 | 240 | 1.2 | Brightfield | High-resolution cellular detail |
| `microscope_100x_oil_phase` | 100× | 1.4 | Oil | 550 | 240 | 1.2 | Phase | Oil immersion phase contrast for unstained live cells |
| `microscope_60x_water` | 60× | 1.2 | Water | 550 | 280 | 1.5 | Brightfield | Water immersion for live tissue imaging |
| `microscope_40x_air` | 40× | 0.9 | Air | 550 | 371 | 2.7 | Brightfield | High-NA air objective for detailed observation |
| `microscope_40x_phase` | 40× | 0.9 | Air | 550 | 371 | 2.7 | Phase | Phase contrast for unstained samples |
| `microscope_40x_dic` | 40× | 0.9 | Air | 550 | 371 | 2.7 | DIC | Differential interference contrast for 3D-like relief |
| `microscope_20x_air` | 20× | 0.75 | Air | 550 | 445 | 3.9 | Brightfield | Medium magnification for tissue sections |
| `microscope_10x_air` | 10× | 0.3 | Air | 550 | 1113 | 24.4 | Brightfield | Low magnification for sample overview |
| `microscope_fluorescence_100x` | 100× | 1.4 | Oil | 488 | 213 | 1.0 | Brightfield* | Fluorescence microscopy with GFP filter (488nm) |

**Note**: `microscope_fluorescence_100x` uses brightfield mode as PRISM doesn't have dedicated fluorescence mode yet. The 488nm wavelength corresponds to GFP excitation.

### Preset Selection Guide

**High Resolution (< 300 nm)**:
- `microscope_100x_oil` - Maximum resolution, brightfield
- `microscope_100x_oil_phase` - Maximum resolution, phase contrast
- `microscope_fluorescence_100x` - Fluorescence at 488nm
- `microscope_60x_water` - Live tissue with high resolution

**Medium Resolution (300-500 nm)**:
- `microscope_40x_air` - Versatile, convenient (no immersion)
- `microscope_40x_phase` - Phase contrast at 40×
- `microscope_40x_dic` - DIC at 40×
- `microscope_20x_air` - Larger FOV, tissue sections

**Overview/Navigation (> 1 µm)**:
- `microscope_10x_air` - Sample overview and navigation

---

## Nyquist Sampling

For proper digital sampling, the pixel size in object space must satisfy:

$$p_{\text{obj}} < \frac{\lambda}{4 \cdot \text{NA}}$$

Where:
- **p_obj** = pixel size in object space = sensor pixel size / magnification
- **λ** = wavelength
- **NA** = numerical aperture

### Recommended Pixel Sizes

For proper Nyquist sampling at 550 nm:

| Objective | NA | Min Sampling (nm) | Recommended (nm) | Sensor Pixel @ Mag | Example Sensor |
|-----------|-----|-------------------|------------------|-------------------|----------------|
| 100×/1.4 Oil | 1.4 | 98 | 65 | 6.5 µm @ 100× | Scientific CMOS |
| 60×/1.2 Water | 1.2 | 114 | 76 | 4.6 µm @ 60× | Scientific CMOS |
| 40×/0.9 Air | 0.9 | 153 | 102 | 4.1 µm @ 40× | Standard CMOS |
| 20×/0.75 Air | 0.75 | 183 | 122 | 2.4 µm @ 20× | High-res CMOS |
| 10×/0.3 Air | 0.3 | 458 | 306 | 3.1 µm @ 10× | Standard CMOS |

**Rule of thumb**: Use sensor pixels that give 2-3× oversampling (pixel size = min_sampling / 2.5)

---

## Usage Examples

### Load Preset

```python
from prism.scenarios import get_scenario_preset

# Load preset
scenario = get_scenario_preset("microscope_100x_oil")

# Check computed parameters
print(f"Lateral resolution: {scenario.lateral_resolution_nm:.1f} nm")
print(f"Axial resolution: {scenario.axial_resolution_um:.2f} µm")
print(f"NA: {scenario.objective_spec.numerical_aperture}")
print(f"Magnification: {scenario.objective_spec.magnification}×")
print(f"Immersion: {scenario.objective_spec.immersion_medium}")
```

### Create Custom Microscope

```python
from prism.scenarios.microscopy import MicroscopeScenarioConfig

# Custom 63× oil objective
scenario = MicroscopeScenarioConfig(
    objective_spec="63x_1.4NA_oil",
    illumination_mode="brightfield",
    wavelength=550e-9,
    n_pixels=2048
)

print(f"Resolution: {scenario.lateral_resolution_nm:.1f} nm")
```

### Convert to Instrument

```python
from prism.scenarios import get_scenario_preset

# Load scenario
scenario = get_scenario_preset("microscope_40x_air")

# Convert to instrument configuration
instrument_config = scenario.to_instrument_config()

# Use with PRISM
from prism.core import Microscope
microscope = Microscope(instrument_config)
```

### CLI Usage

```bash
# List microscope presets
prism-scenario list --category microscope

# Get detailed info
prism-scenario info microscope_100x_oil

# Create instrument from preset
prism-scenario instrument microscope_40x_phase --output config.yaml
```

---

## Source Files

This reference is based on:
- [prism/scenarios/presets.py](../../prism/scenarios/presets.py) (lines 14-70)
- [prism/scenarios/microscopy.py](../../prism/scenarios/microscopy.py)

## Related References

- [Physical Constants](physical_constants.md) - Wavelengths and units
- [Optical Resolution Limits](optical_resolution_limits.md) - Detailed resolution formulas
- [Scenario Preset Catalog](scenario_preset_catalog.md) - All presets in one table

## Related User Guides

- [Optical Engineering Guide](../user_guides/optical-engineering.md) - Understanding optical systems
- [Scenarios User Guide](../user_guides/scenarios.md) - How to use scenario system

---

**Last Updated**: 2025-01-26
**Accuracy**: All values verified against `prism/scenarios/presets.py` and `microscopy.py`
**Formulas**: Standard microscopy physics (Abbe, Rayleigh)
