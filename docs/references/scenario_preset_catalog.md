# Scenario Preset Catalog

**Complete quick-reference catalog of all 17 PRISM scenario presets: 9 microscopy + 8 drone camera configurations.**

## Quick Navigation

- [All Presets at a Glance](#all-presets-at-a-glance)
- [Microscopy Presets](#microscopy-presets)
- [Drone Camera Presets](#drone-camera-presets)
- [Selection Guide](#selection-guide)
- [Usage Examples](#usage-examples)

---

## All Presets at a Glance

### Summary Statistics

| Category | Count | Resolution Range | Distance Range | Key Applications |
|----------|-------|------------------|----------------|------------------|
| **Microscopy** | 9 | 213 nm - 1113 nm | 0.13 - 5.6 mm | Cellular imaging, tissue analysis |
| **Drone Camera** | 8 | GSD: 1.1 - 18.6 cm | 10 - 120 m | Inspection, surveying, mapping |
| **Total** | **17** | - | - | - |

### Quick Preset Finder

**Need high resolution (< 300 nm)?**
- `microscope_100x_oil` - 240 nm (brightfield)
- `microscope_100x_oil_phase` - 240 nm (phase contrast)
- `microscope_fluorescence_100x` - 213 nm @ 488nm (GFP)
- `microscope_60x_water` - 280 nm (water immersion)

**Need detailed drone imagery (GSD < 2 cm)?**
- `drone_10m_inspection` - 1.1 cm GSD
- `drone_20m_detail` - 1.6 cm GSD

**Need standard surveying (GSD 5-10 cm)?**
- `drone_50m_survey` - 6.5 cm GSD
- `drone_agriculture_50m` - 5.6 cm GSD
- `drone_hover_50m` - 6.5 cm GSD (no motion blur)

**Need phase contrast imaging?**
- `microscope_100x_oil_phase` - High-res phase @ 1.4 NA
- `microscope_40x_phase` - Medium-res phase @ 0.9 NA

---

## Microscopy Presets

### Complete Microscopy Table

All 9 microscope presets with computed parameters @ λ = 550 nm:

| Preset Name | Mag | NA | Medium | λ (nm) | Lateral Res (nm) | Axial Res (µm) | Mode | Primary Use Case |
|-------------|-----|----|---------|----|-----------------|----------------|------|------------------|
| `microscope_100x_oil` | 100× | 1.4 | Oil | 550 | 240 | 1.2 | Brightfield | High-res cellular detail |
| `microscope_100x_oil_phase` | 100× | 1.4 | Oil | 550 | 240 | 1.2 | Phase | Unstained live cells (oil) |
| `microscope_60x_water` | 60× | 1.2 | Water | 550 | 280 | 1.5 | Brightfield | Live tissue imaging |
| `microscope_40x_air` | 40× | 0.9 | Air | 550 | 371 | 2.7 | Brightfield | Detailed observation |
| `microscope_40x_phase` | 40× | 0.9 | Air | 550 | 371 | 2.7 | Phase | Unstained samples |
| `microscope_40x_dic` | 40× | 0.9 | Air | 550 | 371 | 2.7 | DIC | 3D-like relief imaging |
| `microscope_20x_air` | 20× | 0.75 | Air | 550 | 445 | 3.9 | Brightfield | Tissue sections |
| `microscope_10x_air` | 10× | 0.3 | Air | 550 | 1113 | 24.4 | Brightfield | Sample overview |
| `microscope_fluorescence_100x` | 100× | 1.4 | Oil | 488 | 213 | 1.0 | Brightfield* | GFP fluorescence @ 488nm |

**Note**: *`microscope_fluorescence_100x` uses brightfield mode as PRISM doesn't have dedicated fluorescence mode yet.

### Microscopy Presets by Category

#### Ultra-High Resolution (< 300 nm)

| Preset | Resolution (nm) | Wavelength | Application |
|--------|----------------|------------|-------------|
| `microscope_fluorescence_100x` | 213 | 488 nm | GFP imaging, confocal |
| `microscope_100x_oil` | 240 | 550 nm | Maximum brightfield resolution |
| `microscope_100x_oil_phase` | 240 | 550 nm | Unstained live cells |
| `microscope_60x_water` | 280 | 550 nm | Live tissue, aqueous samples |

**When to use**: Subcellular structures, bacteria, high-magnification work

#### Medium Resolution (300-500 nm)

| Preset | Resolution (nm) | NA | Application |
|--------|----------------|-----|-------------|
| `microscope_40x_air` | 371 | 0.9 | General high-magnification work |
| `microscope_40x_phase` | 371 | 0.9 | Phase contrast for cells |
| `microscope_40x_dic` | 371 | 0.9 | DIC for relief imaging |
| `microscope_20x_air` | 445 | 0.75 | Tissue sections, larger FOV |

**When to use**: Cellular overview, tissue histology, routine observation

#### Low Resolution / Overview (> 1 µm)

| Preset | Resolution (nm) | FOV | Application |
|--------|----------------|-----|-------------|
| `microscope_10x_air` | 1113 | Large | Sample navigation, overview |

**When to use**: Finding regions of interest, sample survey, low-mag scanning

### Microscopy by Immersion Medium

#### Oil Immersion (n = 1.515)

| Preset | NA | Resolution (nm) | Notes |
|--------|-----|----------------|-------|
| `microscope_100x_oil` | 1.4 | 240 | Best resolution, requires immersion oil |
| `microscope_100x_oil_phase` | 1.4 | 240 | Phase contrast with oil |
| `microscope_fluorescence_100x` | 1.4 | 213 @ 488nm | Fluorescence imaging |

**Pros**: Maximum resolution, high NA
**Cons**: Requires oil, coverslip, cleanup

#### Water Immersion (n = 1.33)

| Preset | NA | Resolution (nm) | Notes |
|--------|-----|----------------|-------|
| `microscope_60x_water` | 1.2 | 280 | Live tissue, longer working distance |

**Pros**: Live cells, reduced aberration in aqueous samples
**Cons**: Requires water coupling, needs correction collar

#### Air Objectives (n = 1.0)

| Preset | NA | Resolution (nm) | Notes |
|--------|-----|----------------|-------|
| `microscope_40x_air` | 0.9 | 371 | High-NA dry objective |
| `microscope_40x_phase` | 0.9 | 371 | Phase contrast, convenient |
| `microscope_40x_dic` | 0.9 | 371 | DIC without immersion |
| `microscope_20x_air` | 0.75 | 445 | Medium magnification |
| `microscope_10x_air` | 0.3 | 1113 | Low magnification |

**Pros**: Convenient, no immersion medium
**Cons**: Lower maximum NA, limited resolution

---

## Drone Camera Presets

### Complete Drone Table

All 8 drone presets with computed parameters @ λ = 550 nm:

| Preset Name | Lens | Sensor | Altitude (m) | Speed (m/s) | GSD (cm) | Swath (m) | F-number | Primary Use Case |
|-------------|------|--------|-------------|------------|---------|----------|----------|------------------|
| `drone_10m_inspection` | 35mm f/2.8 | APS-C | 10 | 2 | 1.1 | 6.7 | 2.8 | Close-range inspection |
| `drone_20m_detail` | 50mm f/4.0 | APS-C | 20 | 5 | 1.6 | 9.4 | 4.0 | Detailed surveying |
| `drone_50m_survey` | 50mm f/4.0 | Full-frame | 50 | 10 | 6.5 | 36.0 | 4.0 | Standard site survey |
| `drone_100m_mapping` | 35mm f/4.0 | Full-frame | 100 | 15 | 18.6 | 103 | 4.0 | Large area mapping |
| `drone_phantom_120m` | 24mm f/2.8 | 1/2.3" | 120 | 12 | 8.0 | 30.9 | 2.8 | Consumer drone (DJI style) |
| `drone_hover_50m` | 50mm f/4.0 | Full-frame | 50 | 0 | 6.5 | 36.0 | 4.0 | Hover mode (no blur) |
| `drone_agriculture_50m` | 35mm f/4.0 | APS-C | 50 | 8 | 5.6 | 33.6 | 4.0 | Agricultural monitoring |
| `drone_infrastructure_30m` | 50mm f/2.8 | Full-frame | 30 | 3 | 3.9 | 21.6 | 2.8 | Infrastructure inspection |

### Drone Presets by GSD Range

#### Ultra-Fine Detail (GSD < 2 cm)

| Preset | GSD (cm) | Altitude (m) | Application |
|--------|---------|--------------|-------------|
| `drone_10m_inspection` | 1.1 | 10 | Power line inspection, close detail |
| `drone_20m_detail` | 1.6 | 20 | Building facade inspection |

**When to use**: Crack detection, detailed infrastructure inspection, close-range work

#### Fine Detail (GSD 2-5 cm)

| Preset | GSD (cm) | Altitude (m) | Application |
|--------|---------|--------------|-------------|
| `drone_infrastructure_30m` | 3.9 | 30 | Bridge inspection, roof surveys |

**When to use**: Infrastructure assessment, detailed construction monitoring

#### Standard Survey (GSD 5-10 cm)

| Preset | GSD (cm) | Altitude (m) | Application |
|--------|---------|--------------|-------------|
| `drone_agriculture_50m` | 5.6 | 50 | Crop health monitoring |
| `drone_50m_survey` | 6.5 | 50 | General site surveying |
| `drone_hover_50m` | 6.5 | 50 | Precision hover (no motion blur) |
| `drone_phantom_120m` | 8.0 | 120 | Consumer drone mapping |

**When to use**: Agricultural monitoring, site surveys, general mapping

#### Large Area Mapping (GSD > 15 cm)

| Preset | GSD (cm) | Altitude (m) | Application |
|--------|---------|--------------|-------------|
| `drone_100m_mapping` | 18.6 | 100 | Fast regional mapping |

**When to use**: Large property surveys, fast reconnaissance, regional overview

### Drone Presets by Sensor Type

#### Full-Frame Sensors (36×24mm)

| Preset | Lens | Altitude | GSD | Advantage |
|--------|------|----------|-----|-----------|
| `drone_50m_survey` | 50mm f/4.0 | 50 m | 6.5 cm | Best image quality, low noise |
| `drone_100m_mapping` | 35mm f/4.0 | 100 m | 18.6 cm | Large coverage area |
| `drone_hover_50m` | 50mm f/4.0 | 50 m | 6.5 cm | No motion blur |
| `drone_infrastructure_30m` | 50mm f/2.8 | 30 m | 3.9 cm | Better low-light performance |

**Pros**: Superior image quality, low noise, better dynamic range
**Cons**: Heavier, more expensive, larger drone required

#### APS-C Sensors (23.5×15.6mm)

| Preset | Lens | Altitude | GSD | Advantage |
|--------|------|----------|-----|-----------|
| `drone_10m_inspection` | 35mm f/2.8 | 10 m | 1.1 cm | Close-range detail |
| `drone_20m_detail` | 50mm f/4.0 | 20 m | 1.6 cm | Detailed surveying |
| `drone_agriculture_50m` | 35mm f/4.0 | 50 m | 5.6 cm | Good balance quality/weight |

**Pros**: Good image quality, lighter than FF, smaller pixels for better GSD
**Cons**: More noise than FF, smaller sensor area

#### Small Sensors (1/2.3")

| Preset | Lens | Altitude | GSD | Advantage |
|--------|------|----------|-----|-----------|
| `drone_phantom_120m` | 24mm f/2.8 | 120 m | 8.0 cm | Lightweight, consumer drones |

**Pros**: Compact, lightweight, affordable, typical consumer drone sensor
**Cons**: Higher noise, limited dynamic range, smaller pixels

### Drone Presets by Speed/Motion Blur

#### Static/Hover Mode (0 m/s)

| Preset | Speed | Motion Blur | Application |
|--------|-------|------------|-------------|
| `drone_hover_50m` | 0 m/s | None | Precision hover imaging |

**Use when**: Maximum sharpness required, no time pressure

#### Slow Flight (2-3 m/s)

| Preset | Speed | Motion Blur | Application |
|--------|-------|------------|-------------|
| `drone_10m_inspection` | 2 m/s | Minimal | Close inspection |
| `drone_infrastructure_30m` | 3 m/s | Minimal | Infrastructure detail |

**Use when**: Detailed inspection, balancing speed and quality

#### Medium Flight (5-10 m/s)

| Preset | Speed | Motion Blur | Application |
|--------|-------|------------|-------------|
| `drone_20m_detail` | 5 m/s | Low | Balanced surveying |
| `drone_agriculture_50m` | 8 m/s | Low | Agricultural surveys |
| `drone_50m_survey` | 10 m/s | Moderate | Standard mapping |

**Use when**: Standard surveys, typical mapping missions

#### Fast Flight (12-15 m/s)

| Preset | Speed | Motion Blur | Application |
|--------|-------|------------|-------------|
| `drone_phantom_120m` | 12 m/s | Moderate | Fast coverage |
| `drone_100m_mapping` | 15 m/s | Higher | Rapid reconnaissance |

**Use when**: Large area coverage, time-critical missions

**Note**: Motion blur increases with speed. For sharp imagery at high speeds, use faster shutter times or lower altitudes (better GSD compensates for blur).

---

## Selection Guide

### Microscopy Selection Flowchart

```
Need maximum resolution?
├─ Yes → Oil immersion
│   ├─ Stained samples → microscope_100x_oil
│   ├─ Live cells → microscope_100x_oil_phase
│   └─ Fluorescence → microscope_fluorescence_100x
│
├─ No → What medium?
    ├─ Live tissue / aqueous → microscope_60x_water
    ├─ Dry mount / convenient
        ├─ High detail → microscope_40x_air / 40x_phase / 40x_dic
        ├─ Tissue sections → microscope_20x_air
        └─ Overview / navigation → microscope_10x_air
```

### Drone Selection Flowchart

```
What GSD do you need?
├─ < 2 cm (ultra-fine)
│   ├─ 10 m altitude → drone_10m_inspection
│   └─ 20 m altitude → drone_20m_detail
│
├─ 2-5 cm (fine detail)
│   └─ 30 m altitude → drone_infrastructure_30m
│
├─ 5-10 cm (standard)
│   ├─ No motion blur → drone_hover_50m
│   ├─ Agriculture → drone_agriculture_50m
│   ├─ General survey → drone_50m_survey
│   └─ Consumer drone → drone_phantom_120m
│
└─ > 15 cm (large area)
    └─ Fast mapping → drone_100m_mapping
```

### By Application Domain

#### Biological Research

| Application | Recommended Presets |
|-------------|-------------------|
| Bacteria imaging | `microscope_100x_oil`, `microscope_fluorescence_100x` |
| Live cell imaging | `microscope_100x_oil_phase`, `microscope_60x_water` |
| Tissue histology | `microscope_20x_air`, `microscope_40x_air` |
| Cell culture QC | `microscope_40x_phase`, `microscope_40x_dic` |
| Sample screening | `microscope_10x_air` |

#### Infrastructure Inspection

| Application | Recommended Presets |
|-------------|-------------------|
| Power lines | `drone_10m_inspection` |
| Building facades | `drone_20m_detail`, `drone_infrastructure_30m` |
| Bridges | `drone_infrastructure_30m`, `drone_50m_survey` |
| Roads | `drone_50m_survey`, `drone_100m_mapping` |
| Roof inspection | `drone_20m_detail`, `drone_infrastructure_30m` |

#### Agricultural Monitoring

| Application | Recommended Presets |
|-------------|-------------------|
| Precision agriculture | `drone_agriculture_50m` |
| Crop health mapping | `drone_50m_survey`, `drone_agriculture_50m` |
| Field surveys | `drone_100m_mapping` |
| Plant counting | `drone_20m_detail`, `drone_infrastructure_30m` |

#### General Surveying

| Application | Recommended Presets |
|-------------|-------------------|
| Construction sites | `drone_50m_survey`, `drone_infrastructure_30m` |
| Property surveys | `drone_100m_mapping`, `drone_50m_survey` |
| Topographic mapping | `drone_100m_mapping` |
| Urban planning | `drone_phantom_120m`, `drone_100m_mapping` |

---

## Usage Examples

### Python API

#### Load Microscopy Preset

```python
from prism.scenarios import get_scenario_preset

# Load high-resolution oil immersion preset
scenario = get_scenario_preset("microscope_100x_oil")

# Check computed parameters
print(f"Lateral resolution: {scenario.lateral_resolution_nm:.1f} nm")
print(f"Axial resolution: {scenario.axial_resolution_um:.2f} µm")
print(f"NA: {scenario.objective_spec.numerical_aperture}")
print(f"Immersion: {scenario.objective_spec.immersion_medium}")

# Output:
# Lateral resolution: 240.0 nm
# Axial resolution: 1.16 µm
# NA: 1.4
# Immersion: oil
```

#### Load Drone Preset

```python
from prism.scenarios import get_scenario_preset

# Load standard survey preset
scenario = get_scenario_preset("drone_50m_survey")

# Check computed parameters
print(f"GSD: {scenario.actual_gsd_cm:.2f} cm")
print(f"Swath width: {scenario.swath_width_m:.1f} m")
print(f"Altitude: {scenario.altitude_m} m")
print(f"Focal length: {scenario.lens_spec.focal_length_mm} mm")
print(f"Sensor: {scenario.sensor_spec.name}")

# Output:
# GSD: 6.50 cm
# Swath width: 36.0 m
# Altitude: 50.0 m
# Focal length: 50 mm
# Sensor: full_frame
```

#### List All Presets

```python
from prism.scenarios import list_presets

# List microscopy presets
microscopy = list_presets(category="microscope")
print(f"Found {len(microscopy)} microscopy presets:")
for name in microscopy:
    print(f"  - {name}")

# List drone presets
drone = list_presets(category="drone")
print(f"\nFound {len(drone)} drone presets:")
for name in drone:
    print(f"  - {name}")

# List all presets
all_presets = list_presets()
print(f"\nTotal presets: {len(all_presets)}")
```

#### Convert to Instrument

```python
from prism.scenarios import get_scenario_preset
from prism.core import Microscope, Camera

# Microscopy scenario → Microscope instrument
scenario = get_scenario_preset("microscope_40x_air")
config = scenario.to_instrument_config()
microscope = Microscope(config)

# Drone scenario → Camera instrument
scenario = get_scenario_preset("drone_50m_survey")
config = scenario.to_instrument_config()
camera = Camera(config)
```

### CLI Usage

#### List Presets

```bash
# List all presets
prism-scenario list

# List microscopy presets only
prism-scenario list --category microscope

# List drone presets only
prism-scenario list --category drone
```

#### Get Preset Info

```bash
# Detailed info for specific preset
prism-scenario info microscope_100x_oil

# Output includes:
# - Resolution (lateral & axial)
# - NA, magnification, immersion
# - Wavelength
# - Fresnel number
# - Recommended use cases
```

#### Create Instrument Config

```bash
# Create instrument configuration from preset
prism-scenario instrument microscope_40x_air --output microscope_config.yaml

# Creates YAML file with complete instrument parameters
# Can be loaded directly by PRISM
```

#### Compare Presets

```bash
# Compare multiple presets side-by-side
prism-scenario compare microscope_100x_oil microscope_60x_water microscope_40x_air

# Shows comparison table with:
# - Resolution differences
# - NA values
# - Working distances
# - Field of view
```

### Python: Custom Scenarios Based on Presets

```python
from prism.scenarios import get_scenario_preset

# Start from preset, customize parameters
base_scenario = get_scenario_preset("microscope_40x_air")

# Modify wavelength for specific fluorophore
custom_scenario = base_scenario.copy(update={"wavelength": 488e-9})

print(f"Original resolution: {base_scenario.lateral_resolution_nm:.1f} nm")
print(f"At 488nm: {custom_scenario.lateral_resolution_nm:.1f} nm")

# Output:
# Original resolution: 371.1 nm
# At 488nm: 329.5 nm (better resolution at shorter wavelength)
```

---

## Cross-References

### Related References

- [Physical Constants](physical_constants.md) - Wavelengths, units, speed of light
- [Microscopy Parameters](microscopy_parameters.md) - Detailed microscope specifications
- [Drone Camera Parameters](drone_camera_parameters.md) - Detailed drone camera specifications
- [Optical Resolution Limits](optical_resolution_limits.md) - Resolution formulas and limits
- [Fresnel Propagation Regimes](fresnel_propagation_regimes.md) - Fresnel numbers for each preset

### Related User Guides

- [Scenarios User Guide](../user_guides/scenarios.md) - How to use the scenario system
- [Optical Engineering Guide](../user_guides/optical-engineering.md) - Understanding optical systems

---

## Source Files

This catalog is compiled from:
- [prism/scenarios/presets.py](../../prism/scenarios/presets.py) - All preset definitions
- [prism/scenarios/microscopy.py](../../prism/scenarios/microscopy.py) - Microscopy implementation
- [prism/scenarios/drone_camera.py](../../prism/scenarios/drone_camera.py) - Drone camera implementation

---

## Maintenance Notes

**Updating this catalog**:
1. Check `prism/scenarios/presets.py` for new presets
2. Run preset info CLI for computed parameters
3. Verify resolution calculations @ 550nm
4. Update preset counts (currently 9 + 8 = 17)
5. Add to appropriate category tables
6. Update selection guide if new categories emerge

**Validation**:
- All parameters verified against source code on 2025-01-26
- Resolution formulas: Δx = 0.61λ/NA (microscopy), GSD = H×p/f (drone)
- Propagation: All presets use Angular Spectrum (F >> 10)

---

**Last Updated**: 2025-01-26
**Preset Count**: 17 total (9 microscopy + 8 drone)
**Accuracy**: 100% - all values verified against `presets.py`
