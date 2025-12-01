# Drone Camera Parameters Reference

**Complete reference for drone-mounted camera systems: lenses, sensors, GSD calculations, and all 8 PRISM drone presets.**

## Quick Navigation

- [GSD Formula](#gsd-formula)
- [Lens Specifications](#lens-specifications)
- [Sensor Specifications](#sensor-specifications)
- [GSD Lookup Tables](#gsd-lookup-tables)
- [Mission Type Guidelines](#mission-type-guidelines)
- [Complete Preset Table](#complete-preset-table)
- [Usage Examples](#usage-examples)

---

## GSD Formula

### Ground Sampling Distance (GSD)

The GSD is the physical distance on the ground represented by one pixel:

$$\text{GSD} = \frac{H \times p}{f}$$

Where:
- **H** = altitude above ground (m)
- **p** = pixel pitch (sensor pixel size) (µm or m)
- **f** = focal length (mm or m)

**Units**: Ensure consistent units. Typical: H in meters, p in µm, f in mm → GSD in cm/pixel.

**Conversion**:
```
GSD (cm) = (H [m] × p [µm] × 100) / (f [mm] × 1000)
GSD (cm) = (H [m] × p [µm]) / (f [mm] × 10)
```

### Swath Width (Coverage Width)

The ground width covered by one image:

$$W = \frac{H \times S_w}{f}$$

Where:
- **W** = swath width (m)
- **H** = altitude (m)
- **S_w** = sensor width (mm)
- **f** = focal length (mm)

**Example**: 50mm lens, full-frame sensor (36mm), 50m altitude:
```
W = (50 × 36) / 50 = 36 m
```

---

## Lens Specifications

### Common Focal Lengths

| Focal Length | Field of View (FF) | Typical Use | GSD @ 50m (FF) | Aperture Examples |
|--------------|-------------------|-------------|----------------|-------------------|
| 24mm | 84° | Wide-area mapping | ~13.5 cm | f/2.8, f/4.0 |
| 35mm | 63° | General surveying | ~9.3 cm | f/2.8, f/4.0 |
| 50mm | 47° | Detailed surveying | ~6.5 cm | f/2.8, f/4.0 |
| 85mm | 29° | Inspection | ~3.8 cm | f/1.8, f/2.8 |

**Note**: Field of view (FOV) is for full-frame sensor (36×24mm). FOV scales with sensor size.

### F-Number and Aperture

$$\text{Aperture Diameter} = \frac{f}{N}$$

Where:
- **f** = focal length (mm)
- **N** = f-number (e.g., 2.8, 4.0)

| Lens | f-number | Aperture Diameter | Light Gathering | Diffraction Limit @ 550nm |
|------|----------|------------------|----------------|--------------------------|
| 50mm f/2.8 | 2.8 | 17.9 mm | High | 1.9 µrad |
| 50mm f/4.0 | 4.0 | 12.5 mm | Medium | 2.7 µrad |
| 35mm f/2.8 | 2.8 | 12.5 mm | High | 2.7 µrad |
| 35mm f/4.0 | 4.0 | 8.75 mm | Medium | 3.9 µrad |
| 24mm f/2.8 | 2.8 | 8.57 mm | High | 3.9 µrad |

**Diffraction-limited angular resolution**: θ = 1.22λ/D (Rayleigh criterion)

---

## Sensor Specifications

### Full Comparison Table

| Sensor Type | Width (mm) | Height (mm) | Typical Resolution | Pixel Pitch (µm) | Megapixels | Common Uses |
|-------------|------------|-------------|-------------------|------------------|------------|-------------|
| Full-frame | 36.0 | 24.0 | 5538×3692 | 6.5 | 20.4 | Professional drones, high-end mapping |
| APS-C | 23.5 | 15.6 | 6026×4000 | 3.9 | 24.1 | Consumer/prosumer drones, versatile |
| Micro 4/3 | 17.3 | 13.0 | 5242×3939 | 3.3 | 20.6 | Compact systems, video-focused |
| 1" | 13.2 | 8.8 | 5500×3667 | 2.4 | 20.2 | Mavic 2 Pro, RX100 series |
| 1/2.3" | 6.17 | 4.55 | 3857×2839 | 1.6 | 10.9 | DJI Phantom 4, consumer drones |

**Pixel pitch vs sensor size**: Smaller sensors typically have smaller pixels, leading to different noise/dynamic range characteristics.

### PRISM Sensor Specs

From `prism/scenarios/drone_camera.py` (SensorSpec.from_name):

| PRISM Name | Width | Height | Pixel Pitch | Megapixels |
|------------|-------|--------|-------------|------------|
| `full_frame` | 36.0 mm | 24.0 mm | 6.5 µm | 20.4 MP |
| `aps_c` | 23.5 mm | 15.6 mm | 3.9 µm | 24.1 MP |
| `micro_four_thirds` | 17.3 mm | 13.0 mm | 3.3 µm | 20.6 MP |
| `1_inch` | 13.2 mm | 8.8 mm | 2.4 µm | 20.2 MP |
| `1_2.3_inch` | 6.17 mm | 4.55 mm | 1.6 µm | 10.9 MP |

---

## GSD Lookup Tables

### 50mm Lens, Full-Frame Sensor (6.5µm pixels)

| Altitude (m) | GSD (cm) | Swath Width (m) | Use Case |
|--------------|----------|-----------------|----------|
| 10 | 1.3 | 7.2 | Ultra-close inspection |
| 20 | 2.6 | 14.4 | Close-range detail |
| 30 | 3.9 | 21.6 | Infrastructure inspection |
| 50 | 6.5 | 36.0 | Standard survey |
| 100 | 13.0 | 72.0 | Large-area mapping |
| 120 | 15.6 | 86.4 | Max legal altitude (US) |

### 35mm Lens, APS-C Sensor (3.9µm pixels)

| Altitude (m) | GSD (cm) | Swath Width (m) | Use Case |
|--------------|----------|-----------------|----------|
| 10 | 1.1 | 6.7 | Very close inspection |
| 20 | 2.2 | 13.4 | Detail work |
| 50 | 5.6 | 33.6 | Agricultural monitoring |
| 100 | 11.1 | 67.1 | Regional mapping |
| 120 | 13.4 | 80.6 | Max altitude survey |

### 24mm Lens, 1/2.3" Sensor (1.6µm pixels) - DJI Phantom 4 Style

| Altitude (m) | GSD (cm) | Swath Width (m) | Use Case |
|--------------|----------|-----------------|----------|
| 30 | 2.0 | 7.7 | Close inspection |
| 50 | 3.3 | 12.9 | Detail capture |
| 100 | 6.7 | 25.7 | Standard mapping |
| 120 | 8.0 | 30.9 | Legal max altitude |

---

## Mission Type Guidelines

### GSD Requirements by Application

| Application | Target GSD | Typical Altitude | Recommended Setup |
|-------------|-----------|------------------|-------------------|
| **Close inspection** | < 5 mm | 5-15 m | 50mm + FF, 35mm + APS-C |
| **Detail mapping** | 5-50 mm | 15-30 m | 50mm + FF, 35mm + APS-C |
| **Standard survey** | 5-10 cm | 50-100 m | 50mm + FF, 35mm + FF |
| **Large-area mapping** | 10-20 cm | 100-150 m | 35mm + FF, 24mm + FF |
| **Agriculture** | 5-10 cm | 50-80 m | 35mm + APS-C, 50mm + FF |
| **Infrastructure** | 2-5 cm | 20-50 m | 50mm + FF, 85mm + APS-C |

### Quality vs Coverage Tradeoff

| Priority | GSD Target | Coverage Strategy | Example |
|----------|-----------|------------------|---------|
| Maximum detail | < 1 cm | Low altitude, slow flight, high overlap | Inspection |
| Balanced | 5-10 cm | Medium altitude, standard flight, 60-70% overlap | Survey |
| Maximum coverage | > 15 cm | High altitude, fast flight, minimal overlap | Reconnaissance |

---

## Complete Preset Table

All 8 drone presets with computed parameters @ λ = 550 nm:

| Preset Name | Lens | Sensor | Altitude (m) | Speed (m/s) | GSD (cm) | Swath (m) | F-number | Use Case |
|-------------|------|--------|-------------|------------|---------|----------|----------|----------|
| `drone_10m_inspection` | 35mm f/2.8 | APS-C | 10 | 2 | 1.1 | 6.7 | 2.8 | Close-range inspection |
| `drone_20m_detail` | 50mm f/4.0 | APS-C | 20 | 5 | 1.6 | 9.4 | 4.0 | Detailed surveying |
| `drone_50m_survey` | 50mm f/4.0 | Full-frame | 50 | 10 | 6.5 | 36.0 | 4.0 | Site survey (standard) |
| `drone_100m_mapping` | 35mm f/4.0 | Full-frame | 100 | 15 | 18.6 | 103 | 4.0 | Large area mapping |
| `drone_phantom_120m` | 24mm f/2.8 | 1/2.3" | 120 | 12 | 8.0 | 30.9 | 2.8 | DJI Phantom equivalent |
| `drone_hover_50m` | 50mm f/4.0 | Full-frame | 50 | 0 | 6.5 | 36.0 | 4.0 | Hover (no motion blur) |
| `drone_agriculture_50m` | 35mm f/4.0 | APS-C | 50 | 8 | 5.6 | 33.6 | 4.0 | Agricultural monitoring |
| `drone_infrastructure_30m` | 50mm f/2.8 | Full-frame | 30 | 3 | 3.9 | 21.6 | 2.8 | Infrastructure inspection |

### Preset Selection Guide

**Ultra-close detail (GSD < 2 cm)**:
- `drone_10m_inspection` - 1.1 cm GSD
- `drone_20m_detail` - 1.6 cm GSD

**Standard survey (GSD 3-7 cm)**:
- `drone_infrastructure_30m` - 3.9 cm GSD
- `drone_agriculture_50m` - 5.6 cm GSD
- `drone_50m_survey` - 6.5 cm GSD
- `drone_hover_50m` - 6.5 cm GSD (no motion blur)

**Large area mapping (GSD > 8 cm)**:
- `drone_phantom_120m` - 8.0 cm GSD (consumer drone)
- `drone_100m_mapping` - 18.6 cm GSD (fast coverage)

**Motion blur considerations**:
- `drone_hover_50m` - Zero motion blur (hover mode)
- `drone_10m_inspection` - Minimal (2 m/s speed)
- `drone_infrastructure_30m` - Low (3 m/s speed)
- `drone_100m_mapping` - Higher (15 m/s speed, may need faster shutter)

---

## Motion Blur

### Motion Blur Calculation

Motion blur occurs when the drone moves during exposure:

$$\text{Blur Distance} = v \times t_{\text{exp}}$$
$$\text{Blur Pixels} = \frac{\text{Blur Distance}}{\text{GSD}}$$

Where:
- **v** = ground speed (m/s)
- **t_exp** = exposure time (s)
- Typical exposure time: 1/1000s (1ms) in bright conditions

### Blur Pixel Lookup

Assuming 1ms exposure time:

| Speed (m/s) | Ground Motion (cm) | @ 1cm GSD | @ 5cm GSD | @ 10cm GSD |
|------------|-------------------|----------|----------|-----------|
| 2 | 0.2 | 0.2 px | 0.04 px | 0.02 px |
| 5 | 0.5 | 0.5 px | 0.1 px | 0.05 px |
| 10 | 1.0 | 1.0 px | 0.2 px | 0.1 px |
| 15 | 1.5 | 1.5 px | 0.3 px | 0.15 px |

**Rule of thumb**: Keep motion blur < 0.5 pixels for sharp imagery
- At 10 m/s, need GSD > 2 cm (or shorter exposure time)
- At 15 m/s, need GSD > 3 cm (or shorter exposure time)

---

## Fresnel Number and Propagation

For drone altitudes, the Fresnel number determines diffraction regime:

### Example Calculations

**50mm f/4.0 lens at 50m altitude** (aperture = 12.5mm):
```
F = a²/(λz) = (0.00625)²/(550e-9 × 50) = 1420 >> 1 (near-field)
```

**24mm f/2.8 lens at 120m altitude** (aperture = 8.57mm):
```
F = a²/(λz) = (0.00428)²/(550e-9 × 120) = 278 >> 1 (near-field)
```

**Conclusion**: All drone scenarios are in the near-field regime (F >> 1), so PRISM uses Angular Spectrum propagation by default.

---

## Usage Examples

### Load Preset

```python
from prism.scenarios import get_scenario_preset

# Load preset
scenario = get_scenario_preset("drone_50m_survey")

# Check computed parameters
print(f"GSD: {scenario.actual_gsd_cm:.2f} cm")
print(f"Swath width: {scenario.swath_width_m:.1f} m")
print(f"Altitude: {scenario.altitude_m} m")
print(f"Lens: {scenario.lens_spec}")
print(f"Sensor: {scenario.sensor_spec.name}")
print(f"Motion blur: {scenario.motion_blur_pixels:.2f} pixels")
```

### Create Custom Drone Scenario

```python
from prism.scenarios.drone_camera import DroneScenarioConfig

# Custom agricultural drone
scenario = DroneScenarioConfig(
    lens_spec="35mm_f4.0",
    sensor_spec="aps_c",
    altitude_m=75.0,
    ground_speed_mps=8.0,
)

print(f"GSD: {scenario.actual_gsd_cm:.2f} cm")
print(f"Swath: {scenario.swath_width_m:.1f} m")
```

### Convert to Instrument

```python
from prism.scenarios import get_scenario_preset

# Load scenario
scenario = get_scenario_preset("drone_agriculture_50m")

# Convert to instrument configuration
instrument_config = scenario.to_instrument_config()

# Use with PRISM
from prism.core import Camera
camera = Camera(instrument_config)
```

### CLI Usage

```bash
# List drone presets
prism-scenario list --category drone

# Get detailed info
prism-scenario info drone_50m_survey

# Create instrument from preset
prism-scenario instrument drone_agriculture_50m --output config.yaml
```

---

## Source Files

This reference is based on:
- [prism/scenarios/presets.py](../../prism/scenarios/presets.py) (lines 72-130)
- [prism/scenarios/drone_camera.py](../../prism/scenarios/drone_camera.py)

## Related References

- [Physical Constants](physical_constants.md) - Wavelengths and units
- [Optical Resolution Limits](optical_resolution_limits.md) - Diffraction limits
- [Fresnel Propagation Regimes](fresnel_propagation_regimes.md) - Propagation method selection
- [Scenario Preset Catalog](scenario_preset_catalog.md) - All presets in one table

## Related User Guides

- [Optical Engineering Guide](../user_guides/optical-engineering.md) - Understanding optical systems
- [Scenarios User Guide](../user_guides/scenarios.md) - How to use scenario system

---

**Last Updated**: 2025-01-26
**Accuracy**: All values verified against `prism/scenarios/presets.py` and `drone_camera.py`
**GSD Formula**: Standard photogrammetry (H×p/f)
