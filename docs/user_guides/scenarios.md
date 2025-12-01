# Scenario Configuration Guide

## Overview

PRISM provides a user-friendly scenario configuration system for simulating real-world optical systems. Instead of manually configuring wavelengths, numerical apertures, and pixel sizes, you can use pre-defined presets or intuitive builder patterns that handle the physics automatically.

The scenario system supports:

- **Microscopy**: High-NA objectives with brightfield, darkfield, phase, and DIC illumination
- **Drone Cameras**: Consumer lenses at finite distances with ground sampling distance (GSD) calculations
- **Satellite Cameras**: Earth observation with diffraction-limited imaging (coming soon)

---

## Quick Start

### Using Presets

The simplest way to configure an optical system is with presets:

```python
from prism.scenarios import get_scenario_preset

# Load a pre-configured microscope
scenario = get_scenario_preset("microscope_100x_oil")

# See what you get
print(f"Resolution: {scenario.lateral_resolution_nm:.0f} nm")
print(f"FOV: {scenario.field_of_view_um:.1f} µm")

# Convert to PRISM instrument config
config = scenario.to_instrument_config()

# Create the instrument
from prism.core.instruments import Microscope
microscope = Microscope(config)
```

### Listing Available Presets

```python
from prism.scenarios import list_scenario_presets, get_preset_description

# List all presets
all_presets = list_scenario_presets()
print(f"Available presets: {len(all_presets)}")

# List by category
microscope_presets = list_scenario_presets("microscope")
drone_presets = list_scenario_presets("drone")

# Get description for a preset
desc = get_preset_description("microscope_100x_oil")
print(desc)  # "High-resolution oil immersion for cellular detail"
```

---

## CLI Integration

### Using Scenarios from Command Line

```bash
# List available presets
python main.py --list-scenarios

# Show details of a specific preset
python main.py --show-scenario microscope_100x_oil

# Run with a scenario preset
python main.py --scenario microscope_40x_air --obj_name usaf1951 --name microscope_test

# Override scenario parameters
python main.py --scenario microscope_40x_air --objective 60x_1.2NA_water --illumination phase
```

### Available CLI Options

| Option | Description |
|--------|-------------|
| `--scenario NAME` | Load scenario preset |
| `--list-scenarios` | List all available presets |
| `--show-scenario NAME` | Show preset details |
| `--objective SPEC` | Override objective (e.g., "100x_1.4NA_oil") |
| `--illumination MODE` | Override illumination mode |
| `--lens SPEC` | Override lens spec (e.g., "50mm_f2.8") |
| `--altitude METERS` | Override altitude for drone scenarios |
| `--sensor TYPE` | Override sensor type |

---

## Microscope Scenarios

### Available Presets

| Preset Name | Objective | Mode | Description |
|-------------|-----------|------|-------------|
| `microscope_10x_air` | 10x 0.25 NA | Brightfield | Low magnification overview |
| `microscope_20x_air` | 20x 0.4 NA | Brightfield | General observation |
| `microscope_40x_air` | 40x 0.9 NA | Brightfield | High-NA air objective |
| `microscope_60x_water` | 60x 1.2 NA | Brightfield | Water immersion for live cells |
| `microscope_100x_oil` | 100x 1.4 NA | Brightfield | High-resolution oil immersion |
| `microscope_phase_20x` | 20x 0.4 NA | Phase | Phase contrast for unstained |
| `microscope_phase_40x` | 40x 0.9 NA | Phase | High-NA phase contrast |
| `microscope_dic_40x` | 40x 0.9 NA | DIC | Differential interference contrast |
| `microscope_darkfield_40x` | 40x 0.9 NA | Darkfield | Darkfield illumination |

### Using the Builder Pattern

For custom configurations, use the builder pattern:

```python
from prism.scenarios import MicroscopeBuilder

scenario = (MicroscopeBuilder()
    .objective("60x_1.2NA_water")
    .illumination("phase")
    .wavelength_nm(488)  # Blue laser
    .sensor_pixels(2048, 6.5)  # Pixels, pixel size in µm
    .name("Custom Phase Microscope")
    .description("488nm phase contrast for live cell imaging")
    .build())

# Get physics parameters
print(f"Lateral resolution: {scenario.lateral_resolution_nm:.0f} nm")
print(f"Axial resolution: {scenario.axial_resolution_um:.2f} µm")
print(f"FOV: {scenario.field_of_view_um:.0f} µm")

# Convert to instrument
config = scenario.to_instrument_config()
```

### Direct Configuration

For full control, configure directly:

```python
from prism.scenarios import MicroscopeScenarioConfig

scenario = MicroscopeScenarioConfig(
    objective_spec="100x_1.4NA_oil",
    illumination_mode="brightfield",
    wavelength=550e-9,  # 550 nm
    n_pixels=2048,
    sensor_pixel_size=6.5e-6,  # 6.5 µm
)

# Automatic calculations are performed:
print(f"Resolution: {scenario.lateral_resolution_nm:.0f} nm")
print(f"Axial DOF: {scenario.axial_resolution_um:.2f} µm")
```

### Objective Specification Format

Objectives are specified as: `{mag}x_{NA}NA_{medium}`

Examples:
- `100x_1.4NA_oil` - 100x magnification, 1.4 NA, oil immersion
- `40x_0.9NA_air` - 40x magnification, 0.9 NA, air
- `60x_1.2NA_water` - 60x magnification, 1.2 NA, water immersion

Supported media:
| Medium | Refractive Index |
|--------|------------------|
| `air` | 1.0 |
| `water` | 1.33 |
| `oil` | 1.515 |
| `silicone` | 1.4 |

### Illumination Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `brightfield` | Standard transmission/reflection | Stained samples, opaque objects |
| `darkfield` | Only scattered light | Unstained, edge detection |
| `phase` | Phase shifts to intensity | Live cells, unstained samples |
| `dic` | Differential interference contrast | 3D-like appearance, surface topology |

---

## Drone Camera Scenarios

### Available Presets

| Preset Name | Lens | Altitude | GSD | Description |
|-------------|------|----------|-----|-------------|
| `drone_10m_inspection` | 35mm f/2.8 | 10 m | ~3 mm | Close-range inspection |
| `drone_30m_mapping` | 35mm f/2.8 | 30 m | ~8 mm | Detailed mapping |
| `drone_50m_survey` | 50mm f/4.0 | 50 m | ~1.5 cm | Site survey |
| `drone_80m_agriculture` | 50mm f/4.0 | 80 m | ~2.5 cm | Agricultural monitoring |
| `drone_100m_overview` | 50mm f/4.0 | 100 m | ~3 cm | Area overview |
| `drone_120m_corridor` | 85mm f/1.8 | 120 m | ~2 cm | Corridor mapping |

### Using the Builder Pattern

```python
from prism.scenarios import DroneBuilder

scenario = (DroneBuilder()
    .lens("50mm_f2.8")
    .sensor("aps_c")
    .altitude(75)
    .ground_speed(12)  # m/s for motion blur calculation
    .wavelength_nm(550)
    .name("Construction Site Survey")
    .description("75m altitude survey with minimal motion blur")
    .build())

# Get mission parameters
print(f"GSD: {scenario.actual_gsd_cm:.2f} cm")
print(f"Swath width: {scenario.swath_width_m:.1f} m")
print(f"Motion blur: {scenario.motion_blur_pixels:.2f} pixels")
print(f"Fresnel number: {scenario.fresnel_number:.0f}")
```

### Direct Configuration

```python
from prism.scenarios import DroneScenarioConfig

scenario = DroneScenarioConfig(
    lens_spec="50mm_f4.0",
    sensor_spec="full_frame",
    altitude_m=50.0,
    ground_speed_mps=10.0,  # For motion blur
    wavelength=550e-9,
)

# Physics are calculated automatically
print(f"GSD: {scenario.actual_gsd_cm:.2f} cm")
```

### Lens Specification Format

Lenses are specified as: `{focal_length}mm_f{aperture}`

Examples:
- `35mm_f2.8` - 35mm focal length, f/2.8 aperture
- `50mm_f4.0` - 50mm focal length, f/4.0 aperture
- `85mm_f1.8` - 85mm focal length, f/1.8 aperture

### Sensor Types

| Type | Width | Height | Pixel Pitch |
|------|-------|--------|-------------|
| `full_frame` | 36 mm | 24 mm | 6.5 µm |
| `aps_c` | 23.5 mm | 15.6 mm | 3.9 µm |
| `1_inch` | 13.2 mm | 8.8 mm | 2.4 µm |
| `m43` | 17.3 mm | 13 mm | 3.75 µm |

---

## Using with USAF-1951 Test Targets

The scenario system integrates with the USAF-1951 test target module for resolution validation:

```python
from prism.core.targets import create_target
from prism.scenarios import get_scenario_preset
from prism.core.instruments import Microscope
import torch

# Create a test target
target = create_target("usaf1951", size=512, groups=(0, 1, 2, 3))
ground_truth = target.generate()

# See resolution elements
print("Resolution elements:")
for key, value in target.resolution_elements.items():
    print(f"  {key}: {value}")

# Load microscope scenario
scenario = get_scenario_preset("microscope_40x_air")
print(f"Microscope resolution: {scenario.lateral_resolution_nm:.0f} nm")

# Create instrument
config = scenario.to_instrument_config()
config.n_pixels = 512  # Match target size
microscope = Microscope(config)

# Image the target
field = ground_truth.to(torch.complex64)
image = microscope.forward(field, illumination_mode="brightfield")
```

---

## Physics Calculations

### Microscope Resolution (Abbe Limit)

The lateral resolution is calculated using the Abbe diffraction limit:

$$\Delta x = \frac{0.61 \lambda}{NA}$$

For example, with λ = 550 nm and NA = 1.4:
- Resolution = 0.61 × 550 nm / 1.4 = 240 nm

### Axial Resolution

The axial (depth) resolution follows:

$$\Delta z = \frac{2 n \lambda}{NA^2}$$

### Ground Sampling Distance (GSD)

For drone cameras, GSD is calculated as:

$$GSD = \frac{H \cdot p}{f}$$

Where:
- H = altitude (meters)
- p = pixel pitch (meters)
- f = focal length (meters)

### Fresnel Number

The Fresnel number determines the propagation regime:

$$F = \frac{a^2}{\lambda z}$$

Where:
- a = aperture radius
- λ = wavelength
- z = propagation distance

- F >> 1: Near-field (use Angular Spectrum propagator)
- F << 1: Far-field (use Fraunhofer propagator)

---

## Validation and Error Messages

The scenario system validates all parameters:

```python
from prism.scenarios import MicroscopeScenarioConfig

# This will raise a validation error
try:
    scenario = MicroscopeScenarioConfig(
        objective_spec="100x_1.5NA_air"  # NA > 1 in air is impossible
    )
except ValueError as e:
    print(f"Error: {e}")
    # Output: "NA (1.5) cannot exceed medium index (1.0)"

# Undersampling detection
try:
    scenario = MicroscopeScenarioConfig(
        objective_spec="100x_1.4NA_oil",
        sensor_pixel_size=20e-6,  # Too large
    )
except ValueError as e:
    print(f"Error: {e}")
    # Output: "Undersampling detected..."
```

---

## Example Workflows

### Workflow 1: Comparing Objectives

```python
from prism.scenarios import MicroscopeBuilder, get_scenario_preset

objectives = [
    "20x_0.4NA_air",
    "40x_0.9NA_air",
    "60x_1.2NA_water",
    "100x_1.4NA_oil",
]

print("Objective Comparison:")
print("-" * 60)
print(f"{'Objective':<20} {'Resolution (nm)':<18} {'FOV (µm)':<15}")
print("-" * 60)

for obj in objectives:
    scenario = MicroscopeBuilder().objective(obj).build()
    print(f"{obj:<20} {scenario.lateral_resolution_nm:<18.0f} {scenario.field_of_view_um:<15.0f}")
```

### Workflow 2: Drone Mission Planning

```python
from prism.scenarios import DroneBuilder

altitudes = [30, 50, 80, 100, 120]

print("Mission Planning - 50mm f/4.0 lens, Full Frame:")
print("-" * 60)
print(f"{'Altitude (m)':<15} {'GSD (cm)':<12} {'Swath (m)':<12}")
print("-" * 60)

for alt in altitudes:
    scenario = DroneBuilder().lens("50mm_f4.0").sensor("full_frame").altitude(alt).build()
    print(f"{alt:<15} {scenario.actual_gsd_cm:<12.2f} {scenario.swath_width_m:<12.1f}")
```

---

## Extending the System

### Adding Custom Presets

The preset system can be extended in `prism/scenarios/presets.py`:

```python
MICROSCOPE_PRESETS["microscope_custom"] = {
    "objective_spec": "40x_0.75NA_air",
    "illumination_mode": "darkfield",
    "description": "Custom darkfield configuration",
}
```

### Creating Custom Scenario Types

For new optical systems (e.g., satellite cameras), create a new scenario class inheriting from `ScenarioConfig`:

```python
from prism.scenarios.base import ScenarioConfig
from dataclasses import dataclass

@dataclass
class SatelliteScenarioConfig(ScenarioConfig):
    orbit_altitude_km: float = 400.0
    aperture_m: float = 0.5

    def __post_init__(self):
        # Calculate physics
        self._calculate_resolution()
```

---

## Troubleshooting

### Common Issues

**"NA cannot exceed medium index"**
- NA > 1.0 requires immersion medium (water, oil)
- Use `_water` or `_oil` suffix in objective spec

**"Undersampling detected"**
- Pixel size too large for objective magnification
- Either increase magnification or decrease pixel size

**"Altitude too low"**
- Drone altitude < 5m is not practical
- Increase altitude or use microscope scenario

**"Unknown preset"**
- Check spelling with `list_scenario_presets()`
- Presets are case-sensitive

---

## Next Steps

- See [API Reference: Scenarios](../api/scenarios.md) for detailed API documentation
- See [Propagator Selection Guide](propagator_selection.md) for propagation method details
- See [USAF-1951 Targets](../api/targets.md) for resolution validation
