# prism.scenarios

Scenario configuration system for PRISM optical simulations.

This module provides user-friendly configuration for real-world optical systems:
- Microscopy (brightfield, darkfield, phase, DIC)
- Drone cameras (finite distance, consumer lenses, GSD)
- Satellite cameras (Earth observation, diffraction limits)

## Quick Start

```python
from prism.scenarios import get_scenario_preset

# Load a preset
scenario = get_scenario_preset("microscope_100x_oil")
print(f"Resolution: {scenario.lateral_resolution_nm:.0f} nm")

# Convert to PRISM instrument config
instrument_config = scenario.to_instrument_config()
```

## Classes

### ScenarioConfig

```python
class ScenarioConfig(ABC)
```

Base configuration for optical imaging scenarios.

All scenario configs provide:
1. User-friendly parameters (objective spec, altitude, lens spec)
2. Auto-computed physics parameters (wavelength, resolution, SNR)
3. Conversion to PRISM instrument configs via `to_instrument_config()`
4. Validation with helpful error messages

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `scenario_type` | `str` | Type identifier ("microscope", "drone", "satellite") |
| `name` | `str` | Human-readable scenario name |
| `description` | `str` | Detailed description of the scenario |
| `wavelength` | `float` | Operating wavelength in meters |
| `object_distance` | `float` | Distance to object in meters |
| `resolution_limit` | `float` | Theoretical resolution in meters |
| `snr` | `float` | Expected signal-to-noise ratio in dB |
| `propagator_method` | `str` | Propagation method ('auto', 'fraunhofer', 'angular_spectrum') |

**Abstract Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_instrument_config()` | `InstrumentConfig` | Convert to PRISM instrument config |
| `validate()` | `None` | Validate scenario parameters |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_info()` | `dict` | Get scenario information summary |

---

### ObjectiveSpec

```python
@dataclass
class ObjectiveSpec
```

Microscope objective specification.

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `magnification` | `int` | Total magnification (e.g., 100) |
| `numerical_aperture` | `float` | Objective NA (e.g., 1.4) |
| `immersion_medium` | `str` | Medium type ('air', 'water', 'oil', 'silicone') |
| `medium_index` | `float` | Refractive index of medium |

**Class Methods:**

```python
@classmethod
def from_string(cls, spec: str) -> ObjectiveSpec
```

Parse objective from string format.

**Parameters:**
- `spec` (str): Specification string (e.g., "100x_1.4NA_oil")

**Returns:**
- `ObjectiveSpec`: Parsed objective specification

**Raises:**
- `ValueError`: If format is invalid

**Example:**

```python
spec = ObjectiveSpec.from_string("100x_1.4NA_oil")
assert spec.magnification == 100
assert spec.numerical_aperture == 1.4
assert spec.immersion_medium == "oil"
assert spec.medium_index == 1.515
```

---

### MicroscopeScenarioConfig

```python
@dataclass
class MicroscopeScenarioConfig(ScenarioConfig)
```

Microscope scenario configuration.

**Attributes:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `objective_spec` | `str` or `ObjectiveSpec` | Required | Objective specification |
| `illumination_mode` | `str` | `"brightfield"` | Illumination mode |
| `wavelength` | `float` | `550e-9` | Operating wavelength (m) |
| `n_pixels` | `int` | `1024` | Detector pixels |
| `sensor_pixel_size` | `float` | `6.5e-6` | Pixel size (m) |

**Computed Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `lateral_resolution_nm` | `float` | Abbe resolution limit (nm) |
| `axial_resolution_um` | `float` | Depth of focus (µm) |
| `field_of_view_um` | `float` | Field of view (µm) |
| `object_pixel_size` | `float` | Object-space pixel size (m) |

**Methods:**

```python
def to_instrument_config(self) -> MicroscopeConfig
```

Convert to PRISM MicroscopeConfig.

**Returns:**
- `MicroscopeConfig`: Configuration for PRISM Microscope instrument

**Example:**

```python
from prism.scenarios import MicroscopeScenarioConfig
from prism.core.instruments import Microscope

scenario = MicroscopeScenarioConfig(
    objective_spec="100x_1.4NA_oil",
    illumination_mode="phase",
    wavelength=532e-9
)

print(f"Resolution: {scenario.lateral_resolution_nm:.0f} nm")
# Output: Resolution: 232 nm

config = scenario.to_instrument_config()
microscope = Microscope(config)
```

---

### MicroscopeBuilder

```python
class MicroscopeBuilder
```

Fluent builder for microscope scenarios.

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `objective(spec)` | `str` | `self` | Set objective specification |
| `illumination(mode)` | `str` | `self` | Set illumination mode |
| `wavelength_nm(nm)` | `float` | `self` | Set wavelength in nm |
| `sensor_pixels(n, size)` | `int, float` | `self` | Set sensor configuration |
| `name(n)` | `str` | `self` | Set scenario name |
| `description(d)` | `str` | `self` | Set description |
| `build()` | - | `MicroscopeScenarioConfig` | Build the scenario |

**Example:**

```python
from prism.scenarios import MicroscopeBuilder

scenario = (MicroscopeBuilder()
    .objective("60x_1.2NA_water")
    .illumination("phase")
    .wavelength_nm(488)
    .sensor_pixels(2048, 6.5)
    .name("Live Cell Imaging")
    .build())
```

---

### LensSpec

```python
@dataclass
class LensSpec
```

Camera lens specification.

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `focal_length_mm` | `float` | Focal length in mm |
| `f_number` | `float` | F-number (aperture) |
| `aperture_diameter_mm` | `float` | Calculated aperture diameter |

**Class Methods:**

```python
@classmethod
def from_string(cls, spec: str) -> LensSpec
```

Parse lens from string format.

**Parameters:**
- `spec` (str): Specification string (e.g., "50mm_f2.8")

**Returns:**
- `LensSpec`: Parsed lens specification

**Example:**

```python
lens = LensSpec.from_string("50mm_f2.8")
assert lens.focal_length_mm == 50
assert lens.f_number == 2.8
assert lens.aperture_diameter_mm == 50 / 2.8
```

---

### SensorSpec

```python
@dataclass
class SensorSpec
```

Camera sensor specification.

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `name` | `str` | Sensor type name |
| `width_mm` | `float` | Sensor width (mm) |
| `height_mm` | `float` | Sensor height (mm) |
| `pixel_pitch_um` | `float` | Pixel pitch (µm) |
| `megapixels` | `float` | Calculated megapixels |

**Class Methods:**

```python
@classmethod
def from_name(cls, name: str) -> SensorSpec
```

Get sensor by type name.

**Parameters:**
- `name` (str): Sensor type ('full_frame', 'aps_c', '1_inch', 'm43')

**Returns:**
- `SensorSpec`: Sensor specification

**Raises:**
- `ValueError`: If sensor type unknown

**Available Types:**

| Name | Width | Height | Pixel Pitch |
|------|-------|--------|-------------|
| `full_frame` | 36 mm | 24 mm | 6.5 µm |
| `aps_c` | 23.5 mm | 15.6 mm | 3.9 µm |
| `1_inch` | 13.2 mm | 8.8 mm | 2.4 µm |
| `m43` | 17.3 mm | 13 mm | 3.75 µm |

---

### DroneScenarioConfig

```python
@dataclass
class DroneScenarioConfig(ScenarioConfig)
```

Drone camera scenario configuration.

**Attributes:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `lens_spec` | `str` or `LensSpec` | Required | Lens specification |
| `sensor_spec` | `str` or `SensorSpec` | `"full_frame"` | Sensor type |
| `altitude_m` | `float` | `50.0` | Flight altitude (m) |
| `ground_speed_mps` | `float` | `0.0` | Ground speed (m/s) |
| `wavelength` | `float` | `550e-9` | Operating wavelength (m) |
| `n_pixels` | `int` | `1024` | Image pixels |

**Computed Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `actual_gsd_cm` | `float` | Ground sampling distance (cm) |
| `swath_width_m` | `float` | Swath width (m) |
| `motion_blur_pixels` | `float` | Motion blur (pixels) |
| `fresnel_number` | `float` | Fresnel number for propagator selection |

**Methods:**

```python
def to_instrument_config(self) -> CameraConfig
```

Convert to PRISM CameraConfig.

**Returns:**
- `CameraConfig`: Configuration for PRISM Camera instrument

**Example:**

```python
from prism.scenarios import DroneScenarioConfig
from prism.core.instruments import Camera

scenario = DroneScenarioConfig(
    lens_spec="50mm_f4.0",
    sensor_spec="full_frame",
    altitude_m=100.0,
    ground_speed_mps=10.0
)

print(f"GSD: {scenario.actual_gsd_cm:.2f} cm")
print(f"Swath: {scenario.swath_width_m:.1f} m")

config = scenario.to_instrument_config()
camera = Camera(config)
```

---

### DroneBuilder

```python
class DroneBuilder
```

Fluent builder for drone camera scenarios.

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `lens(spec)` | `str` | `self` | Set lens specification |
| `sensor(name)` | `str` | `self` | Set sensor type |
| `altitude(m)` | `float` | `self` | Set altitude in meters |
| `ground_speed(mps)` | `float` | `self` | Set ground speed (m/s) |
| `wavelength_nm(nm)` | `float` | `self` | Set wavelength in nm |
| `name(n)` | `str` | `self` | Set scenario name |
| `description(d)` | `str` | `self` | Set description |
| `build()` | - | `DroneScenarioConfig` | Build the scenario |

**Example:**

```python
from prism.scenarios import DroneBuilder

scenario = (DroneBuilder()
    .lens("50mm_f2.8")
    .sensor("aps_c")
    .altitude(75)
    .ground_speed(12)
    .name("Site Survey")
    .build())
```

---

## Functions

### get_scenario_preset

```python
def get_scenario_preset(name: str) -> ScenarioConfig
```

Get scenario preset by name.

**Parameters:**
- `name` (str): Preset name (e.g., "microscope_100x_oil", "drone_50m_survey")

**Returns:**
- `ScenarioConfig`: Configured scenario instance

**Raises:**
- `ValueError`: If preset name not found

**Example:**

```python
scenario = get_scenario_preset("microscope_100x_oil")
print(f"Resolution: {scenario.lateral_resolution_nm:.0f} nm")
```

---

### list_scenario_presets

```python
def list_scenario_presets(category: Optional[str] = None) -> List[str]
```

List available scenario presets.

**Parameters:**
- `category` (str, optional): Filter by category ('microscope', 'drone')

**Returns:**
- `List[str]`: List of preset names

**Example:**

```python
# All presets
all_presets = list_scenario_presets()

# Microscope only
microscope_presets = list_scenario_presets("microscope")

# Drone only
drone_presets = list_scenario_presets("drone")
```

---

### get_preset_description

```python
def get_preset_description(name: str) -> str
```

Get description for a preset.

**Parameters:**
- `name` (str): Preset name

**Returns:**
- `str`: Description string (empty if not found)

**Example:**

```python
desc = get_preset_description("microscope_100x_oil")
print(desc)  # "High-resolution oil immersion for cellular detail"
```

---

### print_all_presets

```python
def print_all_presets() -> None
```

Print all available presets with descriptions.

Outputs a formatted table of all presets grouped by category.

---

## Available Presets

### Microscope Presets

| Name | Objective | Mode | Description |
|------|-----------|------|-------------|
| `microscope_10x_air` | 10x 0.25 NA | Brightfield | Low magnification overview |
| `microscope_20x_air` | 20x 0.4 NA | Brightfield | General observation |
| `microscope_40x_air` | 40x 0.9 NA | Brightfield | High-NA air objective |
| `microscope_60x_water` | 60x 1.2 NA | Brightfield | Water immersion |
| `microscope_100x_oil` | 100x 1.4 NA | Brightfield | Oil immersion |
| `microscope_phase_20x` | 20x 0.4 NA | Phase | Phase contrast |
| `microscope_phase_40x` | 40x 0.9 NA | Phase | High-NA phase |
| `microscope_dic_40x` | 40x 0.9 NA | DIC | DIC imaging |
| `microscope_darkfield_40x` | 40x 0.9 NA | Darkfield | Darkfield |

### Drone Presets

| Name | Lens | Altitude | Description |
|------|------|----------|-------------|
| `drone_10m_inspection` | 35mm f/2.8 | 10 m | Close inspection |
| `drone_30m_mapping` | 35mm f/2.8 | 30 m | Detailed mapping |
| `drone_50m_survey` | 50mm f/4.0 | 50 m | Site survey |
| `drone_80m_agriculture` | 50mm f/4.0 | 80 m | Agricultural |
| `drone_100m_overview` | 50mm f/4.0 | 100 m | Area overview |
| `drone_120m_corridor` | 85mm f/1.8 | 120 m | Corridor mapping |

---

## See Also

- [User Guide: Scenarios](../user_guides/scenarios.md) - Detailed usage guide
- [prism.core.instruments](instruments.md) - Instrument implementations
- [prism.core.targets](targets.md) - USAF-1951 test targets
