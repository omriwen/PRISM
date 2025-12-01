# PRISM Codebase Exploration Report

## Executive Summary

PRISM (Progressive Reconstruction from Imaging with Sparse Measurements) is a sophisticated optical imaging framework combining deep learning with physics-based simulation. The codebase supports multiple imaging domains (astronomy telescopes, microscopy, drone cameras) with a unified, extensible architecture.

**Key Strengths:**
- Modular architecture separating physics simulation, neural networks, and application logic
- Comprehensive configuration system with presets and scenarios
- Rich set of visualization and analysis tools
- Well-documented with existing notebooks demonstrating patterns
- Production-quality code with type hints and testing

---

## PROJECT STRUCTURE

### Directory Organization
```
PRISM/
├── prism/                          # Main package
│   ├── core/                       # Core imaging physics & algorithms
│   │   ├── instruments/            # Optical instruments (Telescope, Microscope, Camera)
│   │   ├── optics/                 # Forward models and light propagation
│   │   ├── propagators/            # FFT, Angular Spectrum, Free Space propagators
│   │   ├── algorithms/             # Classical algorithms (ePIE, etc.)
│   │   ├── targets.py              # Test patterns (USAF-1951, Checkerboard, PointSource)
│   │   ├── patterns.py             # K-space sampling patterns (Fermat, Star, Random)
│   │   ├── pattern_library.py      # Pattern library management
│   │   ├── pattern_loader.py       # Load patterns from files
│   │   ├── apertures.py            # Aperture mask management (27KB)
│   │   ├── measurement_system.py   # Progressive measurement accumulation (25KB)
│   │   ├── convergence.py          # Convergence monitoring (23KB)
│   │   ├── aggregator.py           # Legacy measurement aggregation
│   │   ├── telescope.py            # Telescope simulation (legacy)
│   │   ├── trainers.py             # Training loops and convergence (46KB)
│   │   ├── runner.py               # Main experiment orchestrator (29KB)
│   │   ├── grid.py                 # Computational grid management
│   │   ├── illumination.py         # Illumination source models (22KB)
│   │   └── line_acquisition.py     # Motion blur simulation
│   ├── models/                     # Neural network architectures
│   │   ├── networks.py             # ProgressiveDecoder, LossAggregator (31KB)
│   │   ├── losses.py               # Loss functions (48KB)
│   │   └── layers.py               # Custom layers (21KB)
│   ├── config/                     # Configuration system
│   │   ├── base.py                 # Configuration dataclasses
│   │   ├── presets.py              # Experiment presets (Quick, Production, Debug, etc.)
│   │   ├── constants.py            # Physical constants
│   │   └── validation.py           # Configuration validation
│   ├── scenarios/                  # High-level scenario builders
│   │   ├── microscopy.py           # Microscope scenarios with preset objectives
│   │   ├── drone_camera.py         # Drone/aerial imaging scenarios
│   │   └── presets.py              # Scenario preset library
│   ├── catalog/                    # Component catalog system (NEW)
│   │   ├── components/             # Optical components library
│   │   ├── illumination/           # Illumination presets
│   │   ├── imaging/                # Imaging system presets
│   │   └── targets/                # Target catalog with generators
│   ├── analysis/                   # Result analysis utilities (NEW)
│   │   └── comparison.py           # Result comparison tools
│   ├── validation/                 # Runtime validation (NEW, distinct from tests/)
│   │   └── baselines.py            # Baseline comparisons (25KB)
│   ├── utils/                      # Utility functions
│   │   ├── transforms.py           # FFT/IFFT utilities
│   │   ├── metrics.py              # Image metrics (SSIM, PSNR, RMSE)
│   │   ├── sampling.py             # Sampling utilities
│   │   ├── validation_metrics.py   # Advanced validation metrics (26KB)
│   │   └── measurement_cache.py    # Performance caching
│   ├── visualization/              # Plotting and animation
│   │   ├── animation.py            # Training animation generation
│   │   ├── plotters/               # Specific plot types
│   │   ├── components/             # Reusable plot components
│   │   ├── style/                  # Matplotlib style definitions
│   │   └── config.py               # Visualization configuration
│   ├── cli/                        # Command-line interface
│   │   ├── inspect_pkg/            # Package inspection tools
│   │   └── patterns/               # Pattern CLI tools
│   ├── web/                        # Dashboard and web UI
│   │   ├── assets/                 # Static assets
│   │   ├── callbacks/              # Dash callbacks
│   │   └── layouts/                # Dash layout components
│   └── types.py                    # Custom type definitions (10KB)
├── examples/                       # Example notebooks and scripts
│   ├── notebooks/                  # Jupyter notebooks
│   ├── validation/                 # Validation notebooks
│   ├── python_api/                 # Python API examples
│   ├── baselines/                  # Comparison algorithms
│   ├── patterns/                   # Pattern examples
│   └── paper_figures/              # Publication figure generation
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── benchmarks/                 # Performance benchmarks
├── .memory/                        # Knowledge graph (Claude updates automatically)
│   ├── memory.jsonl                # MCP memory graph data
│   ├── .last_update                # Update metadata
│   ├── update_knowledge_graph.py   # Manual AST-based updater (optional)
│   ├── validate_graph.py           # Sync validation (pre-commit hook)
│   └── test_queries.py             # Graph validation tests
├── configs/                        # YAML configuration files
├── main.py                         # Main PRISM entry point
├── main_epie.py                    # ePIE baseline entry point
└── pyproject.toml                  # Project dependencies and configuration
```

---

## 1. OVERALL ARCHITECTURE & DESIGN PATTERNS

### Unified Instruments API
The system provides a unified interface for different optical instruments through inheritance and factory patterns:

```python
# All instruments share common interface
Instrument (Abstract Base)
├── Telescope (astronomical, far-field)
├── Microscope (near-field, high-NA)
└── Camera (general imaging, thin lens)

# Factory function for automatic type selection
instrument = create_instrument(config)  # Returns appropriate subclass
```

### Progressive Imaging Workflow
PRISM implements progressive neural network training:
1. **Initialization**: Train neural network on first aperture measurement
2. **Progressive Steps**: For each new aperture position:
   - Measure through accumulated previous apertures (old measurement)
   - Measure through new aperture (new measurement)
   - Train network to satisfy both measurement constraints
   - Add aperture to cumulative mask
3. **Convergence**: Monitor loss threshold and sample success rate

### Key Design Decisions
- **Separation of Concerns**: Physics (instruments) separate from algorithms (trainers, runners)
- **Lazy Initialization**: Grid and propagators created on first access
- **Configuration as Data**: All parameters in dataclasses for validation and reproducibility
- **Progressive Accumulation**: Cumulative mask builds synthetic aperture over iterations

---

## 2. EXISTING NOTEBOOKS - PATTERNS & STRUCTURE

### Notebook Organization

The notebooks follow a consistent pedagogical structure:

```
notebooks/
├── quickstart_* (30-60 min)        # Fast onboarding
│   ├── 01_microscopy_basic         # Basic microscope setup
│   ├── 02_drone_basic              # Drone imaging basics
│   └── 03_validation_intro         # Quality metrics intro
├── learning_* (2-3 hours)          # Deep understanding
│   ├── 01_resolution_fundamentals  # Abbe limit, PSF, NA
│   ├── 02_resolution_validation    # USAF testing
│   ├── 03_illumination_modes       # Brightfield/darkfield/phase/DIC
│   ├── 04_gsd_basics               # Ground Sampling Distance
│   ├── 05_drone_altitudes          # Altitude tradeoffs
│   └── 08_scenario_comparison      # Multi-scenario analysis
└── tutorial_* (2-3 hours)          # PRISM features
    ├── 01_quickstart               # PRISM basics
    ├── 02_pattern_design           # K-space sampling
    ├── 03_result_analysis          # Experiment analysis
    ├── 04_dashboard                # Web monitoring
    └── 05_reporting                # Publication figures
```

### Notebook Code Pattern

Standard setup cell pattern:
```python
# Imports (matplotlib, numpy, torch, prism)
import matplotlib.pyplot as plt
import numpy as np
import torch

from prism.config.constants import um
from prism.core import create_usaf_target
from prism.core.instruments import create_instrument
from prism.scenarios import get_scenario_preset

# Setup (plotting, device, logging)
plt.ion()  # Interactive mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load preset/scenario
scenario = get_scenario_preset("microscope_40x_0.9NA_air")
config = scenario.to_instrument_config()
microscope = create_instrument(config)

# Generate target
target = create_usaf_target(field_size=100*um, resolution=512)
image = target.generate()

# Simulate measurement
measurement = microscope.forward(image)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(image, cmap='gray')
axes[1].imshow(measurement, cmap='gray')
```

### Key Patterns in Notebooks

1. **Scenario-Driven Setup**: Load predefined scenario → derive parameters
2. **Physics Validation**: Compare theoretical vs simulated metrics
3. **Progressive Demonstration**: Show concepts building from simple to complex
4. **Visualization-Heavy**: Multiple plots, interactive widgets where available
5. **Metric Reporting**: Show quantitative validation (resolution, SNR, etc.)

---

## 3. KEY MODULES FOR MICROSCOPY SIMULATION

### Instruments Module (`prism/core/instruments/`)

#### MicroscopeConfig - Configuration dataclass
```python
@dataclass
class MicroscopeConfig(InstrumentConfig):
    numerical_aperture: float = 0.9
    magnification: float = 40.0
    medium_index: float = 1.0  # 1.0=air, 1.33=water, 1.515=oil
    tube_lens_focal: float = 0.2  # 200mm standard
    working_distance: Optional[float] = None  # Defaults to focal plane
    forward_model_regime: str = "simplified"  # 'simplified', 'full', or 'auto'
    defocus_threshold: float = 0.01
    padding_factor: float = 2.0  # FFT padding for wraparound prevention
    wavelength: float = 550e-9  # Green light
    n_pixels: int = 1024
    pixel_size: float = 6.5e-6
```

#### Microscope Class - Imaging simulation
```python
class Microscope(Instrument):
    """Supports multiple illumination modes:
    - Brightfield: Direct transmitted/reflected light
    - Darkfield: Only scattered light (direct blocked)
    - Phase contrast: Phase shifts → intensity
    - DIC: Differential interference contrast
    - Custom: User-defined pupils
    """

    def forward(
        self,
        field: torch.Tensor,
        illumination_mode: Optional[str] = None,
        illumination_params: Optional[dict] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward propagation with selectable illumination"""
```

#### Key Properties
- **resolution_limit**: Abbe diffraction limit (0.61λ/NA)
- **objective_focal**: Computed from magnification (f_obj = f_tube / M)
- **forward_model**: Lazy-loaded MicroscopeForwardModel with regime selection

### Optics Module (`prism/core/optics/`)

#### Input Handling (`input_handling.py`)
- Automatic detection of input format (intensity, amplitude, complex)
- FOV consistency validation
- Field preparation and normalization

#### Thin Lens Calculation (`thin_lens.py`)
- Image distance from lens equation
- Magnification and depth of field
- Circle of confusion for defocus

#### Microscope Forward Model (`microscope_forward.py`)
- **SIMPLIFIED regime**: FFT-based 4f optical system (correct for in-focus)
- **FULL regime**: Propagation with lens phases (for defocus/z-stacks)
- **AUTO regime**: Selects based on defocus threshold

### Propagators (`prism/core/propagators/`)

Three propagation engines:
1. **AngularSpectrumPropagator**: High-accuracy near-field (best for microscopy)
2. **FreeSpacePropagator**: Far-field Fraunhofer diffraction (telescopes)
3. **FresselPropagator**: Intermediate Fresnel propagation

Selection based on Fresnel number: F = a²/(λz)
- F >> 1: Fraunhofer (far-field)
- F ~ 1: Fresnel (intermediate)
- F << 1: Angular spectrum (near-field)

---

## 4. CONFIGURATION SYSTEM & PARAMETERS

### Configuration Hierarchy

```
CLI Arguments
    ↓ (highest priority)
Config File (YAML)
    ↓
Preset Dictionary
    ↓
Scenario Preset
    ↓ (lowest priority)
Defaults in Dataclass
```

### Configuration Files (`configs/`)

Standard YAML format:
```yaml
# configs/default.yaml
microscope:
  objective: "40x_0.9NA_air"
  wavelength_nm: 550
  magnification: 40

measurement:
  n_samples: 100
  pattern: "fermat"
  sample_diameter_pix: 50

training:
  max_epochs: 25
  learning_rate: 0.001
  loss_threshold: 0.001
```

### Presets (`prism/config/presets.py`)

Predefined configurations for common workflows:
- **QUICK_TEST**: Fast iteration (64 samples, 1 epoch)
- **PRODUCTION**: High quality (200 samples, 25 epochs)
- **DEBUG**: Minimal setup (16 samples, 1 epoch)
- **LINE_SAMPLING**: Extended aperture measurements

### Scenarios (`prism/scenarios/`)

User-friendly builders for real-world setups:
```python
# Microscopy presets (e.g., "microscope_40x_0.9NA_air")
scenario = get_scenario_preset("microscope_100x_1.4NA_oil")
print(f"Resolution: {scenario.lateral_resolution_nm:.0f} nm")

# Custom configuration
config = MicroscopeScenarioConfig(
    objective_spec="40x_0.9NA_air",
    illumination_mode="phase",
    wavelength=532e-9
)

# Drone presets (e.g., "drone_dji_air2s_60m")
scenario = get_scenario_preset("drone_phantom4_120m")
print(f"GSD: {scenario.gsd_cm:.1f} cm/pixel")
```

**Available Scenario Presets**: Access via `list_scenario_presets()`, `get_preset_description()`

---

## 5. MICROSCOPE OBJECTIVES - LOW MAGNIFICATION SETUP

### Objective Specification System

Objectives specified as strings: `"<mag>x_<NA>NA_<medium>"`

Examples:
- `"4x_0.1NA_air"` - Low power, air
- `"10x_0.25NA_air"` - Standard low power
- `"20x_0.4NA_air"` - Medium low power
- `"40x_0.9NA_air"` - Standard high power
- `"100x_1.4NA_oil"` - Ultra high NA, oil immersion

### Auto-Computed Parameters

From objective spec:
1. **Magnification**: Total magnification (M)
2. **Numerical Aperture**: Light-gathering ability (NA)
3. **Medium Index**: 1.0 (air), 1.33 (water), 1.515 (oil)
4. **Objective Focal Length**: f_obj = f_tube / M
5. **Abbe Resolution**: λ/(2·NA)
6. **Axial Resolution**: λ/(2·n·NA²)
7. **Field of View**: Sensor FOV / Magnification

### Low Magnification (4-10x)

**Characteristics:**
- Lower NA → worse lateral resolution
- Larger field of view
- Deeper depth of field
- Less demanding sampling

**Example Setup:**
```python
from prism.scenarios import get_scenario_preset

# 10x air objective - high depth of field
scenario = get_scenario_preset("microscope_10x_0.25NA_air")
print(f"Resolution: {scenario.lateral_resolution_nm:.0f} nm")  # ~1100 nm
print(f"DOF: {scenario.axial_resolution_um:.1f} µm")           # ~7.2 µm
print(f"FOV: {scenario.field_of_view_um:.0f} µm")              # ~1700 µm
```

### Working Distance Control

Default: Working distance = objective focal plane

For custom working distance (defocus, z-stacks):
```python
config = MicroscopeConfig(
    numerical_aperture=0.9,
    magnification=40,
    working_distance=5e-3,  # 5mm (typically 0.66mm focal)
    forward_model_regime="full"  # Use full propagation for defocus
)
```

---

## 6. PRISM WORKFLOW - HOW IT ALL WORKS TOGETHER

### Complete Progressive Imaging Workflow

```
1. SETUP PHASE
   └─ Load scenario/config → Create instrument → Create measurement system

2. MEASUREMENT SYSTEM INITIALIZATION
   └─ MeasurementSystem wraps instrument
      └─ Tracks cumulative aperture mask
      └─ Caches measurements (15-25% speedup)

3. TARGET & SAMPLING GENERATION
   └─ Generate test target (USAF, synthetic, natural image)
   └─ Generate aperture sampling pattern (Fermat spiral, star, random)

4. PROGRESSIVE TRAINING LOOP (Main Algorithm)
   For each aperture position:
   ├─ OLD MEASUREMENT:
   │  └─ Current reconstruction through accumulated mask → m_old
   ├─ NEW MEASUREMENT:
   │  └─ Ground truth through new aperture → m_new
   ├─ NEURAL NETWORK:
   │  └─ Forward pass: prediction = model(latent_vector)
   ├─ LOSS COMPUTATION:
   │  └─ loss = LossAggregator(prediction, [m_old, m_new])
   ├─ BACKPROPAGATION:
   │  └─ Gradient step updates model weights
   └─ ACCUMULATION:
      └─ Add new aperture to cumulative mask
      └─ Update measurement cache

5. CONVERGENCE & OUTPUT
   └─ Monitor loss threshold (0.001 default)
   └─ Track sample success rate
   └─ Save: checkpoint, metrics, synthetic aperture visualization
```

### Key Algorithm Components

**MeasurementSystem** (orchestrates measurements):
```python
# Create system
measurement_system = MeasurementSystem(microscope)

# First measurement (no previous data)
m_new = measurement_system.measure(ground_truth, None, [[0, 0]])
measurement_system.add_mask([[0, 0]])

# Subsequent measurements (with accumulated mask)
m_old = measurement_system.measure_through_accumulated_mask(reconstruction)
m_new = measurement_system.measure(ground_truth, None, [[10, 10]])
loss = loss_fn(prediction, [m_old, m_new])
measurement_system.add_mask([[10, 10]])
```

**Synthetic Aperture Construction**:
- Starts empty (no measurements)
- Each aperture adds binary mask to cumulative result
- Final mask covers regions measured throughout training
- Visualization shows k-space coverage

**Progressive Training Benefits**:
- Early samples contain coarse information (full image structure)
- Later samples refine details
- Network learns to match ALL measurements simultaneously
- Ensures consistency across different aperture positions

---

## 7. RESOLUTION TARGETS & VALIDATION

### USAF-1951 Resolution Chart

Standard test pattern for optical system validation:
- Groups numbered -2 to 9 (common range: 0-8)
- 6 elements per group
- Spatial frequency: f(g,e) = 2^(g + (e-1)/6) lp/mm
- Bar width: 1/(2f) mm

#### Target Generation API
```python
from prism.core import create_usaf_target, USAF1951Config
from prism.config.constants import um

# Physical parameters (preferred)
target = create_usaf_target(
    field_size=100 * um,      # 100 µm field
    resolution=512,
    groups=(4, 5, 6, 7),
    margin_ratio=0.25          # 25% margin each side
)
image = target.generate()

# Low-level configuration
config = USAF1951Config(
    size=1024,
    field_size=None,           # Can be None for pixel-based sizing
    groups=(0, 1, 2, 3, 4),
    margin_ratio=0.25
)
target = USAF1951Target(config)

# Frequency information
freq = target.get_frequency_lp_mm(group=4, element=3)  # lp/mm
bar_width_m = target.get_bar_width_m(group=4, element=3)  # meters
bar_width_pix = target.get_bar_width_pixels(group=4, element=3)  # pixels
```

### Other Resolution Targets

```python
from prism.core import create_checkerboard_target, PointSourceTarget

# Checkerboard pattern (uniform spatial frequency)
checkerboard = create_checkerboard_target(
    field_size=100 * um,
    square_size=5 * um,
    resolution=512
)

# Point source (PSF characterization)
psf_target = PointSourceTarget(
    PointSourceConfig(size=512, center=(256, 256))
)
```

### Margin System (Critical!)

- **margin_ratio**: Fraction of image on each side left as zero padding
- **Default**: 0.25 (25% margin = 50% active area)
- **Why**: Prevents periodic boundary artifacts in FFT-based propagation
- **Recommendation**: >= 0.25 for diffraction simulation, 0.35+ for critical work

---

## 8. SYNTHETIC APERTURE CONSTRUCTION

### How Progressive Measurements Build Synthetic Aperture

```python
# Initialize empty cumulative mask
cum_mask = torch.zeros((n_pixels, n_pixels), dtype=torch.bool)

# For each aperture position [y, x]
for center in aperture_centers:
    # Create binary mask for circular aperture
    y, x = center
    r = np.sqrt((yy - y)**2 + (xx - x)**2)
    aperture_mask = r <= aperture_radius

    # Accumulate into cumulative mask (union)
    cum_mask = cum_mask | aperture_mask

# Final cumulative mask shows total coverage region
# Visualization shows k-space density distribution
```

### Measurement Integration

Each progressive step:
1. **Measure through accumulated mask**: Previous apertures + current prediction
   - Ensures consistency with all prior constraints
2. **Measure through new aperture**: Only new aperture + ground truth
   - Incorporates new information
3. **Loss combines both**: Network matches both sets of data

### Visualization

Standard output plot: `synthetic_aperture.png`
- Shows cumulative k-space coverage
- Density indicates measurement frequency
- Fermat spiral pattern typically shows well-distributed coverage

---

## 9. CODE PATTERNS FROM EXISTING NOTEBOOKS

### Import Pattern (Standard)
```python
import matplotlib.pyplot as plt
import numpy as np
import torch

from prism.config.constants import um
from prism.core import create_usaf_target
from prism.core.instruments import create_instrument, MicroscopeConfig
from prism.scenarios import get_scenario_preset
```

### Setup Pattern
```python
# Device management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Plotting configuration
plt.ion()  # Interactive mode for Jupyter
fig_size = (12, 8)

# Constants for units
wavelength_nm = 550
wavelength = wavelength_nm * 1e-9
```

### Scenario Loading Pattern
```python
# Via scenario preset (recommended for users)
scenario = get_scenario_preset("microscope_40x_0.9NA_air")
config = scenario.to_instrument_config()

# Via direct configuration (for customization)
config = MicroscopeConfig(
    numerical_aperture=0.9,
    magnification=40,
    medium_index=1.0,
    wavelength=550e-9
)

# Create instrument
microscope = create_instrument(config)
print(f"Resolution: {microscope.resolution_limit * 1e9:.1f} nm")
```

### Target & Measurement Pattern
```python
# Generate standard resolution test
target = create_usaf_target(
    field_size=100 * um,
    resolution=512,
    groups=(4, 5, 6)
)
object_field = target.generate()

# Validate field characteristics
info = microscope.validate_field(
    object_field,
    input_mode='intensity'
)

# Simulate measurement
measurement = microscope.forward(
    object_field,
    illumination_mode="brightfield"
)
```

### Visualization Pattern
```python
# Multi-panel comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(object_field.cpu().numpy(), cmap='gray')
axes[0].set_title('Ground Truth')
axes[0].axis('off')

axes[1].imshow(measurement.cpu().numpy(), cmap='gray')
axes[1].set_title('Measurement')
axes[1].axis('off')

axes[2].imshow(np.abs(fft_result), cmap='viridis')
axes[2].set_title('K-space (log scale)')
axes[2].set_yscale('log')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

### Metrics & Validation Pattern
```python
from prism.utils.metrics import compute_ssim, compute_psnr, compute_rmse

# Compare reconstruction to ground truth
ssim = compute_ssim(reconstruction, ground_truth)
psnr = compute_psnr(reconstruction, ground_truth)
rmse = compute_rmse(reconstruction, ground_truth)

print(f"SSIM: {ssim:.4f} (higher is better, max 1.0)")
print(f"PSNR: {psnr:.2f} dB (higher is better)")
print(f"RMSE: {rmse:.4f} (lower is better)")
```

### Interactive Widget Pattern (where available)
```python
# Using ipywidgets if available
try:
    import ipywidgets as widgets

    wavelength_slider = widgets.FloatSlider(
        value=550e-9,
        min=400e-9,
        max=700e-9,
        step=10e-9,
        description='Wavelength (m)'
    )

    def update_simulation(wavelength):
        microscope.config.wavelength = wavelength
        # Re-run simulation

    widgets.interact(update_simulation, wavelength=wavelength_slider)
except ImportError:
    print("ipywidgets not available - using manual parameter setting")
```

---

## 10. KEY FILE LOCATIONS & QUICK REFERENCE

### Core APIs (What to Import)

```python
# Instruments (primary)
from prism.core.instruments import (
    Telescope, TelescopeConfig,
    Microscope, MicroscopeConfig,
    Camera, CameraConfig,
    create_instrument
)

# Measurement system (PRISM-specific)
from prism.core import MeasurementSystem, MeasurementSystemConfig

# Test targets (resolution validation)
from prism.core import (
    create_usaf_target, USAF1951Target,
    create_checkerboard_target,
    PointSourceTarget
)

# Sampling patterns (aperture positions)
from prism.core import (
    generate_fermat_spiral,
    generate_star_pattern,
    create_pattern
)

# Scenarios (user-friendly presets)
from prism.scenarios import (
    get_scenario_preset,
    list_scenario_presets,
    MicroscopeScenarioConfig,
    DroneScenarioConfig
)

# Configuration & constants
from prism.config import load_config
from prism.config.constants import um, nm, wavelength_green

# Metrics & visualization
from prism.utils.metrics import compute_ssim, compute_psnr
from prism.visualization import plot_reconstruction
```

### File Navigation

| Task | File |
|------|------|
| Microscope config details | `/prism/core/instruments/microscope.py` |
| USAF target generation | `/prism/core/targets.py` (USAF1951Target class) |
| Objective specs | `/prism/scenarios/microscopy.py` (ObjectiveSpec) |
| Sampling patterns | `/prism/core/patterns.py` |
| Measurement accumulation | `/prism/core/measurement_system.py` |
| Forward models | `/prism/core/optics/microscope_forward.py` |
| Illumination modes | `/prism/core/illumination.py` |
| Constants (µm, nm) | `/prism/config/constants.py` |

### Configuration Files

| Preset | File | Use Case |
|--------|------|----------|
| Quick test | `configs/quick_test.yaml` | Fast iteration |
| Production | `configs/production_europa.yaml` | High quality |
| Debug | Via preset system | Minimal setup |
| Custom | Create new YAML or use Python | Domain-specific |

---

## SUMMARY: CREATING NEW NOTEBOOKS

When creating a new notebook, follow this pattern:

1. **Title & Objectives**: Markdown cell with learning goals
2. **Setup Cell**: Import standard modules, set device, enable interactive mode
3. **Load Scenario/Config**: Use `get_scenario_preset()` or custom config
4. **Create Instrument**: `create_instrument(config)`
5. **Generate Target**: `create_usaf_target()` or custom pattern
6. **Simulation**: `instrument.forward(target, illumination_mode=...)`
7. **Visualization**: Multi-panel plots with proper labeling
8. **Metrics**: Print quantitative validation results
9. **Discussion**: Explain physics and results
10. **Save Outputs**: If needed, save figures/data for analysis

Key principles:
- **Pedagogical**: Build understanding progressively
- **Quantitative**: Show metrics, not just plots
- **Reproducible**: Use fixed random seeds, specify all parameters
- **Physical**: Validate against theoretical expectations
- **Practical**: Show real-world usage patterns
