# PRISM User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
   - [Coordinate Conventions](#coordinate-conventions) ⭐ **Important**
5. [Basic Usage](#basic-usage)
6. [Advanced Usage](#advanced-usage)
7. [Configuration](#configuration)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

## Introduction

PRISM (Progressive Reconstruction from Imaging with Sparse Measurements) is a deep learning-based astronomical imaging system that reconstructs high-resolution images from sparse telescope aperture measurements using progressive neural network training.

### Key Features

- **Progressive Reconstruction**: Build images incrementally from sparse measurements
- **Realistic Physics**: Fraunhofer diffraction, shot noise, and propagation effects
- **Flexible Sampling**: Fermat spiral, random, or custom sampling patterns
- **Generative Modeling**: Decoder-only architecture learns from scratch
- **Professional Output**: Publication-ready visualizations and reports

### When to Use PRISM

PRISM is designed for:
- Astronomical imaging with sparse aperture arrays
- Super-resolution reconstruction beyond diffraction limits
- Progressive measurements where data arrives sequentially
- Situations where full aperture coverage is impossible

## Installation

### Prerequisites

- **Python 3.11 or higher** (Python 3.12 recommended)
- **uv package manager** ([Installation instructions](https://github.com/astral-sh/uv))
- **PyTorch 2.0 or higher** (automatically installed via uv sync)
- **CUDA** (optional, for GPU acceleration)

#### Installing uv

If you don't have uv installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/omri/PRISM.git
cd PRISM

# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate

# Verify installation
uv run python -c "import prism; print(prism.__version__)"
```

### Using pip

```bash
# Clone and install
git clone https://github.com/omri/PRISM.git
cd PRISM
pip install -e .

# Verify installation
python -c "import prism; print(prism.__version__)"
```

### Development Installation

For contributing to PRISM development:

```bash
# Install with development dependencies (included by default)
uv sync

# Run tests to verify setup
uv run pytest tests/ -v

# Check code quality
uv run black prism/
uv run ruff check prism/
uv run mypy prism/
```

See the main [README.md](../README.md) for detailed development setup and [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## Quick Start

### Minimal Example

```python
import torch
from prism.models.networks import ProgressiveDecoder
from prism.core.instruments import Telescope, TelescopeConfig
from prism.core.measurement_system import MeasurementSystem
from prism.models.losses import LossAgg
from prism.utils.image import load_image

# Load image
image = load_image("europa.jpg", size=1024)

# Create telescope and measurement system
config = TelescopeConfig(n_pixels=1024, aperture_radius_pixels=50, snr=40)
telescope = Telescope(config)
measurement_system = MeasurementSystem(telescope, obj_size=512)

# Create model
model = ProgressiveDecoder(input_size=1024, output_size=512)

# Create loss function
criterion = LossAgg(loss_type="l1")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Sample centers (Fermat spiral)
from prism.utils.sampling import fermat_spiral_sample
centers = fermat_spiral_sample(n_samples=100, r=300)

# Progressive training
for center in centers:
    # Get measurements
    measurement = measurement_system.measure(image, model(), [center], add_noise=True)

    # Train to match measurements
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model()
        loss_old, loss_new = criterion(output, measurement, measurement_system, [center])
        loss = loss_old + loss_new
        loss.backward()
        optimizer.step()

        if loss_old < 0.001 and loss_new < 0.001:
            break

    # Add measurement to accumulated mask
    measurement_system.add_measurement([center])

# Get final reconstruction
reconstruction = model()
```

### Command-Line Interface

```bash
# Basic reconstruction
uv run python main.py --obj_name europa --n_samples 100 --fermat

# Custom image
uv run python main.py --input_path myimage.jpg --image_size 1024 --n_samples 200

# Resume from checkpoint
uv run python main.py --checkpoint experiment_name --name resumed_run

# Debug mode (no saving)
uv run python main.py --obj_name europa --n_samples 64 --debug --max_epochs 1
```

## Core Concepts

### Progressive Imaging

PRISM builds reconstructions progressively by:

1. **Initial Sample**: Train on first measurement from scratch
2. **Progressive Samples**: For each new sample:
   - Measure ground truth through new aperture
   - Measure reconstruction through accumulated mask
   - Train to match both measurements
   - Add new aperture to accumulated mask

This ensures the reconstruction remains consistent with all previous measurements while incorporating new information.

### Key Components

#### 1. Telescope (`prism.core.instruments.Telescope`)

Simulates realistic telescope measurements:
- Fraunhofer diffraction (FFT to k-space)
- Circular, hexagonal, or obscured aperture masks
- Shot noise modeling
- Configurable propagators

```python
from prism.core.instruments import Telescope, TelescopeConfig

config = TelescopeConfig(
    n_pixels=1024,                   # Image size
    aperture_radius_pixels=50,       # Aperture radius
    snr=40,                          # Signal-to-noise ratio (dB)
)
telescope = Telescope(config)
```

#### 2. MeasurementSystem (`prism.core.measurement_system.MeasurementSystem`)

**Critical for PRISM**: Accumulates aperture masks across measurements.

```python
from prism.core.measurement_system import MeasurementSystem

measurement_system = MeasurementSystem(telescope, obj_size=512)

# First measurement
meas = measurement_system.measure(image, None, [[0, 0]])
measurement_system.add_measurement([[0, 0]])

# Later measurements return [old_mask_meas, new_meas]
meas = measurement_system.measure(image, reconstruction, [[10, 10]])
measurement_system.add_measurement([[10, 10]])
```

#### 3. ProgressiveDecoder (`prism.models.networks.ProgressiveDecoder`)

**Primary model**: Decoder-only generative network.

```python
model = ProgressiveDecoder(
    input_size=1024,     # Measurement size
    output_size=512,     # Object size
    use_bn=True,         # Batch normalization
    max_ch=256,          # Max channels
    complex_data=False   # Complex-valued output
)

# No input needed - generates from learned latent
output = model()
```

#### 4. LossAgg (`prism.models.losses.LossAgg`)

**Progressive loss**: Compares against both old and new measurements.

```python
criterion = LossAgg(loss_type="l1")

# Returns separate losses for dual convergence
loss_old, loss_new = criterion(output, measurement, telescope, centers)

# Both must converge for successful sample
if loss_old < threshold and loss_new < threshold:
    converged = True
```

### Physics Integration

PRISM incorporates realistic physics:

```python
from prism.config.objects import get_obj_params

# Get astronomical object parameters
params = get_obj_params("europa")
# Returns:
# - wavelength: Observation wavelength
# - diameter: Object diameter (m)
# - distance: Object distance (m)
# - obj_size: Angular size (pixels)
```

### Coordinate Conventions

**Important**: PRISM uses **centered FFT conventions** throughout. Understanding these conventions is critical for correctly interpreting results and implementing custom components.

#### Quick Summary

- **Spatial domain**: Origin (0,0) at center of array
- **Fourier domain**: DC component (k=0) at center of array
- **FFT operations**: Always use `fftshift`/`ifftshift` for proper centering
- **Normalization**: Orthonormal FFT (`norm='ortho'`) for energy conservation

#### Spatial and Fourier Domains

**Spatial domain** (object/image plane):
- Center pixel represents coordinates (0, 0)
- For 512×512 array: center is at index [256, 256]
- Coordinates range from -FOV/2 to +FOV/2

**Fourier domain** (k-space):
- DC component (zero frequency) at center
- Low frequencies near center, high frequencies at edges
- Nyquist frequency at array edges

#### Why This Matters

**Correct usage**:
```python
from prism.utils.transforms import fft, ifft

# ✅ Correct: Use centered FFT functions
k_space = fft(image)  # DC at center
reconstructed = ifft(k_space)  # Origin at center
```

**Incorrect usage**:
```python
# ❌ Wrong: Direct PyTorch FFT has DC at corner
k_space = torch.fft.fft2(image)  # DC at [0,0] corner - physically incorrect!
```

#### Energy Conservation

PRISM uses orthonormal FFT to ensure:
- Energy is conserved: `∑|field|² = ∑|k_field|²`
- Forward and inverse transforms are true inverses
- No scaling factors needed between domains

```python
import torch
from prism.utils.transforms import fft, ifft

# Create test field
field = torch.randn(512, 512, dtype=torch.cfloat)

# Energy in spatial domain
energy_spatial = (field.abs() ** 2).sum()

# Energy in Fourier domain
k_field = fft(field)
energy_fourier = (k_field.abs() ** 2).sum()

# These are equal (within numerical precision)
print(f"Energy conserved: {torch.allclose(energy_spatial, energy_fourier)}")
# Output: Energy conserved: True
```

#### For Advanced Users

For detailed information on coordinate systems, propagation conventions, and implementation details, see:
- [Implementation Guide: Propagators](implementation_guides/foundational_revision/02_propagators.md#coordinate-conventions)
- API documentation for `Grid`, `Propagator`, and `FFTCache` classes

## Basic Usage

### Loading Images

```python
from prism.utils.image import load_image

# Load from file
image = load_image("myimage.jpg", size=1024)

# Load with padding
image = load_image("myimage.jpg", size=512, padded_size=1024)

# Invert colors (for bright objects)
image = load_image("star.jpg", size=1024, invert=True)
```

### Sampling Patterns

```python
from prism.utils.sampling import (
    fermat_spiral_sample,
    random_sample,
    star_sample
)

# Fermat spiral (recommended - optimal coverage)
centers = fermat_spiral_sample(n_samples=100, r=300)

# Random sampling
centers = random_sample(n_samples=100, r=300)

# Star pattern (radial lines)
centers = star_sample(n_lines=6, samples_per_line=20, r=300)
```

### Saving and Loading

```python
from prism.utils.io import save_checkpoint, load_checkpoint

# Save checkpoint
save_checkpoint(
    path="experiment/checkpoint.pt",
    model=model,
    optimizer=optimizer,
    epoch=100,
    losses=[...],
    ssims=[...]
)

# Load checkpoint
checkpoint = load_checkpoint("experiment/checkpoint.pt")
model.load_state_dict(checkpoint['model_state_dict'])
reconstruction = checkpoint['current_rec']
```

### Visualization

```python
from prism.visualization.publication import PublicationPlotter

plotter = PublicationPlotter(figsize=(12, 4), dpi=300)

# Create comparison figure
fig = plotter.plot_reconstruction_comparison(
    ground_truth=image,
    reconstruction=reconstruction,
    measurement=measurement
)
fig.savefig("results/comparison.png")
```

## Advanced Usage

### Custom Sampling Patterns

```python
def custom_pattern(n_samples, r):
    """Create custom sampling pattern."""
    centers = []
    for i in range(n_samples):
        # Your pattern logic here
        angle = 2 * np.pi * i / n_samples
        radius = r * np.sqrt(i / n_samples)
        y = radius * np.sin(angle)
        x = radius * np.cos(angle)
        centers.append([y, x])
    return centers

centers = custom_pattern(n_samples=100, r=300)
```

### Configuration-Based Training

```python
from prism.config.base import ExperimentConfig, ImageConfig, TelescopeConfig, TrainingConfig
from prism.config.loader import load_config

# Load from YAML
config = load_config("configs/europa.yaml")

# Or create programmatically
config = ExperimentConfig(
    image=ImageConfig(
        input_path="europa.jpg",
        image_size=1024,
        obj_size=512
    ),
    telescope=TelescopeConfig(
        sample_diameter=100,
        sample_shape="circle",
        snr=40
    ),
    training=TrainingConfig(
        n_epochs=1000,
        lr=0.001,
        loss_type="l1",
        loss_threshold=0.001
    )
)
```

### Memory Management

```python
from prism.utils.memory import MemoryTracker, cleanup_tensors

# Track memory usage
tracker = MemoryTracker()

for epoch in range(n_epochs):
    output = model()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    tracker.log()

    # Cleanup after processing
    cleanup_tensors(output, loss)

# Analyze memory usage
tracker.plot_history()
```

### Parallel Processing

```python
from prism.training.parallel import setup_parallel_model

# Enable multi-GPU training
if torch.cuda.device_count() > 1:
    model = setup_parallel_model(model)
    print(f"Using {torch.cuda.device_count()} GPUs")
```

### Custom Loss Functions

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, output, measurement, telescope, centers):
        # Measure output through telescope
        output_meas = telescope(output, centers)

        # Combined loss
        loss_l1 = self.l1(output_meas[1], measurement[1])
        loss_l2 = self.l2(output_meas[0], measurement[0])

        return loss_l1, loss_l2
```

## Configuration

### Configuration Files

Create YAML configuration files for reproducible experiments:

```yaml
# configs/europa.yaml
name: europa_reconstruction

image:
  obj_name: europa
  image_size: 1024
  obj_size: 512
  invert: false
  crop: true

telescope:
  sample_diameter: 100
  sample_shape: circle
  roi_diameter: 600
  snr: 40

sampling:
  n_samples: 100
  pattern: fermat

training:
  n_epochs: 1000
  max_epochs: 25
  lr: 0.001
  loss_type: l1
  loss_threshold: 0.001

output:
  save_dir: runs
  save_checkpoints: true
  save_interval: 10
  create_report: true
```

Load and use:

```python
from prism.config.loader import load_config

config = load_config("configs/europa.yaml")
```

### Environment Variables

```bash
# Set device
export PRISM_DEVICE=cuda:0

# Set default output directory
export PRISM_OUTPUT_DIR=./results

# Enable debug mode
export PRISM_DEBUG=1
```

## Examples

### Example 1: Basic Reconstruction

```python
import torch
from prism.models.networks import ProgressiveDecoder
from prism.core.instruments import Telescope, TelescopeConfig
from prism.core.measurement_system import MeasurementSystem
from prism.models.losses import LossAgg
from prism.utils.image import load_image
from prism.utils.sampling import fermat_spiral_sample

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = load_image("europa.jpg", size=1024).to(device)

# Create components
config = TelescopeConfig(n_pixels=1024, aperture_radius_pixels=50, snr=40)
telescope = Telescope(config).to(device)
measurement_system = MeasurementSystem(telescope, obj_size=512).to(device)
model = ProgressiveDecoder(input_size=1024, output_size=512).to(device)
criterion = LossAgg(loss_type="l1")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Sample centers
centers = fermat_spiral_sample(n_samples=100, r=300)

# Training loop
for i, center in enumerate(centers):
    measurement = measurement_system.measure(image, model(), [center])

    for epoch in range(1000):
        optimizer.zero_grad()
        output = model()
        loss_old, loss_new = criterion(output, measurement, measurement_system, [center])
        loss = loss_old + loss_new
        loss.backward()
        optimizer.step()

        if loss_old < 0.001 and loss_new < 0.001:
            break

    measurement_system.add_measurement([center])
    print(f"Sample {i+1}/{len(centers)}: Loss={loss.item():.4f}")

# Save result
torch.save({
    'reconstruction': model(),
    'model_state': model.state_dict()
}, 'reconstruction.pt')
```

### Example 2: Configuration-Based Training

```python
from prism.config.loader import load_config
from prism.training.trainer import Trainer

# Load configuration
config = load_config("configs/europa.yaml")

# Create trainer
trainer = Trainer(config)

# Train
trainer.train()

# Get results
results = trainer.get_results()
print(f"Final SSIM: {results['ssim']:.4f}")
print(f"Final PSNR: {results['psnr']:.2f} dB")
```

### Example 3: Custom Object

```python
from prism.config.objects import register_object

# Register new object
register_object(
    name="custom_star",
    wavelength=550e-9,
    diameter=1e9,  # 1 million km
    distance=10 * 9.461e15,  # 10 light years
    obj_size=256
)

# Use in training
config = ExperimentConfig(
    image=ImageConfig(obj_name="custom_star", ...),
    ...
)
```

### Example 4: Progressive Visualization

```python
from prism.visualization.animation import TrainingAnimator

# Create animator
animator = TrainingAnimator("runs/experiment_name/checkpoints")

# Generate animation
animator.create_animation(
    output_path="training_progress.mp4",
    fps=10
)
```

## Realistic Optical Simulations

PRISM now includes a scenario configuration system for simulating real-world optical systems beyond astronomical telescopes.

### Using Scenario Presets

```python
from prism.scenarios import get_scenario_preset, list_scenario_presets

# List available presets
print(list_scenario_presets())
# ['microscope_10x_air', 'microscope_100x_oil', 'drone_50m_survey', ...]

# Load a microscope preset
scenario = get_scenario_preset("microscope_100x_oil")
print(f"Resolution: {scenario.lateral_resolution_nm:.0f} nm")

# Convert to PRISM instrument
from prism.core.instruments import Microscope
config = scenario.to_instrument_config()
microscope = Microscope(config)
```

### Available Optical Systems

| System | Presets | Use Case |
|--------|---------|----------|
| Microscope | 9 presets (10x-100x, air/water/oil) | Cellular imaging, phase contrast |
| Drone Camera | 8 presets (10m-120m altitude) | Aerial surveys, mapping |

### Using USAF-1951 Test Targets

For resolution validation, use the USAF-1951 target module:

```python
from prism.core.targets import create_target

# Create test target
target = create_target("usaf1951", size=512, groups=(0, 1, 2))
image = target.generate()

# Get resolution element info
print(target.resolution_elements)
# {'G0E1': '1.000 lp/mm', 'G0E2': '1.122 lp/mm', ...}
```

### CLI Integration

```bash
# List scenarios
python main.py --list-scenarios

# Run with a scenario
python main.py --scenario microscope_40x_air --name test

# Override parameters
python main.py --scenario drone_50m_survey --altitude 75
```

For detailed documentation, see:
- [Scenarios User Guide](user_guides/scenarios.md)
- [Scenarios API Reference](api/scenarios.md)

## Troubleshooting

### Common Issues

#### Out of Memory

```python
# Reduce batch size or image size
image = load_image("image.jpg", size=512)  # Instead of 1024

# Enable gradient checkpointing
torch.utils.checkpoint.checkpoint_sequential(...)

# Clear cache periodically
torch.cuda.empty_cache()
```

#### Poor Convergence

```python
# Increase max_epochs
max_epochs = 50  # Instead of 25

# Adjust learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Change loss threshold
loss_threshold = 0.01  # Instead of 0.001
```

#### NaN Loss

```python
# Check input normalization
image = image / image.max()

# Reduce learning rate
lr = 1e-4

# Enable gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Getting Help

- **Documentation**: [https://prism.readthedocs.io](https://prism.readthedocs.io)
- **Issues**: [https://github.com/omri/PRISM/issues](https://github.com/omri/PRISM/issues)
- **Discussions**: [https://github.com/omri/PRISM/discussions](https://github.com/omri/PRISM/discussions)

### Performance Tips

1. **Use GPU**: PRISM is 10-50x faster on GPU
2. **Fermat Spiral**: Better coverage than random sampling
3. **Checkpoint Frequently**: Save every 10-20 samples
4. **Monitor Memory**: Use MemoryTracker during development
5. **Parallel Training**: Use DataParallel for multi-GPU

## Next Steps

- Explore [API Reference](api_reference.md) for detailed function documentation
- See [Examples Gallery](examples/) for more use cases
- Read [Architecture Guide](architecture.md) to understand internals
- Check [Contributing Guide](CONTRIBUTING.md) to contribute

---

**Last Updated**: 2025-12-01
**Version**: 0.3.0
**License**: MIT
