# PRISM - Progressive Reconstruction from Imaging with Sparse Measurements

PRISM (Progressive Reconstruction from Imaging with Sparse Measurements) is a deep learning-based astronomical imaging system that reconstructs high-resolution images from sparse telescope aperture measurements using progressive neural network training.

## Installation

### Prerequisites

- **Python 3.11 or higher** (Python 3.12 recommended)
- **uv package manager** - [Installation instructions](https://github.com/astral-sh/uv)
- **CUDA-capable GPU** (optional, recommended for larger experiments)

### Quick Install

The project comes with a pre-configured virtual environment and dependency specification.

```bash
# 1. Clone the repository
git clone https://github.com/omri/PRISM.git
cd PRISM

# 2. Install dependencies (production + development)
uv sync

# This will:
# - Create/activate virtual environment in .venv/
# - Install all dependencies from uv.lock
# - Install PRISM package in editable mode

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import prism; print(f'PRISM version: {prism.__version__}')"
# Should output: PRISM version: 0.3.0

uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Shows PyTorch version and GPU availability
```

### Production-Only Installation

If you only need to run experiments (not develop the package):

```bash
# Install without development dependencies
uv sync --no-dev

# This skips pytest, mypy, black, ruff, jupyter, etc.
```

### Alternative: Traditional pip Installation

If you prefer using pip:

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install from source
pip install -e .

# For development dependencies, generate requirements file:
uv export --no-hashes > requirements-dev.txt
pip install -r requirements-dev.txt
```

### Verifying Installation

Run a quick test to ensure everything is working:

```bash
# Quick smoke test (should complete in ~2 minutes)
uv run python main.py \
    --obj_name europa \
    --n_samples 64 \
    --sample_length 64 \
    --samples_per_line_meas 9 \
    --max_epochs 1 \
    --fermat \
    --debug \
    --name installation_test

# Check output
ls runs/installation_test/
# Should contain: checkpoint.pt, args.txt, sample_points.pt
```

### Troubleshooting Installation

**CUDA not available**:
```bash
# Check PyTorch installation
python -c "import torch; print(torch.version.cuda)"

# If None, reinstall PyTorch with CUDA support
# See: https://pytorch.org/get-started/locally/
```

**Import errors**:
```bash
# Ensure you're in the virtual environment
which python
# Should show: /path/to/PRISM/.venv/bin/python

# Re-run sync if needed
uv sync --reinstall
```

**Slow installation**:
```bash
# uv should be fast, but if slow, check:
# - Internet connection
# - Disk space (need ~2GB for all dependencies)
# - Try with cached packages: uv sync --offline (if previously installed)
```

## Development Setup

If you want to contribute to PRISM development:

```bash
# 1. Fork and clone the repository
git clone https://github.com/omri/PRISM.git
cd PRISM

# 2. Install with development dependencies
uv sync  # Includes pytest, mypy, black, ruff, jupyter, etc.

# 3. Run tests to verify setup
uv run pytest tests/ -v

# 4. Check code quality
uv run black prism/
uv run ruff check prism/
uv run mypy prism/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

## Usage

### Quick Start

Run the main PRISM algorithm with Europa test:
```bash
uv run python main.py --obj_name europa --n_samples 100 --fermat --name test_run
```

For quick testing (sparse sampling for fast iteration):
```bash
uv run python main.py --obj_name europa --n_samples 64 --sample_length 64 --samples_per_line_meas 9 --max_epochs 1 --fermat --debug --name quick_test
```

### Alternative ePIE Algorithm

For comparison with traditional phase retrieval:
```bash
uv run python main_epie.py --obj_name europa --n_samples 100 --name epie_baseline
```

### Monitoring Training

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir runs/
```

## UI/UX Tools & Commands (v0.3.0)

PRISM includes a comprehensive suite of interactive tools for experiment analysis, comparison, and visualization.

### Experiment Comparison

Compare multiple experiments side-by-side:

```bash
# Compare two experiments
prism compare runs/exp1 runs/exp2

# Compare multiple experiments with custom output
prism compare runs/exp_* --output comparison_report.png

# Compare specific metrics only
prism compare runs/exp1 runs/exp2 --metrics loss ssim psnr

# Show configuration differences
prism compare runs/exp1 runs/exp2 --show-config-diff
```

### Checkpoint Inspector

Interactively explore experiment checkpoints:

```bash
# Inspect a checkpoint (summary view)
prism inspect runs/experiment/checkpoint.pt

# Interactive mode with menu navigation
prism inspect runs/experiment/checkpoint.pt --interactive

# Show only metrics
prism inspect runs/experiment/checkpoint.pt --metrics-only

# Export reconstruction as image
prism inspect runs/experiment/checkpoint.pt --export-reconstruction
```

### Web Dashboard

Launch an interactive web dashboard for real-time monitoring and multi-experiment comparison:

```bash
# Launch standalone dashboard
prism dashboard --port 8050

# Dashboard automatically available at http://localhost:8050
# Features:
# - Real-time training monitoring
# - Interactive Plotly visualizations (zoom, pan, hover)
# - Multi-experiment comparison
# - K-space coverage visualization
# - Configuration comparison
```

Integrate with training:
```bash
# Launch dashboard alongside training
python main.py --obj_name europa --n_samples 100 --dashboard
```

### Training Animations

Generate MP4 or GIF animations showing training progression:

```bash
# Generate MP4 animation
prism animate runs/experiment --output training.mp4

# Customize animation parameters
prism animate runs/experiment --fps 10 --format gif --duration 10

# Side-by-side comparison animation
prism animate runs/exp1 runs/exp2 --side-by-side --output comparison.mp4
```

### Automatic Report Generation

Generate comprehensive HTML or PDF reports:

```bash
# Generate HTML report
prism report runs/experiment --format html

# Generate PDF report (publication-ready)
prism report runs/experiment --format pdf --output report.pdf

# Multi-experiment comparison report
prism report runs/exp1 runs/exp2 runs/exp3 --format pdf --output comparison.pdf

# Custom template
prism report runs/experiment --template custom_template.html
```

Reports include:
- Executive summary with key metrics
- Configuration details
- Training curves and visualizations
- K-space coverage analysis
- High-DPI figures (300 DPI for publications)

### Sampling Pattern Library

Browse and compare k-space sampling patterns:

```bash
# List all available patterns
prism patterns list

# Visualize a specific pattern
prism patterns show fermat --n-samples 100 --output fermat.png

# Compare multiple patterns
prism patterns compare fermat random star --n-samples 100

# Show pattern statistics
prism patterns stats fermat --n-samples 100

# Generate interactive pattern gallery
prism patterns gallery --output pattern_gallery.html
```

## Examples

The `examples/` directory contains reference implementations and reproducibility resources:

### Baseline Algorithms

**`examples/baselines/`** - Comparison algorithms for scientific validation

Compare PRISM with traditional phase retrieval methods:

```bash
cd examples/baselines

# Run ePIE baseline
uv run python epie_baseline.py --obj_name europa --n_samples 100 --fermat --name epie_test

# Compare with PRISM
cd ../..
uv run python main.py --obj_name europa --n_samples 100 --fermat --name prism_test

# Compare results in runs/ directory
```

See [examples/baselines/README.md](examples/baselines/README.md) for details on:
- ePIE (Extended Ptychographic Iterative Engine)
- Algorithm comparison methodology
- Performance metrics and benchmarking

### Paper Figures

**`examples/paper_figures/`** - Reproduce publication figures

Generate figures from PRISM publications:

```bash
cd examples/paper_figures

# Generate all figures
uv run python generate_figures.py --output_dir ./output

# Generate specific figure
uv run python generate_figures.py --figure 1 --output_dir ./output
```

See [examples/paper_figures/README.md](examples/paper_figures/README.md) for:
- Figure specifications and requirements
- Required experiment data ([required_data.txt](examples/paper_figures/required_data.txt))
- Reproducibility instructions

### Pattern Examples

**`examples/patterns/`** - Sampling pattern demonstrations

Explore different k-space sampling strategies for telescope measurements.

## Key Features

- **ProgressiveDecoder**: Primary generative model with decoder-only architecture
- **Progressive Training**: Iterative improvement using telescopic measurements
- **Realistic Physics**: Fraunhofer diffraction simulation with noise modeling
- **Multiple Sampling Patterns**: Fermat spiral, star, and random sampling
- **Astronomical Objects**: Predefined physics parameters for Europa, Titan, Betelgeuse, Neptune
- **Interactive UI/UX Tools** (v0.3.0):
  - Web dashboard for real-time monitoring and multi-experiment comparison
  - Experiment comparison and checkpoint inspection CLI tools
  - Automated report generation (HTML/PDF) for publications
  - Training animation generation (MP4/GIF)
  - Pattern library browser with interactive visualization
  - Enhanced error messages with intelligent suggestions

## Multi-Instrument Support

PRISM now supports multiple optical instrument types beyond telescopes, enabling multi-scale optical simulations from microscopy to astronomy.

### Supported Instruments

#### Telescopes (Astronomical Imaging)
- Far-field (Fraunhofer) propagation for astronomical objects
- Large apertures (meters scale)
- Angular resolution following Rayleigh criterion (1.22λ/D)
- Support for various aperture types (circular, hexagonal, segmented)
- SNR-based noise modeling

#### Microscopes (Near-field Imaging)
- High numerical aperture (NA) optical systems
- Angular spectrum propagation for high accuracy
- Multiple illumination modes:
  - Brightfield: Standard transmission microscopy
  - Darkfield: Enhanced contrast for transparent specimens
  - Phase contrast: Visualize phase objects
  - DIC (Differential Interference Contrast): 3D-like relief contrast
- 3D PSF computation for volumetric imaging
- Resolution following Abbe limit (0.61λ/NA)
- Support for oil immersion (NA > 1.0) and water immersion objectives

#### Cameras (General Imaging)
- Automatic propagation regime selection based on Fresnel number
- Thin lens equation calculations:
  - Image distance and magnification
  - Depth of field (DOF) computation
  - Circle of confusion modeling
- Defocus aberration modeling
- Realistic sensor noise:
  - Shot noise (Poisson statistics)
  - Read noise (Gaussian)
  - Dark current
- Support for various focal lengths (wide-angle to telephoto)

### Usage Examples

#### Using the Factory Function
```python
from prism.core.instruments import create_instrument, MicroscopeConfig, CameraConfig

# Create a high-NA oil immersion microscope
microscope_config = MicroscopeConfig(
    numerical_aperture=1.4,
    magnification=100,
    wavelength=532e-9,      # Green laser
    medium_index=1.515,     # Oil immersion
    n_pixels=512,
    pixel_size=6.5e-6
)
microscope = create_instrument(microscope_config)
print(f"Resolution: {microscope.resolution_limit * 1e9:.1f} nm")

# Create a standard 50mm camera
camera_config = CameraConfig(
    focal_length=50e-3,     # 50mm
    f_number=1.4,           # f/1.4 fast lens
    sensor_size=(36e-3, 24e-3),  # Full frame
    object_distance=2.0,    # 2 meters
    wavelength=550e-9,
    n_pixels=512
)
camera = create_instrument(camera_config)
near, far = camera.calculate_depth_of_field()
print(f"DOF: {near:.2f}m to {far:.2f}m")
```

#### Dynamic Illumination Modes (Microscopy)
```python
# Use different illumination modes with the same microscope
brightfield = microscope.forward(sample, illumination_mode="brightfield")
darkfield = microscope.forward(sample, illumination_mode="darkfield")
phase_contrast = microscope.forward(sample, illumination_mode="phase")
dic_image = microscope.forward(sample, illumination_mode="dic")
```

#### PSF Computation
```python
# 2D PSF for any instrument
psf_2d = instrument.compute_psf()

# 3D PSF for microscopes
psf_3d = microscope.compute_psf(z_slices=41)

# Defocused PSF for cameras
psf_defocus = camera.compute_psf(defocus=1e-3)  # 1mm defocus
```

### Interactive Notebooks

Explore the multi-instrument capabilities with our example notebooks:
- [`microscope_simulation.ipynb`](examples/notebooks/microscope_simulation.ipynb): Complete microscopy simulation tutorial
- [`camera_simulation.ipynb`](examples/notebooks/camera_simulation.ipynb): Camera optics and imaging demonstration

### Physical Validation

All instruments are validated against theoretical formulas:
- **Telescope resolution**: Rayleigh criterion (1.22λ/D)
- **Microscope resolution**: Abbe diffraction limit (0.61λ/NA)
- **Camera spot size**: Airy disk diameter (2.44λ × f/#)
- **Sampling validation**: Automatic Nyquist criterion checking
- **Energy conservation**: Normalized PSFs with proper energy scaling

## Documentation

### User Guide

For comprehensive usage instructions, physics details, and best practices:
- **[User Guide](docs/user_guide.md)** - Complete documentation for users

**Important sections**:
- [Coordinate Conventions](docs/user_guide.md#coordinate-conventions) - **Critical for understanding PRISM physics**
  - Centered FFT conventions (DC at center)
  - Energy conservation and orthonormal transforms
  - Correct vs incorrect FFT usage
- [Core Concepts](docs/user_guide.md#core-concepts) - Progressive imaging fundamentals
- [Configuration](docs/user_guide.md#configuration) - Complete config system reference

### Implementation Guides

For developers and contributors:
- [Foundational Modules Revision](docs/implementation_guides/foundational_revision/) - Architecture and refactoring guides
- [Propagators Guide](docs/implementation_guides/foundational_revision/02_propagators.md#coordinate-conventions) - Detailed coordinate system documentation

### Performance

- [Phase 1 Performance Report](docs/performance/phase1_week3_performance_report.md) - Optimization results (16x measurement cache speedup!)
- [Benchmarks](benchmarks/) - Performance benchmarking scripts

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute to PRISM.
