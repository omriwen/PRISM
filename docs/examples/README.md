# PRISM Examples

This directory contains example scripts demonstrating key features of PRISM.

## Prerequisites

Before running these examples, ensure you have:

1. Installed PRISM and its dependencies:
   ```bash
   cd /path/to/PRISM
   uv sync
   ```

2. Activated the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Examples Overview

### 01_basic_reconstruction.py

**Basic Progressive Reconstruction**

Demonstrates the core PRISM workflow:
- Loading astronomical images
- Creating telescope and model components
- Progressive training with Fermat spiral sampling
- Computing quality metrics (SSIM, PSNR)
- Saving results and checkpoints

**Usage:**
```bash
# Edit the script to set your image path
uv run python docs/examples/01_basic_reconstruction.py
```

**What you'll learn:**
- Core PRISM training loop
- TelescopeAgg and LossAgg usage
- Fermat spiral sampling
- Metric calculation
- Result visualization

---

### 02_config_based_training.py

**Configuration-Based Training**

Shows how to use YAML configuration files for reproducible experiments.

**Usage:**
```bash
uv run python docs/examples/02_config_based_training.py
```

**What you'll learn:**
- Creating ExperimentConfig objects
- Saving/loading YAML configurations
- Organizing experiment parameters
- Using predefined astronomical objects

---

### 03_custom_sampling.py

**Custom Sampling Patterns**

Demonstrates creating and visualizing different k-space sampling patterns.

**Usage:**
```bash
uv run python docs/examples/03_custom_sampling.py
```

**What you'll learn:**
- Built-in sampling patterns (Fermat, random, star)
- Creating custom sampling patterns
- Visualizing sampling patterns
- Saving/loading patterns from JSON

---

### 04_visualization.py

**Advanced Visualization**

Shows how to create publication-quality figures and reports.

**Usage:**
```bash
# Run after example 01 to use its checkpoint
uv run python docs/examples/04_visualization.py
```

**What you'll learn:**
- Publication-quality plotting
- Interactive visualizations (with plotly)
- Statistical analysis
- HTML report generation
- Multi-panel figures for papers

---

## Running the Examples

### Quick Start

Run all examples in sequence:

```bash
# 1. Basic reconstruction
uv run python docs/examples/01_basic_reconstruction.py

# 2. Config-based training
uv run python docs/examples/02_config_based_training.py

# 3. Custom sampling
uv run python docs/examples/03_custom_sampling.py

# 4. Visualization (uses results from example 01)
uv run python docs/examples/04_visualization.py
```

### Customization

Each example is self-contained and can be modified:

1. **Change image paths**: Edit the `image_path` variable
2. **Adjust parameters**: Modify constants at the top of each script
3. **Add your own logic**: Examples use clear, commented code

### Expected Output

After running the examples, you'll find:

```
results/
├── basic_reconstruction/
│   ├── checkpoint.pt
│   └── comparison.png
├── visualizations/
│   ├── training_curves.png
│   ├── reconstruction.png
│   ├── publication_figure.png
│   └── interactive_curves.html
configs/
└── europa_example.yaml
sampling_patterns.png
custom_pattern.json
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'prism'**
```bash
# Make sure you're in the PRISM directory and have activated the environment
cd /path/to/PRISM
source .venv/bin/activate
```

**FileNotFoundError: Image not found**
```python
# Edit the image_path in the example scripts
image_path = "path/to/your/image.jpg"  # Change this line
```

**CUDA out of memory**
```python
# Reduce image size or disable GPU
IMAGE_SIZE = 512  # Instead of 1024
device = torch.device("cpu")  # Force CPU
```

**Plotly not available**
```bash
# Install plotly for interactive visualizations
uv add plotly
```

## Additional Resources

### Learning Path
- **[Complete Learning Path](../user_guides/learning_path.md)**: Full 12-15 hour curriculum with assessments
- **[Notebooks README](../../examples/notebooks/README.md)**: Interactive notebook tutorials

### Python API Scripts
Located in [examples/python_api/](../../examples/python_api/):

| Script | Purpose |
|--------|---------|
| 06_microscope_reconstruction.py | Microscopy workflow |
| 07_drone_mapping.py | Drone imaging workflow |
| 08_custom_scenario_builder.py | Custom scenario configuration |
| 09_resolution_validation.py | Automated validation suite |

### Validation Notebooks
Located in [examples/validation/notebooks/](../../examples/validation/notebooks/):

| Notebook | Purpose |
|----------|---------|
| 01_microscope_resolution_validation.ipynb | Resolution vs Abbe limit |
| 02_snr_reconstruction_quality.ipynb | SNR effects on quality |
| 03_propagator_accuracy_validation.ipynb | Wave propagation accuracy |
| 04_drone_gsd_validation.ipynb | GSD validation |
| 05_sampling_density_validation.ipynb | Nyquist sampling validation |

### Reference Documentation
- **User Guide**: See [docs/user_guide.md](../user_guide.md) for comprehensive documentation
- **API Reference**: See [docs/api/index.md](../api/index.md) for detailed API docs
- **Main Scripts**: Check `main.py` and `main_epie.py` for full-featured implementations

## Contributing Examples

Have a useful example? Please contribute!

1. Create a new example script following the naming convention: `XX_descriptive_name.py`
2. Add clear comments and docstrings
3. Update this README with your example
4. Submit a pull request

## Questions?

- Open an issue: [GitHub Issues](https://github.com/omri/PRISM/issues)
- Start a discussion: [GitHub Discussions](https://github.com/omri/PRISM/discussions)

---

**Last Updated**: 2025-11-04
