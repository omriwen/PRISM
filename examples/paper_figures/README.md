# Paper Figures Generation

Scripts to reproduce figures from SPIDS publications.

## Overview

This directory contains code for generating publication-quality figures for the SPIDS paper. The figures follow Optica Publishing Group guidelines and demonstrate:
- Neural network architecture
- Detector configuration and sampling trajectories
- Validation results and resolution analysis

## Generated Figures

### Figure 1: Neural Network Architecture and Algorithm Flowchart
- Network structure diagram (GenCropSpidsNet)
- Progressive training algorithm flowchart
- Shows latent vector, decoder, and measurement feedback loop

### Figure 2: Detector Configuration and Motion Trajectories
- Telescope aperture configuration
- K-space sampling patterns (Fermat spiral, star, random)
- Coverage maps and trajectory visualization

### Figure 4: Comprehensive Validation and Resolution Analysis
- Reconstruction quality vs number of samples
- SSIM/RMSE metrics across different objects
- Comparison with Mo-PIE baseline
- Resolution analysis and frequency response

## Requirements

### Software Dependencies
```bash
# Already installed if you ran uv sync
# Key packages:
# - matplotlib (plotting)
# - numpy (numerical operations)
# - torch (loading checkpoints)
# - PIL/Pillow (image processing)
```

### Experiment Data

The script requires completed SPIDS experiment runs. See [required_data.txt](required_data.txt) for specific experiments needed.

**Typical requirements**:
1. Europa baseline (100 samples, Fermat spiral)
2. Comparison experiments (SPIDS vs Mo-PIE)
3. Multi-object validation runs (Europa, Titan, Betelgeuse, Neptune)
4. Resolution scaling experiments (varying sample counts)

### Setting Up Data

```bash
# Option 1: Download pre-run experiments (if available)
# [URL to data repository]

# Option 2: Re-run experiments yourself
cd ../..

# Europa baseline
uv run python main.py \
    --obj_name europa \
    --n_samples 100 \
    --fermat \
    --name europa_baseline_100

# Add more experiments as needed...
```

## Usage

### Generate All Figures

```bash
cd examples/paper_figures

# Generate all figures for a publication
uv run python generate_figures.py \
    --output_dir ./paper_figures_output

# This creates:
# - figure_1_architecture.pdf
# - figure_2_trajectories.pdf
# - figure_4_validation.pdf
```

### Generate Specific Figure

```bash
# Generate only Figure 1
uv run python generate_figures.py \
    --figure 1 \
    --output_dir ./output

# Generate only Figure 4 (requires experiment data)
uv run python generate_figures.py \
    --figure 4 \
    --experiment ../../runs/europa_baseline_100 \
    --output_dir ./output
```

### Custom Configuration

```bash
# High-resolution output (for print)
uv run python generate_figures.py \
    --dpi 300 \
    --format pdf \
    --output_dir ./high_res_output

# Quick preview (lower quality, faster)
uv run python generate_figures.py \
    --dpi 100 \
    --format png \
    --output_dir ./preview
```

## Command-line Arguments

- `--figure`: Specific figure number to generate (1, 2, or 4), or omit for all
- `--experiment`: Path to experiment run directory (for figures requiring data)
- `--output_dir`: Directory to save generated figures (default: `./paper_figures_output`)
- `--dpi`: Resolution in dots per inch (default: 300 for publication quality)
- `--format`: Output format - pdf, png, svg (default: pdf)
- `--no-show`: Don't display figures interactively (just save)

## Output Structure

```
paper_figures_output/
├── figure_1_architecture.pdf       # Network architecture diagram
├── figure_2_trajectories.pdf       # Sampling patterns
├── figure_4_validation.pdf         # Validation results
└── metadata.json                   # Figure generation metadata
```

## Publication Specifications

**Optica Publishing Group Requirements**:
- Format: PDF or high-resolution PNG (≥300 DPI)
- Fonts: Arial or similar sans-serif, embedded
- Line weights: ≥0.5 pt
- Color mode: RGB for online, CMYK for print
- Size: Column width (3.45") or full width (7.125")

**Implementation**:
- All figures use 300 DPI by default
- Fonts embedded in PDF output
- Color schemes tested for colorblind accessibility
- Figure dimensions match journal specifications

## Modifying Figures

### Adjusting Plot Styles

Edit the `FigureGenerator` class in [generate_figures.py](generate_figures.py):

```python
# Change color scheme
self.color_scheme = {
    'spids': '#2E86AB',      # Blue
    'mopie': '#A23B72',       # Purple
    'baseline': '#F18F01',   # Orange
}

# Adjust font sizes
self.font_sizes = {
    'title': 14,
    'label': 12,
    'tick': 10,
    'legend': 10,
}
```

### Adding New Panels

1. Create new method: `_create_panel_X()`
2. Add to figure composition in main generation method
3. Update documentation and metadata

### Using Different Data

```python
# Load different experiment
results = torch.load('path/to/different/experiment/checkpoint.pt')

# Pass to figure generator
generator.create_figure_4(results)
```

## Version Compatibility

⚠️ **Important**: These scripts are **frozen at publication time**

- Scripts reflect SPIDS codebase at time of paper submission
- May not work with current SPIDS versions if API changed
- For reproducibility, use git tag corresponding to publication

### Using Publication Version

```bash
# Check out code at publication time
git checkout v0.2.0  # or appropriate publication tag

# Run figure generation with that version
cd examples/paper_figures
uv run python generate_figures.py
```

## Reproducibility Notes

### Exact Reproduction

To exactly reproduce published figures:

1. Use same SPIDS version (git tag)
2. Use same random seeds (set in experiment configs)
3. Use same experiment parameters
4. Use same plotting parameters (DPI, fonts, colors)

### Expected Variations

Minor variations may occur due to:
- Random initialization (even with same seed, different PyTorch versions may vary)
- GPU vs CPU computation (floating point differences)
- Different experiment runs (if re-running from scratch)

These should be minimal and not affect scientific conclusions.

## Troubleshooting

### Missing Experiment Data

```
Error: Experiment directory not found: runs/europa_baseline_100
```

**Solution**: Run the required experiments or download pre-run data

### Import Errors

```
ModuleNotFoundError: No module named 'spids'
```

**Solution**: Install SPIDS package
```bash
cd ../..
uv sync
```

### Font Warnings

```
UserWarning: findfont: Font family 'Arial' not found
```

**Solution**: Install Arial or use matplotlib default fonts (automatic fallback)

### Memory Issues

Large figures may require significant RAM. If running out of memory:
- Generate figures one at a time (`--figure 1`, etc.)
- Reduce DPI for preview (`--dpi 100`)
- Close other applications

## Maintenance Status

**Status**: Frozen at publication version

These scripts are maintained for reproducibility but frozen at the version used for the published paper. They may not be updated for new SPIDS features.

For new figures using current SPIDS versions:
- Create new scripts in a dated subdirectory
- Document differences from publication version
- Maintain backward compatibility where possible

## References

**Paper**: [Citation to be added when published]

**Code Version**: v0.2.0 (or tag at time of publication)

**Data Repository**: [Link to archived experiment data, if published]

## Contact

For questions about reproducing these figures:
- Check git history: `git log examples/paper_figures/`
- Review commit messages for changes
- See main SPIDS documentation for general usage
