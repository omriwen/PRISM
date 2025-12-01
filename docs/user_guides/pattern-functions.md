# Pattern Functions Guide

## Overview

PRISM pattern functions allow you to define arbitrary sampling patterns as executable Python code. This provides unlimited flexibility while maintaining reproducibility and supporting AI-assisted pattern generation.

## Quick Start

### Using Builtin Patterns

```bash
# Fermat spiral (recommended default)
python main.py --pattern-fn builtin:fermat --obj-name europa

# Star pattern
python main.py --pattern-fn builtin:star --obj-name europa

# Random uniform
python main.py --pattern-fn builtin:random --obj-name europa
```

### Creating Custom Patterns

1. Create a Python file with a `generate_pattern` function:

```python
# my_pattern.py
import torch
import numpy as np

def generate_pattern(config):
    """Your pattern description (saved in metadata)."""
    n = config.n_samples
    roi_r = config.roi_diameter / 2

    # Your pattern generation logic
    # ...

    # Return torch.Tensor of shape (n_samples, 1, 2)
    return sample_centers
```

2. Preview the pattern:

```bash
python main.py --pattern-fn my_pattern.py --n-samples 100 --preview-pattern
```

3. Run experiment:

```bash
python main.py --pattern-fn my_pattern.py --obj-name europa
```

## Pattern Function Contract

### Input: Configuration Object

Your function receives a `config` object with these attributes:

- `config.n_samples` (int): Number of sampling positions
- `config.roi_diameter` (float): K-space region diameter in pixels
- `config.sample_diameter` (float): Telescope aperture size
- `config.sample_length` (float): Line length (0 = point sampling)
- `config.line_angle` (Optional[float]): Fixed line angle in radians
- `config.obj_size` (int): Object size in pixels

### Output: Sample Positions Tensor

Return a `torch.Tensor` with:

- **Shape**: `(n_samples, 1, 2)` for point sampling OR `(n_samples, 2, 2)` for line sampling
- **Values**: `[x, y]` coordinates in pixels, centered at origin `(0, 0)`
- **Coordinate system**: Standard Cartesian (not image coordinates)
- **Range**: Typically within `[-roi_diameter/2, roi_diameter/2]`

### Example

```python
def generate_pattern(config):
    """Uniform circle sampling."""
    n = config.n_samples
    r_max = config.roi_diameter / 2

    # Random angles and radii
    theta = torch.rand(n) * 2 * np.pi
    r = torch.sqrt(torch.rand(n)) * r_max  # sqrt for uniform area

    # Convert to Cartesian
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    # Stack to required shape
    return torch.stack([x, y], dim=-1)[:, None, :]
```

## Example Patterns

See `examples/patterns/` for complete examples:

- `continuous_spiral.py`: Archimedean spiral with adjustable turns
- `concentric_circles.py`: Multiple circles with center weighting
- `jittered_grid.py`: Regular grid with random perturbations
- `logarithmic_spiral.py`: Exponential radial growth

## Preview and Verification

Always preview patterns before running expensive experiments:

```bash
# Preview with default parameters
python main.py --pattern-fn my_pattern.py --preview-pattern

# Preview with custom parameters
python main.py --pattern-fn my_pattern.py --n-samples 200 \
    --roi-diameter 1024 --preview-pattern
```

The preview shows:
- Sample position scatter plot
- K-space coverage heatmap
- Radial distribution histogram
- Pattern statistics (coverage %, density, etc.)

## Reproducibility

Pattern specifications are automatically saved with experiment results:

```
runs/experiment_name/
├── checkpoint.pt              # Contains pattern_metadata
└── pattern_preview.png        # Visualization (if generated)
```

This ensures perfect reproducibility - anyone can see exactly how the pattern was generated.

## AI-Assisted Pattern Generation

You can ask Claude Code to generate patterns from natural language:

**Example prompt:**
> "Create a sampling pattern with 5 concentric circles, with denser sampling in the center (30, 25, 20, 15, 10 samples per circle)"

Claude will generate the appropriate pattern function code for you.

## Backward Compatibility

Legacy flags still work:

```bash
# Old way
python main.py --fermat --obj-name europa

# New equivalent
python main.py --pattern-fn builtin:fermat --obj-name europa
```

## Best Practices

1. **Add docstrings**: They're saved in experiment metadata
2. **Preview first**: Catch issues before expensive experiments
3. **Use config attributes**: Don't hardcode n_samples
4. **Validate range**: Keep points within ROI
5. **Test incrementally**: Start with small n_samples
6. **Save pattern files**: Keep custom patterns in version control

## Troubleshooting

**Error: "Pattern function must return torch.Tensor"**
- Make sure you return a PyTorch tensor, not numpy array

**Error: "Pattern output must be 3D tensor"**
- Check your output shape is `(n_samples, 1, 2)` or `(n_samples, 2, 2)`

**Pattern looks wrong in preview**
- Check coordinate system (should be centered at 0,0)
- Verify scale matches `roi_diameter`

**Import errors in pattern file**
- Pattern files are executed in isolation
- Import all dependencies at top of file
