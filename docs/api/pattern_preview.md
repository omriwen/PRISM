# prism.core.pattern_preview

Pattern visualization and verification tools.

Provides utilities for previewing sampling patterns before running
expensive experiments.

## Classes

## Functions

### compute_pattern_statistics

```python
compute_pattern_statistics(sample_centers: torch.Tensor, roi_diameter: float) -> Dict[str, Any]
```

Compute statistical properties of sampling pattern.

Args:
    sample_centers: Pattern positions (n_samples, n_points, 2)
    roi_diameter: ROI diameter for coverage calculation

Returns:
    Dictionary with statistics:
    - n_samples: Number of sampling positions
    - is_line_sampling: Whether using line sampling
    - radial_mean: Mean distance from center
    - radial_std: Std of distance from center
    - radial_min: Minimum distance from center
    - radial_max: Maximum distance from center
    - coverage_percentage: Percentage of ROI covered
    - inter_sample_distances: Statistics on distances between samples

### preview_pattern

```python
preview_pattern(pattern_spec: str, config: Any, save_path: Optional[pathlib.Path] = None) -> Dict[str, Any]
```

Preview a pattern from specification.

Args:
    pattern_spec: Pattern specification (builtin:name or file path)
    config: Configuration object
    save_path: Optional path to save visualization

Returns:
    Dictionary with:
    - 'sample_centers': Generated pattern tensor
    - 'statistics': Pattern statistics
    - 'metadata': Pattern metadata (source, hash, etc.)
    - 'figure': Matplotlib figure (if not saved and closed)

### visualize_pattern

```python
visualize_pattern(sample_centers: torch.Tensor, roi_diameter: float, save_path: Optional[pathlib.Path] = None, show_statistics: bool = True) -> matplotlib.figure.Figure
```

Create comprehensive visualization of sampling pattern.

Args:
    sample_centers: Pattern positions (n_samples, n_points, 2)
    roi_diameter: ROI diameter
    save_path: Optional path to save figure
    show_statistics: Whether to include statistics panel

Returns:
    Matplotlib figure
