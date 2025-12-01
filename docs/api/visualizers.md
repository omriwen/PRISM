# prism.cli.patterns.visualizers

Visualization helpers for pattern plotting.

## Classes

## Functions

### plot_coverage_heatmap

```python
plot_coverage_heatmap(ax: matplotlib.axes._axes.Axes, sample_centers: torch.Tensor, roi_diameter: float, title: str) -> None
```

Plot coverage heatmap on an axes.

Parameters
----------
ax : Axes
    Matplotlib axes to plot on
sample_centers : Tensor
    Sample center coordinates
roi_diameter : float
    ROI diameter in pixels
title : str
    Plot title

### plot_sample_positions

```python
plot_sample_positions(ax: matplotlib.axes._axes.Axes, sample_centers: torch.Tensor, roi_diameter: float, title: str) -> None
```

Plot sample positions on an axes.

Parameters
----------
ax : Axes
    Matplotlib axes to plot on
sample_centers : Tensor
    Sample center coordinates
roi_diameter : float
    ROI diameter in pixels
title : str
    Plot title
