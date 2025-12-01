# prism.visualization.publication

Publication-quality visualizations for PRISM results.

This module provides tools for creating publication-ready figures with
proper styling, fonts, and layout suitable for scientific papers.

## Classes

### PublicationPlotter

```python
PublicationPlotter(figsize: Tuple[float, float] = (8, 6), dpi: int = 300, use_latex: bool = False)
```

Create publication-ready figures with consistent styling.

This class provides methods for creating high-quality scientific figures
suitable for publication. It handles styling, layout, and common plot types.

Parameters
----------
figsize : tuple, optional
    Default figure size (width, height) in inches, by default (8, 6)
dpi : int, optional
    Resolution in dots per inch, by default 300
use_latex : bool, optional
    Whether to use LaTeX for text rendering, by default False
    (requires LaTeX installation)

Attributes
----------
figsize : tuple
    Default figure size
dpi : int
    Resolution for saved figures
use_latex : bool
    Whether LaTeX rendering is enabled

Examples
--------
>>> plotter = PublicationPlotter(figsize=(10, 6), dpi=300)
>>> fig = plotter.plot_reconstruction_comparison(
...     ground_truth, reconstruction, measurement
... )
>>> plotter.save_figure(fig, "comparison.pdf")

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

##### `plot_k_space_coverage`

Plot k-space sampling pattern coverage.

Parameters
----------
sample_points : Tensor, ndarray, or list
    Sampling points coordinates
grid_size : int, optional
    Size of the k-space grid, by default 256
title : str, optional
    Figure title

Returns
-------
Figure
    Matplotlib figure object

##### `plot_reconstruction_comparison`

Create three-panel comparison figure.

Creates a publication-ready comparison showing ground truth,
reconstruction, and static measurement side by side with metrics.

Parameters
----------
ground_truth : Tensor or ndarray
    Ground truth image
reconstruction : Tensor or ndarray
    Reconstructed image
measurement : Tensor or ndarray
    Static telescope measurement
title : str, optional
    Overall figure title, by default None
metrics : dict, optional
    Dictionary of metrics to display, by default None
    Will calculate SSIM, RMSE, PSNR if not provided

Returns
-------
Figure
    Matplotlib figure object

##### `plot_training_curves`

Plot training curves with multiple metrics.

Parameters
----------
losses : list of float
    Loss values over training
ssims : list of float, optional
    SSIM values over training, by default None
psnrs : list of float, optional
    PSNR values over training, by default None
title : str, optional
    Figure title, by default "Training Progress"

Returns
-------
Figure
    Matplotlib figure object

##### `save_figure`

Save figure with publication settings.

Parameters
----------
fig : Figure
    Matplotlib figure to save
filepath : str or Path
    Output file path (supports .pdf, .png, .svg, .eps)
**kwargs
    Additional arguments passed to fig.savefig()

##### `setup_style`

Configure matplotlib for publication-quality output.

Sets up consistent styling including fonts, sizes, and appearance
parameters suitable for scientific publications.

## Functions

### compare_rmse

```python
compare_rmse(img1, img2)
```

Fallback RMSE calculation

### psnr

```python
psnr(img1, img2)
```

Fallback PSNR calculation

### ssim_skimage

```python
ssim_skimage(img1, img2)
```

Fallback SSIM calculation
