# prism.visualization.plotting

Matplotlib plotting utilities with memory leak prevention.

This module provides context managers and utilities to prevent
matplotlib memory leaks during training and visualization.

## Classes

## Functions

### cleanup_matplotlib

```python
cleanup_matplotlib() -> None
```

Perform aggressive matplotlib cleanup.

Call this periodically during long training runs to ensure
matplotlib resources are released.

Examples
--------
>>> for epoch in range(1000):
...     # Training code
...     if epoch % 100 == 0:
...         cleanup_matplotlib()

### matplotlib_figure

```python
matplotlib_figure(*args: Any, **kwargs: Any)
```

Context manager for matplotlib figures with automatic cleanup.

This prevents memory leaks by ensuring proper cleanup of matplotlib
objects, which is critical for long-running training loops.

Parameters
----------
*args : Any
    Positional arguments passed to plt.figure()
**kwargs : Any
    Keyword arguments passed to plt.figure()

Yields
------
matplotlib.figure.Figure
    The created figure object

Examples
--------
>>> with matplotlib_figure(figsize=(10, 8)) as fig:
...     ax = fig.add_subplot(111)
...     ax.plot([1, 2, 3], [1, 4, 9])
...     fig.savefig('output.png')
# Figure is automatically cleaned up here

Notes
-----
Without this context manager, matplotlib figures can accumulate
in memory during training, leading to memory leaks. Always use
this when creating figures in loops.

### matplotlib_subplots

```python
matplotlib_subplots(*args: Any, **kwargs: Any)
```

Context manager for matplotlib subplots with automatic cleanup.

Similar to matplotlib_figure but for plt.subplots().

Parameters
----------
*args : Any
    Positional arguments passed to plt.subplots()
**kwargs : Any
    Keyword arguments passed to plt.subplots()

Yields
------
tuple
    (fig, axes) tuple from plt.subplots()

Examples
--------
>>> with matplotlib_subplots(2, 2, figsize=(12, 10)) as (fig, axes):
...     axes[0, 0].imshow(image1)
...     axes[0, 1].imshow(image2)
...     fig.savefig('comparison.png')

### save_figure_safely

```python
save_figure_safely(fig: matplotlib.figure.Figure, path: str, dpi: int = 300, bbox_inches: str = 'tight', **kwargs: Any) -> None
```

Save matplotlib figure with proper cleanup.

Parameters
----------
fig : matplotlib.figure.Figure
    Figure to save
path : str
    Output path
dpi : int, optional
    DPI for saved figure (default: 300)
bbox_inches : str, optional
    Bounding box setting (default: 'tight')
**kwargs : Any
    Additional arguments passed to fig.savefig()

Examples
--------
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3])
>>> save_figure_safely(fig, 'output.png')
