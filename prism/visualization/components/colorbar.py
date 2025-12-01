"""
Module: spids.visualization.components.colorbar
Purpose: Enhanced colorbar component with scientific formatting
Dependencies: matplotlib, dataclasses

Description:
    Provides publication-quality colorbars with scientific notation,
    custom labels, and configurable positioning for astronomical images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray


if TYPE_CHECKING:
    from matplotlib.image import AxesImage


@dataclass
class ColorbarConfig:
    """Configuration for enhanced colorbar.

    Parameters
    ----------
    label : str
        Colorbar label (supports LaTeX)
    orientation : str
        Colorbar orientation ('vertical', 'horizontal')
    location : str
        Location relative to axes ('right', 'left', 'top', 'bottom')
    fraction : float
        Fraction of axes taken by colorbar
    pad : float
        Padding between axes and colorbar
    aspect : float
        Aspect ratio of colorbar (length/width)
    shrink : float
        Shrink factor for colorbar
    extend : str
        Extend colorbar ends ('neither', 'both', 'min', 'max')
    use_scientific : bool
        Use scientific notation for labels
    scientific_limits : tuple[int, int]
        Range for scientific notation (e.g., (-2, 3) means 10^-2 to 10^3)
    n_ticks : int
        Number of ticks on colorbar
    tick_format : str
        Format string for tick labels
    font_size : int
        Font size for labels
    label_pad : float
        Padding for label
    """

    label: str = ""
    orientation: Literal["vertical", "horizontal"] = "vertical"
    location: Literal["right", "left", "top", "bottom"] = "right"
    fraction: float = 0.046
    pad: float = 0.04
    aspect: float = 20.0
    shrink: float = 1.0
    extend: Literal["neither", "both", "min", "max"] = "neither"
    use_scientific: bool = False
    scientific_limits: tuple[int, int] = (-2, 3)
    n_ticks: int = 5
    tick_format: str = "{:.2g}"
    font_size: int = 10
    label_pad: float = 10.0


def add_colorbar(
    fig: Figure,
    ax: Axes,
    image: AxesImage,
    config: ColorbarConfig,
) -> Colorbar:
    """Add enhanced colorbar to figure.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Axes containing the image
    image : AxesImage
        Image to create colorbar for
    config : ColorbarConfig
        Colorbar configuration

    Returns
    -------
    Colorbar
        Created colorbar instance

    Examples
    --------
    >>> im = ax.imshow(data)
    >>> config = ColorbarConfig(label="Intensity [counts]")
    >>> cbar = add_colorbar(fig, ax, im, config)
    """
    cbar = fig.colorbar(
        image,
        ax=ax,
        orientation=config.orientation,
        location=config.location,
        fraction=config.fraction,
        pad=config.pad,
        aspect=config.aspect,
        shrink=config.shrink,
        extend=config.extend,
    )

    # Set label
    if config.label:
        cbar.set_label(config.label, fontsize=config.font_size, labelpad=config.label_pad)

    # Configure ticks
    if config.use_scientific:
        from matplotlib.ticker import ScalarFormatter

        formatter = ScalarFormatter()
        formatter.set_scientific(True)
        formatter.set_powerlimits(config.scientific_limits)
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.update_ticks()

    # Set tick properties
    cbar.ax.tick_params(labelsize=config.font_size - 2)

    return cbar


def add_colorbar_with_units(
    fig: Figure,
    ax: Axes,
    image: AxesImage,
    label: str,
    unit: str,
    *,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    **kwargs: float | int | str,
) -> Colorbar:
    """Add colorbar with unit label.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Axes containing the image
    image : AxesImage
        Image to create colorbar for
    label : str
        Physical quantity label
    unit : str
        Physical unit
    orientation : str
        Colorbar orientation
    **kwargs
        Additional arguments for ColorbarConfig

    Returns
    -------
    Colorbar
        Created colorbar instance

    Examples
    --------
    >>> cbar = add_colorbar_with_units(fig, ax, im, "Flux", "Jy/beam")
    """
    full_label = f"{label} [{unit}]" if unit else label
    config = ColorbarConfig(label=full_label, orientation=orientation, **kwargs)  # type: ignore[arg-type]
    return add_colorbar(fig, ax, image, config)


def create_log_colorbar(
    fig: Figure,
    ax: Axes,
    data: NDArray[np.floating],
    config: ColorbarConfig,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> tuple[AxesImage, Colorbar]:
    """Create image with log-scale colorbar.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Axes to plot image
    data : NDArray
        2D data array (must be positive)
    config : ColorbarConfig
        Colorbar configuration
    vmin : float, optional
        Minimum value for normalization
    vmax : float, optional
        Maximum value for normalization
    cmap : str
        Colormap name

    Returns
    -------
    tuple[AxesImage, Colorbar]
        Image and colorbar instances

    Examples
    --------
    >>> config = ColorbarConfig(label="Log Intensity")
    >>> im, cbar = create_log_colorbar(fig, ax, kspace_magnitude, config)
    """
    # Handle zeros and negatives
    data_positive = np.maximum(data, np.finfo(float).tiny)

    if vmin is None:
        vmin = float(data_positive[data_positive > 0].min())
    if vmax is None:
        vmax = float(data_positive.max())

    norm = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(data_positive, norm=norm, cmap=cmap)
    cbar = add_colorbar(fig, ax, im, config)

    return im, cbar


def create_symmetric_colorbar(
    fig: Figure,
    ax: Axes,
    data: NDArray[np.floating],
    config: ColorbarConfig,
    *,
    symmetric_around: float = 0.0,
    cmap: str = "RdBu_r",
) -> tuple[AxesImage, Colorbar]:
    """Create image with symmetric colorbar (useful for phase or difference maps).

    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Axes to plot image
    data : NDArray
        2D data array
    config : ColorbarConfig
        Colorbar configuration
    symmetric_around : float
        Center value for symmetric normalization
    cmap : str
        Colormap name (diverging colormaps recommended)

    Returns
    -------
    tuple[AxesImage, Colorbar]
        Image and colorbar instances

    Examples
    --------
    >>> config = ColorbarConfig(label="Phase [rad]")
    >>> im, cbar = create_symmetric_colorbar(fig, ax, phase_data, config)
    """
    # Calculate symmetric limits
    max_deviation = float(np.abs(data - symmetric_around).max())
    vmin = symmetric_around - max_deviation
    vmax = symmetric_around + max_deviation

    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(data, norm=norm, cmap=cmap)
    cbar = add_colorbar(fig, ax, im, config)

    return im, cbar


def add_inset_colorbar(
    ax: Axes,
    image: AxesImage,
    config: ColorbarConfig,
    *,
    bounds: tuple[float, float, float, float] = (0.65, 0.05, 0.3, 0.03),
) -> Colorbar:
    """Add colorbar as inset within axes.

    Parameters
    ----------
    ax : Axes
        Parent axes
    image : AxesImage
        Image to create colorbar for
    config : ColorbarConfig
        Colorbar configuration
    bounds : tuple[float, float, float, float]
        (x, y, width, height) in axes coordinates

    Returns
    -------
    Colorbar
        Created colorbar instance

    Examples
    --------
    >>> cbar = add_inset_colorbar(ax, im, config, bounds=(0.7, 0.1, 0.25, 0.02))
    """
    cax = ax.inset_axes(bounds)
    cbar = plt.colorbar(image, cax=cax, orientation=config.orientation)

    if config.label:
        cbar.set_label(config.label, fontsize=config.font_size)

    cbar.ax.tick_params(labelsize=config.font_size - 2)

    return cbar
