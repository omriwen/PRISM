"""
Module: spids.visualization.components.scalebar
Purpose: Astronomical scale bar component for visualization
Dependencies: matplotlib, dataclasses

Description:
    Provides configurable scale bars for astronomical images with
    support for physical units (arcseconds, arcminutes, parsecs, etc.)
    and customizable styling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Rectangle


@dataclass
class ScaleBarConfig:
    """Configuration for astronomical scale bar.

    Parameters
    ----------
    length_pixels : float
        Length of scale bar in pixels
    length_physical : float
        Length of scale bar in physical units
    unit : str
        Physical unit label (e.g., "arcsec", "arcmin", "pc", "km")
    position : str
        Position in figure ('bottom-left', 'bottom-right', 'top-left', 'top-right')
    color : str
        Color of scale bar and text
    background_color : str | None
        Background color (None for transparent)
    background_alpha : float
        Background transparency (0-1)
    font_size : int
        Font size for label
    bar_height_fraction : float
        Height of bar as fraction of length
    padding : float
        Padding from edge in pixels
    show_label : bool
        Whether to show the unit label
    label_position : str
        Position of label relative to bar ('above', 'below', 'right')
    """

    length_pixels: float = 50.0
    length_physical: float = 1.0
    unit: str = "arcsec"
    position: Literal["bottom-left", "bottom-right", "top-left", "top-right"] = "bottom-right"
    color: str = "white"
    background_color: str | None = "black"
    background_alpha: float = 0.6
    font_size: int = 10
    bar_height_fraction: float = 0.1
    padding: float = 10.0
    show_label: bool = True
    label_position: Literal["above", "below", "right"] = "above"


def add_scalebar(
    ax: Axes,
    config: ScaleBarConfig,
    *,
    image_extent: tuple[float, float, float, float] | None = None,
) -> None:
    """Add scale bar to axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to add scale bar to
    config : ScaleBarConfig
        Scale bar configuration
    image_extent : tuple[float, float, float, float], optional
        Image extent (xmin, xmax, ymin, ymax). If None, uses axes limits.

    Examples
    --------
    >>> config = ScaleBarConfig(length_pixels=100, length_physical=10, unit="arcsec")
    >>> add_scalebar(ax, config)
    """
    # Get image bounds
    if image_extent is not None:
        xmin, xmax, ymin, ymax = image_extent
    else:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

    # Note: xmin, xmax, ymin, ymax are used for positioning calculations below

    # Calculate bar dimensions
    bar_length = config.length_pixels
    bar_height = bar_length * config.bar_height_fraction

    # Calculate position based on config
    padding = config.padding
    if config.position == "bottom-left":
        bar_x = xmin + padding
        bar_y = ymin + padding
    elif config.position == "bottom-right":
        bar_x = xmax - padding - bar_length
        bar_y = ymin + padding
    elif config.position == "top-left":
        bar_x = xmin + padding
        bar_y = ymax - padding - bar_height
    else:  # top-right
        bar_x = xmax - padding - bar_length
        bar_y = ymax - padding - bar_height

    # Add background rectangle if specified
    if config.background_color is not None:
        bg_padding = padding * 0.5
        bg_width = bar_length + 2 * bg_padding
        bg_height = bar_height + config.font_size * 1.5 + 2 * bg_padding

        if config.label_position == "above":
            bg_y = bar_y - bg_padding
        elif config.label_position == "below":
            bg_y = bar_y - config.font_size * 1.5 - bg_padding
        else:  # right
            bg_height = bar_height + 2 * bg_padding

        bg_rect = FancyBboxPatch(
            (bar_x - bg_padding, bg_y),
            bg_width,
            bg_height,
            boxstyle="round,pad=0.02",
            facecolor=config.background_color,
            alpha=config.background_alpha,
            edgecolor="none",
            zorder=10,
        )
        ax.add_patch(bg_rect)

    # Add scale bar
    bar = Rectangle(
        (bar_x, bar_y),
        bar_length,
        bar_height,
        facecolor=config.color,
        edgecolor=config.color,
        zorder=11,
    )
    ax.add_patch(bar)

    # Add label
    if config.show_label:
        label_text = f"{config.length_physical:.3g} {config.unit}"

        if config.label_position == "above":
            text_x = bar_x + bar_length / 2
            text_y = bar_y + bar_height + 2
            ha = "center"
            va = "bottom"
        elif config.label_position == "below":
            text_x = bar_x + bar_length / 2
            text_y = bar_y - 2
            ha = "center"
            va = "top"
        else:  # right
            text_x = bar_x + bar_length + 5
            text_y = bar_y + bar_height / 2
            ha = "left"
            va = "center"

        ax.text(
            text_x,
            text_y,
            label_text,
            fontsize=config.font_size,
            color=config.color,
            ha=ha,
            va=va,
            zorder=11,
        )


def calculate_scale_bar_length(
    pixel_scale: float,
    desired_physical_length: float,
) -> float:
    """Calculate scale bar length in pixels.

    Parameters
    ----------
    pixel_scale : float
        Physical size per pixel (e.g., arcsec/pixel)
    desired_physical_length : float
        Desired physical length of scale bar

    Returns
    -------
    float
        Scale bar length in pixels
    """
    return desired_physical_length / pixel_scale


def format_physical_unit(value: float, unit: str) -> tuple[float, str]:
    """Format physical value with appropriate unit prefix.

    Parameters
    ----------
    value : float
        Physical value
    unit : str
        Base unit

    Returns
    -------
    tuple[float, str]
        Formatted (value, unit) pair

    Examples
    --------
    >>> format_physical_unit(3600, "arcsec")
    (1.0, "degree")
    >>> format_physical_unit(0.001, "pc")
    (1.0, "mpc")
    """
    # Arcsec conversions
    if unit == "arcsec":
        if value >= 3600:
            return value / 3600, "degree"
        elif value >= 60:
            return value / 60, "arcmin"
        elif value < 0.001:
            return value * 1000, "mas"
        return value, unit

    # Distance conversions
    if unit in ("pc", "parsec"):
        if value >= 1e6:
            return value / 1e6, "Mpc"
        elif value >= 1e3:
            return value / 1e3, "kpc"
        elif value < 1:
            return value * 1e3, "mpc"
        return value, "pc"

    # Length conversions
    if unit == "km":
        if value >= 1e6:
            return value / 1e6, "Mm"
        elif value >= 1e3:
            return value / 1e3, "×10³ km"
        elif value < 1:
            return value * 1e3, "m"
        return value, unit

    return value, unit
