"""
Module: spids.visualization.components.annotations
Purpose: Annotation components for visualization (circles, text boxes, arrows)
Dependencies: matplotlib, dataclasses

Description:
    Provides reusable annotation components for scientific figures including
    ROI circles, metrics text boxes, sample position markers, and arrows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


if TYPE_CHECKING:
    from matplotlib.text import Text


@dataclass
class CircleAnnotationConfig:
    """Configuration for circle annotation (ROI, aperture, etc.).

    Parameters
    ----------
    color : str
        Circle color
    linestyle : str
        Line style ('--', '-', ':', '-.')
    linewidth : float
        Line width
    fill : bool
        Whether to fill the circle
    fill_alpha : float
        Fill transparency (0-1)
    label : str | None
        Label for legend
    zorder : int
        Drawing order
    """

    color: str = "red"
    linestyle: str = "--"
    linewidth: float = 2.0
    fill: bool = False
    fill_alpha: float = 0.2
    label: str | None = None
    zorder: int = 5


@dataclass
class TextBoxConfig:
    """Configuration for text box annotation.

    Parameters
    ----------
    font_size : int
        Font size
    font_family : str
        Font family
    text_color : str
        Text color
    background_color : str
        Background color
    background_alpha : float
        Background transparency (0-1)
    border_color : str
        Border color
    border_width : float
        Border width
    box_style : str
        Box style ('round', 'square', 'roundtooth', 'sawtooth')
    padding : float
        Internal padding
    position : str
        Position in axes ('top-left', 'top-right', 'bottom-left', 'bottom-right')
    zorder : int
        Drawing order
    """

    font_size: int = 10
    font_family: str = "monospace"
    text_color: str = "black"
    background_color: str = "white"
    background_alpha: float = 0.8
    border_color: str = "gray"
    border_width: float = 0.5
    box_style: str = "round"
    padding: float = 0.3
    position: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "bottom-right"
    zorder: int = 10


@dataclass
class MarkerConfig:
    """Configuration for position markers.

    Parameters
    ----------
    marker : str
        Marker style
    size : float
        Marker size
    color : str
        Marker color
    edge_color : str
        Edge color
    edge_width : float
        Edge width
    alpha : float
        Transparency (0-1)
    label : str | None
        Label for legend
    zorder : int
        Drawing order
    """

    marker: str = "o"
    size: float = 50.0
    color: str = "red"
    edge_color: str = "white"
    edge_width: float = 1.0
    alpha: float = 1.0
    label: str | None = None
    zorder: int = 5


@dataclass
class ArrowConfig:
    """Configuration for arrow annotation.

    Parameters
    ----------
    color : str
        Arrow color
    linewidth : float
        Line width
    head_width : float
        Arrow head width
    head_length : float
        Arrow head length
    style : str
        Arrow style ('simple', 'fancy', 'wedge')
    connection_style : str
        Connection style ('arc3', 'angle', 'angle3')
    zorder : int
        Drawing order
    """

    color: str = "black"
    linewidth: float = 1.5
    head_width: float = 10.0
    head_length: float = 10.0
    style: str = "simple"
    connection_style: str = "arc3,rad=0.1"
    zorder: int = 5


def add_roi_circle(
    ax: Axes,
    center: tuple[float, float],
    radius: float,
    config: CircleAnnotationConfig | None = None,
) -> Circle:
    """Add ROI circle to axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    center : tuple[float, float]
        (x, y) center coordinates
    radius : float
        Circle radius in data coordinates
    config : CircleAnnotationConfig, optional
        Circle configuration

    Returns
    -------
    Circle
        Created circle patch

    Examples
    --------
    >>> config = CircleAnnotationConfig(color="blue", linestyle="-")
    >>> circle = add_roi_circle(ax, (256, 256), 100, config)
    """
    if config is None:
        config = CircleAnnotationConfig()

    circle = Circle(
        center,
        radius,
        color=config.color,
        fill=config.fill,
        alpha=config.fill_alpha if config.fill else 1.0,
        linewidth=config.linewidth,
        linestyle=config.linestyle,
        label=config.label,
        zorder=config.zorder,
    )
    ax.add_patch(circle)
    return circle


def add_multiple_roi_circles(
    ax: Axes,
    centers: Sequence[tuple[float, float]],
    radius: float,
    config: CircleAnnotationConfig | None = None,
    *,
    alpha_gradient: bool = False,
) -> list[Circle]:
    """Add multiple ROI circles.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    centers : Sequence[tuple[float, float]]
        List of (x, y) center coordinates
    radius : float
        Circle radius
    config : CircleAnnotationConfig, optional
        Circle configuration
    alpha_gradient : bool
        Whether to apply alpha gradient (newer = more opaque)

    Returns
    -------
    list[Circle]
        List of created circle patches
    """
    if config is None:
        config = CircleAnnotationConfig()

    circles: list[Circle] = []
    n = len(centers)

    for i, center in enumerate(centers):
        # Create copy of config for potential alpha modification
        alpha = (i + 1) / n if alpha_gradient else 1.0

        circle = Circle(
            center,
            radius,
            color=config.color,
            fill=config.fill,
            alpha=alpha * (config.fill_alpha if config.fill else 1.0),
            linewidth=config.linewidth,
            linestyle=config.linestyle,
            zorder=config.zorder,
        )
        ax.add_patch(circle)
        circles.append(circle)

    return circles


def add_text_box(
    ax: Axes,
    text: str,
    config: TextBoxConfig | None = None,
) -> Text:
    """Add text box annotation to axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    text : str
        Text content (can be multiline)
    config : TextBoxConfig, optional
        Text box configuration

    Returns
    -------
    Text
        Created text object

    Examples
    --------
    >>> config = TextBoxConfig(position="top-left")
    >>> text = add_text_box(ax, "SSIM: 0.95\\nPSNR: 32.1 dB", config)
    """
    if config is None:
        config = TextBoxConfig()

    # Determine position
    pos_map: dict[str, tuple[float, float, str, str]] = {
        "top-left": (0.02, 0.98, "left", "top"),
        "top-right": (0.98, 0.98, "right", "top"),
        "bottom-left": (0.02, 0.02, "left", "bottom"),
        "bottom-right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = pos_map[config.position]

    text_obj = ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=config.font_size,
        fontfamily=config.font_family,
        color=config.text_color,
        zorder=config.zorder,
        bbox={
            "boxstyle": f"{config.box_style},pad={config.padding}",
            "facecolor": config.background_color,
            "alpha": config.background_alpha,
            "edgecolor": config.border_color,
            "linewidth": config.border_width,
        },
    )
    return text_obj


def add_metrics_overlay(
    ax: Axes,
    metrics: dict[str, float],
    config: TextBoxConfig | None = None,
    *,
    format_spec: dict[str, str] | None = None,
) -> Text:
    """Add metrics overlay text box.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    metrics : dict[str, float]
        Dictionary of metric names to values
    config : TextBoxConfig, optional
        Text box configuration
    format_spec : dict[str, str], optional
        Format specifiers for each metric

    Returns
    -------
    Text
        Created text object

    Examples
    --------
    >>> metrics = {"SSIM": 0.95, "PSNR": 32.1, "Loss": 0.001}
    >>> text = add_metrics_overlay(ax, metrics)
    """
    if format_spec is None:
        format_spec = {
            "SSIM": ".4f",
            "PSNR": ".1f",
            "Loss": ".6f",
            "RMSE": ".4f",
        }

    lines: list[str] = []
    for name, value in metrics.items():
        fmt = format_spec.get(name, ".4g")
        unit = " dB" if name == "PSNR" else ""
        lines.append(f"{name}: {value:{fmt}}{unit}")

    text = "\n".join(lines)
    return add_text_box(ax, text, config)


def add_sample_markers(
    ax: Axes,
    positions: Sequence[tuple[float, float]],
    config: MarkerConfig | None = None,
    *,
    highlight_last: bool = True,
    highlight_config: MarkerConfig | None = None,
) -> None:
    """Add sample position markers.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    positions : Sequence[tuple[float, float]]
        List of (x, y) positions
    config : MarkerConfig, optional
        Marker configuration for all points
    highlight_last : bool
        Whether to highlight the last point differently
    highlight_config : MarkerConfig, optional
        Configuration for highlighted (last) point
    """
    if config is None:
        config = MarkerConfig()

    if len(positions) == 0:
        return

    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]

    if highlight_last:
        # Plot all but last
        ax.scatter(
            x_coords[:-1],
            y_coords[:-1],
            marker=config.marker,
            s=config.size,
            c=config.color,
            edgecolors=config.edge_color,
            linewidths=config.edge_width,
            alpha=config.alpha,
            label=config.label,
            zorder=config.zorder,
        )

        # Plot last with highlight
        if highlight_config is None:
            highlight_config = MarkerConfig(
                marker="*",
                size=config.size * 2,
                color="yellow",
                edge_color="red",
                edge_width=1.5,
            )

        ax.scatter(
            [x_coords[-1]],
            [y_coords[-1]],
            marker=highlight_config.marker,
            s=highlight_config.size,
            c=highlight_config.color,
            edgecolors=highlight_config.edge_color,
            linewidths=highlight_config.edge_width,
            alpha=highlight_config.alpha,
            label=highlight_config.label,
            zorder=highlight_config.zorder,
        )
    else:
        ax.scatter(
            x_coords,
            y_coords,
            marker=config.marker,
            s=config.size,
            c=config.color,
            edgecolors=config.edge_color,
            linewidths=config.edge_width,
            alpha=config.alpha,
            label=config.label,
            zorder=config.zorder,
        )


def add_arrow_annotation(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    config: ArrowConfig | None = None,
    *,
    text: str | None = None,
    text_offset: tuple[float, float] = (0, 10),
) -> FancyArrowPatch:
    """Add arrow annotation to axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    start : tuple[float, float]
        Start position (x, y)
    end : tuple[float, float]
        End position (x, y)
    config : ArrowConfig, optional
        Arrow configuration
    text : str, optional
        Text label for arrow
    text_offset : tuple[float, float]
        Offset for text label

    Returns
    -------
    FancyArrowPatch
        Created arrow patch
    """
    if config is None:
        config = ArrowConfig()

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=f"{config.style},head_width={config.head_width},head_length={config.head_length}",
        connectionstyle=config.connection_style,
        color=config.color,
        linewidth=config.linewidth,
        zorder=config.zorder,
    )
    ax.add_patch(arrow)

    if text:
        mid_x = (start[0] + end[0]) / 2 + text_offset[0]
        mid_y = (start[1] + end[1]) / 2 + text_offset[1]
        ax.annotate(text, (mid_x, mid_y), fontsize=9)

    return arrow


def add_rectangle_roi(
    ax: Axes,
    corner: tuple[float, float],
    width: float,
    height: float,
    *,
    color: str = "red",
    linestyle: str = "--",
    linewidth: float = 2.0,
    fill: bool = False,
    fill_alpha: float = 0.2,
    label: str | None = None,
) -> Rectangle:
    """Add rectangular ROI annotation.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    corner : tuple[float, float]
        Bottom-left corner (x, y)
    width : float
        Rectangle width
    height : float
        Rectangle height
    color : str
        Rectangle color
    linestyle : str
        Line style
    linewidth : float
        Line width
    fill : bool
        Whether to fill
    fill_alpha : float
        Fill transparency
    label : str, optional
        Label for legend

    Returns
    -------
    Rectangle
        Created rectangle patch
    """
    rect = Rectangle(
        corner,
        width,
        height,
        color=color,
        fill=fill,
        alpha=fill_alpha if fill else 1.0,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    ax.add_patch(rect)
    return rect
