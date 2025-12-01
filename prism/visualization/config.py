"""
Module: spids.visualization.config
Purpose: Configuration dataclasses for visualization system
Dependencies: dataclasses, typing

Description:
    Provides typed configuration for all visualization components including
    figure settings, style options, k-space display, and metrics overlays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class FigureConfig:
    """Configuration for figure creation and layout.

    Parameters
    ----------
    figsize : tuple[float, float]
        Figure size in inches (width, height)
    dpi : int
        Dots per inch for rasterization
    tight_layout : bool
        Whether to use tight_layout for spacing
    constrained_layout : bool
        Whether to use constrained_layout (alternative to tight_layout)

    Examples
    --------
    >>> config = FigureConfig(figsize=(12, 8), dpi=300)
    >>> config.validate()
    """

    figsize: tuple[float, float] = (10.0, 8.0)
    dpi: int = 300
    tight_layout: bool = True
    constrained_layout: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.dpi <= 0:
            raise ValueError(f"dpi must be positive, got {self.dpi}")
        if any(s <= 0 for s in self.figsize):
            raise ValueError(f"figsize dimensions must be positive, got {self.figsize}")
        if self.tight_layout and self.constrained_layout:
            raise ValueError("Cannot use both tight_layout and constrained_layout")


@dataclass
class StyleConfig:
    """Configuration for plot styling.

    Parameters
    ----------
    colormap : str
        Default colormap name for images
    font_family : str
        Font family ('serif', 'sans-serif', 'monospace')
    font_size : int
        Base font size in points
    title_size : int
        Title font size in points
    label_size : int
        Axis label font size in points
    tick_size : int
        Tick label font size in points
    line_width : float
        Default line width for plots
    marker_size : float
        Default marker size for scatter plots
    grid_alpha : float
        Grid line transparency (0-1)
    use_latex : bool
        Whether to use LaTeX for text rendering
    """

    colormap: str = "gray"
    font_family: str = "serif"
    font_size: int = 12
    title_size: int = 14
    label_size: int = 10
    tick_size: int = 9
    line_width: float = 2.0
    marker_size: float = 6.0
    grid_alpha: float = 0.3
    use_latex: bool = False

    def apply(self) -> None:
        """Apply style to matplotlib rcParams."""
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.family": self.font_family,
                "font.size": self.font_size,
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "lines.linewidth": self.line_width,
                "lines.markersize": self.marker_size,
                "grid.alpha": self.grid_alpha,
                "text.usetex": self.use_latex,
            }
        )


@dataclass
class KSpaceConfig:
    """Configuration for k-space visualization.

    Parameters
    ----------
    show_aperture_mask : bool
        Whether to show the accumulated aperture mask
    mask_color : tuple[float, float, float, float]
        RGBA color for mask overlay (default: semi-transparent green)
    show_sample_centers : bool
        Whether to show sample center positions
    center_color : str
        Color for sample center markers
    center_marker_size : float
        Size of center markers
    log_scale : bool
        Whether to use log scale for k-space magnitude
    roi_line_style : str
        Line style for ROI circle ('--', '-', ':', '-.')
    roi_line_width : float
        Line width for ROI circle
    roi_color : str
        Color for ROI circle
    """

    show_aperture_mask: bool = True
    mask_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5)
    show_sample_centers: bool = True
    center_color: str = "red"
    center_marker_size: float = 10.0
    log_scale: bool = True
    roi_line_style: str = "--"
    roi_line_width: float = 3.0
    roi_color: str = "red"


@dataclass
class MetricsOverlayConfig:
    """Configuration for metrics text overlays.

    Parameters
    ----------
    show_ssim : bool
        Whether to show SSIM metric
    show_psnr : bool
        Whether to show PSNR metric
    show_loss : bool
        Whether to show loss value
    position : str
        Position for overlay ('top-left', 'top-right', 'bottom-left', 'bottom-right')
    font_size : int
        Font size for metrics text
    background_alpha : float
        Background transparency (0-1)
    box_style : str
        Style for text box ('round', 'square', 'roundtooth', 'sawtooth')
    decimal_places_ssim : int
        Decimal places for SSIM display
    decimal_places_psnr : int
        Decimal places for PSNR display
    """

    show_ssim: bool = True
    show_psnr: bool = True
    show_loss: bool = False
    position: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "bottom-right"
    font_size: int = 10
    background_alpha: float = 0.8
    box_style: str = "round"
    decimal_places_ssim: int = 4
    decimal_places_psnr: int = 1


@dataclass
class VisualizationConfig:
    """Master configuration for visualization system.

    Parameters
    ----------
    figure : FigureConfig
        Figure configuration
    style : StyleConfig
        Style configuration
    kspace : KSpaceConfig
        K-space display configuration
    metrics : MetricsOverlayConfig
        Metrics overlay configuration
    crop_to_obj_size : bool
        Whether to crop images to object size
    show_colorbars : bool
        Whether to show colorbars on images
    memory_cleanup : bool
        Whether to perform gc.collect() after plotting

    Examples
    --------
    >>> config = VisualizationConfig()
    >>> config.figure.dpi = 600  # High-res for publication
    >>> config.style.use_latex = True
    """

    figure: FigureConfig = field(default_factory=FigureConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    kspace: KSpaceConfig = field(default_factory=KSpaceConfig)
    metrics: MetricsOverlayConfig = field(default_factory=MetricsOverlayConfig)

    crop_to_obj_size: bool = True
    show_colorbars: bool = True
    memory_cleanup: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
