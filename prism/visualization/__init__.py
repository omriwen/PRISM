"""
Module: spids.visualization
Purpose: Modern visualization system for SPIDS

Description:
    Publication-quality visualization for astronomical imaging with:
    - Configuration dataclasses for all components
    - Protocol-based plotter interfaces
    - Style presets (PUBLICATION, PRESENTATION, DASHBOARD, INTERACTIVE)
    - Astronomy-specific colormaps
    - Reusable components (scalebar, colorbar, annotations)

Usage:
    >>> from prism.visualization import (
    ...     ReconstructionComparisonPlotter,
    ...     PUBLICATION,
    ... )
    >>> with ReconstructionComparisonPlotter(PUBLICATION) as plotter:
    ...     plotter.plot(ground_truth=gt, reconstruction=rec, ...)
    ...     plotter.save("result.pdf")
"""

from __future__ import annotations

# Existing animation module
from prism.visualization.animation import TrainingAnimator

# Phase 1 exports - Foundation
from prism.visualization.base import BasePlotter

# Phase 4 exports - Components
from prism.visualization.components import (
    ArrowConfig,
    CircleAnnotationConfig,
    ColorbarConfig,
    MarkerConfig,
    ScaleBarConfig,
    TextBoxConfig,
    add_arrow_annotation,
    add_colorbar,
    add_colorbar_with_units,
    add_inset_colorbar,
    add_metrics_overlay,
    add_multiple_roi_circles,
    add_rectangle_roi,
    add_roi_circle,
    add_sample_markers,
    add_scalebar,
    add_text_box,
    calculate_scale_bar_length,
    create_log_colorbar,
    create_symmetric_colorbar,
    format_physical_unit,
)
from prism.visualization.config import (
    FigureConfig,
    KSpaceConfig,
    MetricsOverlayConfig,
    StyleConfig,
    VisualizationConfig,
)
from prism.visualization.helpers import (
    add_metrics_text,
    compute_kspace_display,
    create_aperture_overlay,
    create_roi_circle,
    ensure_4d_tensor,
    get_cpu_device,
    prepare_tensor_for_display,
)

# Legacy backward-compatible functions (deprecated)
from prism.visualization.legacy import plot_meas_agg

# Phase 3 exports - Plotters
from prism.visualization.plotters import (
    LearningCurvesPlotter,
    ReconstructionComparisonPlotter,
    SyntheticAperturePlotter,
    TrainingVisualizer,
)
from prism.visualization.protocols import (
    LearningCurvesData,
    MetricsData,
    PlotData,
    PlotterProtocol,
    SaveFigureOptions,
)

# Phase 2 exports - Style System
from prism.visualization.style import (
    ASTRONOMY_COLORMAPS,
    DARK_THEME,
    DASHBOARD,
    INTERACTIVE,
    LIGHT_THEME,
    PRESENTATION,
    PUBLICATION,
    Theme,
    apply_theme,
    create_dashboard_config,
    create_interactive_config,
    create_presentation_config,
    create_publication_config,
    get_colormap,
    get_plotly_template,
    get_theme_colors,
    list_astronomy_colormaps,
    register_astronomy_colormaps,
)


__all__ = [
    # Animation
    "TrainingAnimator",
    # Config
    "FigureConfig",
    "StyleConfig",
    "KSpaceConfig",
    "MetricsOverlayConfig",
    "VisualizationConfig",
    # Protocols
    "PlotterProtocol",
    "PlotData",
    "MetricsData",
    "SaveFigureOptions",
    "LearningCurvesData",
    # Base
    "BasePlotter",
    # Helpers
    "prepare_tensor_for_display",
    "compute_kspace_display",
    "create_aperture_overlay",
    "ensure_4d_tensor",
    "add_metrics_text",
    "create_roi_circle",
    "get_cpu_device",
    # Style - Presets
    "PUBLICATION",
    "PRESENTATION",
    "DASHBOARD",
    "INTERACTIVE",
    "create_publication_config",
    "create_presentation_config",
    "create_dashboard_config",
    "create_interactive_config",
    # Style - Themes
    "Theme",
    "LIGHT_THEME",
    "DARK_THEME",
    "apply_theme",
    "get_theme_colors",
    "get_plotly_template",
    # Style - Colormaps
    "ASTRONOMY_COLORMAPS",
    "register_astronomy_colormaps",
    "get_colormap",
    "list_astronomy_colormaps",
    # Plotters
    "TrainingVisualizer",
    "ReconstructionComparisonPlotter",
    "SyntheticAperturePlotter",
    "LearningCurvesPlotter",
    # Components - Scalebar
    "ScaleBarConfig",
    "add_scalebar",
    "calculate_scale_bar_length",
    "format_physical_unit",
    # Components - Colorbar
    "ColorbarConfig",
    "add_colorbar",
    "add_colorbar_with_units",
    "add_inset_colorbar",
    "create_log_colorbar",
    "create_symmetric_colorbar",
    # Components - Annotations
    "CircleAnnotationConfig",
    "TextBoxConfig",
    "MarkerConfig",
    "ArrowConfig",
    "add_roi_circle",
    "add_multiple_roi_circles",
    "add_text_box",
    "add_metrics_overlay",
    "add_sample_markers",
    "add_arrow_annotation",
    "add_rectangle_roi",
    # Legacy (deprecated)
    "plot_meas_agg",
]
