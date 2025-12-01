"""
Module: spids.visualization.components
Purpose: Reusable visualization components

Description:
    Provides publication-quality components for scientific figures:
    - Scale bars with physical units
    - Enhanced colorbars with scientific notation
    - Annotation helpers (circles, text boxes, arrows, markers)
"""

from __future__ import annotations

from prism.visualization.components.annotations import (
    ArrowConfig,
    CircleAnnotationConfig,
    MarkerConfig,
    TextBoxConfig,
    add_arrow_annotation,
    add_metrics_overlay,
    add_multiple_roi_circles,
    add_rectangle_roi,
    add_roi_circle,
    add_sample_markers,
    add_text_box,
)
from prism.visualization.components.colorbar import (
    ColorbarConfig,
    add_colorbar,
    add_colorbar_with_units,
    add_inset_colorbar,
    create_log_colorbar,
    create_symmetric_colorbar,
)
from prism.visualization.components.scalebar import (
    ScaleBarConfig,
    add_scalebar,
    calculate_scale_bar_length,
    format_physical_unit,
)


__all__ = [
    # Scalebar
    "ScaleBarConfig",
    "add_scalebar",
    "calculate_scale_bar_length",
    "format_physical_unit",
    # Colorbar
    "ColorbarConfig",
    "add_colorbar",
    "add_colorbar_with_units",
    "add_inset_colorbar",
    "create_log_colorbar",
    "create_symmetric_colorbar",
    # Annotations
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
]
