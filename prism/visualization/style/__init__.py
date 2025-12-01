"""
Module: spids.visualization.style
Purpose: Style system for publication-quality visualizations

Description:
    Provides a comprehensive style system including:
    - Pre-configured presets (PUBLICATION, PRESENTATION, DASHBOARD, INTERACTIVE)
    - Light and dark color themes
    - Astronomy-specific colormaps (cosmic_gray, europa_ice, betelgeuse_fire, etc.)
"""

from __future__ import annotations

from prism.visualization.style.colormaps import (
    ASTRONOMY_COLORMAPS,
    get_colormap,
    list_astronomy_colormaps,
    register_astronomy_colormaps,
)
from prism.visualization.style.presets import (
    DASHBOARD,
    INTERACTIVE,
    PRESENTATION,
    PUBLICATION,
    create_dashboard_config,
    create_interactive_config,
    create_presentation_config,
    create_publication_config,
)
from prism.visualization.style.themes import (
    DARK_THEME,
    LIGHT_THEME,
    Theme,
    apply_theme,
    get_plotly_template,
    get_theme_colors,
)


__all__ = [
    # Presets
    "PUBLICATION",
    "PRESENTATION",
    "DASHBOARD",
    "INTERACTIVE",
    "create_publication_config",
    "create_presentation_config",
    "create_dashboard_config",
    "create_interactive_config",
    # Themes
    "Theme",
    "LIGHT_THEME",
    "DARK_THEME",
    "apply_theme",
    "get_theme_colors",
    "get_plotly_template",
    # Colormaps
    "ASTRONOMY_COLORMAPS",
    "register_astronomy_colormaps",
    "get_colormap",
    "list_astronomy_colormaps",
]

# Register custom colormaps on import
register_astronomy_colormaps()
