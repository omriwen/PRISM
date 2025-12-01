"""
Module: spids.visualization.style.themes
Purpose: Color themes for visualization (light/dark)
Dependencies: matplotlib

Description:
    Provides light and dark color themes primarily for dashboard
    and interactive visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Theme:
    """Color theme configuration.

    Parameters
    ----------
    name : str
        Theme name
    background_color : str
        Background color (hex)
    surface_color : str
        Surface/panel color (hex)
    text_color : str
        Primary text color (hex)
    secondary_text_color : str
        Secondary text color (hex)
    grid_color : str
        Grid line color (hex)
    accent_color : str
        Accent/highlight color (hex)
    success_color : str
        Success/positive indicator (hex)
    warning_color : str
        Warning indicator (hex)
    error_color : str
        Error/negative indicator (hex)
    """

    name: str
    background_color: str
    surface_color: str
    text_color: str
    secondary_text_color: str
    grid_color: str
    accent_color: str
    success_color: str
    warning_color: str
    error_color: str


LIGHT_THEME = Theme(
    name="light",
    background_color="#FFFFFF",
    surface_color="#F5F5F5",
    text_color="#1A1A1A",
    secondary_text_color="#6B6B6B",
    grid_color="#E0E0E0",
    accent_color="#2196F3",
    success_color="#4CAF50",
    warning_color="#FF9800",
    error_color="#F44336",
)


DARK_THEME = Theme(
    name="dark",
    background_color="#1E1E2E",  # Catppuccin Mocha base
    surface_color="#313244",
    text_color="#CDD6F4",
    secondary_text_color="#A6ADC8",
    grid_color="#45475A",
    accent_color="#89B4FA",
    success_color="#A6E3A1",
    warning_color="#F9E2AF",
    error_color="#F38BA8",
)


def apply_theme(theme: Theme) -> dict[str, Any]:
    """Apply theme to matplotlib rcParams.

    Parameters
    ----------
    theme : Theme
        Theme to apply

    Returns
    -------
    dict[str, Any]
        Dictionary of rcParams changes for reverting
    """
    import matplotlib.pyplot as plt

    # Store original values
    original = {
        "figure.facecolor": plt.rcParams["figure.facecolor"],
        "axes.facecolor": plt.rcParams["axes.facecolor"],
        "text.color": plt.rcParams["text.color"],
        "axes.labelcolor": plt.rcParams["axes.labelcolor"],
        "xtick.color": plt.rcParams["xtick.color"],
        "ytick.color": plt.rcParams["ytick.color"],
        "grid.color": plt.rcParams["grid.color"],
    }

    # Apply theme
    plt.rcParams.update(
        {
            "figure.facecolor": theme.background_color,
            "axes.facecolor": theme.surface_color,
            "text.color": theme.text_color,
            "axes.labelcolor": theme.text_color,
            "xtick.color": theme.text_color,
            "ytick.color": theme.text_color,
            "grid.color": theme.grid_color,
        }
    )

    return original


def get_theme_colors(theme: Theme) -> dict[str, str]:
    """Get theme colors as a dictionary.

    Parameters
    ----------
    theme : Theme
        Theme to extract colors from

    Returns
    -------
    dict[str, str]
        Dictionary of color names to hex values
    """
    return {
        "background": theme.background_color,
        "surface": theme.surface_color,
        "text": theme.text_color,
        "secondary_text": theme.secondary_text_color,
        "grid": theme.grid_color,
        "accent": theme.accent_color,
        "success": theme.success_color,
        "warning": theme.warning_color,
        "error": theme.error_color,
    }


def get_plotly_template(theme: Theme) -> dict[str, Any]:
    """Get Plotly layout dict from theme for dashboard integration.

    Parameters
    ----------
    theme : Theme
        Theme to convert to Plotly template

    Returns
    -------
    dict[str, Any]
        Dictionary suitable for plotly figure.update_layout(...)

    Example
    -------
    >>> from prism.visualization.style import DARK_THEME, get_plotly_template
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> template = get_plotly_template(DARK_THEME)
    >>> fig.update_layout(**template)
    """
    return {
        "paper_bgcolor": theme.background_color,
        "plot_bgcolor": theme.surface_color,
        "font": {"color": theme.text_color},
        "xaxis": {
            "gridcolor": theme.grid_color,
            "linecolor": theme.grid_color,
            "tickcolor": theme.text_color,
        },
        "yaxis": {
            "gridcolor": theme.grid_color,
            "linecolor": theme.grid_color,
            "tickcolor": theme.text_color,
        },
        "colorway": [
            theme.accent_color,
            theme.success_color,
            theme.warning_color,
            theme.error_color,
            "#A6E3A1",  # Additional colors
            "#F9E2AF",
            "#FAB387",
            "#CBA6F7",
        ],
    }
