"""
Module: spids.visualization.style.presets
Purpose: Pre-configured visualization presets for different use cases
Dependencies: spids.visualization.config

Description:
    Provides ready-to-use configuration presets optimized for:
    - PUBLICATION: Nature/Science journal quality (300 DPI, LaTeX, serif fonts)
    - PRESENTATION: Conference talks (150 DPI, large fonts, high contrast)
    - DASHBOARD: Web monitoring (100 DPI, fast rendering)
    - INTERACTIVE: Real-time training display (low DPI, fast updates)
"""

from __future__ import annotations

from prism.visualization.config import (
    FigureConfig,
    KSpaceConfig,
    MetricsOverlayConfig,
    StyleConfig,
    VisualizationConfig,
)


def create_publication_config() -> VisualizationConfig:
    """Create configuration for publication-quality figures.

    Returns
    -------
    VisualizationConfig
        Configuration optimized for journal publications

    Notes
    -----
    - 300 DPI for high-resolution rasterization
    - LaTeX rendering enabled for mathematical labels
    - Serif fonts (Computer Modern style)
    - Golden ratio-based figure sizes
    """
    return VisualizationConfig(
        figure=FigureConfig(
            figsize=(7.0, 4.33),  # Golden ratio width for single column
            dpi=300,
            tight_layout=True,
        ),
        style=StyleConfig(
            colormap="prism_cosmic_gray",
            font_family="serif",
            font_size=10,
            title_size=12,
            label_size=10,
            tick_size=8,
            line_width=1.5,
            marker_size=6.0,
            grid_alpha=0.2,
            use_latex=True,
        ),
        kspace=KSpaceConfig(
            show_aperture_mask=True,
            mask_color=(0.0, 0.8, 0.0, 0.4),
            log_scale=True,
            roi_line_width=2.0,
        ),
        metrics=MetricsOverlayConfig(
            show_ssim=True,
            show_psnr=True,
            font_size=9,
            decimal_places_ssim=3,
            decimal_places_psnr=1,
        ),
        crop_to_obj_size=True,
        show_colorbars=True,
        memory_cleanup=True,
    )


def create_presentation_config() -> VisualizationConfig:
    """Create configuration for presentation slides.

    Returns
    -------
    VisualizationConfig
        Configuration optimized for conference presentations
    """
    return VisualizationConfig(
        figure=FigureConfig(
            figsize=(10.0, 5.625),  # 16:9 aspect ratio
            dpi=150,
            tight_layout=True,
        ),
        style=StyleConfig(
            colormap="prism_cosmic_gray",
            font_family="sans-serif",
            font_size=14,
            title_size=18,
            label_size=14,
            tick_size=12,
            line_width=2.5,
            marker_size=10.0,
            grid_alpha=0.3,
            use_latex=False,
        ),
        kspace=KSpaceConfig(
            show_aperture_mask=True,
            mask_color=(0.0, 1.0, 0.0, 0.5),
            log_scale=True,
            roi_line_width=3.0,
        ),
        metrics=MetricsOverlayConfig(
            show_ssim=True,
            show_psnr=True,
            font_size=12,
        ),
        crop_to_obj_size=True,
        show_colorbars=True,
        memory_cleanup=True,
    )


def create_dashboard_config() -> VisualizationConfig:
    """Create configuration for web dashboard.

    Returns
    -------
    VisualizationConfig
        Configuration optimized for web display
    """
    return VisualizationConfig(
        figure=FigureConfig(
            figsize=(8.0, 5.0),
            dpi=100,
            tight_layout=True,
        ),
        style=StyleConfig(
            colormap="prism_cosmic_gray",
            font_family="sans-serif",
            font_size=11,
            title_size=13,
            label_size=10,
            tick_size=9,
            line_width=2.0,
            marker_size=6.0,
            grid_alpha=0.3,
            use_latex=False,
        ),
        kspace=KSpaceConfig(
            show_aperture_mask=True,
            mask_color=(0.0, 1.0, 0.0, 0.5),
            log_scale=True,
        ),
        metrics=MetricsOverlayConfig(
            show_ssim=True,
            show_psnr=True,
            font_size=10,
        ),
        crop_to_obj_size=True,
        show_colorbars=True,
        memory_cleanup=True,
    )


def create_interactive_config() -> VisualizationConfig:
    """Create configuration for interactive/real-time display.

    Returns
    -------
    VisualizationConfig
        Configuration optimized for fast real-time updates
    """
    return VisualizationConfig(
        figure=FigureConfig(
            figsize=(10.0, 6.0),
            dpi=100,
            tight_layout=True,
        ),
        style=StyleConfig(
            colormap="gray",
            font_family="sans-serif",
            font_size=10,
            title_size=12,
            label_size=9,
            tick_size=8,
            line_width=1.5,
            marker_size=5.0,
            grid_alpha=0.3,
            use_latex=False,
        ),
        kspace=KSpaceConfig(
            show_aperture_mask=True,
            mask_color=(0.0, 1.0, 0.0, 0.5),
            log_scale=True,
        ),
        metrics=MetricsOverlayConfig(
            show_ssim=True,
            show_psnr=True,
            font_size=9,
        ),
        crop_to_obj_size=True,
        show_colorbars=False,  # Faster without colorbars
        memory_cleanup=False,  # Reuse figures for speed
    )


# Pre-instantiated presets
PUBLICATION = create_publication_config()
PRESENTATION = create_presentation_config()
DASHBOARD = create_dashboard_config()
INTERACTIVE = create_interactive_config()
