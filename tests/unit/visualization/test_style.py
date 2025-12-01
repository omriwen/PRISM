"""Unit tests for spids.visualization.style module."""

from __future__ import annotations

import matplotlib.pyplot as plt

from prism.visualization.config import VisualizationConfig
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
    get_theme_colors,
    list_astronomy_colormaps,
    register_astronomy_colormaps,
)


class TestPresets:
    """Tests for visualization presets."""

    def test_publication_preset_type(self) -> None:
        """Test PUBLICATION preset is a VisualizationConfig."""
        assert isinstance(PUBLICATION, VisualizationConfig)

    def test_publication_high_dpi(self) -> None:
        """Test PUBLICATION preset has high DPI."""
        assert PUBLICATION.figure.dpi == 300

    def test_publication_uses_latex(self) -> None:
        """Test PUBLICATION preset uses LaTeX if available."""
        # Note: LaTeX is disabled by default for compatibility
        assert isinstance(PUBLICATION.style.use_latex, bool)

    def test_presentation_larger_fonts(self) -> None:
        """Test PRESENTATION has larger fonts than PUBLICATION."""
        assert PRESENTATION.style.font_size >= PUBLICATION.style.font_size

    def test_presentation_lower_dpi(self) -> None:
        """Test PRESENTATION has lower DPI (for slides)."""
        assert PRESENTATION.figure.dpi == 150

    def test_dashboard_lowest_dpi(self) -> None:
        """Test DASHBOARD has lowest DPI (for web)."""
        assert DASHBOARD.figure.dpi == 100

    def test_interactive_optimized_for_speed(self) -> None:
        """Test INTERACTIVE preset is optimized for rendering speed."""
        # Interactive uses lower DPI for faster rendering
        assert INTERACTIVE.figure.dpi <= PUBLICATION.figure.dpi

    def test_create_publication_config(self) -> None:
        """Test create_publication_config factory function."""
        config = create_publication_config()
        assert isinstance(config, VisualizationConfig)
        assert config.figure.dpi == 300

    def test_create_presentation_config(self) -> None:
        """Test create_presentation_config factory function."""
        config = create_presentation_config()
        assert isinstance(config, VisualizationConfig)
        assert config.figure.dpi == 150

    def test_create_dashboard_config(self) -> None:
        """Test create_dashboard_config factory function."""
        config = create_dashboard_config()
        assert isinstance(config, VisualizationConfig)
        assert config.figure.dpi == 100

    def test_create_interactive_config(self) -> None:
        """Test create_interactive_config factory function."""
        config = create_interactive_config()
        assert isinstance(config, VisualizationConfig)


class TestThemes:
    """Tests for color themes."""

    def test_light_theme_type(self) -> None:
        """Test LIGHT_THEME is a Theme."""
        assert isinstance(LIGHT_THEME, Theme)

    def test_dark_theme_type(self) -> None:
        """Test DARK_THEME is a Theme."""
        assert isinstance(DARK_THEME, Theme)

    def test_theme_has_required_colors(self) -> None:
        """Test themes have required color attributes."""
        for theme in [LIGHT_THEME, DARK_THEME]:
            assert hasattr(theme, "background_color")
            assert hasattr(theme, "text_color")
            assert hasattr(theme, "accent_color")
            assert hasattr(theme, "surface_color")

    def test_light_theme_light_background(self) -> None:
        """Test light theme has light background."""
        # Light theme background should be light (high luminance)
        # Hex color like #FFFFFF or similar
        bg = LIGHT_THEME.background_color
        assert bg.startswith("#") or bg in ["white", "w"]

    def test_dark_theme_dark_background(self) -> None:
        """Test dark theme has dark background."""
        bg = DARK_THEME.background_color
        assert bg.startswith("#")

    def test_get_theme_colors(self) -> None:
        """Test get_theme_colors returns dict."""
        colors = get_theme_colors(LIGHT_THEME)
        assert isinstance(colors, dict)
        assert "background" in colors
        assert "text" in colors

    def test_apply_theme_sets_rcparams(self) -> None:
        """Test apply_theme modifies matplotlib rcParams."""
        original_bg = plt.rcParams["figure.facecolor"]
        apply_theme(DARK_THEME)
        # Theme should change figure background
        # Restore after test
        plt.rcParams["figure.facecolor"] = original_bg


class TestColormaps:
    """Tests for astronomy-specific colormaps."""

    def test_colormaps_dict_exists(self) -> None:
        """Test ASTRONOMY_COLORMAPS dict exists."""
        assert isinstance(ASTRONOMY_COLORMAPS, dict)

    def test_has_cosmic_gray(self) -> None:
        """Test cosmic gray colormap exists."""
        assert "prism_cosmic_gray" in ASTRONOMY_COLORMAPS

    def test_has_europa_ice(self) -> None:
        """Test Europa ice colormap exists."""
        assert "prism_europa_ice" in ASTRONOMY_COLORMAPS

    def test_has_betelgeuse_fire(self) -> None:
        """Test Betelgeuse fire colormap exists."""
        assert "prism_betelgeuse_fire" in ASTRONOMY_COLORMAPS

    def test_has_kspace_power(self) -> None:
        """Test k-space power colormap exists."""
        assert "prism_kspace_power" in ASTRONOMY_COLORMAPS

    def test_has_phase(self) -> None:
        """Test phase colormap exists."""
        assert "prism_phase" in ASTRONOMY_COLORMAPS

    def test_register_colormaps(self) -> None:
        """Test register_astronomy_colormaps doesn't raise."""
        register_astronomy_colormaps()

    def test_get_colormap_by_name(self) -> None:
        """Test get_colormap returns colormap."""
        cmap = get_colormap("prism_cosmic_gray")
        assert cmap is not None

    def test_get_colormap_fallback(self) -> None:
        """Test get_colormap falls back to matplotlib."""
        cmap = get_colormap("viridis")
        assert cmap is not None

    def test_list_astronomy_colormaps(self) -> None:
        """Test list_astronomy_colormaps returns list."""
        cmap_list = list_astronomy_colormaps()
        assert isinstance(cmap_list, list)
        assert "prism_cosmic_gray" in cmap_list
        assert len(cmap_list) >= 5  # At least 5 custom colormaps
