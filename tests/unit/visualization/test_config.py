"""Unit tests for spids.visualization.config module."""

from __future__ import annotations

import pytest

from prism.visualization.config import (
    FigureConfig,
    KSpaceConfig,
    MetricsOverlayConfig,
    StyleConfig,
    VisualizationConfig,
)


class TestFigureConfig:
    """Tests for FigureConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = FigureConfig()
        assert config.figsize == (10.0, 8.0)
        assert config.dpi == 300
        assert config.tight_layout is True
        assert config.constrained_layout is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = FigureConfig(figsize=(12.0, 6.0), dpi=600)
        assert config.figsize == (12.0, 6.0)
        assert config.dpi == 600

    def test_invalid_dpi_raises(self) -> None:
        """Test that invalid DPI raises ValueError."""
        with pytest.raises(ValueError, match="dpi must be positive"):
            FigureConfig(dpi=0)
        with pytest.raises(ValueError, match="dpi must be positive"):
            FigureConfig(dpi=-100)

    def test_invalid_figsize_raises(self) -> None:
        """Test that invalid figsize raises ValueError."""
        with pytest.raises(ValueError, match="figsize dimensions must be positive"):
            FigureConfig(figsize=(0, 8))
        with pytest.raises(ValueError, match="figsize dimensions must be positive"):
            FigureConfig(figsize=(10, -5))

    def test_conflicting_layout_raises(self) -> None:
        """Test that both layout options raises ValueError."""
        with pytest.raises(ValueError, match="Cannot use both tight_layout and constrained_layout"):
            FigureConfig(tight_layout=True, constrained_layout=True)


class TestStyleConfig:
    """Tests for StyleConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default style values."""
        config = StyleConfig()
        assert config.colormap == "gray"
        assert config.font_family == "serif"
        assert config.font_size == 12
        assert config.use_latex is False

    def test_apply_updates_rcparams(self) -> None:
        """Test that apply() modifies matplotlib rcParams."""
        import matplotlib.pyplot as plt

        # Store original values
        original_font_size = plt.rcParams["font.size"]

        config = StyleConfig(font_size=14)
        config.apply()

        assert plt.rcParams["font.size"] == 14

        # Restore original
        plt.rcParams["font.size"] = original_font_size


class TestKSpaceConfig:
    """Tests for KSpaceConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default k-space config values."""
        config = KSpaceConfig()
        assert config.show_aperture_mask is True
        assert config.log_scale is True
        assert config.roi_color == "red"

    def test_mask_color_rgba(self) -> None:
        """Test mask color RGBA format."""
        config = KSpaceConfig(mask_color=(1.0, 0.0, 0.0, 0.8))
        assert config.mask_color == (1.0, 0.0, 0.0, 0.8)


class TestMetricsOverlayConfig:
    """Tests for MetricsOverlayConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default metrics overlay values."""
        config = MetricsOverlayConfig()
        assert config.show_ssim is True
        assert config.show_psnr is True
        assert config.show_loss is False
        assert config.position == "bottom-right"

    def test_decimal_places(self) -> None:
        """Test decimal places configuration."""
        config = MetricsOverlayConfig(decimal_places_ssim=5, decimal_places_psnr=2)
        assert config.decimal_places_ssim == 5
        assert config.decimal_places_psnr == 2


class TestVisualizationConfig:
    """Tests for VisualizationConfig dataclass."""

    def test_default_nested_configs(self) -> None:
        """Test that nested configs have defaults."""
        config = VisualizationConfig()
        assert isinstance(config.figure, FigureConfig)
        assert isinstance(config.style, StyleConfig)
        assert isinstance(config.kspace, KSpaceConfig)
        assert isinstance(config.metrics, MetricsOverlayConfig)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = VisualizationConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "figure" in d
        assert "style" in d
        assert d["figure"]["dpi"] == 300

    def test_modify_nested_config(self) -> None:
        """Test that nested configs can be modified."""
        config = VisualizationConfig()
        config.figure.dpi = 600
        assert config.figure.dpi == 600
