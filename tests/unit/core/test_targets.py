"""Unit tests for test target generation."""

import pytest
import torch

from prism.core.targets import (
    TargetConfig,
    USAF1951Config,
    USAF1951Target,
    create_target,
)


class TestTargetConfig:
    """Test TargetConfig dataclass."""

    def test_default_config(self):
        config = TargetConfig()
        assert config.size == 1024
        assert config.contrast == 1.0
        assert config.polarity == "positive"
        assert config.device == "cpu"

    def test_invalid_contrast(self):
        with pytest.raises(ValueError, match="Contrast must be"):
            TargetConfig(contrast=-0.1)
        with pytest.raises(ValueError, match="Contrast must be"):
            TargetConfig(contrast=1.5)

    def test_invalid_polarity(self):
        with pytest.raises(ValueError, match="Polarity must be"):
            TargetConfig(polarity="invalid")


class TestUSAF1951Target:
    """Test USAF-1951 target generation."""

    def test_frequency_formula(self):
        """Verify USAF-1951 frequency formula."""
        # Known values from USAF-1951 standard
        assert USAF1951Target.get_frequency_lp_mm(0, 1) == pytest.approx(1.0)
        assert USAF1951Target.get_frequency_lp_mm(0, 6) == pytest.approx(1.78, rel=0.01)
        assert USAF1951Target.get_frequency_lp_mm(1, 1) == pytest.approx(2.0)
        assert USAF1951Target.get_frequency_lp_mm(2, 1) == pytest.approx(4.0)

    def test_bar_width(self):
        """Verify bar width calculations."""
        # Group 0, Element 1: 1 lp/mm -> bar width = 0.5mm
        assert USAF1951Target.get_bar_width_mm(0, 1) == pytest.approx(0.5)
        # Group 1, Element 1: 2 lp/mm -> bar width = 0.25mm
        assert USAF1951Target.get_bar_width_mm(1, 1) == pytest.approx(0.25)

    def test_generate_shape(self):
        """Test generated target has correct shape."""
        config = USAF1951Config(size=512, groups=(0, 1, 2))
        target = USAF1951Target(config)
        image = target.generate()

        assert image.shape == (512, 512)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_resolution_elements(self):
        """Test resolution elements dictionary."""
        config = USAF1951Config(groups=(0, 1))
        target = USAF1951Target(config)
        elements = target.resolution_elements

        assert "G0E1" in elements
        assert "G1E6" in elements
        # Check that element contains expected resolution info
        assert "frequency_lp_mm" in elements["G0E1"]
        assert elements["G0E1"]["frequency_lp_mm"] == pytest.approx(1.0, rel=1e-3)

    def test_polarity(self):
        """Test positive vs negative polarity."""
        config_pos = USAF1951Config(size=256, groups=(0,), polarity="positive")
        config_neg = USAF1951Config(size=256, groups=(0,), polarity="negative")

        target_pos = USAF1951Target(config_pos)
        target_neg = USAF1951Target(config_neg)

        img_pos = target_pos.generate()
        img_neg = target_neg.generate()

        # Inverted images
        assert torch.allclose(img_pos, 1.0 - img_neg, atol=1e-6)


class TestCreateTarget:
    """Test target factory function."""

    def test_create_usaf1951(self):
        target = create_target("usaf1951", size=512, groups=(0, 1))
        assert isinstance(target, USAF1951Target)
        assert target.config.size == 512

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown target type"):
            create_target("invalid_type")
