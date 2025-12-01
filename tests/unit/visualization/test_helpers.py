"""Unit tests for spids.visualization.helpers module."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.patches import Circle

from prism.visualization.helpers import (
    create_aperture_overlay,
    create_roi_circle,
    ensure_4d_tensor,
    get_cpu_device,
    prepare_tensor_for_display,
)


class TestPrepareForDisplay:
    """Tests for prepare_tensor_for_display function."""

    def test_2d_tensor(self) -> None:
        """Test with 2D tensor input."""
        tensor = torch.randn(64, 64)
        result = prepare_tensor_for_display(tensor)
        assert result.shape == (64, 64)
        assert isinstance(result, np.ndarray)

    def test_3d_tensor(self) -> None:
        """Test with 3D tensor input (C, H, W)."""
        tensor = torch.randn(1, 64, 64)
        result = prepare_tensor_for_display(tensor)
        assert result.shape == (64, 64)

    def test_4d_tensor(self) -> None:
        """Test with 4D tensor input (B, C, H, W)."""
        tensor = torch.randn(1, 1, 64, 64)
        result = prepare_tensor_for_display(tensor)
        assert result.shape == (64, 64)

    def test_normalization(self) -> None:
        """Test that output is normalized to [0, 1]."""
        tensor = torch.randn(64, 64) * 10 + 5  # Random values around 5
        result = prepare_tensor_for_display(tensor, normalize=True)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_no_normalization(self) -> None:
        """Test without normalization."""
        tensor = torch.ones(64, 64) * 5
        result = prepare_tensor_for_display(tensor, normalize=False)
        assert np.allclose(result, 5.0)

    def test_log_scale(self) -> None:
        """Test log scale transformation."""
        tensor = torch.ones(64, 64) * 100
        result = prepare_tensor_for_display(tensor, log_scale=True, normalize=False)
        # log10(100) = 2
        assert np.allclose(result, 2.0, atol=0.01)

    def test_complex_tensor(self) -> None:
        """Test with complex tensor (takes absolute value)."""
        tensor = torch.complex(torch.ones(64, 64) * 3, torch.ones(64, 64) * 4)
        result = prepare_tensor_for_display(tensor, normalize=False, take_abs=True)
        # |3+4i| = 5
        assert np.allclose(result, 5.0)

    def test_crop_size(self) -> None:
        """Test center cropping."""
        tensor = torch.randn(128, 128)
        result = prepare_tensor_for_display(tensor, crop_size=64)
        assert result.shape == (64, 64)

    def test_percentile_clipping(self) -> None:
        """Test percentile clipping."""
        tensor = torch.randn(1000, 1000)  # Need enough values for percentiles
        # Add some outliers
        tensor[0, 0] = 1000
        tensor[0, 1] = -1000
        result = prepare_tensor_for_display(tensor, percentile_clip=(1, 99), normalize=False)
        # Outliers should be clipped
        assert result.max() < 1000
        assert result.min() > -1000


class TestCreateApertureOverlay:
    """Tests for create_aperture_overlay function."""

    def test_output_shape(self) -> None:
        """Test output shape is RGBA."""
        mask = torch.zeros(64, 64)
        mask[20:40, 20:40] = 1
        result = create_aperture_overlay(mask)
        assert result.shape == (64, 64, 4)

    def test_color_application(self) -> None:
        """Test that color is correctly applied."""
        mask = torch.ones(10, 10)
        color = (1.0, 0.5, 0.0, 0.8)
        result = create_aperture_overlay(mask, color)
        # Check RGB channels
        assert np.allclose(result[0, 0, 0], 1.0)  # R
        assert np.allclose(result[0, 0, 1], 0.5)  # G
        assert np.allclose(result[0, 0, 2], 0.0)  # B
        assert np.allclose(result[0, 0, 3], 0.8)  # A (mask is 1)

    def test_alpha_respects_mask(self) -> None:
        """Test that alpha channel respects mask values."""
        mask = torch.zeros(10, 10)
        mask[5, 5] = 1
        color = (0.0, 1.0, 0.0, 0.5)
        result = create_aperture_overlay(mask, color)
        # Alpha should be 0.5 where mask is 1
        assert result[5, 5, 3] == 0.5
        # Alpha should be 0 where mask is 0
        assert result[0, 0, 3] == 0.0


class TestEnsure4dTensor:
    """Tests for ensure_4d_tensor function."""

    def test_2d_to_4d(self) -> None:
        """Test 2D tensor becomes 4D."""
        tensor = torch.randn(64, 64)
        result = ensure_4d_tensor(tensor)
        assert result.ndim == 4
        assert result.shape == (1, 1, 64, 64)

    def test_3d_to_4d(self) -> None:
        """Test 3D tensor becomes 4D."""
        tensor = torch.randn(3, 64, 64)
        result = ensure_4d_tensor(tensor)
        assert result.ndim == 4
        assert result.shape == (1, 3, 64, 64)

    def test_4d_unchanged(self) -> None:
        """Test 4D tensor remains unchanged."""
        tensor = torch.randn(2, 3, 64, 64)
        result = ensure_4d_tensor(tensor)
        assert result.ndim == 4
        assert result.shape == (2, 3, 64, 64)


class TestCreateRoiCircle:
    """Tests for create_roi_circle function."""

    def test_creates_circle(self) -> None:
        """Test that a Circle patch is created."""
        circle = create_roi_circle((50, 50), 25)
        assert isinstance(circle, Circle)

    def test_circle_properties(self) -> None:
        """Test circle properties are set correctly."""
        circle = create_roi_circle(
            (100, 100),
            50,
            color="blue",
            linestyle="-",
            linewidth=2.0,
            label="Test ROI",
        )
        assert circle.center == (100, 100)
        assert circle.radius == 50
        assert circle.get_label() == "Test ROI"

    def test_default_no_fill(self) -> None:
        """Test that circle is not filled by default."""
        circle = create_roi_circle((50, 50), 25)
        assert circle.get_fill() is False


class TestGetCpuDevice:
    """Tests for get_cpu_device function."""

    def test_returns_cpu_device(self) -> None:
        """Test that CPU device is returned."""
        device = get_cpu_device()
        assert device == torch.device("cpu")
