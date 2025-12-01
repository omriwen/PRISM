"""
Unit tests for loss functions and SSIM helper functions.

Tests cover:
    - Gaussian window generation
    - Gaussian filtering
    - SSIM computation (to be added in Phase 2)
    - MS-SSIM computation (to be added in Phase 3)
    - LossAggregator with SSIM (to be added in Phase 4)
"""

from __future__ import annotations

import pytest
import torch

from prism.models.losses import (
    CompositeLossStrategy,
    L1LossStrategy,
    L2LossStrategy,
    LossAggregator,
    MSSSIMLossStrategy,
    SSIMLossStrategy,
    _compute_ssim,
    _create_gaussian_window,
    _gaussian_filter,
    _ms_ssim,
    _ssim_single,
)


class TestGaussianWindow:
    """Test Gaussian window generation."""

    def test_window_shape(self):
        """Test window has correct shape."""
        window = _create_gaussian_window(11, 1.5, channels=1)
        assert window.shape == (1, 1, 11, 11), f"Expected (1, 1, 11, 11), got {window.shape}"

    def test_window_shape_multichannel(self):
        """Test window shape with multiple channels."""
        window = _create_gaussian_window(11, 1.5, channels=3)
        assert window.shape == (3, 1, 11, 11), f"Expected (3, 1, 11, 11), got {window.shape}"

    def test_window_normalization(self):
        """Test window sums to 1."""
        window = _create_gaussian_window(11, 1.5, channels=1)
        # Each channel's window should sum to 1
        window_sum = window[0, 0].sum()
        assert torch.allclose(window_sum, torch.tensor(1.0), atol=1e-6), (
            f"Window sum should be 1.0, got {window_sum}"
        )

    def test_window_symmetry(self):
        """Test window is symmetric."""
        window = _create_gaussian_window(11, 1.5, channels=1)
        w = window[0, 0]

        # Test horizontal symmetry
        assert torch.allclose(w, w.T, atol=1e-6), "Window should be symmetric (transpose)"

        # Test vertical flip symmetry
        assert torch.allclose(w, w.flip(0), atol=1e-6), "Window should be symmetric (vertical flip)"

        # Test horizontal flip symmetry
        assert torch.allclose(w, w.flip(1), atol=1e-6), (
            "Window should be symmetric (horizontal flip)"
        )

    def test_window_center_peak(self):
        """Test that window has maximum value at center."""
        window = _create_gaussian_window(11, 1.5, channels=1)
        w = window[0, 0]

        # Center should be the maximum value
        center_value = w[5, 5]  # Center of 11x11 window
        max_value = w.max()

        assert torch.allclose(center_value, max_value, atol=1e-6), (
            "Center should have maximum value"
        )

    def test_window_different_sizes(self):
        """Test window generation with different sizes."""
        for size in [7, 9, 11, 13, 15]:
            window = _create_gaussian_window(size, 1.5, channels=1)
            assert window.shape == (1, 1, size, size)
            assert torch.allclose(window.sum(), torch.tensor(1.0), atol=1e-6)

    def test_window_different_sigma(self):
        """Test window generation with different sigma values."""
        for sigma in [0.5, 1.0, 1.5, 2.0, 3.0]:
            window = _create_gaussian_window(11, sigma, channels=1)
            assert window.shape == (1, 1, 11, 11)
            assert torch.allclose(window.sum(), torch.tensor(1.0), atol=1e-6)

    def test_window_contiguous(self):
        """Test that window is contiguous in memory."""
        window = _create_gaussian_window(11, 1.5, channels=1)
        assert window.is_contiguous(), "Window should be contiguous"


class TestGaussianFilter:
    """Test Gaussian filtering function."""

    def test_filter_preserves_shape(self):
        """Test that filtering preserves input shape."""
        img = torch.rand(1, 1, 128, 128)
        window = _create_gaussian_window(11, 1.5, channels=1)
        filtered = _gaussian_filter(img, window)

        assert filtered.shape == img.shape, f"Shape mismatch: {filtered.shape} vs {img.shape}"

    def test_filter_preserves_shape_multichannel(self):
        """Test filtering with multiple channels."""
        img = torch.rand(2, 3, 64, 64)
        window = _create_gaussian_window(11, 1.5, channels=3)
        window = window.to(img.device)
        filtered = _gaussian_filter(img, window)

        assert filtered.shape == img.shape, f"Shape mismatch: {filtered.shape} vs {img.shape}"

    def test_filter_constant_image(self):
        """Test that constant images remain approximately constant after filtering."""
        constant_value = 0.5
        img = torch.full((1, 1, 64, 64), constant_value)
        window = _create_gaussian_window(11, 1.5, channels=1)
        filtered = _gaussian_filter(img, window)

        # Filtered constant image should be approximately constant in the interior
        # (edges are affected by padding mode)
        interior = filtered[:, :, 10:-10, 10:-10]
        assert torch.allclose(interior, torch.full_like(interior, constant_value), atol=1e-5), (
            "Constant image should remain constant in interior after filtering"
        )

    def test_filter_smoothing(self):
        """Test that filtering reduces variance (smoothing property)."""
        torch.manual_seed(42)
        img = torch.rand(1, 1, 128, 128)
        window = _create_gaussian_window(11, 1.5, channels=1)
        filtered = _gaussian_filter(img, window)

        # Variance should decrease after smoothing
        original_var = img.var()
        filtered_var = filtered.var()

        assert filtered_var < original_var, (
            f"Filtering should reduce variance: {filtered_var} >= {original_var}"
        )

    def test_filter_mean_preservation(self):
        """Test that filtering approximately preserves mean value."""
        torch.manual_seed(42)
        img = torch.rand(1, 1, 128, 128)
        window = _create_gaussian_window(11, 1.5, channels=1)
        filtered = _gaussian_filter(img, window)

        # Mean should be approximately preserved (within 2% due to padding effects)
        original_mean = img.mean()
        filtered_mean = filtered.mean()
        relative_error = abs(original_mean - filtered_mean) / original_mean

        assert relative_error < 0.02, (
            f"Mean should be approximately preserved: {original_mean} vs {filtered_mean} (error: {relative_error:.4f})"
        )

    def test_filter_range_preservation(self):
        """Test that filtering doesn't create values outside input range."""
        img = torch.rand(1, 1, 64, 64)
        window = _create_gaussian_window(11, 1.5, channels=1)
        filtered = _gaussian_filter(img, window)

        # Filtered values should be within original range
        assert filtered.min() >= 0.0, "Filtered min should be >= 0"
        assert filtered.max() <= 1.0, "Filtered max should be <= 1"

    def test_filter_different_window_sizes(self):
        """Test filtering with different window sizes."""
        img = torch.rand(1, 1, 128, 128)

        for size in [7, 11, 15]:
            window = _create_gaussian_window(size, 1.5, channels=1)
            filtered = _gaussian_filter(img, window)
            assert filtered.shape == img.shape

    def test_filter_batch_processing(self):
        """Test filtering with batch dimension."""
        batch_size = 4
        img = torch.rand(batch_size, 1, 64, 64)
        window = _create_gaussian_window(11, 1.5, channels=1)
        filtered = _gaussian_filter(img, window)

        assert filtered.shape == img.shape
        assert filtered.shape[0] == batch_size


class TestGaussianWindowFilterIntegration:
    """Test integration between window generation and filtering."""

    def test_window_device_compatibility(self):
        """Test that window and image can be on different devices initially."""
        img = torch.rand(1, 1, 64, 64)
        window = _create_gaussian_window(11, 1.5, channels=1)

        # Move window to same device as image
        window = window.to(img.device)
        filtered = _gaussian_filter(img, window)

        assert filtered.device == img.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Test SSIM helper functions work on CUDA."""
        img = torch.rand(1, 1, 64, 64).cuda()
        window = _create_gaussian_window(11, 1.5, channels=1).cuda()
        filtered = _gaussian_filter(img, window)

        assert filtered.is_cuda
        assert filtered.shape == img.shape

    def test_different_dtypes(self):
        """Test filtering with different data types."""
        for dtype in [torch.float32, torch.float64]:
            img = torch.rand(1, 1, 64, 64, dtype=dtype)
            window = _create_gaussian_window(11, 1.5, channels=1).to(dtype)
            filtered = _gaussian_filter(img, window)

            assert filtered.dtype == dtype

    def test_gradient_flow(self):
        """Test that gradients flow through filtering."""
        img = torch.rand(1, 1, 64, 64, requires_grad=True)
        window = _create_gaussian_window(11, 1.5, channels=1)
        filtered = _gaussian_filter(img, window)

        # Compute some loss
        loss = filtered.sum()
        loss.backward()

        assert img.grad is not None, "Gradients should flow through filter"
        assert not torch.isnan(img.grad).any(), "Gradients should not contain NaN"
        assert img.grad.shape == img.shape


class TestSSIMSingle:
    """Test single-scale SSIM computation (_ssim_single)."""

    def test_perfect_similarity(self):
        """Test SSIM = 1 for identical images."""
        torch.manual_seed(42)
        img = torch.rand(1, 1, 128, 128)
        window = _create_gaussian_window(11, 1.5, channels=1)
        window = window.to(img.device).type_as(img)

        ssim_val = _ssim_single(img, img, window)

        assert torch.allclose(ssim_val, torch.tensor(1.0), atol=1e-6), (
            f"SSIM should be 1.0 for identical images, got {ssim_val}"
        )

    def test_ssim_range(self):
        """Test SSIM is in valid range [0, 1]."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 128, 128)
        img2 = torch.rand(1, 1, 128, 128)
        window = _create_gaussian_window(11, 1.5, channels=1)
        window = window.to(img1.device).type_as(img1)

        ssim_val = _ssim_single(img1, img2, window)

        assert 0 <= ssim_val <= 1, f"SSIM should be in [0, 1], got {ssim_val}"

    def test_ssim_symmetry(self):
        """Test SSIM(x, y) = SSIM(y, x)."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 64, 64)
        img2 = torch.rand(1, 1, 64, 64)
        window = _create_gaussian_window(11, 1.5, channels=1)
        window = window.to(img1.device).type_as(img1)

        ssim_12 = _ssim_single(img1, img2, window)
        ssim_21 = _ssim_single(img2, img1, window)

        assert torch.allclose(ssim_12, ssim_21, atol=1e-6), (
            f"SSIM should be symmetric: {ssim_12} vs {ssim_21}"
        )

    def test_ssim_similar_images(self):
        """Test SSIM for slightly perturbed images."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 128, 128)
        # Add small noise
        img2 = img1 + 0.05 * torch.randn(1, 1, 128, 128)
        img2 = torch.clamp(img2, 0, 1)

        window = _create_gaussian_window(11, 1.5, channels=1)
        window = window.to(img1.device).type_as(img1)

        ssim_val = _ssim_single(img1, img2, window)

        # Similar images should have high SSIM (> 0.9)
        assert ssim_val > 0.9, f"Similar images should have high SSIM, got {ssim_val}"

    def test_ssim_different_images(self):
        """Test SSIM for very different images."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 128, 128)
        img2 = torch.rand(1, 1, 128, 128)

        window = _create_gaussian_window(11, 1.5, channels=1)
        window = window.to(img1.device).type_as(img1)

        ssim_val = _ssim_single(img1, img2, window)

        # Random different images should have lower SSIM
        assert ssim_val < 1.0, f"Different images should have SSIM < 1, got {ssim_val}"

    def test_ssim_multichannel(self):
        """Test SSIM with multiple channels."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 64, 64)
        img2 = torch.rand(1, 3, 64, 64)

        window = _create_gaussian_window(11, 1.5, channels=3)
        window = window.to(img1.device).type_as(img1)

        ssim_val = _ssim_single(img1, img2, window)

        assert isinstance(ssim_val, torch.Tensor)
        assert ssim_val.dim() == 0  # Scalar
        # SSIM can range from -1 to 1, but for non-negative images typically [0, 1]
        assert -1 <= ssim_val <= 1

    def test_ssim_batch_processing(self):
        """Test SSIM with batch dimension."""
        torch.manual_seed(42)
        batch_size = 4
        img1 = torch.rand(batch_size, 1, 64, 64)
        img2 = torch.rand(batch_size, 1, 64, 64)

        window = _create_gaussian_window(11, 1.5, channels=1)
        window = window.to(img1.device).type_as(img1)

        ssim_val = _ssim_single(img1, img2, window)

        assert isinstance(ssim_val, torch.Tensor)
        assert ssim_val.dim() == 0  # Scalar (averaged over batch)
        assert 0 <= ssim_val <= 1

    def test_ssim_different_window_sizes(self):
        """Test SSIM with different window sizes."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 128, 128)
        img2 = torch.rand(1, 1, 128, 128)

        for window_size in [7, 11, 15]:
            window = _create_gaussian_window(window_size, 1.5, channels=1)
            window = window.to(img1.device).type_as(img1)

            ssim_val = _ssim_single(img1, img2, window)

            assert 0 <= ssim_val <= 1, f"SSIM with window_size={window_size} out of range"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ssim_cuda(self):
        """Test SSIM on CUDA device."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 128, 128).cuda()
        img2 = torch.rand(1, 1, 128, 128).cuda()

        window = _create_gaussian_window(11, 1.5, channels=1).cuda()

        ssim_val = _ssim_single(img1, img2, window)

        assert ssim_val.is_cuda
        assert 0 <= ssim_val <= 1


class TestSSIMDifferentiability:
    """Test SSIM is differentiable."""

    def test_gradients_exist(self):
        """Test gradients can be computed."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 64, 64, requires_grad=True)
        img2 = torch.rand(1, 1, 64, 64)

        window = _create_gaussian_window(11, 1.5, channels=1)
        window = window.to(img1.device).type_as(img1)

        ssim_val = _ssim_single(img1, img2, window)
        ssim_val.backward()

        assert img1.grad is not None, "Gradients should exist"
        assert not torch.isnan(img1.grad).any(), "Gradients should not contain NaN"
        assert not torch.isinf(img1.grad).any(), "Gradients should not contain inf"

    def test_gradient_shape(self):
        """Test gradient has same shape as input."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 64, 64, requires_grad=True)
        img2 = torch.rand(1, 1, 64, 64)

        window = _create_gaussian_window(11, 1.5, channels=1)
        window = window.to(img1.device).type_as(img1)

        ssim_val = _ssim_single(img1, img2, window)
        ssim_val.backward()

        assert img1.grad.shape == img1.shape, "Gradient shape should match input shape"

    def test_gradient_check(self):
        """Test gradients are numerically correct using gradcheck."""
        torch.manual_seed(42)
        # Use smaller image for gradcheck (faster)
        img1 = torch.rand(1, 1, 32, 32, dtype=torch.double, requires_grad=True)
        img2 = torch.rand(1, 1, 32, 32, dtype=torch.double)

        window = _create_gaussian_window(11, 1.5, channels=1).to(torch.double)

        def ssim_func(x):
            return _ssim_single(x, img2, window)

        # gradcheck verifies numerical gradient vs analytical gradient
        result = torch.autograd.gradcheck(ssim_func, img1, eps=1e-6, atol=1e-4)

        assert result, "Gradient check failed"

    def test_gradient_flow_through_wrapper(self):
        """Test gradients flow through _compute_ssim wrapper."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 64, 64, requires_grad=True)
        img2 = torch.rand(1, 1, 64, 64)

        ssim_val = _compute_ssim(img1, img2)
        ssim_val.backward()

        assert img1.grad is not None
        assert not torch.isnan(img1.grad).any()


class TestComputeSSIM:
    """Test _compute_ssim wrapper function."""

    def test_basic_computation(self):
        """Test basic SSIM computation."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 128, 128)
        img2 = torch.rand(1, 1, 128, 128)

        ssim_val = _compute_ssim(img1, img2)

        assert isinstance(ssim_val, torch.Tensor)
        assert ssim_val.dim() == 0  # Scalar
        assert 0 <= ssim_val <= 1

    def test_3d_input_handling(self):
        """Test that 3D inputs are handled correctly."""
        torch.manual_seed(42)
        # 3D input: [C, H, W]
        img1 = torch.rand(1, 128, 128)
        img2 = torch.rand(1, 128, 128)

        ssim_val = _compute_ssim(img1, img2)

        assert isinstance(ssim_val, torch.Tensor)
        assert 0 <= ssim_val <= 1

    def test_4d_input_handling(self):
        """Test that 4D inputs work correctly."""
        torch.manual_seed(42)
        # 4D input: [B, C, H, W]
        img1 = torch.rand(2, 1, 128, 128)
        img2 = torch.rand(2, 1, 128, 128)

        ssim_val = _compute_ssim(img1, img2)

        assert isinstance(ssim_val, torch.Tensor)
        assert 0 <= ssim_val <= 1

    def test_perfect_match(self):
        """Test SSIM = 1 for identical images."""
        torch.manual_seed(42)
        img = torch.rand(1, 1, 128, 128)

        ssim_val = _compute_ssim(img, img)

        assert torch.allclose(ssim_val, torch.tensor(1.0), atol=1e-6)

    def test_different_parameters(self):
        """Test SSIM with different window_size and sigma."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 128, 128)
        img2 = torch.rand(1, 1, 128, 128)

        # Test with different window size
        ssim_7 = _compute_ssim(img1, img2, window_size=7, sigma=1.5)
        ssim_11 = _compute_ssim(img1, img2, window_size=11, sigma=1.5)

        # Both should be valid
        assert 0 <= ssim_7 <= 1
        assert 0 <= ssim_11 <= 1

        # Test with different sigma
        ssim_s1 = _compute_ssim(img1, img2, window_size=11, sigma=1.0)
        ssim_s2 = _compute_ssim(img1, img2, window_size=11, sigma=2.0)

        assert 0 <= ssim_s1 <= 1
        assert 0 <= ssim_s2 <= 1

    def test_data_range_parameter(self):
        """Test SSIM with different data_range values."""
        torch.manual_seed(42)
        # Image in range [0, 255]
        img1 = torch.rand(1, 1, 64, 64) * 255
        img2 = torch.rand(1, 1, 64, 64) * 255

        ssim_val = _compute_ssim(img1, img2, data_range=255.0)

        assert 0 <= ssim_val <= 1


class TestSSIMAccuracy:
    """Test SSIM matches scikit-image implementation."""

    @pytest.fixture
    def test_images(self):
        """Create test image pairs."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 256, 256)
        img2 = img1 + 0.1 * torch.randn(1, 1, 256, 256)
        img2 = torch.clamp(img2, 0, 1)
        return img1, img2

    def test_ssim_vs_skimage(self, test_images):
        """Compare PyTorch SSIM to scikit-image SSIM."""
        pytest.importorskip("skimage", reason="scikit-image not available")

        from prism.utils.metrics import compute_ssim

        img1, img2 = test_images

        # Compute with PyTorch
        ssim_torch = _compute_ssim(img1, img2)

        # Compute with scikit-image
        ssim_sk = compute_ssim(img1, img2)

        # Should match within numerical precision
        # Using reflection padding to match scikit-image
        # Tolerance of 2e-4 accounts for float32 precision and minor implementation differences
        assert torch.allclose(
            ssim_torch, torch.tensor(ssim_sk, dtype=ssim_torch.dtype), atol=2e-4
        ), f"SSIM mismatch: PyTorch={ssim_torch:.6f}, scikit-image={ssim_sk:.6f}"

    def test_ssim_vs_skimage_random_images(self):
        """Test SSIM vs scikit-image on multiple random image pairs."""
        pytest.importorskip("skimage", reason="scikit-image not available")

        from prism.utils.metrics import compute_ssim

        torch.manual_seed(123)

        for i in range(5):
            img1 = torch.rand(1, 1, 128, 128)
            img2 = torch.rand(1, 1, 128, 128)

            ssim_torch = _compute_ssim(img1, img2)
            ssim_sk = compute_ssim(img1, img2)

            # Should match closely with reflection padding
            # Higher tolerance for random images due to low absolute SSIM values
            # SSIM can be negative for very dissimilar images
            # Use absolute difference check
            diff = abs(ssim_torch.item() - ssim_sk)
            assert diff < 5e-3, (
                f"Test {i}: SSIM mismatch: PyTorch={ssim_torch:.6f}, scikit-image={ssim_sk:.6f}, diff={diff:.6f}"
            )


class TestMSSSIM:
    """Test Multi-Scale SSIM (_ms_ssim)."""

    def test_perfect_similarity(self):
        """Test MS-SSIM = 1 for identical images."""
        torch.manual_seed(42)
        img = torch.rand(1, 1, 256, 256)
        ms_ssim = _ms_ssim(img, img)
        assert torch.allclose(ms_ssim, torch.tensor(1.0), atol=1e-5), (
            f"MS-SSIM should be 1.0 for identical images, got {ms_ssim}"
        )

    def test_ms_ssim_range(self):
        """Test MS-SSIM is in valid range [0, 1]."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 256, 256)
        img2 = img1 + 0.1 * torch.randn(1, 1, 256, 256)
        img2 = torch.clamp(img2, 0, 1)

        ms_ssim = _ms_ssim(img1, img2)

        assert 0 <= ms_ssim <= 1, f"MS-SSIM should be in [0, 1], got {ms_ssim}"

    def test_small_images(self):
        """Test MS-SSIM handles small images gracefully."""
        torch.manual_seed(42)
        # Image too small for 5 scales (64x64 can only do 2-3 scales)
        img1 = torch.rand(1, 1, 64, 64)
        img2 = torch.rand(1, 1, 64, 64)

        # Should not crash, should use fewer scales
        ms_ssim = _ms_ssim(img1, img2)
        assert 0 <= ms_ssim <= 1, f"MS-SSIM should be in [0, 1] for small images, got {ms_ssim}"
        assert isinstance(ms_ssim, torch.Tensor)
        assert ms_ssim.dim() == 0  # Scalar

    def test_ms_ssim_symmetry(self):
        """Test MS-SSIM(x, y) = MS-SSIM(y, x)."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 256, 256)
        img2 = torch.rand(1, 1, 256, 256)

        ms_ssim_12 = _ms_ssim(img1, img2)
        ms_ssim_21 = _ms_ssim(img2, img1)

        assert torch.allclose(ms_ssim_12, ms_ssim_21, atol=1e-6), (
            f"MS-SSIM should be symmetric: {ms_ssim_12} vs {ms_ssim_21}"
        )

    def test_ms_ssim_vs_ssim(self):
        """Test MS-SSIM is comparable to single-scale SSIM."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 256, 256)
        img2 = img1 + 0.05 * torch.randn(1, 1, 256, 256)
        img2 = torch.clamp(img2, 0, 1)

        ssim = _compute_ssim(img1, img2)
        ms_ssim = _ms_ssim(img1, img2)

        # Both should be high for similar images
        assert ssim > 0.9
        assert ms_ssim > 0.9
        # MS-SSIM should be in reasonable range compared to SSIM
        # (they don't have to be exactly equal)
        assert isinstance(ms_ssim, torch.Tensor)

    def test_custom_weights(self):
        """Test MS-SSIM with custom weights."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 256, 256)
        img2 = torch.rand(1, 1, 256, 256)

        # 3-scale MS-SSIM
        custom_weights = [0.33, 0.33, 0.34]
        ms_ssim = _ms_ssim(img1, img2, weights=custom_weights)

        assert 0 <= ms_ssim <= 1
        assert isinstance(ms_ssim, torch.Tensor)

    def test_different_scales(self):
        """Test MS-SSIM with different number of scales."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 256, 256)
        img2 = torch.rand(1, 1, 256, 256)

        # Test 2, 3, 4, and 5 scales
        for num_scales in [2, 3, 4, 5]:
            weights = [1.0 / num_scales] * num_scales
            ms_ssim = _ms_ssim(img1, img2, weights=weights)
            assert 0 <= ms_ssim <= 1, f"MS-SSIM with {num_scales} scales out of range"

    def test_3d_input_handling(self):
        """Test MS-SSIM handles 3D inputs correctly."""
        torch.manual_seed(42)
        # 3D input: [C, H, W]
        img1 = torch.rand(1, 256, 256)
        img2 = torch.rand(1, 256, 256)

        ms_ssim = _ms_ssim(img1, img2)

        assert isinstance(ms_ssim, torch.Tensor)
        assert 0 <= ms_ssim <= 1

    def test_multichannel(self):
        """Test MS-SSIM with multiple channels."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 256, 256)
        img2 = torch.rand(1, 3, 256, 256)

        ms_ssim = _ms_ssim(img1, img2)

        assert isinstance(ms_ssim, torch.Tensor)
        assert ms_ssim.dim() == 0  # Scalar
        # MS-SSIM can range from -1 to 1, but typically [0, 1] for non-negative images
        assert -1 <= ms_ssim <= 1

    def test_batch_processing(self):
        """Test MS-SSIM with batch dimension."""
        torch.manual_seed(42)
        batch_size = 4
        img1 = torch.rand(batch_size, 1, 256, 256)
        img2 = torch.rand(batch_size, 1, 256, 256)

        ms_ssim = _ms_ssim(img1, img2)

        assert isinstance(ms_ssim, torch.Tensor)
        assert ms_ssim.dim() == 0  # Scalar (averaged over batch)
        assert 0 <= ms_ssim <= 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Test MS-SSIM on CUDA device."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 256, 256).cuda()
        img2 = torch.rand(1, 1, 256, 256).cuda()

        ms_ssim = _ms_ssim(img1, img2)

        assert ms_ssim.is_cuda
        assert 0 <= ms_ssim <= 1

    def test_different_data_ranges(self):
        """Test MS-SSIM with different data ranges."""
        torch.manual_seed(42)
        # Image in range [0, 255]
        img1 = torch.rand(1, 1, 256, 256) * 255
        img2 = torch.rand(1, 1, 256, 256) * 255

        ms_ssim = _ms_ssim(img1, img2, data_range=255.0)

        assert 0 <= ms_ssim <= 1


class TestMSSSIMDifferentiability:
    """Test MS-SSIM is differentiable."""

    def test_gradients_exist(self):
        """Test gradients can be computed."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 256, 256, requires_grad=True)
        img2 = torch.rand(1, 1, 256, 256)

        ms_ssim = _ms_ssim(img1, img2)
        ms_ssim.backward()

        assert img1.grad is not None, "Gradients should exist"
        assert not torch.isnan(img1.grad).any(), "Gradients should not contain NaN"
        assert not torch.isinf(img1.grad).any(), "Gradients should not contain inf"

    def test_gradient_shape(self):
        """Test gradient has same shape as input."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 256, 256, requires_grad=True)
        img2 = torch.rand(1, 1, 256, 256)

        ms_ssim = _ms_ssim(img1, img2)
        ms_ssim.backward()

        assert img1.grad.shape == img1.shape, "Gradient shape should match input shape"

    def test_gradient_check(self):
        """Test gradients are numerically correct using gradcheck."""
        torch.manual_seed(42)
        # Use smaller image for gradcheck (faster)
        # Need at least 128x128 for 4 scales
        img1 = torch.rand(1, 1, 128, 128, dtype=torch.double, requires_grad=True)
        img2 = torch.rand(1, 1, 128, 128, dtype=torch.double)

        def ms_ssim_func(x):
            # Use 3 scales to speed up gradcheck
            return _ms_ssim(x, img2, weights=[0.33, 0.33, 0.34])

        # gradcheck verifies numerical gradient vs analytical gradient
        result = torch.autograd.gradcheck(ms_ssim_func, img1, eps=1e-6, atol=1e-4)

        assert result, "Gradient check failed"

    def test_backward_pass(self):
        """Test backward pass works without errors."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 256, 256, requires_grad=True)
        img2 = torch.rand(1, 1, 256, 256)

        ms_ssim = _ms_ssim(img1, img2)
        loss = 1.0 - ms_ssim  # DSSIM-like loss
        loss.backward()

        assert img1.grad is not None
        assert not torch.isnan(img1.grad).any()


class TestMSSSIMEdgeCases:
    """Test MS-SSIM edge cases and robustness."""

    def test_very_small_image(self):
        """Test MS-SSIM with image smaller than default scales."""
        torch.manual_seed(42)
        # Very small image (32x32) - can only do 1-2 scales with window_size=11
        img1 = torch.rand(1, 1, 32, 32)
        img2 = torch.rand(1, 1, 32, 32)

        # Should handle gracefully with early termination
        ms_ssim = _ms_ssim(img1, img2)
        assert 0 <= ms_ssim <= 1
        assert not torch.isnan(ms_ssim)
        assert not torch.isinf(ms_ssim)

    def test_single_scale_fallback(self):
        """Test that MS-SSIM with 1 scale is similar to regular SSIM."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 128, 128)
        img2 = torch.rand(1, 1, 128, 128)

        # Single scale MS-SSIM
        ms_ssim_1 = _ms_ssim(img1, img2, weights=[1.0])

        # Regular SSIM
        ssim = _compute_ssim(img1, img2)

        # Should be approximately equal (within numerical precision)
        assert torch.allclose(ms_ssim_1, ssim, atol=1e-4), (
            f"Single-scale MS-SSIM should match SSIM: {ms_ssim_1} vs {ssim}"
        )

    def test_constant_images(self):
        """Test MS-SSIM with constant images."""
        # Two identical constant images should have MS-SSIM = 1
        img = torch.full((1, 1, 256, 256), 0.5)

        ms_ssim = _ms_ssim(img, img)
        assert torch.allclose(ms_ssim, torch.tensor(1.0), atol=1e-5)

    def test_black_vs_white(self):
        """Test MS-SSIM with completely different images."""
        img1 = torch.zeros(1, 1, 256, 256)
        img2 = torch.ones(1, 1, 256, 256)

        ms_ssim = _ms_ssim(img1, img2)

        # Should be very low but not necessarily 0
        assert ms_ssim < 0.5, f"MS-SSIM of black vs white should be low, got {ms_ssim}"

    def test_different_window_sizes(self):
        """Test MS-SSIM with different window sizes."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 256, 256)
        img2 = torch.rand(1, 1, 256, 256)

        for window_size in [7, 11, 15]:
            ms_ssim = _ms_ssim(img1, img2, window_size=window_size)
            assert 0 <= ms_ssim <= 1, f"MS-SSIM with window_size={window_size} out of range"

    def test_similar_images_high_score(self):
        """Test MS-SSIM gives high score for very similar images."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 256, 256)
        # Very small noise
        img2 = img1 + 0.01 * torch.randn(1, 1, 256, 256)
        img2 = torch.clamp(img2, 0, 1)

        ms_ssim = _ms_ssim(img1, img2)

        # Very similar images should have very high MS-SSIM
        assert ms_ssim > 0.95, f"Very similar images should have high MS-SSIM, got {ms_ssim}"


class TestLossAggregatorExisting:
    """Test existing LossAggregator functionality (regression tests)."""

    def test_l1_loss_initialization(self):
        """Test L1 loss initialization."""
        criterion = LossAggregator(loss_type="l1")
        assert criterion.loss_type == "l1"

    def test_l2_loss_initialization(self):
        """Test L2 loss initialization."""
        criterion = LossAggregator(loss_type="l2")
        assert criterion.loss_type == "l2"

    def test_invalid_loss_type(self):
        """Test that invalid loss type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            LossAggregator(loss_type="invalid")  # type: ignore[arg-type]

    def test_dual_loss_output(self):
        """Test that LossAggregator returns dual losses."""
        criterion = LossAggregator(loss_type="l1")

        inputs = torch.rand(1, 1, 64, 64)
        target = torch.rand(2, 1, 64, 64)

        loss_old, loss_new = criterion(inputs, target, telescope=None)

        assert isinstance(loss_old, torch.Tensor)
        assert isinstance(loss_new, torch.Tensor)
        assert loss_old.dim() == 0, "loss_old should be scalar"
        assert loss_new.dim() == 0, "loss_new should be scalar"

    def test_loss_positive(self):
        """Test that losses are positive."""
        criterion = LossAggregator(loss_type="l1")

        inputs = torch.rand(1, 1, 64, 64)
        target = torch.rand(2, 1, 64, 64)

        loss_old, loss_new = criterion(inputs, target, telescope=None)

        assert loss_old >= 0, "L1 loss should be non-negative"
        assert loss_new >= 0, "L1 loss should be non-negative"

    def test_backward_pass(self):
        """Test that backward pass works."""
        criterion = LossAggregator(loss_type="l1")

        inputs = torch.rand(1, 1, 64, 64, requires_grad=True)
        target = torch.rand(2, 1, 64, 64)

        loss_old, loss_new = criterion(inputs, target, telescope=None)
        loss = loss_old + loss_new
        loss.backward()

        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()


class TestLossAggregatorSSIM:
    """Test LossAggregator with SSIM and MS-SSIM loss types."""

    def test_ssim_loss_initialization(self):
        """Test SSIM loss initialization."""
        criterion = LossAggregator(loss_type="ssim")
        assert criterion.loss_type == "ssim"
        assert criterion.window_size == 11
        assert criterion.sigma == 1.5
        assert criterion.data_range == 1.0
        assert criterion.window is None  # Created on first forward pass

    def test_ms_ssim_loss_initialization(self):
        """Test MS-SSIM loss initialization."""
        criterion = LossAggregator(loss_type="ms-ssim")
        assert criterion.loss_type == "ms-ssim"
        assert criterion.window_size == 11
        assert criterion.sigma == 1.5
        assert criterion.data_range == 1.0

    def test_ssim_dual_loss_output(self):
        """Test that SSIM returns dual losses."""
        criterion = LossAggregator(loss_type="ssim")

        inputs = torch.rand(1, 1, 128, 128)
        target = torch.rand(2, 1, 128, 128)

        loss_old, loss_new = criterion(inputs, target)

        assert isinstance(loss_old, torch.Tensor)
        assert isinstance(loss_new, torch.Tensor)
        assert loss_old.dim() == 0, "loss_old should be scalar"
        assert loss_new.dim() == 0, "loss_new should be scalar"

    def test_ms_ssim_dual_loss_output(self):
        """Test that MS-SSIM returns dual losses."""
        criterion = LossAggregator(loss_type="ms-ssim")

        inputs = torch.rand(1, 1, 256, 256)  # Larger for MS-SSIM
        target = torch.rand(2, 1, 256, 256)

        loss_old, loss_new = criterion(inputs, target)

        assert isinstance(loss_old, torch.Tensor)
        assert isinstance(loss_new, torch.Tensor)
        assert loss_old.dim() == 0
        assert loss_new.dim() == 0

    def test_dssim_range(self):
        """Test DSSIM losses are in expected range (approximately [0, 0.5])."""
        criterion = LossAggregator(loss_type="ssim")

        inputs = torch.rand(1, 1, 128, 128)
        target = torch.rand(2, 1, 128, 128)

        loss_old, loss_new = criterion(inputs, target)

        # DSSIM = (1 - SSIM) / 2 → typically [0, 0.5]
        # SSIM can go slightly below -1 for very dissimilar images
        # so allow slight tolerance above 0.5
        # In practice, for random images, should be around 0.5
        assert 0 <= loss_old <= 0.6, f"DSSIM should be approximately in [0, 0.5], got {loss_old}"
        assert 0 <= loss_new <= 0.6, f"DSSIM should be approximately in [0, 0.5], got {loss_new}"

    def test_ssim_perfect_similarity(self):
        """Test SSIM loss is ~0 for identical images."""
        criterion = LossAggregator(loss_type="ssim")

        img = torch.rand(1, 1, 128, 128)
        inputs = img.clone()
        # Create target with correct shape [2, C, H, W]
        target = torch.cat([img, img], dim=0)

        loss_old, loss_new = criterion(inputs, target)

        # Perfect similarity → SSIM = 1 → DSSIM = 0
        assert loss_old < 1e-5, f"Perfect match should have loss ~0, got {loss_old}"
        assert loss_new < 1e-5, f"Perfect match should have loss ~0, got {loss_new}"

    def test_ms_ssim_perfect_similarity(self):
        """Test MS-SSIM loss is ~0 for identical images."""
        criterion = LossAggregator(loss_type="ms-ssim")

        img = torch.rand(1, 1, 256, 256)
        inputs = img.clone()
        # Create target with correct shape [2, C, H, W]
        target = torch.cat([img, img], dim=0)

        loss_old, loss_new = criterion(inputs, target)

        # Perfect similarity → MS-SSIM = 1 → DSSIM = 0
        assert loss_old < 1e-4, f"Perfect match should have loss ~0, got {loss_old}"
        assert loss_new < 1e-4, f"Perfect match should have loss ~0, got {loss_new}"

    def test_ssim_backward_pass(self):
        """Test SSIM loss works in backward pass."""
        criterion = LossAggregator(loss_type="ssim")

        inputs = torch.rand(1, 1, 128, 128, requires_grad=True)
        target = torch.rand(2, 1, 128, 128)

        loss_old, loss_new = criterion(inputs, target)
        loss = loss_old + loss_new
        loss.backward()

        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()

    def test_ms_ssim_backward_pass(self):
        """Test MS-SSIM loss works in backward pass."""
        criterion = LossAggregator(loss_type="ms-ssim")

        inputs = torch.rand(1, 1, 256, 256, requires_grad=True)
        target = torch.rand(2, 1, 256, 256)

        loss_old, loss_new = criterion(inputs, target)
        loss = loss_old + loss_new
        loss.backward()

        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()

    def test_ssim_window_caching(self):
        """Test that Gaussian window is cached after first forward pass."""
        criterion = LossAggregator(loss_type="ssim")

        assert criterion.window is None

        inputs = torch.rand(1, 1, 128, 128)
        target = torch.rand(2, 1, 128, 128)

        # First forward pass creates window
        criterion(inputs, target)
        assert criterion.window is not None
        first_window = criterion.window

        # Second forward pass reuses window
        criterion(inputs, target)
        assert criterion.window is first_window

    def test_ssim_different_targets(self):
        """Test SSIM with different old and new targets."""
        criterion = LossAggregator(loss_type="ssim")

        inputs = torch.rand(1, 1, 128, 128)
        target_old = torch.rand(1, 1, 128, 128)
        target_new = torch.rand(1, 1, 128, 128)
        target = torch.cat([target_old, target_new], dim=0)

        loss_old, loss_new = criterion(inputs, target)

        # Losses should be different (very unlikely to be equal)
        assert not torch.allclose(loss_old, loss_new, atol=1e-5)

    def test_ssim_vs_l1_different_behavior(self):
        """Test that SSIM and L1 losses can be computed independently."""
        criterion_ssim = LossAggregator(loss_type="ssim")
        criterion_l1 = LossAggregator(loss_type="l1")

        inputs = torch.rand(1, 1, 128, 128)
        target = torch.rand(2, 1, 128, 128)

        # Both should compute successfully
        loss_ssim_old, loss_ssim_new = criterion_ssim(inputs, target)
        loss_l1_old, loss_l1_new = criterion_l1(inputs, target, telescope=None)

        # Verify both return valid scalar tensors
        assert loss_ssim_old.dim() == 0 and loss_l1_old.dim() == 0
        assert loss_ssim_new.dim() == 0 and loss_l1_new.dim() == 0

        # SSIM should be approximately in [0, 0.5] range (can exceed slightly for very dissimilar images)
        # L1 is unbounded above
        assert 0 <= loss_ssim_old <= 0.6
        assert 0 <= loss_l1_old  # L1 is unbounded above

    def test_ssim_batch_dimension_handling(self):
        """Test SSIM handles 3D target tensors correctly."""
        criterion = LossAggregator(loss_type="ssim")

        inputs = torch.rand(1, 1, 128, 128)
        # Create target with 3D tensors (will be unsqueezed to 4D)
        target_old = torch.rand(1, 128, 128)
        target_new = torch.rand(1, 128, 128)
        target = torch.stack([target_old, target_new], dim=0)

        # Should not raise an error
        loss_old, loss_new = criterion(inputs, target)

        assert isinstance(loss_old, torch.Tensor)
        assert isinstance(loss_new, torch.Tensor)


class TestL1LossStrategy:
    """Test L1LossStrategy."""

    def test_strategy_initialization(self):
        """Test L1 strategy can be initialized."""
        strategy = L1LossStrategy()
        assert strategy.name == "l1"

    def test_perfect_match(self):
        """Test L1 loss is zero for identical inputs."""
        strategy = L1LossStrategy()
        pred = torch.rand(1, 1, 64, 64)
        loss = strategy(pred, pred)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_loss_positive(self):
        """Test L1 loss is non-negative."""
        strategy = L1LossStrategy()
        pred = torch.rand(1, 1, 64, 64)
        target = torch.rand(1, 1, 64, 64)
        loss = strategy(pred, target)
        assert loss >= 0

    def test_loss_symmetry(self):
        """Test L1(pred, target) = L1(target, pred)."""
        strategy = L1LossStrategy()
        pred = torch.rand(1, 1, 64, 64)
        target = torch.rand(1, 1, 64, 64)
        loss1 = strategy(pred, target)
        loss2 = strategy(target, pred)
        assert torch.allclose(loss1, loss2)

    def test_backward_pass(self):
        """Test gradients flow through L1 strategy."""
        strategy = L1LossStrategy()
        pred = torch.rand(1, 1, 64, 64, requires_grad=True)
        target = torch.rand(1, 1, 64, 64)
        loss = strategy(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_different_shapes(self):
        """Test L1 with different input shapes."""
        strategy = L1LossStrategy()
        for shape in [(1, 1, 32, 32), (2, 3, 64, 64), (4, 1, 128, 128)]:
            pred = torch.rand(*shape)
            target = torch.rand(*shape)
            loss = strategy(pred, target)
            assert loss.dim() == 0  # Scalar
            assert loss >= 0


class TestL2LossStrategy:
    """Test L2LossStrategy."""

    def test_strategy_initialization(self):
        """Test L2 strategy can be initialized."""
        strategy = L2LossStrategy()
        assert strategy.name == "l2"

    def test_perfect_match(self):
        """Test L2 loss is zero for identical inputs."""
        strategy = L2LossStrategy()
        pred = torch.rand(1, 1, 64, 64)
        loss = strategy(pred, pred)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_loss_positive(self):
        """Test L2 loss is non-negative."""
        strategy = L2LossStrategy()
        pred = torch.rand(1, 1, 64, 64)
        target = torch.rand(1, 1, 64, 64)
        loss = strategy(pred, target)
        assert loss >= 0

    def test_loss_symmetry(self):
        """Test L2(pred, target) = L2(target, pred)."""
        strategy = L2LossStrategy()
        pred = torch.rand(1, 1, 64, 64)
        target = torch.rand(1, 1, 64, 64)
        loss1 = strategy(pred, target)
        loss2 = strategy(target, pred)
        assert torch.allclose(loss1, loss2)

    def test_backward_pass(self):
        """Test gradients flow through L2 strategy."""
        strategy = L2LossStrategy()
        pred = torch.rand(1, 1, 64, 64, requires_grad=True)
        target = torch.rand(1, 1, 64, 64)
        loss = strategy(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_l2_vs_l1(self):
        """Test L2 penalizes large errors more than L1."""
        l1_strategy = L1LossStrategy()
        l2_strategy = L2LossStrategy()

        # Small error
        pred1 = torch.tensor([[0.1]], dtype=torch.float32)
        target1 = torch.tensor([[0.0]], dtype=torch.float32)
        l1_small = l1_strategy(pred1, target1)
        l2_small = l2_strategy(pred1, target1)
        # For small errors, L1 > L2
        assert l1_small > l2_small

        # Large error (>1 so L2 exceeds L1)
        # For L1: |2.0 - 0| = 2.0
        # For L2: (2.0 - 0)² = 4.0
        # L2 > L1 for errors > 1
        pred2 = torch.tensor([[2.0]], dtype=torch.float32)
        target2 = torch.tensor([[0.0]], dtype=torch.float32)
        l1_large = l1_strategy(pred2, target2)
        l2_large = l2_strategy(pred2, target2)
        # For large errors (>1), L2 > L1
        assert l2_large > l1_large


class TestSSIMLossStrategy:
    """Test SSIMLossStrategy."""

    def test_strategy_initialization(self):
        """Test SSIM strategy initialization."""
        strategy = SSIMLossStrategy()
        assert strategy.name == "ssim"
        assert strategy.window_size == 11
        assert strategy.sigma == 1.5
        assert strategy.data_range == 1.0

    def test_custom_parameters(self):
        """Test SSIM strategy with custom parameters."""
        strategy = SSIMLossStrategy(window_size=7, sigma=1.0, data_range=255.0)
        assert strategy.window_size == 7
        assert strategy.sigma == 1.0
        assert strategy.data_range == 255.0

    def test_perfect_match(self):
        """Test SSIM loss is ~0 for identical inputs."""
        strategy = SSIMLossStrategy()
        pred = torch.rand(1, 1, 128, 128)
        loss = strategy(pred, pred)
        # Perfect match → SSIM = 1 → DSSIM = (1 - 1) / 2 = 0
        assert loss < 1e-5

    def test_loss_range(self):
        """Test SSIM loss is in expected range [0, 0.5]."""
        strategy = SSIMLossStrategy()
        pred = torch.rand(1, 1, 128, 128)
        target = torch.rand(1, 1, 128, 128)
        loss = strategy(pred, target)
        # DSSIM = (1 - SSIM) / 2 → typically [0, 0.5]
        assert 0 <= loss <= 0.6  # Allow slight tolerance

    def test_window_caching(self):
        """Test that Gaussian windows are cached per device."""
        strategy = SSIMLossStrategy()
        pred = torch.rand(1, 1, 128, 128)
        target = torch.rand(1, 1, 128, 128)

        # First call creates cache
        _ = strategy(pred, target)
        assert len(strategy._window_cache) == 1

        # Second call reuses cache
        _ = strategy(pred, target)
        assert len(strategy._window_cache) == 1

    def test_backward_pass(self):
        """Test gradients flow through SSIM strategy."""
        strategy = SSIMLossStrategy()
        pred = torch.rand(1, 1, 128, 128, requires_grad=True)
        target = torch.rand(1, 1, 128, 128)
        loss = strategy(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Test SSIM strategy on CUDA."""
        strategy = SSIMLossStrategy()
        pred = torch.rand(1, 1, 128, 128).cuda()
        target = torch.rand(1, 1, 128, 128).cuda()
        loss = strategy(pred, target)
        assert loss.is_cuda
        assert 0 <= loss <= 0.6

    def test_different_channels(self):
        """Test SSIM with different channel counts."""
        strategy = SSIMLossStrategy()
        for channels in [1, 3]:
            pred = torch.rand(1, channels, 128, 128)
            target = torch.rand(1, channels, 128, 128)
            loss = strategy(pred, target)
            assert 0 <= loss <= 0.6


class TestMSSSIMLossStrategy:
    """Test MSSSIMLossStrategy."""

    def test_strategy_initialization(self):
        """Test MS-SSIM strategy initialization."""
        strategy = MSSSIMLossStrategy()
        assert strategy.name == "msssim"
        assert strategy.window_size == 11
        assert strategy.sigma == 1.5
        assert strategy.data_range == 1.0
        assert len(strategy.weights) == 5

    def test_custom_weights(self):
        """Test MS-SSIM with custom weights."""
        custom_weights = [0.33, 0.33, 0.34]
        strategy = MSSSIMLossStrategy(weights=custom_weights)
        assert strategy.weights == custom_weights
        assert strategy.num_scales == 3

    def test_perfect_match(self):
        """Test MS-SSIM loss is ~0 for identical inputs."""
        strategy = MSSSIMLossStrategy()
        pred = torch.rand(1, 1, 256, 256)
        loss = strategy(pred, pred)
        # Perfect match → MS-SSIM = 1 → DSSIM = 0
        assert loss < 1e-4

    def test_loss_range(self):
        """Test MS-SSIM loss is in expected range."""
        strategy = MSSSIMLossStrategy()
        pred = torch.rand(1, 1, 256, 256)
        target = torch.rand(1, 1, 256, 256)
        loss = strategy(pred, target)
        assert 0 <= loss <= 0.6

    def test_input_validation(self):
        """Test MS-SSIM raises error for small inputs."""
        strategy = MSSSIMLossStrategy()
        pred = torch.rand(1, 1, 32, 32)  # Too small for 5 scales
        target = torch.rand(1, 1, 32, 32)
        with pytest.raises(ValueError, match="Input too small"):
            strategy(pred, target)

    def test_backward_pass(self):
        """Test gradients flow through MS-SSIM strategy."""
        strategy = MSSSIMLossStrategy()
        pred = torch.rand(1, 1, 256, 256, requires_grad=True)
        target = torch.rand(1, 1, 256, 256)
        loss = strategy(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_fewer_scales(self):
        """Test MS-SSIM with fewer scales doesn't raise validation error."""
        strategy = MSSSIMLossStrategy(weights=[0.5, 0.5])  # 2 scales
        pred = torch.rand(1, 1, 128, 128)
        target = torch.rand(1, 1, 128, 128)
        # Should work (128 is enough for 2 scales with window_size=11)
        loss = strategy(pred, target)
        assert 0 <= loss <= 0.6


class TestCompositeLossStrategy:
    """Test CompositeLossStrategy."""

    def test_strategy_initialization(self):
        """Test composite strategy initialization."""
        losses = {
            "l1": (L1LossStrategy(), 0.7),
            "l2": (L2LossStrategy(), 0.3),
        }
        strategy = CompositeLossStrategy(losses)
        assert strategy.name == "0.7*l1+0.3*l2"

    def test_weight_validation(self):
        """Test composite raises error if weights don't sum to 1."""
        losses = {
            "l1": (L1LossStrategy(), 0.5),
            "l2": (L2LossStrategy(), 0.4),  # Sum = 0.9
        }
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            CompositeLossStrategy(losses)

    def test_weight_tolerance(self):
        """Test composite accepts weights within tolerance."""
        # Sum = 1.005 (within 0.01 tolerance)
        losses = {
            "l1": (L1LossStrategy(), 0.505),
            "l2": (L2LossStrategy(), 0.5),
        }
        strategy = CompositeLossStrategy(losses)
        assert strategy is not None

    def test_l1_l2_composite(self):
        """Test composite of L1 and L2."""
        losses = {
            "l1": (L1LossStrategy(), 0.6),
            "l2": (L2LossStrategy(), 0.4),
        }
        strategy = CompositeLossStrategy(losses)

        pred = torch.rand(1, 1, 64, 64)
        target = torch.rand(1, 1, 64, 64)

        # Compute composite loss
        composite_loss = strategy(pred, target)

        # Compute individual losses
        l1_loss = L1LossStrategy()(pred, target)
        l2_loss = L2LossStrategy()(pred, target)

        # Check weighted sum
        expected = 0.6 * l1_loss + 0.4 * l2_loss
        assert torch.allclose(composite_loss, expected, atol=1e-6)

    def test_l1_ssim_composite(self):
        """Test composite of L1 and SSIM."""
        losses = {
            "l1": (L1LossStrategy(), 0.7),
            "ssim": (SSIMLossStrategy(), 0.3),
        }
        strategy = CompositeLossStrategy(losses)

        pred = torch.rand(1, 1, 128, 128)
        target = torch.rand(1, 1, 128, 128)

        composite_loss = strategy(pred, target)

        # Verify weighted sum
        l1_loss = L1LossStrategy()(pred, target)
        ssim_loss = SSIMLossStrategy()(pred, target)
        expected = 0.7 * l1_loss + 0.3 * ssim_loss

        assert torch.allclose(composite_loss, expected, atol=1e-5)

    def test_triple_composite(self):
        """Test composite with three loss types."""
        losses = {
            "l1": (L1LossStrategy(), 0.5),
            "l2": (L2LossStrategy(), 0.3),
            "ssim": (SSIMLossStrategy(), 0.2),
        }
        strategy = CompositeLossStrategy(losses)

        pred = torch.rand(1, 1, 128, 128)
        target = torch.rand(1, 1, 128, 128)

        composite_loss = strategy(pred, target)

        # Verify it's a scalar and in reasonable range
        assert composite_loss.dim() == 0
        assert composite_loss >= 0

    def test_backward_pass(self):
        """Test gradients flow through composite strategy."""
        losses = {
            "l1": (L1LossStrategy(), 0.6),
            "ssim": (SSIMLossStrategy(), 0.4),
        }
        strategy = CompositeLossStrategy(losses)

        pred = torch.rand(1, 1, 128, 128, requires_grad=True)
        target = torch.rand(1, 1, 128, 128)

        loss = strategy(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_perfect_match(self):
        """Test composite loss is ~0 for identical inputs."""
        losses = {
            "l1": (L1LossStrategy(), 0.5),
            "l2": (L2LossStrategy(), 0.5),
        }
        strategy = CompositeLossStrategy(losses)

        pred = torch.rand(1, 1, 64, 64)
        loss = strategy(pred, pred)

        # Both L1 and L2 should be 0 for identical inputs
        assert loss < 1e-6


class TestLossAggregatorWithComposite:
    """Test LossAggregator with composite loss type."""

    def test_composite_initialization(self):
        """Test LossAggregator with composite loss."""
        criterion = LossAggregator(
            loss_type="composite",
            loss_weights={"l1": 0.7, "l2": 0.3},
        )
        assert criterion.loss_type == "composite"
        assert criterion.use_strategy is True

    def test_composite_missing_weights(self):
        """Test composite requires loss_weights."""
        with pytest.raises(ValueError, match="loss_weights must be provided"):
            LossAggregator(loss_type="composite")

    def test_composite_dual_loss_output(self):
        """Test composite returns dual losses."""
        criterion = LossAggregator(
            loss_type="composite",
            loss_weights={"l1": 0.6, "l2": 0.4},
        )

        inputs = torch.rand(1, 1, 64, 64)
        target = torch.rand(2, 1, 64, 64)

        loss_old, loss_new = criterion(inputs, target, telescope=None)

        assert isinstance(loss_old, torch.Tensor)
        assert isinstance(loss_new, torch.Tensor)
        assert loss_old.dim() == 0
        assert loss_new.dim() == 0

    def test_composite_with_ssim(self):
        """Test composite with L1 and SSIM."""
        criterion = LossAggregator(
            loss_type="composite",
            loss_weights={"l1": 0.7, "ssim": 0.3},
        )

        inputs = torch.rand(1, 1, 128, 128)
        target = torch.rand(2, 1, 128, 128)

        loss_old, loss_new = criterion(inputs, target)

        assert loss_old >= 0
        assert loss_new >= 0

    def test_composite_backward_pass(self):
        """Test composite loss works in backward pass."""
        criterion = LossAggregator(
            loss_type="composite",
            loss_weights={"l1": 0.5, "l2": 0.5},
        )

        inputs = torch.rand(1, 1, 64, 64, requires_grad=True)
        target = torch.rand(2, 1, 64, 64)

        loss_old, loss_new = criterion(inputs, target, telescope=None)
        loss = loss_old + loss_new
        loss.backward()

        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()

    def test_composite_perfect_similarity(self):
        """Test composite loss is low for identical images."""
        criterion = LossAggregator(
            loss_type="composite",
            loss_weights={"l1": 0.5, "l2": 0.5},
        )

        img = torch.rand(1, 1, 64, 64)
        inputs = img.clone()
        target = torch.cat([img, img], dim=0)

        loss_old, loss_new = criterion(inputs, target, telescope=None)

        # Perfect match should have very low loss
        assert loss_old < 1e-5
        assert loss_new < 1e-5

    def test_composite_custom_strategy_kwargs(self):
        """Test composite with custom SSIM parameters."""
        criterion = LossAggregator(
            loss_type="composite",
            loss_weights={"l1": 0.5, "ssim": 0.5},
            window_size=7,
            sigma=1.0,
        )

        inputs = torch.rand(1, 1, 128, 128)
        target = torch.rand(2, 1, 128, 128)

        # Should work with custom SSIM parameters
        loss_old, loss_new = criterion(inputs, target)

        assert isinstance(loss_old, torch.Tensor)
        assert isinstance(loss_new, torch.Tensor)


class TestLossAggregatorStrategyKwargs:
    """Test LossAggregator with strategy-specific kwargs."""

    def test_ssim_custom_window_size(self):
        """Test SSIM with custom window size."""
        criterion = LossAggregator(loss_type="ssim", window_size=7, sigma=1.0)
        assert criterion.window_size == 7
        assert criterion.sigma == 1.0

    def test_ssim_custom_data_range(self):
        """Test SSIM with custom data range."""
        criterion = LossAggregator(loss_type="ssim", data_range=255.0)
        assert criterion.data_range == 255.0

    def test_ms_ssim_custom_parameters(self):
        """Test MS-SSIM with custom parameters."""
        criterion = LossAggregator(loss_type="ms-ssim", window_size=7, sigma=1.0)
        assert criterion.window_size == 7
        assert criterion.sigma == 1.0


class TestLossStrategyDeviceConsistency:
    """Test loss strategies maintain device consistency."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_l1_cuda(self):
        """Test L1 strategy on CUDA."""
        strategy = L1LossStrategy()
        pred = torch.rand(1, 1, 64, 64).cuda()
        target = torch.rand(1, 1, 64, 64).cuda()
        loss = strategy(pred, target)
        assert loss.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_composite_cuda(self):
        """Test composite strategy on CUDA."""
        losses = {
            "l1": (L1LossStrategy(), 0.5),
            "l2": (L2LossStrategy(), 0.5),
        }
        strategy = CompositeLossStrategy(losses)
        pred = torch.rand(1, 1, 64, 64).cuda()
        target = torch.rand(1, 1, 64, 64).cuda()
        loss = strategy(pred, target)
        assert loss.is_cuda


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
