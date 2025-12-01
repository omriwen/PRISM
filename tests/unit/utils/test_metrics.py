"""Tests for image quality metrics."""

from __future__ import annotations

import pytest
import torch

from prism.utils.metrics import compute_rmse, compute_ssim, psnr


class TestSSIM:
    """Test SSIM metric."""

    def test_identical_images(self):
        """Test SSIM of identical images is 1.0."""
        img = torch.randn(1, 1, 64, 64)
        ssim_value = compute_ssim(img, img)
        assert ssim_value == pytest.approx(1.0, abs=1e-6)

    def test_different_images(self):
        """Test SSIM of different images is less than 1.0."""
        img1 = torch.randn(1, 1, 64, 64)
        img2 = torch.randn(1, 1, 64, 64)
        ssim_value = compute_ssim(img1, img2)
        assert ssim_value < 1.0

    def test_ssim_with_crop(self):
        """Test SSIM with cropping."""
        img1 = torch.randn(1, 1, 128, 128)
        img2 = img1.clone()
        # Add noise to corners
        img2[:, :, :20, :20] += 1.0

        # SSIM with crop should be higher (ignores noisy corners)
        ssim_cropped = compute_ssim(img1, img2, size=64)
        ssim_full = compute_ssim(img1, img2)

        assert ssim_cropped > ssim_full

    def test_ssim_range(self):
        """Test SSIM is in valid range."""
        img1 = torch.randn(1, 1, 64, 64)
        img2 = torch.randn(1, 1, 64, 64)
        ssim_value = compute_ssim(img1, img2)
        assert -1.0 <= ssim_value <= 1.0

    def test_ssim_symmetric(self):
        """Test SSIM is symmetric."""
        img1 = torch.randn(1, 1, 64, 64)
        img2 = torch.randn(1, 1, 64, 64)
        ssim_12 = compute_ssim(img1, img2)
        ssim_21 = compute_ssim(img2, img1)
        assert ssim_12 == pytest.approx(ssim_21)

    def test_ssim_normalized_images(self):
        """Test SSIM with normalized images."""
        img1 = torch.rand(1, 1, 64, 64)  # Range [0, 1]
        img2 = img1 + 0.1 * torch.randn(1, 1, 64, 64)
        img2 = torch.clamp(img2, 0, 1)
        ssim_value = compute_ssim(img1, img2)
        assert 0 < ssim_value < 1.0


class TestRMSE:
    """Test RMSE metric."""

    def test_identical_images(self):
        """Test RMSE of identical images is 0."""
        img = torch.randn(1, 1, 64, 64)
        rmse_value = compute_rmse(img, img)
        assert rmse_value == pytest.approx(0.0, abs=1e-6)

    def test_different_images(self):
        """Test RMSE of different images is positive."""
        img1 = torch.randn(1, 1, 64, 64)
        img2 = torch.randn(1, 1, 64, 64)
        rmse_value = compute_rmse(img1, img2)
        assert rmse_value > 0

    def test_rmse_with_crop(self):
        """Test RMSE with cropping."""
        img1 = torch.randn(1, 1, 128, 128)
        img2 = img1.clone()
        # Add noise to corners
        img2[:, :, :20, :20] += 5.0

        # RMSE with crop should be lower (ignores noisy corners)
        rmse_cropped = compute_rmse(img1, img2, size=64)
        rmse_full = compute_rmse(img1, img2)

        assert rmse_cropped < rmse_full

    def test_rmse_scale_invariance(self):
        """Test RMSE scaling behavior."""
        img1 = torch.ones(1, 1, 64, 64)
        img2 = torch.ones(1, 1, 64, 64) * 0.9  # 10% error

        rmse_value = compute_rmse(img1, img2)
        # Should be roughly 10% error
        assert 5 < rmse_value < 15

    def test_rmse_returns_percentage(self):
        """Test RMSE is returned as percentage."""
        img1 = torch.randn(1, 1, 64, 64)
        img2 = img1 + 0.01 * torch.randn(1, 1, 64, 64)  # Small noise
        rmse_value = compute_rmse(img1, img2)
        # For small noise, RMSE should be a small percentage
        assert rmse_value > 0


class TestPSNR:
    """Test PSNR metric."""

    def test_identical_images(self):
        """Test PSNR of identical images is infinite."""
        img = torch.randn(1, 1, 64, 64)
        psnr_value = psnr(img, img)
        assert psnr_value == float("inf")

    def test_different_images(self):
        """Test PSNR of different images is finite."""
        img1 = torch.rand(1, 1, 64, 64)
        img2 = torch.rand(1, 1, 64, 64)
        psnr_value = psnr(img1, img2)
        assert psnr_value < float("inf")
        assert psnr_value > 0  # Should be positive in dB

    def test_psnr_with_crop(self):
        """Test PSNR with cropping."""
        img1 = torch.randn(1, 1, 128, 128)
        img2 = img1.clone()
        # Add noise to corners
        img2[:, :, :20, :20] += 5.0

        # PSNR with crop should be higher (ignores noisy corners)
        psnr_cropped = psnr(img1, img2, size=64)
        psnr_full = psnr(img1, img2)

        assert psnr_cropped > psnr_full

    def test_psnr_higher_for_similar_images(self):
        """Test PSNR is higher for more similar images."""
        img = torch.rand(1, 1, 64, 64)

        # Image with small noise
        img_small_noise = img + 0.01 * torch.randn(1, 1, 64, 64)
        psnr_small = psnr(img, img_small_noise)

        # Image with large noise
        img_large_noise = img + 0.1 * torch.randn(1, 1, 64, 64)
        psnr_large = psnr(img, img_large_noise)

        assert psnr_small > psnr_large

    def test_psnr_custom_data_range(self):
        """Test PSNR with custom data range."""
        img1 = torch.rand(1, 1, 64, 64) * 255  # Range [0, 255]
        img2 = img1 + torch.randn(1, 1, 64, 64) * 10

        # Calculate with correct data range
        psnr_value = psnr(img1, img2, data_range=255.0)
        assert psnr_value > 0

    def test_psnr_unit_range(self):
        """Test PSNR with unit range images."""
        img1 = torch.rand(1, 1, 64, 64)  # Range [0, 1]
        img2 = img1 + 0.05 * torch.randn(1, 1, 64, 64)

        psnr_value = psnr(img1, img2, data_range=1.0)
        assert psnr_value > 0


class TestMetricsIntegration:
    """Integration tests for metrics."""

    def test_all_metrics_on_same_images(self):
        """Test all metrics on the same image pair."""
        img1 = torch.rand(1, 1, 64, 64)
        img2 = img1 + 0.05 * torch.randn(1, 1, 64, 64)
        img2 = torch.clamp(img2, 0, 1)

        ssim_val = compute_ssim(img1, img2)
        rmse_val = compute_rmse(img1, img2)
        psnr_val = psnr(img1, img2)

        # All metrics should indicate some similarity
        assert 0.5 < ssim_val < 1.0  # High similarity
        assert rmse_val < 50  # Low error
        assert psnr_val > 10  # Reasonable PSNR

    def test_metrics_perfect_reconstruction(self):
        """Test metrics for perfect reconstruction."""
        img = torch.rand(1, 1, 128, 128)

        ssim_val = compute_ssim(img, img)
        rmse_val = compute_rmse(img, img)
        psnr_val = psnr(img, img)

        assert ssim_val == pytest.approx(1.0)
        assert rmse_val == pytest.approx(0.0)
        assert psnr_val == float("inf")

    def test_metrics_with_various_sizes(self):
        """Test metrics work with different image sizes."""
        for size in [32, 64, 128]:
            img1 = torch.rand(1, 1, size, size)
            img2 = img1 + 0.01 * torch.randn(1, 1, size, size)

            ssim_val = compute_ssim(img1, img2)
            rmse_val = compute_rmse(img1, img2)
            psnr_val = psnr(img1, img2)

            assert 0 < ssim_val <= 1.0
            assert rmse_val > 0
            assert 0 < psnr_val < float("inf")
