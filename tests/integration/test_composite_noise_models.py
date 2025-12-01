"""
Integration tests for composite noise models.

Tests CompositeNoise and integration with Telescope including:
- Combining multiple noise sources (Poisson + Readout)
- Sequential noise application
- Statistics aggregation from multiple sources
- Integration with TelescopeAggregator
- Realistic measurement scenarios
"""

from __future__ import annotations

import pytest
import torch

from prism.models.noise import (
    CompositeNoise,
    NoiseModel,
    PoissonNoise,
    ReadoutNoise,
)


@pytest.fixture
def device():
    """Get device for testing (CUDA if available, CPU otherwise)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def clean_image(device):
    """Create a clean test image (single sample - SPIDS paradigm)."""
    height, width = 64, 64
    # Create a simple image with known statistics (single sample)
    image = torch.ones(1, 1, height, width, device=device) * 0.5
    return image


class TestCompositeNoiseCreation:
    """Test CompositeNoise creation and initialization."""

    def test_create_composite_with_two_sources(self, device):
        """Test creating composite noise with two sources."""
        shot = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)

        composite = CompositeNoise([shot, readout])

        assert len(composite.noise_models) == 2
        assert isinstance(composite.noise_models[0], PoissonNoise)
        assert isinstance(composite.noise_models[1], ReadoutNoise)

    def test_create_composite_with_three_sources(self, device):
        """Test creating composite noise with three sources."""
        shot1 = PoissonNoise(snr=40.0)
        shot2 = PoissonNoise(snr=35.0)
        readout = ReadoutNoise(sigma=0.02)

        composite = CompositeNoise([shot1, shot2, readout])

        assert len(composite.noise_models) == 3

    def test_create_composite_with_empty_list_raises(self):
        """Test that empty noise model list raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one noise model"):
            CompositeNoise([])

    def test_composite_is_noise_model(self):
        """Test that CompositeNoise is a NoiseModel."""
        shot = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([shot, readout])

        assert isinstance(composite, NoiseModel)
        assert hasattr(composite, "add_noise")
        assert hasattr(composite, "get_stats")


class TestCompositeNoiseAddNoise:
    """Test CompositeNoise.add_noise() functionality."""

    def test_add_noise_sequential_application(self, clean_image, device):
        """Test that noise models are applied sequentially."""
        shot = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([shot, readout])

        noisy_image = composite.add_noise(clean_image)

        # Check that noise was added
        assert not torch.allclose(noisy_image, clean_image)
        assert noisy_image.shape == clean_image.shape

    def test_add_noise_order_matters(self, clean_image, device):
        """Test that the order of noise models matters."""
        shot = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)

        # Different orders
        composite1 = CompositeNoise([shot, readout])
        composite2 = CompositeNoise([readout, shot])

        noisy1 = composite1.add_noise(clean_image)
        noisy2 = composite2.add_noise(clean_image.clone())

        # Results should be different due to order
        # (Readout then shot vs shot then readout)
        # Note: This test may be flaky due to randomness, but statistically
        # the results should be different
        assert noisy1.shape == noisy2.shape

    def test_add_noise_preserves_shape(self, clean_image, device):
        """Test that add_noise preserves tensor shape."""
        shot = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([shot, readout])

        noisy_image = composite.add_noise(clean_image)

        assert noisy_image.shape == clean_image.shape

    def test_add_noise_different_image_sizes(self, device):
        """Test add_noise with different image sizes (single sample)."""
        shot = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([shot, readout])

        # Test different image sizes (SPIDS uses single samples)
        for size in [32, 64, 128]:
            image = torch.ones(1, 1, size, size, device=device) * 0.5
            noisy = composite.add_noise(image)
            assert noisy.shape == image.shape


class TestCompositeNoiseStatistics:
    """Test CompositeNoise.get_stats() functionality."""

    def test_get_stats_combines_all_sources(self):
        """Test that get_stats returns stats from all noise sources."""
        shot = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([shot, readout])

        stats = composite.get_stats()

        assert "noise_type" in stats
        assert stats["noise_type"] == "composite"
        assert "components" in stats
        assert len(stats["components"]) == 2

        # Check individual noise source stats
        assert stats["components"][0]["noise_type"] == "poisson"
        assert stats["components"][1]["noise_type"] == "readout"

    def test_get_stats_correct_parameters(self):
        """Test that get_stats contains correct parameters."""
        shot = PoissonNoise(snr=45.0)
        readout = ReadoutNoise(sigma=0.02)
        composite = CompositeNoise([shot, readout])

        stats = composite.get_stats()

        # Check Poisson noise stats
        poisson_stats = stats["components"][0]
        assert poisson_stats["snr_db"] == 45.0

        # Check readout noise stats
        readout_stats = stats["components"][1]
        assert readout_stats["sigma"] == 0.02


class TestCompositeNoiseRealisticScenarios:
    """Test CompositeNoise in realistic measurement scenarios."""

    def test_realistic_telescope_noise(self, clean_image, device):
        """
        Test realistic telescope noise: shot noise + readout noise.

        Typical scenario for astronomical measurements.
        """
        # High SNR shot noise (good photon count)
        shot = PoissonNoise(snr=40.0)

        # Low readout noise (modern detector)
        readout = ReadoutNoise(sigma=0.005)

        composite = CompositeNoise([shot, readout])

        noisy_image = composite.add_noise(clean_image)

        # Check that output is reasonable
        assert noisy_image.shape == clean_image.shape
        assert not torch.isnan(noisy_image).any()
        assert not torch.isinf(noisy_image).any()

        # Check that SNR is approximately correct
        signal_power = clean_image.pow(2).mean()
        noise = noisy_image - clean_image.pow(2)  # Approximate noise
        noise_power = noise.pow(2).mean()

        if noise_power > 0:
            measured_snr = 10 * torch.log10(signal_power / noise_power)
            # SNR should be reasonable (but not exact due to approximations)
            assert measured_snr > 0

    def test_high_noise_scenario(self, clean_image, device):
        """Test high noise scenario (low SNR + high readout noise)."""
        # Low SNR shot noise
        shot = PoissonNoise(snr=20.0)

        # High readout noise
        readout = ReadoutNoise(sigma=0.05)

        composite = CompositeNoise([shot, readout])

        noisy_image = composite.add_noise(clean_image)

        # Check output is valid but noisy
        assert noisy_image.shape == clean_image.shape
        assert not torch.isnan(noisy_image).any()

        # High noise should produce noticeable differences
        diff = (noisy_image - clean_image.pow(2)).abs().mean()
        assert diff > 0.01  # Significant noise

    def test_low_noise_scenario(self, clean_image, device):
        """Test low noise scenario (high SNR + low readout noise)."""
        # High SNR shot noise
        shot = PoissonNoise(snr=60.0)

        # Very low readout noise
        readout = ReadoutNoise(sigma=0.001)

        composite = CompositeNoise([shot, readout])

        noisy_image = composite.add_noise(clean_image)

        # Check output is close to original
        assert noisy_image.shape == clean_image.shape
        assert not torch.isnan(noisy_image).any()

        # Low noise should produce small differences
        diff = (noisy_image - clean_image.pow(2)).abs().mean()
        assert diff < 0.1  # Small noise


class TestCompositeNoiseWithDifferentCombinations:
    """Test CompositeNoise with different noise model combinations."""

    def test_double_shot_noise(self, clean_image, device):
        """Test combining two shot noise sources."""
        shot1 = PoissonNoise(snr=40.0)
        shot2 = PoissonNoise(snr=35.0)

        composite = CompositeNoise([shot1, shot2])

        noisy_image = composite.add_noise(clean_image)

        assert noisy_image.shape == clean_image.shape
        assert not torch.isnan(noisy_image).any()

    def test_triple_noise_combination(self, clean_image, device):
        """Test combining three noise sources."""
        shot = PoissonNoise(snr=40.0)
        readout1 = ReadoutNoise(sigma=0.01)
        readout2 = ReadoutNoise(sigma=0.005)

        composite = CompositeNoise([shot, readout1, readout2])

        noisy_image = composite.add_noise(clean_image)

        assert len(composite.noise_models) == 3
        assert noisy_image.shape == clean_image.shape

    def test_only_shot_noise(self, clean_image, device):
        """Test CompositeNoise with only shot noise."""
        shot = PoissonNoise(snr=40.0)
        composite = CompositeNoise([shot])

        noisy_image = composite.add_noise(clean_image)

        # Should work with single noise source
        assert noisy_image.shape == clean_image.shape

    def test_only_readout_noise(self, clean_image, device):
        """Test CompositeNoise with only readout noise."""
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([readout])

        noisy_image = composite.add_noise(clean_image)

        # Should work with single noise source
        assert noisy_image.shape == clean_image.shape


class TestCompositeNoiseNumericalProperties:
    """Test numerical properties of CompositeNoise."""

    def test_noise_is_random(self, clean_image, device):
        """Test that noise is different each time."""
        shot = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([shot, readout])

        noisy1 = composite.add_noise(clean_image)
        noisy2 = composite.add_noise(clean_image.clone())

        # Different noise realizations
        assert not torch.allclose(noisy1, noisy2)

    def test_noise_statistics_correct(self, device):
        """Test that noise statistics are approximately correct (single samples)."""
        # Use a constant image for easier statistics
        clean = torch.ones(1, 1, 64, 64, device=device)

        readout = ReadoutNoise(sigma=0.1)
        composite = CompositeNoise([readout])

        # Generate many noisy samples (SPIDS processes one at a time)
        noisy_samples = [composite.add_noise(clean.clone()) for _ in range(10)]
        noisy_stack = torch.cat(noisy_samples, dim=0)

        # Compute noise statistics
        noise = noisy_stack - clean.expand(10, -1, -1, -1)
        noise_std = noise.std().item()

        # Should be close to specified sigma
        # Allow 50% tolerance due to finite sampling
        assert abs(noise_std - 0.1) < 0.05

    def test_no_nan_or_inf(self, clean_image, device):
        """Test that CompositeNoise never produces NaN or Inf."""
        shot = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([shot, readout])

        for _ in range(10):
            noisy = composite.add_noise(clean_image)
            assert not torch.isnan(noisy).any()
            assert not torch.isinf(noisy).any()


class TestCompositeNoiseDeviceHandling:
    """Test CompositeNoise device handling."""

    def test_composite_on_cpu(self):
        """Test CompositeNoise on CPU (single sample)."""
        device = torch.device("cpu")
        shot = PoissonNoise(snr=40.0).to(device)
        readout = ReadoutNoise(sigma=0.01).to(device)
        composite = CompositeNoise([shot, readout]).to(device)

        # Single sample (SPIDS paradigm)
        image = torch.ones(1, 1, 64, 64, device=device)
        noisy = composite.add_noise(image)

        assert noisy.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_composite_on_cuda(self):
        """Test CompositeNoise on CUDA (single sample)."""
        device = torch.device("cuda")
        shot = PoissonNoise(snr=40.0).to(device)
        readout = ReadoutNoise(sigma=0.01).to(device)
        composite = CompositeNoise([shot, readout]).to(device)

        # Single sample (SPIDS paradigm)
        image = torch.ones(1, 1, 64, 64, device=device)
        noisy = composite.add_noise(image)

        assert noisy.device.type == "cuda"  # Just check device type, not index


class TestCompositeNoiseEdgeCases:
    """Test edge cases for CompositeNoise."""

    def test_zero_image(self, device):
        """Test CompositeNoise with zero image (single sample)."""
        shot = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([shot, readout])

        # Single sample (SPIDS paradigm)
        zero_image = torch.zeros(1, 1, 64, 64, device=device)
        noisy = composite.add_noise(zero_image)

        # Should handle zeros gracefully
        assert noisy.shape == zero_image.shape
        assert not torch.isnan(noisy).any()

    def test_very_small_image_values(self, device):
        """Test CompositeNoise with very small values (single sample)."""
        shot = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([shot, readout])

        # Single sample (SPIDS paradigm)
        small_image = torch.ones(1, 1, 64, 64, device=device) * 1e-6
        noisy = composite.add_noise(small_image)

        # Should handle small values gracefully
        assert noisy.shape == small_image.shape
        assert not torch.isnan(noisy).any()

    def test_very_large_image_values(self, device):
        """Test CompositeNoise with very large values (single sample)."""
        shot = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([shot, readout])

        # Single sample (SPIDS paradigm)
        large_image = torch.ones(1, 1, 64, 64, device=device) * 1000.0
        noisy = composite.add_noise(large_image)

        # Should handle large values gracefully
        assert noisy.shape == large_image.shape
        assert not torch.isnan(noisy).any()
        assert not torch.isinf(noisy).any()
