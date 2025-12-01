"""
Unit tests for noise models.

Tests NoiseModel ABC, PoissonNoise, ReadoutNoise, and CompositeNoise.
"""

from __future__ import annotations

import pytest
import torch

from prism.models.noise import (
    CompositeNoise,
    NoiseModel,
    PoissonNoise,
    ReadoutNoise,
    ShotNoise,
)


class TestNoiseModelABC:
    """Test NoiseModel abstract base class."""

    def test_cannot_instantiate_abc(self):
        """Test that NoiseModel ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            NoiseModel()  # type: ignore


class TestPoissonNoise:
    """Test PoissonNoise implementation."""

    def test_initialization(self):
        """Test PoissonNoise initialization."""
        noise = PoissonNoise(snr=40.0)
        assert noise.snr == 40.0
        assert noise.snr_linear == 10 ** (40.0 / 10)

    def test_add_noise_shape(self):
        """Test that add_noise preserves tensor shape."""
        noise = PoissonNoise(snr=40.0)
        tensor = torch.randn(1, 1, 256, 256)
        noisy = noise.add_noise(tensor)

        assert noisy.shape == tensor.shape

    def test_add_noise_increases_variance(self):
        """Test that noise increases signal variance."""
        noise = PoissonNoise(snr=40.0)
        tensor = torch.ones(1, 1, 256, 256)

        noisy = noise.add_noise(tensor)
        clean_variance = tensor.var().item()
        noisy_variance = noisy.var().item()

        assert noisy_variance > clean_variance

    def test_get_stats(self):
        """Test get_stats returns correct information."""
        noise = PoissonNoise(snr=40.0)
        stats = noise.get_stats()

        assert stats["noise_type"] == "poisson"
        assert stats["snr_db"] == 40.0
        assert "snr_linear" in stats

    @pytest.mark.parametrize("snr", [10.0, 20.0, 30.0, 40.0, 50.0])
    def test_different_snr_levels(self, snr):
        """Test different SNR levels."""
        noise = PoissonNoise(snr=snr)
        tensor = torch.ones(1, 1, 256, 256)
        noisy = noise.add_noise(tensor)

        assert noisy.shape == tensor.shape
        assert not torch.allclose(noisy, tensor)  # Noise was added


class TestReadoutNoise:
    """Test ReadoutNoise implementation."""

    def test_initialization(self):
        """Test ReadoutNoise initialization."""
        noise = ReadoutNoise(sigma=0.01)
        assert noise.sigma == 0.01

    def test_add_noise_shape(self):
        """Test that add_noise preserves tensor shape."""
        noise = ReadoutNoise(sigma=0.01)
        tensor = torch.randn(1, 1, 256, 256)
        noisy = noise.add_noise(tensor)

        assert noisy.shape == tensor.shape

    def test_add_noise_is_additive(self):
        """Test that readout noise is additive."""
        noise = ReadoutNoise(sigma=0.01)
        tensor = torch.zeros(1, 1, 256, 256)
        noisy = noise.add_noise(tensor)

        # For zero input, output should only contain noise
        assert not torch.allclose(noisy, tensor)
        assert noisy.mean().abs() < 0.1  # Mean should be near zero

    def test_get_stats(self):
        """Test get_stats returns correct information."""
        noise = ReadoutNoise(sigma=0.01)
        stats = noise.get_stats()

        assert stats["noise_type"] == "readout"
        assert stats["sigma"] == 0.01
        assert stats["variance"] == 0.01**2

    @pytest.mark.parametrize("sigma", [0.001, 0.01, 0.1, 1.0])
    def test_different_sigma_levels(self, sigma):
        """Test different noise levels."""
        noise = ReadoutNoise(sigma=sigma)
        tensor = torch.zeros(1, 1, 256, 256)
        noisy = noise.add_noise(tensor)

        # Higher sigma should produce more variance
        assert noisy.std() > 0


class TestCompositeNoise:
    """Test CompositeNoise implementation."""

    def test_initialization_with_models(self):
        """Test CompositeNoise initialization with noise models."""
        poisson = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([poisson, readout])

        assert len(composite.noise_models) == 2

    def test_initialization_empty_list(self):
        """Test that empty noise model list raises error."""
        with pytest.raises(ValueError, match="at least one noise model"):
            CompositeNoise([])

    def test_add_noise_applies_all_models(self):
        """Test that all noise models are applied."""
        poisson = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([poisson, readout])

        tensor = torch.ones(1, 1, 256, 256)
        noisy = composite.add_noise(tensor)

        assert noisy.shape == tensor.shape
        assert not torch.allclose(noisy, tensor)

    def test_get_stats(self):
        """Test get_stats returns combined statistics."""
        poisson = PoissonNoise(snr=40.0)
        readout = ReadoutNoise(sigma=0.01)
        composite = CompositeNoise([poisson, readout])

        stats = composite.get_stats()

        assert stats["noise_type"] == "composite"
        assert stats["n_components"] == 2
        assert len(stats["components"]) == 2
        assert stats["components"][0]["noise_type"] == "poisson"
        assert stats["components"][1]["noise_type"] == "readout"

    def test_sequential_application(self):
        """Test that noise models are applied sequentially."""
        # Create two readout noise models with different sigmas
        noise1 = ReadoutNoise(sigma=0.01)
        noise2 = ReadoutNoise(sigma=0.02)
        composite = CompositeNoise([noise1, noise2])

        tensor = torch.zeros(1, 1, 256, 256)
        noisy = composite.add_noise(tensor)

        # Variance should be sum of individual variances (for additive noise)
        # Approximately: sigma^2 = 0.01^2 + 0.02^2 = 0.0005
        expected_variance = 0.01**2 + 0.02**2
        actual_variance = noisy.var().item()

        # Allow some tolerance due to finite sample size
        assert abs(actual_variance - expected_variance) < 0.0002


class TestShotNoiseBackwardCompatibility:
    """Test that ShotNoise maintains backward compatibility."""

    def test_shot_noise_still_works(self):
        """Test that existing ShotNoise class still works."""
        noise = ShotNoise(desired_snr_db=40.0)
        assert noise.desired_snr_db == 40.0

    def test_forward_method(self):
        """Test forward method with add_noise parameter."""
        noise = ShotNoise(desired_snr_db=40.0)
        tensor = torch.ones(1, 1, 256, 256)

        # Test with add_noise=True
        noisy = noise(tensor, add_noise=True)
        assert not torch.allclose(noisy, tensor**2)

        # Test with add_noise=False
        clean = noise(tensor, add_noise=False)
        assert torch.allclose(clean, tensor)

    def test_add_noise_method(self):
        """Test that add_noise method is available."""
        noise = ShotNoise(desired_snr_db=40.0)
        tensor = torch.ones(1, 1, 256, 256)
        noisy = noise.add_noise(tensor)

        assert noisy.shape == (1, 1, 256, 256)

    def test_get_stats_method(self):
        """Test that get_stats method is available."""
        noise = ShotNoise(desired_snr_db=40.0)
        stats = noise.get_stats()

        assert "noise_type" in stats
        assert stats["noise_type"] == "shot"


class TestNoiseModelDeviceCompatibility:
    """Test noise models on different devices."""

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
    def test_poisson_on_device(self, device):
        """Test PoissonNoise on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        noise = PoissonNoise(snr=40.0).to(device)
        tensor = torch.ones(1, 1, 256, 256, device=device)
        noisy = noise.add_noise(tensor)

        assert noisy.device.type == device

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
    def test_readout_on_device(self, device):
        """Test ReadoutNoise on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        noise = ReadoutNoise(sigma=0.01).to(device)
        tensor = torch.zeros(1, 1, 256, 256, device=device)
        noisy = noise.add_noise(tensor)

        assert noisy.device.type == device
