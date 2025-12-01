"""Unit tests for MeasurementSystem.

Tests the new MeasurementSystem class that handles SPIDS-specific progressive
measurement logic and works with any Instrument instance.
"""

from __future__ import annotations

import pytest
import torch

from prism.core.instruments import Telescope, TelescopeConfig
from prism.core.measurement_system import MeasurementSystem, MeasurementSystemConfig
from prism.utils.measurement_cache import MeasurementCache


class TestMeasurementSystemConfig:
    """Test MeasurementSystemConfig validation."""

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        config = MeasurementSystemConfig()
        assert config.enable_caching is True
        assert config.cache_max_size == 1000
        assert config.add_noise_by_default is True

    def test_config_custom(self) -> None:
        """Test custom configuration values."""
        config = MeasurementSystemConfig(
            enable_caching=False,
            cache_max_size=500,
            add_noise_by_default=False,
        )
        assert config.enable_caching is False
        assert config.cache_max_size == 500
        assert config.add_noise_by_default is False

    def test_config_validation_negative_cache_size(self) -> None:
        """Test validation fails for negative cache size."""
        config = MeasurementSystemConfig(cache_max_size=-10)
        with pytest.raises(ValueError, match="Cache max size must be positive"):
            config.validate()


class TestMeasurementSystemCreation:
    """Test MeasurementSystem instantiation."""

    def test_creation_with_telescope(self) -> None:
        """Test creation with Telescope instrument."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=10.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        assert ms is not None
        assert ms.instrument is telescope
        assert ms.sample_count == 0

    def test_creation_with_config(self) -> None:
        """Test creation with custom config."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms_config = MeasurementSystemConfig(enable_caching=False)
        ms = MeasurementSystem(telescope, config=ms_config)

        assert ms.config.enable_caching is False

    def test_creation_with_shared_cache(self) -> None:
        """Test creation with shared measurement cache."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        shared_cache = MeasurementCache()
        ms = MeasurementSystem(telescope, measurement_cache=shared_cache)

        assert ms.measurement_cache is shared_cache

    def test_cumulative_mask_initialized(self) -> None:
        """Test cumulative mask is initialized correctly."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        assert ms.cum_mask.shape == (128, 128)
        assert ms.cum_mask.dtype == torch.bool
        assert ms.cum_mask.sum() == 0  # Initially empty


class TestMeasurementSystemMaskManagement:
    """Test aperture mask management."""

    def test_add_single_mask(self) -> None:
        """Test adding a single aperture mask."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=5.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        ms.add_mask([[0, 0]])

        assert ms.sample_count == 1
        assert ms.cum_mask.sum() > 0  # Some pixels covered

    def test_add_multiple_masks(self) -> None:
        """Test adding multiple aperture masks."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=5.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        ms.add_mask([[0, 0], [10, 10]])

        assert ms.sample_count == 2
        assert ms.cum_mask.sum() > 0

    def test_masks_accumulate(self) -> None:
        """Test masks accumulate (union operation)."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=5.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        ms.add_mask([[0, 0]])
        coverage_1 = ms.cum_mask.sum()

        ms.add_mask([[20, 20]])
        coverage_2 = ms.cum_mask.sum()

        # Coverage should increase
        assert coverage_2 > coverage_1

    def test_add_mask_with_tensor(self) -> None:
        """Test adding masks with tensor centers."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=5.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        centers = torch.tensor([[0, 0], [10, 10]], dtype=torch.float32)
        ms.add_mask(centers)

        assert ms.sample_count == 2


class TestMeasurementSystemMeasurements:
    """Test measurement generation."""

    def test_get_measurements_single(self) -> None:
        """Test getting single measurement."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=5.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        field = torch.ones(128, 128, dtype=torch.complex64)
        meas = ms.get_measurements(field, [[0, 0]], add_noise=False)

        assert meas.shape == (128, 128)
        assert meas.dtype == torch.float32

    def test_get_measurements_multiple(self) -> None:
        """Test getting multiple measurements."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=5.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        field = torch.ones(128, 128, dtype=torch.complex64)
        meas = ms.get_measurements(field, [[0, 0], [10, 10]], add_noise=False)

        assert meas.shape == (2, 128, 128)

    def test_measure_first_sample(self) -> None:
        """Test measure for first sample (no reconstruction)."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=5.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        ground_truth = torch.ones(128, 128, dtype=torch.complex64)
        dual_meas = ms.measure(ground_truth, None, [[0, 0]], add_noise=False)

        # Should return [new_meas, new_meas] for first sample
        # Output shape is [2, 1, H, W] with channel dimension for loss function
        assert dual_meas.shape == (2, 1, 128, 128)
        assert torch.allclose(dual_meas[0], dual_meas[1])

    def test_measure_with_reconstruction(self) -> None:
        """Test measure with reconstruction (accumulated mask)."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=5.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        # Add first mask
        ms.add_mask([[0, 0]])

        ground_truth = torch.ones(128, 128, dtype=torch.complex64)
        reconstruction = torch.ones(128, 128, dtype=torch.complex64) * 0.5
        dual_meas = ms.measure(ground_truth, reconstruction, [[10, 10]], add_noise=False)

        # Output shape is [2, 1, H, W] with channel dimension for loss function
        assert dual_meas.shape == (2, 1, 128, 128)
        # Measurements should be different (old mask vs new aperture)
        assert not torch.allclose(dual_meas[0], dual_meas[1])

    def test_measure_uses_cache(self) -> None:
        """Test measure uses cache for repeated calls."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=5.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        ground_truth = torch.ones(128, 128, dtype=torch.complex64)

        # First call - cache miss
        meas1 = ms.measure(ground_truth, None, [[0, 0]], add_noise=False)

        # Second call with same parameters - cache hit
        meas2 = ms.measure(ground_truth, None, [[0, 0]], add_noise=False)

        assert torch.allclose(meas1, meas2)

        # Check cache stats
        stats = ms.get_cache_stats()
        assert stats["cache_hits"] >= 1


class TestMeasurementSystemAccumulatedMask:
    """Test measurement through accumulated mask."""

    def test_measure_through_empty_mask(self) -> None:
        """Test measuring through empty cumulative mask."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        field = torch.ones(128, 128, dtype=torch.complex64)
        meas = ms.measure_through_accumulated_mask(field)

        # Empty mask should give zero or very small measurement
        assert meas.abs().max() < 1e-6

    def test_measure_through_populated_mask(self) -> None:
        """Test measuring through populated cumulative mask."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=10.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        # Add some masks
        ms.add_mask([[0, 0], [20, 20]])

        field = torch.ones(128, 128, dtype=torch.complex64)
        meas = ms.measure_through_accumulated_mask(field)

        # Should get non-zero measurement
        assert meas.abs().max() > 0


class TestMeasurementSystemCaching:
    """Test measurement caching functionality."""

    def test_caching_enabled_by_default(self) -> None:
        """Test caching is enabled by default."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        assert ms.measurement_cache is not None

    def test_caching_disabled(self) -> None:
        """Test caching can be disabled."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        config = MeasurementSystemConfig(enable_caching=False)
        ms = MeasurementSystem(telescope, config=config)

        assert ms.measurement_cache is None

    def test_get_cache_stats(self) -> None:
        """Test getting cache statistics."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        stats = ms.get_cache_stats()
        assert isinstance(stats, dict)
        assert "cache_hits" in stats or len(stats) == 0

    def test_clear_cache(self) -> None:
        """Test clearing cache."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        # Generate some cached measurements
        field = torch.ones(128, 128, dtype=torch.complex64)
        ms.measure(field, None, [[0, 0]], add_noise=False)

        # Clear cache
        ms.clear_cache()

        stats = ms.get_cache_stats()
        assert stats.get("cache_size", 0) == 0


class TestMeasurementSystemReset:
    """Test resetting measurement system."""

    def test_reset_clears_mask(self) -> None:
        """Test reset clears cumulative mask."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=10.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        # Add masks
        ms.add_mask([[0, 0], [10, 10]])
        assert ms.cum_mask.sum() > 0

        # Reset
        ms.reset()
        assert ms.cum_mask.sum() == 0

    def test_reset_clears_sample_count(self) -> None:
        """Test reset clears sample count."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        ms.add_mask([[0, 0]])
        assert ms.sample_count > 0

        ms.reset()
        assert ms.sample_count == 0


class TestMeasurementSystemDeviceTransfer:
    """Test device transfer."""

    def test_to_device_cpu(self) -> None:
        """Test moving to CPU."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        ms_cpu = ms.to(torch.device("cpu"))
        assert ms_cpu.cum_mask.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_cuda(self) -> None:
        """Test moving to CUDA."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        ms_cuda = ms.to(torch.device("cuda"))
        assert ms_cuda.cum_mask.device.type == "cuda"


class TestMeasurementSystemInfo:
    """Test get_info method."""

    def test_get_info_structure(self) -> None:
        """Test get_info returns correct structure."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        info = ms.get_info()

        assert "sample_count" in info
        assert "caching_enabled" in info
        assert "cumulative_mask_coverage" in info

    def test_get_info_sample_count(self) -> None:
        """Test get_info includes sample count."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        ms.add_mask([[0, 0]])
        info = ms.get_info()

        assert info["sample_count"] == 1

    def test_get_info_coverage(self) -> None:
        """Test get_info includes coverage percentage."""
        telescope_config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=10.0)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        ms.add_mask([[0, 0]])
        info = ms.get_info()

        assert 0 <= info["cumulative_mask_coverage"] <= 1


class TestMeasurementSystemRepr:
    """Test string representation."""

    def test_repr_contains_instrument_type(self) -> None:
        """Test __repr__ contains instrument type."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        repr_str = repr(ms)

        assert "MeasurementSystem" in repr_str
        assert "Telescope" in repr_str

    def test_repr_contains_sample_count(self) -> None:
        """Test __repr__ contains sample count."""
        telescope_config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(telescope_config)
        ms = MeasurementSystem(telescope)

        ms.add_mask([[0, 0]])
        repr_str = repr(ms)

        assert "samples=1" in repr_str
