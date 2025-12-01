"""Unit tests for GPU-optimized line acquisition module."""

import pytest
import torch

from prism.core.instruments.telescope import Telescope, TelescopeConfig
from prism.core.line_acquisition import IncoherentLineAcquisition, LineAcquisitionConfig


@pytest.fixture
def telescope():
    """Create a small telescope for testing."""
    config = TelescopeConfig(
        n_pixels=64,
        aperture_radius_pixels=8,
        snr=None,  # No noise for deterministic tests
    )
    return Telescope(config)


@pytest.fixture
def line_acq_config():
    """Create line acquisition config for testing."""
    return LineAcquisitionConfig(
        mode="accurate",
        samples_per_pixel=1.0,
        min_samples=2,
        batch_size=32,
    )


@pytest.fixture
def line_acq(line_acq_config, telescope):
    """Create line acquisition instance."""
    return IncoherentLineAcquisition(line_acq_config, telescope)


class TestLineAcquisitionConfig:
    """Tests for LineAcquisitionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LineAcquisitionConfig()
        assert config.samples_per_pixel == 1.0
        assert config.min_samples == 2
        assert config.batch_size == 64
        assert config.mode == "accurate"

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = LineAcquisitionConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_samples_per_pixel(self):
        """Test validation fails for invalid samples_per_pixel."""
        config = LineAcquisitionConfig(samples_per_pixel=0)
        with pytest.raises(ValueError, match="samples_per_pixel must be > 0"):
            config.validate()

    def test_validate_invalid_min_samples(self):
        """Test validation fails for invalid min_samples."""
        config = LineAcquisitionConfig(min_samples=1)
        with pytest.raises(ValueError, match="min_samples must be >= 2"):
            config.validate()

    def test_validate_invalid_batch_size(self):
        """Test validation fails for invalid batch_size."""
        config = LineAcquisitionConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            config.validate()

    def test_validate_invalid_mode(self):
        """Test validation fails for invalid mode."""
        config = LineAcquisitionConfig(mode="invalid")  # type: ignore
        with pytest.raises(ValueError, match="mode must be 'accurate' or 'fast'"):
            config.validate()


class TestIncoherentLineAcquisition:
    """Tests for IncoherentLineAcquisition class."""

    def test_initialization(self, line_acq_config, telescope):
        """Test successful initialization."""
        line_acq = IncoherentLineAcquisition(line_acq_config, telescope)
        assert line_acq.config == line_acq_config
        assert line_acq.instrument == telescope
        assert line_acq._effective_batch_size > 0

    def test_compute_n_samples_accurate_mode(self, line_acq):
        """Test sample count in accurate mode."""
        line_length = 32.0
        n_samples = line_acq.compute_n_samples(line_length)
        # 32 pixels * 1.0 sample/pixel = 32 samples
        assert n_samples == 32

    def test_compute_n_samples_fast_mode(self, telescope):
        """Test sample count in fast mode."""
        config = LineAcquisitionConfig(mode="fast")
        line_acq = IncoherentLineAcquisition(config, telescope)

        # Diameter = 8 * 2 = 16, half = 8
        # Line length 32 / 8 + 1 = 5 samples
        line_length = 32.0
        n_samples = line_acq.compute_n_samples(line_length)
        assert n_samples == 5

    def test_compute_n_samples_min_samples(self, line_acq):
        """Test minimum sample count is enforced."""
        line_length = 0.5  # Very short line
        n_samples = line_acq.compute_n_samples(line_length)
        assert n_samples >= line_acq.config.min_samples

    def test_compute_line_positions(self, line_acq):
        """Test line position generation."""
        start = torch.tensor([10.0, 10.0])
        end = torch.tensor([20.0, 20.0])
        n_samples = 5

        positions = line_acq.compute_line_positions(start, end, n_samples)

        assert positions.shape == (5, 2)
        assert torch.allclose(positions[0], start)
        assert torch.allclose(positions[-1], end)

        # Check evenly spaced
        diffs = positions[1:] - positions[:-1]
        assert torch.allclose(diffs, diffs[0])

    def test_compute_line_positions_auto_samples(self, line_acq):
        """Test automatic sample count calculation."""
        start = torch.tensor([0.0, 0.0])
        end = torch.tensor([32.0, 0.0])  # Horizontal line, 32 pixels

        positions = line_acq.compute_line_positions(start, end)

        # Should get 32 samples (1 per pixel in accurate mode)
        assert positions.shape[0] == 32
        assert torch.allclose(positions[0], start)
        assert torch.allclose(positions[-1], end)

    def test_forward_shape(self, line_acq, telescope):
        """Test forward pass output shape."""
        n_pixels = telescope.config.n_pixels
        field_kspace = torch.randn(n_pixels, n_pixels, dtype=torch.complex64)
        start = torch.tensor([0.0, 0.0])
        end = torch.tensor([10.0, 10.0])

        measurement = line_acq.forward(field_kspace, start, end)

        assert measurement.shape == (n_pixels, n_pixels)
        assert measurement.dtype == torch.float32

    def test_forward_real_output(self, line_acq, telescope):
        """Test forward pass produces real-valued intensity."""
        n_pixels = telescope.config.n_pixels
        field_kspace = torch.randn(n_pixels, n_pixels, dtype=torch.complex64)
        start = torch.tensor([0.0, 0.0])
        end = torch.tensor([10.0, 10.0])

        measurement = line_acq.forward(field_kspace, start, end)

        assert torch.all(measurement >= 0)  # Intensity must be non-negative

    def test_forward_batching(self, telescope):
        """Test batched processing with small batch size."""
        config = LineAcquisitionConfig(batch_size=2)
        line_acq = IncoherentLineAcquisition(config, telescope)

        n_pixels = telescope.config.n_pixels
        field_kspace = torch.randn(n_pixels, n_pixels, dtype=torch.complex64)
        start = torch.tensor([0.0, 0.0])
        end = torch.tensor([20.0, 0.0])  # Should need more than 2 samples

        measurement = line_acq.forward(field_kspace, start, end)

        assert measurement.shape == (n_pixels, n_pixels)
        assert torch.all(measurement >= 0)

    def test_forward_single_position(self, line_acq, telescope):
        """Test that single position gives similar result to point measurement."""
        n_pixels = telescope.config.n_pixels
        field_kspace = torch.randn(n_pixels, n_pixels, dtype=torch.complex64)
        center = torch.tensor([0.0, 0.0])

        # Line with same start and end (single point)
        line_meas = line_acq.forward(field_kspace, center, center)

        # Compare to direct mask + IFFT
        mask = telescope.generate_aperture_mask([0.0, 0.0])
        masked = field_kspace * mask.to(field_kspace.dtype)
        from prism.utils.transforms import ifft

        point_meas = ifft(masked).abs() ** 2

        # Should be very similar (allowing for numerical precision)
        assert torch.allclose(line_meas, point_meas, rtol=1e-5, atol=1e-7)

    def test_generate_line_mask_shape(self, line_acq, telescope):
        """Test line mask generation shape."""
        start = torch.tensor([0.0, 0.0])
        end = torch.tensor([10.0, 10.0])

        mask = line_acq.generate_line_mask(start, end)

        n_pixels = telescope.config.n_pixels
        assert mask.shape == (n_pixels, n_pixels)
        assert mask.dtype == torch.bool

    def test_generate_line_mask_coverage(self, line_acq):
        """Test that line mask covers expected region."""
        start = torch.tensor([0.0, 0.0])
        end = torch.tensor([10.0, 0.0])  # Horizontal line

        mask = line_acq.generate_line_mask(start, end)

        # Should have non-zero coverage
        assert torch.any(mask)

        # Should be larger than single aperture
        single_mask = line_acq.instrument.generate_aperture_mask([0.0, 0.0])
        assert torch.sum(mask) >= torch.sum(single_mask)

    @pytest.mark.parametrize("mode", ["accurate", "fast"])
    def test_modes(self, telescope, mode):
        """Test both accurate and fast modes work."""
        config = LineAcquisitionConfig(mode=mode)
        line_acq = IncoherentLineAcquisition(config, telescope)

        n_pixels = telescope.config.n_pixels
        field_kspace = torch.randn(n_pixels, n_pixels, dtype=torch.complex64)
        start = torch.tensor([0.0, 0.0])
        end = torch.tensor([16.0, 16.0])

        measurement = line_acq.forward(field_kspace, start, end)

        assert measurement.shape == (n_pixels, n_pixels)
        assert torch.all(measurement >= 0)

    def test_different_sample_densities(self, telescope):
        """Test different samples_per_pixel values."""
        for spp in [0.5, 1.0, 2.0]:
            config = LineAcquisitionConfig(samples_per_pixel=spp)
            line_acq = IncoherentLineAcquisition(config, telescope)

            line_length = 20.0
            n_samples = line_acq.compute_n_samples(line_length)
            expected = max(2, int(line_length * spp))
            assert n_samples == expected
