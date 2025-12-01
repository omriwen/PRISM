"""Integration tests for MeasurementSystem with line acquisition."""

from __future__ import annotations

import pytest
import torch

from prism.core.instruments import Telescope, TelescopeConfig
from prism.core.line_acquisition import IncoherentLineAcquisition, LineAcquisitionConfig
from prism.core.measurement_system import MeasurementSystem, MeasurementSystemConfig


@pytest.fixture
def device():
    """Test device (CPU for CI compatibility)."""
    return torch.device("cpu")


@pytest.fixture
def telescope(device):
    """Create test telescope."""
    config = TelescopeConfig(
        n_pixels=128,
        aperture_radius_pixels=15.0,
        pixel_size=1.0e-6,  # 1.0 µm in meters (SI units)
        wavelength=0.5e-6,  # 0.5 µm in meters (SI units)
    )
    return Telescope(config).to(device)


@pytest.fixture
def line_acquisition(telescope):
    """Create line acquisition module."""
    config = LineAcquisitionConfig(
        mode="accurate",
        samples_per_pixel=1.0,
        min_samples=2,
        batch_size=16,
    )
    return IncoherentLineAcquisition(config, telescope)


@pytest.fixture
def measurement_system(telescope, line_acquisition):
    """Create measurement system with line acquisition."""
    config = MeasurementSystemConfig(
        enable_caching=False,  # Disable cache for testing
        add_noise_by_default=False,
    )
    return MeasurementSystem(telescope, config, line_acquisition=line_acquisition)


@pytest.fixture
def test_field(device):
    """Create test field."""
    field = torch.randn(128, 128, dtype=torch.complex64, device=device)
    return field


class TestMeasurementSystemLineMode:
    """Test MeasurementSystem with line acquisition."""

    def test_constructor_with_line_acquisition(self, telescope, line_acquisition):
        """Test that line acquisition can be added to MeasurementSystem."""
        ms = MeasurementSystem(telescope, line_acquisition=line_acquisition)
        assert ms.line_acquisition is line_acquisition

    def test_measure_line_first_sample(self, measurement_system, test_field, device):
        """Test first line measurement returns dual measurement."""
        start = torch.tensor([20.0, 20.0], device=device)
        end = torch.tensor([40.0, 60.0], device=device)

        # First sample: reconstruction is None
        measurement = measurement_system.measure(
            test_field,
            reconstruction=None,
            line_endpoints=(start, end),
        )

        # Should return [2, 1, H, W] with channel dimension
        assert measurement.shape == (2, 1, 128, 128)

        # First sample: both measurements should be identical
        assert torch.allclose(measurement[0], measurement[1])

    def test_measure_line_later_samples(self, measurement_system, test_field, device):
        """Test later line measurements use cumulative mask."""
        start1 = torch.tensor([20.0, 20.0], device=device)
        end1 = torch.tensor([40.0, 60.0], device=device)
        start2 = torch.tensor([60.0, 20.0], device=device)
        end2 = torch.tensor([80.0, 60.0], device=device)

        # First measurement
        measurement_system.measure(
            test_field,
            reconstruction=None,
            line_endpoints=(start1, end1),
        )
        measurement_system.add_mask(line_endpoints=(start1, end1))

        # Create a different reconstruction
        reconstruction = test_field * 0.8 + 0.2

        # Second measurement
        meas2 = measurement_system.measure(
            test_field,
            reconstruction=reconstruction,
            line_endpoints=(start2, end2),
        )

        # Should return [2, 1, H, W] with channel dimension
        assert meas2.shape == (2, 1, 128, 128)

        # Second sample: measurements should be different
        # meas2[0] = reconstruction through cum_mask
        # meas2[1] = ground_truth through new line
        assert not torch.allclose(meas2[0], meas2[1])

    def test_add_mask_line_mode(self, measurement_system, device):
        """Test add_mask with line endpoints."""
        start = torch.tensor([20.0, 20.0], device=device)
        end = torch.tensor([40.0, 60.0], device=device)

        # Check initial state
        assert measurement_system.sample_count == 0
        assert measurement_system.cum_mask.sum() == 0

        # Add line mask
        measurement_system.add_mask(line_endpoints=(start, end))

        # Should increment sample count
        assert measurement_system.sample_count == 1

        # Cumulative mask should have coverage
        assert measurement_system.cum_mask.sum() > 0

    def test_add_mask_multiple_lines(self, measurement_system, device):
        """Test adding multiple line masks."""
        lines = [
            (torch.tensor([10.0, 10.0], device=device), torch.tensor([30.0, 30.0], device=device)),
            (torch.tensor([40.0, 10.0], device=device), torch.tensor([60.0, 30.0], device=device)),
            (torch.tensor([70.0, 10.0], device=device), torch.tensor([90.0, 30.0], device=device)),
        ]

        # Add all lines
        for start, end in lines:
            measurement_system.add_mask(line_endpoints=(start, end))

        # Should have 3 samples
        assert measurement_system.sample_count == 3

        # Cumulative mask should have coverage from all lines
        assert measurement_system.cum_mask.sum() > 0

    def test_measure_line_without_line_acquisition_raises(self, telescope, test_field, device):
        """Test that using line_endpoints without line_acquisition raises error."""
        # Create MeasurementSystem without line acquisition
        ms = MeasurementSystem(telescope)

        start = torch.tensor([20.0, 20.0], device=device)
        end = torch.tensor([40.0, 60.0], device=device)

        # Should raise ValueError
        with pytest.raises(ValueError, match="no line_acquisition module configured"):
            ms.measure(test_field, line_endpoints=(start, end))

    def test_add_mask_line_without_line_acquisition_raises(self, telescope, device):
        """Test that add_mask with line_endpoints without line_acquisition raises error."""
        # Create MeasurementSystem without line acquisition
        ms = MeasurementSystem(telescope)

        start = torch.tensor([20.0, 20.0], device=device)
        end = torch.tensor([40.0, 60.0], device=device)

        # Should raise ValueError
        with pytest.raises(ValueError, match="no line_acquisition module configured"):
            ms.add_mask(line_endpoints=(start, end))

    def test_measure_point_mode_still_works(self, measurement_system, test_field):
        """Test that point acquisition mode still works."""
        # Point mode should still work
        centers = [[30.0, 40.0]]
        measurement = measurement_system.measure(test_field, centers=centers)

        # Should return [2, 1, H, W] with channel dimension
        assert measurement.shape == (2, 1, 128, 128)

    def test_add_mask_point_mode_still_works(self, measurement_system):
        """Test that add_mask in point mode still works."""
        centers = [[30.0, 40.0], [50.0, 60.0]]

        # Check initial state
        initial_count = measurement_system.sample_count

        # Add point masks
        measurement_system.add_mask(centers=centers)

        # Should increment by 2
        assert measurement_system.sample_count == initial_count + 2

    def test_measure_with_noise(self, measurement_system, test_field, device):
        """Test line measurement with noise."""
        start = torch.tensor([20.0, 20.0], device=device)
        end = torch.tensor([40.0, 60.0], device=device)

        # Measure with noise
        meas_with_noise = measurement_system.measure(
            test_field,
            reconstruction=None,
            line_endpoints=(start, end),
            add_noise=True,
        )

        # Should still return valid shape
        assert meas_with_noise.shape == (2, 1, 128, 128)

        # Values should be finite
        assert torch.isfinite(meas_with_noise).all()

    def test_line_acquisition_integration_full_workflow(self, measurement_system, device):
        """Test full workflow: measure -> add_mask -> measure -> add_mask."""
        # Create test fields
        ground_truth = torch.randn(128, 128, dtype=torch.complex64, device=device)
        reconstruction = ground_truth.clone() * 0.9

        # First line
        start1 = torch.tensor([20.0, 20.0], device=device)
        end1 = torch.tensor([40.0, 60.0], device=device)

        meas1 = measurement_system.measure(ground_truth, None, line_endpoints=(start1, end1))
        assert meas1.shape == (2, 1, 128, 128)

        measurement_system.add_mask(line_endpoints=(start1, end1))
        assert measurement_system.sample_count == 1

        # Second line
        start2 = torch.tensor([60.0, 20.0], device=device)
        end2 = torch.tensor([80.0, 60.0], device=device)

        meas2 = measurement_system.measure(
            ground_truth, reconstruction, line_endpoints=(start2, end2)
        )
        assert meas2.shape == (2, 1, 128, 128)

        measurement_system.add_mask(line_endpoints=(start2, end2))
        assert measurement_system.sample_count == 2

        # Cumulative mask should have coverage from both lines
        assert measurement_system.cum_mask.sum() > 0

    def test_backward_compatibility_with_none_centers(self, measurement_system, test_field):
        """Test that centers=None defaults to [[0, 0]] (backward compatibility)."""
        # Call without centers (should default to [[0, 0]])
        measurement = measurement_system.measure(test_field, centers=None)

        # Should work and return valid measurement
        assert measurement.shape == (2, 1, 128, 128)

    def test_add_mask_requires_either_centers_or_endpoints(self, measurement_system):
        """Test that add_mask requires either centers or line_endpoints."""
        # Neither provided should raise error
        with pytest.raises(ValueError, match="Either centers or line_endpoints must be provided"):
            measurement_system.add_mask()


class TestMeasurementSystemLineModeModes:
    """Test different line acquisition modes."""

    def test_fast_mode_integration(self, telescope, device):
        """Test fast mode integration with MeasurementSystem."""
        # Create fast mode line acquisition
        config = LineAcquisitionConfig(mode="fast", min_samples=2)
        line_acq = IncoherentLineAcquisition(config, telescope)
        ms = MeasurementSystem(telescope, line_acquisition=line_acq)

        # Create test field
        field = torch.randn(128, 128, dtype=torch.complex64, device=device)

        # Measure with line
        start = torch.tensor([20.0, 20.0], device=device)
        end = torch.tensor([100.0, 100.0], device=device)

        measurement = ms.measure(field, line_endpoints=(start, end))

        # Should return valid measurement
        assert measurement.shape == (2, 1, 128, 128)
        assert torch.isfinite(measurement).all()

    def test_accurate_mode_integration(self, telescope, device):
        """Test accurate mode integration with MeasurementSystem."""
        # Create accurate mode line acquisition
        config = LineAcquisitionConfig(mode="accurate", samples_per_pixel=1.0)
        line_acq = IncoherentLineAcquisition(config, telescope)
        ms = MeasurementSystem(telescope, line_acquisition=line_acq)

        # Create test field
        field = torch.randn(128, 128, dtype=torch.complex64, device=device)

        # Measure with line
        start = torch.tensor([20.0, 20.0], device=device)
        end = torch.tensor([100.0, 100.0], device=device)

        measurement = ms.measure(field, line_endpoints=(start, end))

        # Should return valid measurement
        assert measurement.shape == (2, 1, 128, 128)
        assert torch.isfinite(measurement).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
