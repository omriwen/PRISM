"""Integration tests for SPIDS reconstruction with scanning illumination mode.

End-to-end tests verifying that SPIDS reconstruction works correctly when
using scanning illumination instead of scanning aperture.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.measurement_system import (
    MeasurementSystem,
    MeasurementSystemConfig,
    ScanningMode,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def microscope() -> Microscope:
    """Create microscope for SPIDS tests."""
    config = MicroscopeConfig(
        n_pixels=64,  # Smaller for faster tests
        pixel_size=1e-6,
        numerical_aperture=0.5,
        wavelength=520e-9,
        magnification=40.0,
    )
    return Microscope(config)


@pytest.fixture
def test_target(microscope: Microscope) -> Tensor:
    """Create test target with fine structure.

    Uses a pattern that has both low and high spatial frequency content
    to test synthetic aperture reconstruction.
    """
    n = microscope.config.n_pixels

    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n),
        torch.linspace(-1, 1, n),
        indexing="ij",
    )
    r = torch.sqrt(x**2 + y**2)

    # Base object: circular disc
    disc = (r < 0.5).float()

    # Add fine structure (grating)
    grating = 0.3 * torch.sin(20 * torch.pi * x) * (r < 0.5).float()

    # Combine
    obj = (disc + grating + 0.1).clamp(0.1, 1.0)

    return obj.to(torch.complex64)


@pytest.fixture
def scan_positions() -> list:
    """Generate scanning positions for SPIDS measurements.

    Returns positions in a grid pattern centered on DC.
    """
    positions = []
    # 3x3 grid of positions
    for ky_idx in range(-1, 2):
        for kx_idx in range(-1, 2):
            positions.append([ky_idx * 8, kx_idx * 8])  # Pixel offsets
    return positions


# =============================================================================
# SPIDS Pipeline Tests
# =============================================================================


class TestSPIDSIlluminationPipeline:
    """Integration tests for SPIDS with illumination mode."""

    def test_measurement_collection(
        self,
        microscope: Microscope,
        test_target: Tensor,
    ) -> None:
        """Test that measurements can be collected in illumination mode."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Collect measurements at multiple positions
        positions = [[0, 0], [5, 5], [-5, 5], [5, -5], [-5, -5]]
        measurements = ms.get_measurements(test_target, positions, add_noise=False)

        assert measurements.shape[0] == len(positions)
        assert measurements.shape[1:] == test_target.shape

    def test_cumulative_mask_builds_correctly(
        self,
        microscope: Microscope,
        test_target: Tensor,
        scan_positions: list,
    ) -> None:
        """Test that cumulative mask is built correctly in illumination mode.

        Note: measure() does not automatically update cum_mask - add_mask()
        must be called separately. This is by design for flexibility.
        """
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Collect measurements and build mask
        for pos in scan_positions:
            ms.measure(test_target, None, [pos], add_noise=False)
            ms.add_mask([pos])  # Must call add_mask separately

        # Cumulative mask should have coverage
        coverage = ms.cum_mask.float().mean().item()
        assert coverage > 0, "Cumulative mask should have non-zero coverage"
        assert ms.sample_count == len(scan_positions)

    def test_gaussian_illumination_measurements(
        self,
        microscope: Microscope,
        test_target: Tensor,
    ) -> None:
        """Test measurements with Gaussian illumination source."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=0.05e6,  # sigma in k-space
        )
        ms = MeasurementSystem(microscope, config=config)

        # Collect measurement
        measurement = ms.get_measurements(test_target, [[0, 0]], add_noise=False)

        assert measurement.shape == test_target.shape
        assert measurement.sum() > 0

    def test_circular_illumination_measurements(
        self,
        microscope: Microscope,
        test_target: Tensor,
    ) -> None:
        """Test measurements with circular illumination source."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=0.05e6,  # radius in k-space
        )
        ms = MeasurementSystem(microscope, config=config)

        # Collect measurement
        measurement = ms.get_measurements(test_target, [[0, 0]], add_noise=False)

        assert measurement.shape == test_target.shape
        assert measurement.sum() > 0


# =============================================================================
# Reconstruction Quality Tests
# =============================================================================


class TestReconstructionQuality:
    """Tests for reconstruction quality with illumination mode."""

    def test_dc_measurement_captures_low_frequencies(
        self,
        microscope: Microscope,
        test_target: Tensor,
    ) -> None:
        """Test that DC measurement captures low spatial frequencies."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # DC measurement
        meas_dc = ms.get_measurements(test_target, [[0, 0]], add_noise=False)

        # DC measurement should capture the overall structure
        # Correlation with smoothed target should be high
        smoothed_target = torch.abs(test_target) ** 2
        smoothed_target = smoothed_target / smoothed_target.max()
        meas_norm = meas_dc / (meas_dc.max() + 1e-10)

        # Both should be non-zero and have similar structure
        assert meas_dc.sum() > 0
        assert (
            torch.corrcoef(torch.stack([smoothed_target.flatten(), meas_norm.flatten()]))[0, 1]
            > 0.3
        )

    def test_off_axis_measurement_captures_high_frequencies(
        self,
        microscope: Microscope,
        test_target: Tensor,
    ) -> None:
        """Test that off-axis measurements capture different content than DC."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # DC measurement
        meas_dc = ms.get_measurements(test_target, [[0, 0]], add_noise=False)

        # Off-axis measurement
        meas_offaxis = ms.get_measurements(test_target, [[10, 10]], add_noise=False)

        # The measurements should be different
        diff = torch.abs(meas_dc - meas_offaxis).mean()
        assert diff > 1e-6, "Off-axis measurement should differ from DC"

    def test_multiple_measurements_increase_coverage(
        self,
        microscope: Microscope,
        test_target: Tensor,
    ) -> None:
        """Test that multiple measurements increase k-space coverage.

        Note: measure() does not automatically update cum_mask - add_mask()
        must be called separately.
        """
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Single measurement
        ms.measure(test_target, None, [[0, 0]], add_noise=False)
        ms.add_mask([[0, 0]])
        coverage_1 = ms.cum_mask.float().sum().item()

        # Add more measurements
        ms.measure(test_target, None, [[10, 0]], add_noise=False)
        ms.add_mask([[10, 0]])
        ms.measure(test_target, None, [[0, 10]], add_noise=False)
        ms.add_mask([[0, 10]])
        coverage_3 = ms.cum_mask.float().sum().item()

        # Coverage should increase
        assert coverage_3 > coverage_1


# =============================================================================
# Comparison Tests
# =============================================================================


class TestIlluminationVsAperture:
    """Tests comparing illumination mode with aperture mode."""

    def test_point_illumination_similar_to_aperture(
        self,
        microscope: Microscope,
        test_target: Tensor,
    ) -> None:
        """Test that POINT illumination gives similar results to aperture mode.

        For point sources, scanning illumination should be equivalent to
        scanning aperture by reciprocity (Fourier shift theorem).
        """
        # Aperture mode measurement
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)
        meas_aperture = ms_aperture.get_measurements(test_target, [[0, 0]], add_noise=False)

        # Illumination mode measurement (POINT source)
        microscope2 = Microscope(microscope.config)
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms_illum = MeasurementSystem(microscope2, config=config_illum)
        meas_illum = ms_illum.get_measurements(test_target, [[0, 0]], add_noise=False)

        # Normalize for comparison
        meas_aperture_norm = meas_aperture / (meas_aperture.max() + 1e-10)
        meas_illum_norm = meas_illum / (meas_illum.max() + 1e-10)

        # Should be similar (correlation > 0.5)
        correlation = torch.corrcoef(
            torch.stack([meas_aperture_norm.flatten(), meas_illum_norm.flatten()])
        )[0, 1]
        assert correlation > 0.5, f"Correlation too low: {correlation}"

    def test_gaussian_differs_from_aperture(
        self,
        microscope: Microscope,
        test_target: Tensor,
    ) -> None:
        """Test that GAUSSIAN illumination differs from aperture mode.

        Finite-size sources introduce partial coherence effects that
        cannot be replicated with simple aperture scanning.
        """
        # Aperture mode measurement
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)
        meas_aperture = ms_aperture.get_measurements(test_target, [[0, 0]], add_noise=False)

        # Illumination mode measurement (GAUSSIAN source)
        microscope2 = Microscope(microscope.config)
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=0.2e6,  # Significant width
        )
        ms_illum = MeasurementSystem(microscope2, config=config_illum)
        meas_illum = ms_illum.get_measurements(test_target, [[0, 0]], add_noise=False)

        # Should be different (finite source != hard aperture)
        diff = torch.abs(meas_aperture - meas_illum).mean()
        assert diff > 1e-6, "Gaussian illumination should differ from aperture"


# =============================================================================
# Edge Cases
# =============================================================================


class TestSPIDSIlluminationEdgeCases:
    """Edge case tests for SPIDS illumination mode."""

    def test_empty_scan_positions(
        self,
        microscope: Microscope,
    ) -> None:
        """Test handling of empty scan positions."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Should start with zero coverage
        assert ms.sample_count == 0

    def test_repeated_position_measurements(
        self,
        microscope: Microscope,
        test_target: Tensor,
    ) -> None:
        """Test that repeated measurements at same position are consistent."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Multiple measurements at same position
        meas1 = ms.get_measurements(test_target, [[0, 0]], add_noise=False)
        meas2 = ms.get_measurements(test_target, [[0, 0]], add_noise=False)

        # Should be identical (no noise)
        assert torch.allclose(meas1, meas2)

    def test_large_number_of_measurements(
        self,
        microscope: Microscope,
        test_target: Tensor,
    ) -> None:
        """Test collecting many measurements."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Generate many positions
        positions = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                positions.append([i * 5, j * 5])

        # Collect all measurements
        measurements = ms.get_measurements(test_target, positions, add_noise=False)

        assert measurements.shape[0] == len(positions)
        assert ms.sample_count == 0  # get_measurements doesn't update sample_count


# =============================================================================
# Noise Tests
# =============================================================================


class TestSPIDSIlluminationNoise:
    """Tests for noise handling in SPIDS illumination mode."""

    def test_noisy_measurements(
        self,
        microscope: Microscope,
        test_target: Tensor,
    ) -> None:
        """Test that noise can be added to measurements."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        torch.manual_seed(42)
        meas_clean = ms.get_measurements(test_target, [[0, 0]], add_noise=False)

        torch.manual_seed(42)
        meas_noisy = ms.get_measurements(test_target, [[0, 0]], add_noise=True)

        # Noisy measurement should differ from clean
        diff = torch.abs(meas_noisy - meas_clean).mean()
        assert diff > 0, "Noisy measurement should differ from clean"

    def test_noise_is_random(
        self,
        microscope: Microscope,
        test_target: Tensor,
    ) -> None:
        """Test that noise varies between measurements."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        torch.manual_seed(42)
        meas1 = ms.get_measurements(test_target, [[0, 0]], add_noise=True)

        torch.manual_seed(43)  # Different seed
        meas2 = ms.get_measurements(test_target, [[0, 0]], add_noise=True)

        # Two noisy measurements should differ
        diff = torch.abs(meas1 - meas2).mean()
        assert diff > 0, "Noise should vary between measurements"
