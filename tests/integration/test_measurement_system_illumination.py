"""Integration tests for MeasurementSystem with scanning illumination mode.

Tests the cumulative mask behavior and measurement routing when using
scanning illumination mode instead of scanning aperture mode.
"""

from __future__ import annotations

import pytest
import torch

from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.measurement_system import (
    MeasurementSystem,
    MeasurementSystemConfig,
    ScanningMode,
)


@pytest.fixture
def microscope() -> Microscope:
    """Create a microscope for testing."""
    config = MicroscopeConfig(
        n_pixels=128,
        pixel_size=1e-6,  # 1 micron
        numerical_aperture=0.5,
        wavelength=520e-9,
        magnification=40.0,
    )
    return Microscope(config)


@pytest.fixture
def test_object(microscope: Microscope) -> torch.Tensor:
    """Create a simple test object."""
    n = microscope.config.n_pixels
    # Simple circular object
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n),
        torch.linspace(-1, 1, n),
        indexing="ij",
    )
    r = torch.sqrt(x**2 + y**2)
    obj = (r < 0.3).float() + 0.5
    return obj.to(torch.complex64)


class TestIlluminationModeConfig:
    """Test illumination mode configuration."""

    def test_illumination_mode_point_source(self, microscope: Microscope) -> None:
        """Test creating MeasurementSystem with POINT illumination."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        assert ms.scanning_mode == ScanningMode.ILLUMINATION
        assert ms.config.illumination_source_type == "POINT"

    def test_illumination_mode_gaussian_source(self, microscope: Microscope) -> None:
        """Test creating MeasurementSystem with GAUSSIAN illumination."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=0.05e6,  # sigma in k-space
        )
        ms = MeasurementSystem(microscope, config=config)

        assert ms.config.illumination_source_type == "GAUSSIAN"
        assert ms.config.illumination_radius == 0.05e6

    def test_illumination_mode_circular_source(self, microscope: Microscope) -> None:
        """Test creating MeasurementSystem with CIRCULAR illumination."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=0.05e6,  # radius in k-space
        )
        ms = MeasurementSystem(microscope, config=config)

        assert ms.config.illumination_source_type == "CIRCULAR"

    def test_illumination_mode_requires_radius_for_gaussian(self, microscope: Microscope) -> None:
        """Test GAUSSIAN source requires illumination_radius."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=None,  # Missing required radius
        )
        with pytest.raises(ValueError, match="illumination_radius is required"):
            MeasurementSystem(microscope, config=config)

    def test_illumination_mode_requires_radius_for_circular(self, microscope: Microscope) -> None:
        """Test CIRCULAR source requires illumination_radius."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=None,  # Missing required radius
        )
        with pytest.raises(ValueError, match="illumination_radius is required"):
            MeasurementSystem(microscope, config=config)


class TestCumulativeMaskIlluminationMode:
    """Test cumulative mask behavior in illumination mode."""

    def test_add_mask_illumination_point_source(self, microscope: Microscope) -> None:
        """Test add_mask works with POINT illumination source."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Add mask at center
        ms.add_mask([[0, 0]])

        assert ms.sample_count == 1
        assert ms.cum_mask.sum() > 0

    def test_add_mask_illumination_gaussian_source(self, microscope: Microscope) -> None:
        """Test add_mask with GAUSSIAN illumination uses 2*sigma for mask."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=0.05e6,  # sigma
        )
        ms = MeasurementSystem(microscope, config=config)

        # Add mask at center
        ms.add_mask([[0, 0]])

        assert ms.sample_count == 1
        # Mask should cover more than zero pixels
        coverage = ms.cum_mask.sum().item()
        assert coverage > 0

    def test_add_mask_illumination_circular_source(self, microscope: Microscope) -> None:
        """Test add_mask with CIRCULAR illumination uses exact radius."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=0.05e6,  # radius
        )
        ms = MeasurementSystem(microscope, config=config)

        ms.add_mask([[0, 0]])

        assert ms.sample_count == 1
        assert ms.cum_mask.sum() > 0

    def test_gaussian_mask_larger_than_circular(self, microscope: Microscope) -> None:
        """Test GAUSSIAN mask is ~2x larger than CIRCULAR for same radius.

        For GAUSSIAN sources, the mask uses 2*sigma radius.
        For CIRCULAR sources, the mask uses the radius directly.
        So GAUSSIAN should cover ~4x the area for the same parameter value.
        """
        # Use a larger k-space parameter to ensure measurable coverage
        # Grid resolution: dk = 1/(n*dx) = 1/(128 * 1e-6) = 7812.5 1/m
        # We need k_radius >> dk for meaningful coverage
        k_radius = 0.5e6  # Much larger than dk

        # Create GAUSSIAN measurement system
        config_gauss = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=k_radius,  # sigma for Gaussian
        )
        ms_gauss = MeasurementSystem(microscope, config=config_gauss)
        ms_gauss.add_mask([[0, 0]])
        coverage_gauss = ms_gauss.cum_mask.sum().item()

        # Create CIRCULAR measurement system
        config_circ = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=k_radius,  # radius for circular
        )
        # Need fresh microscope to avoid grid caching issues
        microscope2 = Microscope(microscope.config)
        ms_circ = MeasurementSystem(microscope2, config=config_circ)
        ms_circ.add_mask([[0, 0]])
        coverage_circ = ms_circ.cum_mask.sum().item()

        # Gaussian uses 2*sigma, so should cover ~4x area (area = pi*r^2)
        # Allow for discretization effects
        assert coverage_gauss > coverage_circ, (
            f"Gaussian coverage ({coverage_gauss}) should be larger than "
            f"circular coverage ({coverage_circ})"
        )
        # Both should have meaningful coverage
        assert coverage_gauss > 10, f"Gaussian coverage too small: {coverage_gauss}"
        assert coverage_circ > 5, f"Circular coverage too small: {coverage_circ}"

    def test_add_mask_off_center(self, microscope: Microscope) -> None:
        """Test add_mask places mask at correct off-center position."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=0.02e6,
        )
        ms = MeasurementSystem(microscope, config=config)

        # Add mask at off-center position
        off_center = [10, 15]  # pixel coordinates
        ms.add_mask([off_center])

        # Verify mask is not centered (more coverage on one side)
        n = microscope.config.n_pixels
        center = n // 2

        # Get the mask and check the position of covered pixels
        mask = ms.cum_mask
        covered_y, covered_x = torch.where(mask)

        if len(covered_y) > 0:
            # Mean position should be offset from center
            mean_y = (covered_y.float() - center).mean().item()
            mean_x = (covered_x.float() - center).mean().item()

            # Should be offset in the direction of off_center
            assert abs(mean_y - off_center[0]) < 5, f"Y offset {mean_y} != {off_center[0]}"
            assert abs(mean_x - off_center[1]) < 5, f"X offset {mean_x} != {off_center[1]}"

    def test_cumulative_mask_accumulates(self, microscope: Microscope) -> None:
        """Test masks accumulate correctly in illumination mode."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=0.02e6,
        )
        ms = MeasurementSystem(microscope, config=config)

        # Add first mask
        ms.add_mask([[0, 0]])
        coverage_1 = ms.cum_mask.sum().item()

        # Add second mask at different position
        ms.add_mask([[20, 20]])
        coverage_2 = ms.cum_mask.sum().item()

        # Coverage should increase
        assert coverage_2 > coverage_1
        assert ms.sample_count == 2


class TestMeasurementRoutingIlluminationMode:
    """Test measurement routing in illumination mode."""

    def test_measure_routes_to_illumination(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test measure() routes to illumination mode correctly."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Generate measurement
        dual_meas = ms.measure(test_object, None, [[0, 0]], add_noise=False)

        # Should return dual measurement [2, H, W]
        assert dual_meas.ndim == 3 or dual_meas.ndim == 4
        if dual_meas.ndim == 3:
            assert dual_meas.shape[0] == 2
        else:
            assert dual_meas.shape[0] == 2

    def test_get_measurements_illumination(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test get_measurements() works in illumination mode."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Get single measurement
        meas = ms.get_measurements(test_object, [[0, 0]], add_noise=False)

        assert meas.shape == (128, 128)
        assert meas.dtype == torch.float32

    def test_get_measurements_multiple_illumination(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test get_measurements() with multiple positions in illumination mode."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Get multiple measurements
        meas = ms.get_measurements(test_object, [[0, 0], [10, 10]], add_noise=False)

        assert meas.shape == (2, 128, 128)


class TestIlluminationApertureEquivalence:
    """Test equivalence between illumination and aperture modes for point sources."""

    def test_point_illumination_vs_aperture_similar(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test POINT illumination gives similar results to aperture mode.

        For point-like illumination, scanning illumination should be
        equivalent to scanning aperture by the Fourier shift theorem.
        """
        # Aperture mode
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)
        meas_aperture = ms_aperture.get_measurements(test_object, [[0, 0]], add_noise=False)

        # Illumination mode with POINT source
        microscope2 = Microscope(microscope.config)
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms_illum = MeasurementSystem(microscope2, config=config_illum)
        meas_illum = ms_illum.get_measurements(test_object, [[0, 0]], add_noise=False)

        # Results should be similar (not exact due to implementation differences)
        # At DC (center), both should give similar intensity
        # Check that the central region is similar
        n = microscope.config.n_pixels
        center_slice = slice(n // 2 - 10, n // 2 + 10)
        aperture_center = meas_aperture[center_slice, center_slice]
        illum_center = meas_illum[center_slice, center_slice]

        # Normalize and compare
        aperture_norm = aperture_center / (aperture_center.max() + 1e-10)
        illum_norm = illum_center / (illum_center.max() + 1e-10)

        # Should be similar (within 50% for this test - exact equivalence
        # depends on implementation details)
        correlation = torch.corrcoef(torch.stack([aperture_norm.flatten(), illum_norm.flatten()]))[
            0, 1
        ]
        assert correlation > 0.5, f"Correlation {correlation} too low"


class TestCumulativeMaskGetInfo:
    """Test get_info includes illumination configuration."""

    def test_get_info_illumination_config(self, microscope: Microscope) -> None:
        """Test get_info includes illumination configuration."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=0.05e6,
        )
        ms = MeasurementSystem(microscope, config=config)

        info = ms.get_info()

        assert info["scanning_mode"] == "ILLUMINATION"
        assert "illumination_config" in info
        assert info["illumination_config"]["source_type"] == "GAUSSIAN"
        assert info["illumination_config"]["radius"] == 0.05e6

    def test_repr_includes_illumination_mode(self, microscope: Microscope) -> None:
        """Test __repr__ shows illumination mode."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        repr_str = repr(ms)

        assert "ILLUMINATION" in repr_str


class TestMaskRadiusConversion:
    """Test mask radius conversion from k-space to pixels."""

    def test_k_radius_to_pixels_correct(self, microscope: Microscope) -> None:
        """Test _k_radius_to_pixels gives correct conversion."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=0.1e6,  # 0.1 million 1/m
        )
        ms = MeasurementSystem(microscope, config=config)

        # Get pixel radius
        pixel_radius = ms._k_radius_to_pixels(0.1e6)

        # Verify conversion
        # dk = 1 / (n * dx), pixel_radius = k_radius / dk = k_radius * n * dx
        n = microscope.config.n_pixels
        dx = microscope.grid.dx
        expected_radius = 0.1e6 * n * dx

        assert abs(pixel_radius - expected_radius) < 1e-6

    def test_compute_illumination_mask_radius_gaussian(self, microscope: Microscope) -> None:
        """Test _compute_illumination_mask_radius for GAUSSIAN source."""
        sigma = 0.05e6
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=sigma,
        )
        ms = MeasurementSystem(microscope, config=config)

        # Should return 2*sigma in pixel units
        mask_radius = ms._compute_illumination_mask_radius()

        expected_k_radius = 2.0 * sigma
        expected_pixel_radius = ms._k_radius_to_pixels(expected_k_radius)

        assert mask_radius is not None
        assert abs(mask_radius - expected_pixel_radius) < 1e-10

    def test_compute_illumination_mask_radius_circular(self, microscope: Microscope) -> None:
        """Test _compute_illumination_mask_radius for CIRCULAR source."""
        radius = 0.05e6
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=radius,
        )
        ms = MeasurementSystem(microscope, config=config)

        # Should return radius directly in pixel units
        mask_radius = ms._compute_illumination_mask_radius()

        expected_pixel_radius = ms._k_radius_to_pixels(radius)

        assert mask_radius is not None
        assert abs(mask_radius - expected_pixel_radius) < 1e-10

    def test_compute_illumination_mask_radius_point(self, microscope: Microscope) -> None:
        """Test _compute_illumination_mask_radius for POINT source returns None."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Should return None (use instrument default)
        mask_radius = ms._compute_illumination_mask_radius()

        assert mask_radius is None
