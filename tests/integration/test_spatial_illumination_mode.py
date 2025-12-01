"""Integration tests for spatial illumination scanning mode."""

from __future__ import annotations

import pytest
import torch

from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.measurement_system import (
    IlluminationScanMethod,
    MeasurementSystem,
    MeasurementSystemConfig,
    ScanningMode,
)


class TestMeasurementSystemSpatialMode:
    """Integration tests for MeasurementSystem with spatial illumination."""

    @pytest.fixture
    def microscope(self) -> Microscope:
        """Create microscope for spatial illumination tests."""
        return Microscope(
            MicroscopeConfig(
                n_pixels=64,
                pixel_size=1e-6,
                numerical_aperture=0.5,
                wavelength=520e-9,
                magnification=40.0,
            )
        )

    @pytest.fixture
    def test_object(self, microscope: Microscope) -> torch.Tensor:
        """Create a simple test object."""
        n = microscope.config.n_pixels
        return torch.ones(n, n, dtype=torch.complex64)

    def test_spatial_mode_config(self, microscope: Microscope) -> None:
        """Test creating MeasurementSystem with spatial mode."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)
        assert ms.config.illumination_scan_method == IlluminationScanMethod.SPATIAL

    def test_get_measurements_spatial(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test get_measurements with spatial mode."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        # Centers in pixel units
        meas = ms.get_measurements(test_object, [[0, 0]], add_noise=False)

        assert meas.shape[-2:] == (64, 64)
        assert meas.sum() > 0

    def test_mask_accumulation_spatial(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test mask accumulation with spatial mode."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        # Add masks at different positions
        ms.add_mask([[0, 0]])
        ms.add_mask([[5, 5]])

        # Cumulative mask should have content
        assert ms.cum_mask.sum() > 0

    def test_backward_compatibility_angular(self, microscope: Microscope) -> None:
        """Test that default (ANGULAR) mode still works."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Should default to ANGULAR
        assert ms.config.illumination_scan_method == IlluminationScanMethod.ANGULAR


class TestSpatialIlluminationSourceTypes:
    """Test spatial illumination with different source types."""

    @pytest.fixture
    def microscope(self) -> Microscope:
        """Create microscope for spatial illumination tests."""
        return Microscope(
            MicroscopeConfig(
                n_pixels=64,
                pixel_size=1e-6,
                numerical_aperture=0.5,
                wavelength=520e-9,
                magnification=40.0,
            )
        )

    @pytest.fixture
    def test_object(self, microscope: Microscope) -> torch.Tensor:
        """Create a simple test object."""
        n = microscope.config.n_pixels
        return torch.ones(n, n, dtype=torch.complex64)

    def test_spatial_point_source(self, microscope: Microscope, test_object: torch.Tensor) -> None:
        """Test spatial mode with POINT source."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_type="POINT",
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        meas = ms.get_measurements(test_object, [[0, 0]], add_noise=False)

        assert meas.shape[-2:] == (64, 64)
        assert meas.sum() > 0

    def test_spatial_gaussian_source(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test spatial mode with GAUSSIAN source."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_type="GAUSSIAN",
            illumination_radius=5e-6,  # sigma in meters (physical units for spatial mode)
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        meas = ms.get_measurements(test_object, [[0, 0]], add_noise=False)

        assert meas.shape[-2:] == (64, 64)
        assert meas.sum() > 0

    def test_spatial_circular_source(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test spatial mode with CIRCULAR source."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_type="CIRCULAR",
            illumination_radius=5e-6,  # radius in meters (physical units for spatial mode)
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        meas = ms.get_measurements(test_object, [[0, 0]], add_noise=False)

        assert meas.shape[-2:] == (64, 64)
        assert meas.sum() > 0


class TestSpatialIlluminationPositions:
    """Test spatial illumination at different source positions."""

    @pytest.fixture
    def microscope(self) -> Microscope:
        """Create microscope for spatial illumination tests."""
        return Microscope(
            MicroscopeConfig(
                n_pixels=64,
                pixel_size=1e-6,
                numerical_aperture=0.5,
                wavelength=520e-9,
                magnification=40.0,
            )
        )

    @pytest.fixture
    def test_object(self, microscope: Microscope) -> torch.Tensor:
        """Create a simple test object."""
        n = microscope.config.n_pixels
        return torch.ones(n, n, dtype=torch.complex64)

    def test_centered_source(self, microscope: Microscope, test_object: torch.Tensor) -> None:
        """Test measurement with centered source position."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        meas = ms.get_measurements(test_object, [[0, 0]], add_noise=False)

        assert meas.shape[-2:] == (64, 64)
        assert meas.sum() > 0

    def test_shifted_source_changes_output(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test that different source positions produce different measurements."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        # Centered measurement
        meas_center = ms.get_measurements(test_object, [[0, 0]], add_noise=False)

        # Shifted measurement
        meas_shifted = ms.get_measurements(test_object, [[10, 10]], add_noise=False)

        # Measurements should differ
        diff = (meas_center - meas_shifted).abs().mean()
        assert diff > 1e-6, f"Measurements should differ, got diff={diff}"

    def test_multiple_positions(self, microscope: Microscope, test_object: torch.Tensor) -> None:
        """Test measurements at multiple positions."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        positions = [[0, 0], [5, 5], [-5, 5], [5, -5], [-5, -5]]
        meas = ms.get_measurements(test_object, positions, add_noise=False)

        assert meas.shape[0] == len(positions)
        assert meas.shape[1:] == (64, 64)


class TestSpatialIlluminationMaskTracking:
    """Test cumulative mask tracking with spatial illumination."""

    @pytest.fixture
    def microscope(self) -> Microscope:
        """Create microscope for spatial illumination tests."""
        return Microscope(
            MicroscopeConfig(
                n_pixels=64,
                pixel_size=1e-6,
                numerical_aperture=0.5,
                wavelength=520e-9,
                magnification=40.0,
            )
        )

    @pytest.fixture
    def test_object(self, microscope: Microscope) -> torch.Tensor:
        """Create a simple test object."""
        n = microscope.config.n_pixels
        return torch.ones(n, n, dtype=torch.complex64)

    def test_mask_at_center(self, microscope: Microscope) -> None:
        """Test adding mask at center position."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        ms.add_mask([[0, 0]])

        assert ms.sample_count == 1
        assert ms.cum_mask.sum() > 0

    def test_mask_accumulation_multiple_positions(self, microscope: Microscope) -> None:
        """Test mask accumulation at multiple positions."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        # Add masks at different positions (far enough apart to not overlap)
        ms.add_mask([[0, 0]])
        coverage_1 = ms.cum_mask.sum().item()

        ms.add_mask([[20, 20]])
        coverage_2 = ms.cum_mask.sum().item()

        # Coverage should increase (or at least not decrease)
        assert coverage_2 >= coverage_1
        assert ms.sample_count == 2

    def test_mask_with_gaussian_source(self, microscope: Microscope) -> None:
        """Test mask accumulation with Gaussian source."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_type="GAUSSIAN",
            illumination_radius=5e-6,
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        ms.add_mask([[0, 0]])

        assert ms.sample_count == 1
        assert ms.cum_mask.sum() > 0


class TestSpatialIlluminationValidation:
    """Test validation and error handling for spatial illumination."""

    @pytest.fixture
    def microscope(self) -> Microscope:
        """Create microscope for spatial illumination tests."""
        return Microscope(
            MicroscopeConfig(
                n_pixels=64,
                pixel_size=1e-6,
                numerical_aperture=0.5,
                wavelength=520e-9,
                magnification=40.0,
            )
        )

    def test_spatial_requires_source_distance(self, microscope: Microscope) -> None:
        """Test that spatial mode requires illumination_source_distance."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            # Missing illumination_source_distance
        )
        with pytest.raises(ValueError, match="illumination_source_distance"):
            MeasurementSystem(microscope, config=config)

    def test_source_distance_must_be_positive(self, microscope: Microscope) -> None:
        """Test that source distance must be positive."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=-10e-3,  # Negative distance
        )
        with pytest.raises(ValueError, match="must be positive"):
            MeasurementSystem(microscope, config=config)

    def test_gaussian_requires_radius_spatial_mode(self, microscope: Microscope) -> None:
        """Test GAUSSIAN source requires illumination_radius in spatial mode."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_type="GAUSSIAN",
            illumination_radius=None,  # Missing required radius
            illumination_source_distance=10e-3,
        )
        with pytest.raises(ValueError, match="illumination_radius is required"):
            MeasurementSystem(microscope, config=config)

    def test_circular_requires_radius_spatial_mode(self, microscope: Microscope) -> None:
        """Test CIRCULAR source requires illumination_radius in spatial mode."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_type="CIRCULAR",
            illumination_radius=None,  # Missing required radius
            illumination_source_distance=10e-3,
        )
        with pytest.raises(ValueError, match="illumination_radius is required"):
            MeasurementSystem(microscope, config=config)


class TestSpatialIlluminationPhysics:
    """Test physical properties of spatial illumination."""

    @pytest.fixture
    def microscope(self) -> Microscope:
        """Create microscope for spatial illumination tests."""
        return Microscope(
            MicroscopeConfig(
                n_pixels=64,
                pixel_size=1e-6,
                numerical_aperture=0.5,
                wavelength=520e-9,
                magnification=40.0,
            )
        )

    @pytest.fixture
    def test_object(self, microscope: Microscope) -> torch.Tensor:
        """Create a simple test object."""
        n = microscope.config.n_pixels
        # Circular object
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, n),
            torch.linspace(-1, 1, n),
            indexing="ij",
        )
        r = torch.sqrt(x**2 + y**2)
        obj = (r < 0.3).float() + 0.5
        return obj.to(torch.complex64)

    def test_different_source_distances(self, microscope: Microscope) -> None:
        """Test that different source distances produce different measurements.

        Uses a structured test object to make the phase differences more apparent.
        """
        # Create structured test object with high-frequency content
        n = microscope.config.n_pixels
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, n),
            torch.linspace(-1, 1, n),
            indexing="ij",
        )
        # Add high-frequency grating to make phase-sensitive
        grating = torch.sin(10 * torch.pi * x) * torch.sin(10 * torch.pi * y)
        test_obj = (grating + 1.0).to(torch.complex64)

        # Very near source (1mm)
        config_near = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=1e-3,  # 1mm
        )
        ms_near = MeasurementSystem(microscope, config=config_near)
        meas_near = ms_near.get_measurements(test_obj, [[15, 15]], add_noise=False)

        # Far source (100mm)
        microscope2 = Microscope(microscope.config)
        config_far = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=100e-3,  # 100mm
        )
        ms_far = MeasurementSystem(microscope2, config=config_far)
        meas_far = ms_far.get_measurements(test_obj, [[15, 15]], add_noise=False)

        # Measurements should differ due to different phase curvature
        # The difference might be small but should be measurable
        diff = (meas_near - meas_far).abs().mean()
        assert diff > 1e-12, f"Measurements should differ for different distances, got diff={diff}"

    def test_spatial_vs_angular_mode(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test that spatial and angular modes produce different measurements.

        Spatial mode has position-dependent phase (spherical wavefront) while
        angular mode has uniform phase tilt. These should produce different results.
        """
        # Spatial mode
        config_spatial = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=10e-3,
        )
        ms_spatial = MeasurementSystem(microscope, config=config_spatial)
        meas_spatial = ms_spatial.get_measurements(test_object, [[10, 10]], add_noise=False)

        # Angular mode
        microscope2 = Microscope(microscope.config)
        config_angular = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.ANGULAR,
            illumination_source_type="POINT",
        )
        ms_angular = MeasurementSystem(microscope2, config=config_angular)
        meas_angular = ms_angular.get_measurements(test_object, [[10, 10]], add_noise=False)

        # Measurements should differ (spatial has quadratic phase, angular has linear phase)
        diff = (meas_spatial - meas_angular).abs().mean()
        assert diff > 1e-6, f"Spatial and angular modes should differ, got diff={diff}"


class TestSpatialIlluminationIntegrationWorkflow:
    """Integration test for complete spatial illumination workflow."""

    @pytest.fixture
    def microscope(self) -> Microscope:
        """Create microscope for spatial illumination tests."""
        return Microscope(
            MicroscopeConfig(
                n_pixels=64,
                pixel_size=1e-6,
                numerical_aperture=0.5,
                wavelength=520e-9,
                magnification=40.0,
            )
        )

    @pytest.fixture
    def test_object(self, microscope: Microscope) -> torch.Tensor:
        """Create test object with structure."""
        n = microscope.config.n_pixels
        # Circular disc with grating
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, n),
            torch.linspace(-1, 1, n),
            indexing="ij",
        )
        r = torch.sqrt(x**2 + y**2)
        disc = (r < 0.5).float()
        grating = 0.3 * torch.sin(20 * torch.pi * x) * (r < 0.5).float()
        obj = (disc + grating + 0.1).clamp(0.1, 1.0)
        return obj.to(torch.complex64)

    def test_complete_measurement_workflow(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test complete workflow: collect measurements and track masks."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        # Define scan positions
        positions = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                positions.append([i * 5, j * 5])

        # Collect measurements
        measurements = ms.get_measurements(test_object, positions, add_noise=False)

        # Verify measurements
        assert measurements.shape[0] == len(positions)
        assert measurements.shape[1:] == (64, 64)
        assert measurements.sum() > 0

        # Build cumulative mask
        for pos in positions:
            ms.add_mask([pos])

        # Verify mask coverage
        assert ms.sample_count == len(positions)
        assert ms.cum_mask.sum() > 0

        # Coverage should be reasonable (not empty)
        coverage_ratio = ms.cum_mask.float().mean().item()
        assert coverage_ratio > 0.001, f"Coverage too sparse: {coverage_ratio}"

    def test_progressive_scan_coverage(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test that coverage increases progressively with more positions."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=10e-3,
        )
        ms = MeasurementSystem(microscope, config=config)

        coverages = []

        # Add measurements progressively
        for i in range(5):
            pos = [i * 5, 0]
            ms.measure(test_object, None, [pos], add_noise=False)
            ms.add_mask([pos])
            coverages.append(ms.cum_mask.sum().item())

        # Coverage should generally increase (or stay same if overlapping)
        for i in range(1, len(coverages)):
            assert coverages[i] >= coverages[i - 1], f"Coverage should not decrease: {coverages}"
