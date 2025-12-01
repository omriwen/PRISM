"""Unit tests for spatial illumination scanning mode."""

import pytest
import torch

from prism.core.grid import Grid
from prism.core.optics.illumination import (
    IlluminationSource,
    IlluminationSourceType,
    create_spatial_illumination_field,
)


class TestSpatialIlluminationField:
    """Tests for create_spatial_illumination_field()."""

    @pytest.fixture
    def grid(self) -> Grid:
        return Grid(nx=128, dx=1e-6, wavelength=520e-9)

    def test_centered_source_shape(self, grid: Grid) -> None:
        """Test output shape for centered source."""
        source = IlluminationSource(
            source_type=IlluminationSourceType.GAUSSIAN,
            k_center=[0, 0],
            sigma=5e-6,
        )
        field = create_spatial_illumination_field(
            grid, source, spatial_center=[0.0, 0.0], source_distance=10e-3
        )
        assert field.shape == (grid.ny, grid.nx)
        assert field.dtype == torch.complex64

    def test_gaussian_envelope_shifted(self, grid: Grid) -> None:
        """Test Gaussian envelope is centered at spatial_center."""
        source = IlluminationSource(
            source_type=IlluminationSourceType.GAUSSIAN,
            k_center=[0, 0],
            sigma=10e-6,
        )
        offset = 20e-6  # 20 micrometers
        field = create_spatial_illumination_field(
            grid, source, spatial_center=[offset, 0.0], source_distance=10e-3
        )

        # Peak should be shifted from center
        amplitude = field.abs()
        peak_idx = torch.argmax(amplitude)
        peak_y = peak_idx // grid.nx
        center_y = grid.ny // 2

        # Peak should be offset from center
        expected_offset_pixels = offset / grid.dy
        assert abs(peak_y - center_y - expected_offset_pixels) < 2

    def test_phase_varies_spatially(self, grid: Grid) -> None:
        """Test that phase varies across the field (spherical wavefront).

        Use CIRCULAR source with large radius to ensure we have amplitude
        at edge pixels where we measure phase.
        """
        source = IlluminationSource(
            source_type=IlluminationSourceType.CIRCULAR,
            k_center=[0, 0],
            radius=100e-6,  # Large radius to cover entire grid
        )
        field = create_spatial_illumination_field(
            grid, source, spatial_center=[0.0, 0.0], source_distance=10e-3
        )

        phase = torch.angle(field)
        # Phase should vary from center to edge (quadratic phase)
        center_phase = phase[grid.ny // 2, grid.nx // 2]
        # Check phase at a point offset from center (not corner, which may be outside radius)
        offset_phase = phase[grid.ny // 2 + 20, grid.nx // 2 + 20]

        # Phase difference should be non-zero due to spherical wavefront
        assert abs(offset_phase - center_phase) > 0.1

    def test_far_source_approaches_plane_wave(self, grid: Grid) -> None:
        """Test that very far source creates nearly uniform phase.

        Use CIRCULAR source with large radius to ensure we have amplitude
        across the grid where we measure phase variation.
        """
        source = IlluminationSource(
            source_type=IlluminationSourceType.CIRCULAR,
            k_center=[0, 0],
            radius=100e-6,  # Large radius to cover entire grid
        )
        # Very far source (1 meter)
        field = create_spatial_illumination_field(
            grid, source, spatial_center=[0.0, 0.0], source_distance=1.0
        )

        # Only measure phase where we have amplitude (inside the circular aperture)
        amplitude = field.abs()
        mask = amplitude > amplitude.max() * 0.5
        phase = torch.angle(field)
        phase_masked = phase[mask]

        # Phase variation should be very small for far source
        phase_std = phase_masked.std()
        assert phase_std < 0.02  # Less than 0.02 radians variation


class TestIlluminationScanMethodEnum:
    """Tests for IlluminationScanMethod enum."""

    def test_enum_values_exist(self) -> None:
        from prism.core.measurement_system import IlluminationScanMethod

        assert hasattr(IlluminationScanMethod, "ANGULAR")
        assert hasattr(IlluminationScanMethod, "SPATIAL")

    def test_config_accepts_method(self) -> None:
        from prism.core.measurement_system import (
            IlluminationScanMethod,
            MeasurementSystemConfig,
            ScanningMode,
        )

        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_scan_method=IlluminationScanMethod.SPATIAL,
            illumination_source_distance=10e-3,
        )
        assert config.illumination_scan_method == IlluminationScanMethod.SPATIAL

    def test_spatial_requires_distance(self) -> None:
        from prism.core.measurement_system import (
            IlluminationScanMethod,
            MeasurementSystemConfig,
            ScanningMode,
        )

        with pytest.raises(ValueError, match="illumination_source_distance"):
            config = MeasurementSystemConfig(
                scanning_mode=ScanningMode.ILLUMINATION,
                illumination_scan_method=IlluminationScanMethod.SPATIAL,
                # Missing illumination_source_distance
            )
            config.validate()
