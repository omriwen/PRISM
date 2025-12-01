"""Smoke tests for Wave 4: Cumulative mask in illumination mode.

These tests verify that add_mask() works correctly for both APERTURE and
ILLUMINATION scanning modes, including proper handling of finite-size sources.
"""

import pytest
import torch

from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.measurement_system import (
    MeasurementSystem,
    MeasurementSystemConfig,
    ScanningMode,
)


@pytest.fixture
def microscope():
    """Create a test microscope.

    Uses parameters that satisfy Nyquist sampling requirements:
    - object_pixel_size = pixel_size / magnification = 6.5e-6 / 100 = 6.5e-8 m
    - min_sampling = wavelength / (4 * NA) = 520e-9 / (4 * 0.5) = 2.6e-7 m
    - 6.5e-8 < 2.6e-7, so Nyquist is satisfied
    """
    config = MicroscopeConfig(
        n_pixels=64,
        pixel_size=6.5e-6,
        wavelength=520e-9,
        numerical_aperture=0.5,
        magnification=100.0,  # Higher magnification to satisfy Nyquist
    )
    return Microscope(config)


class TestCumulativeMaskApertureMode:
    """Tests for cumulative mask in APERTURE mode (default)."""

    def test_add_mask_aperture_mode_basic(self, microscope):
        """Test basic add_mask in default aperture mode."""
        ms = MeasurementSystem(microscope)

        # Initially mask should be all zeros
        assert ms.cum_mask.sum() == 0
        assert ms.sample_count == 0

        # Add first mask at DC
        ms.add_mask([[0, 0]])

        # Mask should have some coverage
        assert ms.cum_mask.sum() > 0
        assert ms.sample_count == 1

    def test_add_mask_aperture_mode_multiple(self, microscope):
        """Test adding multiple masks accumulates coverage."""
        ms = MeasurementSystem(microscope)

        # Add first mask
        ms.add_mask([[0, 0]])
        coverage_1 = ms.cum_mask.sum().item()

        # Add second mask at different position
        ms.add_mask([[10, 10]])
        coverage_2 = ms.cum_mask.sum().item()

        # Coverage should increase
        assert coverage_2 > coverage_1
        assert ms.sample_count == 2


class TestCumulativeMaskIlluminationMode:
    """Tests for cumulative mask in ILLUMINATION mode."""

    def test_add_mask_illumination_mode_point(self, microscope):
        """Test add_mask in illumination mode with point source."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Initially mask should be all zeros
        assert ms.cum_mask.sum() == 0
        assert ms.sample_count == 0

        # Add mask at DC
        ms.add_mask([[0, 0]])

        # Mask should have some coverage
        assert ms.cum_mask.sum() > 0
        assert ms.sample_count == 1

    def test_add_mask_illumination_mode_gaussian(self, microscope):
        """Test add_mask in illumination mode with Gaussian source."""
        # Get k-space resolution for setting illumination radius
        grid = microscope.grid
        dk = 1.0 / (grid.nx * grid.dx)
        illum_radius_k = 5 * dk  # 5 pixels worth in k-space

        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=illum_radius_k,
        )
        ms = MeasurementSystem(microscope, config=config)

        # Add mask at DC
        ms.add_mask([[0, 0]])

        # Mask should have some coverage
        assert ms.cum_mask.sum() > 0
        assert ms.sample_count == 1

    def test_add_mask_illumination_mode_circular(self, microscope):
        """Test add_mask in illumination mode with circular source."""
        grid = microscope.grid
        dk = 1.0 / (grid.nx * grid.dx)
        illum_radius_k = 8 * dk  # 8 pixels worth in k-space

        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=illum_radius_k,
        )
        ms = MeasurementSystem(microscope, config=config)

        # Add mask at DC
        ms.add_mask([[0, 0]])

        # Mask should have some coverage
        assert ms.cum_mask.sum() > 0
        assert ms.sample_count == 1

    def test_illumination_radius_affects_mask_size(self, microscope):
        """Test that illumination_radius affects cumulative mask size."""
        grid = microscope.grid
        dk = 1.0 / (grid.nx * grid.dx)

        # Small radius
        config_small = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=3 * dk,
        )
        ms_small = MeasurementSystem(microscope, config=config_small)
        ms_small.add_mask([[0, 0]])
        coverage_small = ms_small.cum_mask.sum().item()

        # Large radius
        config_large = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=10 * dk,
        )
        ms_large = MeasurementSystem(microscope, config=config_large)
        ms_large.add_mask([[0, 0]])
        coverage_large = ms_large.cum_mask.sum().item()

        # Larger radius should have more coverage
        assert coverage_large > coverage_small


class TestMaskEquivalence:
    """Tests verifying aperture and illumination mode produce equivalent masks."""

    def test_same_center_same_mask_position(self, microscope):
        """Verify that same center produces mask at same position in both modes."""
        # Aperture mode
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)

        # Illumination mode with point source (should be equivalent)
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms_illum = MeasurementSystem(microscope, config=config_illum)

        # Add mask at same center
        center = [5, -3]
        ms_aperture.add_mask([center])
        ms_illum.add_mask([center])

        # The masks should cover the same region (same non-zero elements)
        # Note: exact values may differ slightly, but center should be same
        # Check that they have the same number of covered pixels
        assert ms_aperture.cum_mask.sum() == ms_illum.cum_mask.sum()

        # Check that they cover the same pixels
        assert torch.all(ms_aperture.cum_mask == ms_illum.cum_mask)


class TestKRadiusConversion:
    """Tests for k-radius to pixel conversion."""

    def test_k_radius_conversion(self, microscope):
        """Test that _k_radius_to_pixels works correctly."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=1.0,  # Will be overridden in test
        )
        ms = MeasurementSystem(microscope, config=config)

        grid = microscope.grid
        dk = 1.0 / (grid.nx * grid.dx)

        # Convert 10 pixels worth of k-space
        k_radius = 10 * dk
        pixel_radius = ms._k_radius_to_pixels(k_radius)

        # Should convert back to approximately 10 pixels
        assert abs(pixel_radius - 10.0) < 0.01


class TestEdgeCases:
    """Test edge cases for cumulative mask."""

    def test_reset_clears_mask(self, microscope):
        """Test that reset() clears cumulative mask."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms = MeasurementSystem(microscope, config=config)

        # Add some masks
        ms.add_mask([[0, 0]])
        ms.add_mask([[5, 5]])
        assert ms.cum_mask.sum() > 0
        assert ms.sample_count == 2

        # Reset
        ms.reset()

        # Mask should be cleared
        assert ms.cum_mask.sum() == 0
        assert ms.sample_count == 0

    def test_radius_override_takes_priority(self, microscope):
        """Test that explicit radius override takes priority over config."""
        grid = microscope.grid
        dk = 1.0 / (grid.nx * grid.dx)

        # Configure with small illumination radius
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=3 * dk,
        )
        ms = MeasurementSystem(microscope, config=config)

        # Add mask with explicit larger radius override
        ms.add_mask([[0, 0]], radius=15)
        coverage_override = ms.cum_mask.sum().item()

        # Create another MS and use default (from config)
        ms2 = MeasurementSystem(microscope, config=config)
        ms2.add_mask([[0, 0]])  # No override, uses illumination_radius
        coverage_default = ms2.cum_mask.sum().item()

        # Override should produce larger mask
        assert coverage_override > coverage_default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
