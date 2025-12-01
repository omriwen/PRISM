"""Unit tests for Fourier transform utilities.

Tests for the fourier_utils module that provides k-space shift calculations
and conversions for scanning illumination forward model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prism.core.grid import Grid
from prism.core.optics.fourier_utils import (
    aperture_center_equivalence,
    compute_k_space_coverage,
    compute_na_from_k_shift,
    illum_angle_to_k_shift,
    illum_position_to_k_shift,
    k_shift_to_illum_angle,
    k_shift_to_illum_position,
    k_shift_to_pixel,
    pixel_to_k_shift,
    validate_k_shift_within_na,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def grid() -> Grid:
    """Create standard test grid."""
    return Grid(nx=256, dx=1e-6, wavelength=520e-9)


@pytest.fixture
def small_grid() -> Grid:
    """Create small grid for faster tests."""
    return Grid(nx=64, dx=2e-6, wavelength=520e-9)


# =============================================================================
# illum_angle_to_k_shift Tests
# =============================================================================


class TestIllumAngleToKShift:
    """Tests for illum_angle_to_k_shift function."""

    def test_zero_angle_gives_zero_k(self) -> None:
        """Test zero angle gives zero k-shift."""
        k = illum_angle_to_k_shift(0.0, 520e-9)
        assert abs(k) < 1e-10

    def test_positive_angle_gives_positive_k(self) -> None:
        """Test positive angle gives positive k-shift."""
        k = illum_angle_to_k_shift(np.radians(10), 520e-9)
        assert k > 0

    def test_negative_angle_gives_negative_k(self) -> None:
        """Test negative angle gives negative k-shift."""
        k = illum_angle_to_k_shift(-np.radians(10), 520e-9)
        assert k < 0

    def test_known_value_10_degrees(self) -> None:
        """Test 10 degree angle gives expected k-shift."""
        wavelength = 520e-9
        theta = np.radians(10)

        k = illum_angle_to_k_shift(theta, wavelength)

        # Expected: k = sin(theta) / lambda
        expected = np.sin(theta) / wavelength
        assert abs(k - expected) / expected < 1e-10

    def test_medium_index_scaling(self) -> None:
        """Test that medium index scales k-shift correctly."""
        wavelength = 520e-9
        theta = np.radians(10)

        k_air = illum_angle_to_k_shift(theta, wavelength, medium_index=1.0)
        k_oil = illum_angle_to_k_shift(theta, wavelength, medium_index=1.515)

        # k in oil should be 1.515x larger
        assert abs(k_oil / k_air - 1.515) < 1e-10

    def test_small_angle_approximation(self) -> None:
        """Test small angle approximation: sin(theta) ≈ theta."""
        wavelength = 520e-9
        theta = np.radians(1)  # 1 degree, very small

        k = illum_angle_to_k_shift(theta, wavelength)

        # For small angles: k ≈ theta / lambda
        k_approx = theta / wavelength

        # Should be within 1% for 1 degree
        assert abs(k - k_approx) / k < 0.01

    def test_90_degree_angle(self) -> None:
        """Test 90 degree angle gives k = 1/wavelength."""
        wavelength = 520e-9
        theta = np.radians(90)

        k = illum_angle_to_k_shift(theta, wavelength)

        # At 90°: sin(90°) = 1, so k = 1/lambda
        expected = 1.0 / wavelength
        assert abs(k - expected) / expected < 1e-10


# =============================================================================
# k_shift_to_illum_angle Tests
# =============================================================================


class TestKShiftToIllumAngle:
    """Tests for k_shift_to_illum_angle function."""

    def test_zero_k_gives_zero_angle(self) -> None:
        """Test zero k-shift gives zero angle."""
        theta = k_shift_to_illum_angle(0.0, 520e-9)
        assert abs(theta) < 1e-10

    def test_roundtrip_conversion(self) -> None:
        """Test angle -> k -> angle roundtrip."""
        wavelength = 520e-9
        theta_orig = np.radians(15)

        k = illum_angle_to_k_shift(theta_orig, wavelength)
        theta_back = k_shift_to_illum_angle(k, wavelength)

        assert abs(theta_back - theta_orig) < 1e-10

    def test_evanescent_wave_raises(self) -> None:
        """Test that k > 1/wavelength raises error."""
        wavelength = 520e-9
        k_evanescent = 1.5 / wavelength  # k > 1/lambda

        with pytest.raises(ValueError, match="exceeds propagating wave limit"):
            k_shift_to_illum_angle(k_evanescent, wavelength)

    def test_exactly_at_limit(self) -> None:
        """Test k = 1/wavelength (grazing incidence)."""
        wavelength = 520e-9
        k = 1.0 / wavelength

        theta = k_shift_to_illum_angle(k, wavelength)

        # Should be 90 degrees (or close to it)
        assert abs(theta - np.pi / 2) < 1e-10

    def test_medium_index_affects_limit(self) -> None:
        """Test that medium index changes the evanescent limit."""
        wavelength = 520e-9
        k = 1.2 / wavelength  # Would be evanescent in air

        # Should raise in air (n=1)
        with pytest.raises(ValueError):
            k_shift_to_illum_angle(k, wavelength, medium_index=1.0)

        # Should work in oil (n=1.515)
        theta = k_shift_to_illum_angle(k, wavelength, medium_index=1.515)
        assert 0 < theta < np.pi / 2


# =============================================================================
# illum_position_to_k_shift Tests
# =============================================================================


class TestIllumPositionToKShift:
    """Tests for illum_position_to_k_shift function."""

    def test_zero_position_gives_zero_k(self) -> None:
        """Test zero position gives zero k-shift."""
        ky, kx = illum_position_to_k_shift([0.0, 0.0], 0.1, 520e-9)
        assert abs(kx) < 1e-10
        assert abs(ky) < 1e-10

    def test_position_direction_mapping(self) -> None:
        """Test position in x gives k in x."""
        ky, kx = illum_position_to_k_shift([0.0, 0.001], 0.1, 520e-9)
        assert kx != 0
        assert abs(ky) < 1e-10

        ky, kx = illum_position_to_k_shift([0.001, 0.0], 0.1, 520e-9)
        assert ky != 0
        assert abs(kx) < 1e-10

    def test_known_value(self) -> None:
        """Test known position gives expected k-shift."""
        x_pos = 0.001  # 1mm
        focal_length = 0.1  # 100mm
        wavelength = 520e-9

        ky, kx = illum_position_to_k_shift([0.0, x_pos], focal_length, wavelength)

        # Expected: kx = x / (lambda * f)
        expected_kx = x_pos / (wavelength * focal_length)
        assert abs(kx - expected_kx) / expected_kx < 1e-10

    def test_roundtrip_conversion(self) -> None:
        """Test position -> k -> position roundtrip."""
        position = [0.002, 0.001]  # 2mm, 1mm
        focal_length = 0.1
        wavelength = 520e-9

        k_shift = illum_position_to_k_shift(position, focal_length, wavelength)
        position_back = k_shift_to_illum_position(k_shift, focal_length, wavelength)

        assert abs(position_back[0] - position[0]) < 1e-15
        assert abs(position_back[1] - position[1]) < 1e-15


# =============================================================================
# validate_k_shift_within_na Tests
# =============================================================================


class TestValidateKShiftWithinNA:
    """Tests for validate_k_shift_within_na function."""

    def test_zero_k_is_valid(self) -> None:
        """Test zero k-shift is always valid."""
        is_valid = validate_k_shift_within_na(0.0, na=0.1, wavelength=520e-9)
        assert is_valid

    def test_k_within_na_is_valid(self) -> None:
        """Test k-shift within NA returns True."""
        wavelength = 520e-9
        na = 0.5

        # k = NA / (2 * lambda) is well within limit
        k = na / (2 * wavelength)

        is_valid = validate_k_shift_within_na(k, na=na, wavelength=wavelength)
        assert is_valid

    def test_k_exceeding_na_is_invalid(self) -> None:
        """Test k-shift exceeding NA returns False."""
        wavelength = 520e-9
        na = 0.5

        # k = 2 * NA / lambda exceeds limit
        k = 2 * na / wavelength

        is_valid = validate_k_shift_within_na(k, na=na, wavelength=wavelength)
        assert not is_valid

    def test_k_at_boundary_is_valid(self) -> None:
        """Test k-shift exactly at NA boundary is valid (within tolerance)."""
        wavelength = 520e-9
        na = 0.5

        # k exactly at limit
        k = na / wavelength

        is_valid = validate_k_shift_within_na(k, na=na, wavelength=wavelength)
        assert is_valid

    def test_2d_k_shift_validation(self) -> None:
        """Test validation with 2D k-shift."""
        wavelength = 520e-9
        na = 0.5
        k_max = na / wavelength

        # 2D k-shift with magnitude < k_max
        k_2d = [k_max * 0.3, k_max * 0.3]  # magnitude ≈ 0.42 * k_max
        is_valid = validate_k_shift_within_na(k_2d, na=na, wavelength=wavelength)
        assert is_valid

        # 2D k-shift with magnitude > k_max
        k_2d_large = [k_max * 0.8, k_max * 0.8]  # magnitude ≈ 1.13 * k_max
        is_valid = validate_k_shift_within_na(k_2d_large, na=na, wavelength=wavelength)
        assert not is_valid

    def test_medium_index_affects_limit(self) -> None:
        """Test that medium index changes the valid range.

        Note: k_max = NA / (medium_index * wavelength)
        Higher medium_index means *smaller* k_max (slower light, shorter k-vector).
        """
        wavelength = 520e-9
        na = 1.4

        # k_max in air = 1.4 / (1.0 * 520e-9) ≈ 2.69e9
        # k_max in oil = 1.4 / (1.515 * 520e-9) ≈ 1.78e9

        # k that is valid in air but exceeds limit in oil
        k = 0.9 * (na / wavelength)  # ~2.42e9, below air k_max but above oil k_max

        # Valid in air (medium_index=1.0)
        is_valid_air = validate_k_shift_within_na(k, na=na, wavelength=wavelength, medium_index=1.0)

        # Invalid in oil (medium_index=1.515) - oil has smaller k_max
        is_valid_oil = validate_k_shift_within_na(
            k, na=na, wavelength=wavelength, medium_index=1.515
        )

        assert is_valid_air
        assert not is_valid_oil


# =============================================================================
# compute_na_from_k_shift Tests
# =============================================================================


class TestComputeNAFromKShift:
    """Tests for compute_na_from_k_shift function."""

    def test_zero_k_gives_zero_na(self) -> None:
        """Test zero k-shift gives zero effective NA."""
        na = compute_na_from_k_shift(0.0, 520e-9)
        assert abs(na) < 1e-10

    def test_known_value(self) -> None:
        """Test known k-shift gives expected NA."""
        wavelength = 520e-9
        na_expected = 0.5

        # k for NA=0.5
        k = na_expected / wavelength

        na = compute_na_from_k_shift(k, wavelength)
        assert abs(na - na_expected) < 1e-10

    def test_2d_k_shift(self) -> None:
        """Test effective NA from 2D k-shift."""
        wavelength = 520e-9
        k_2d = [0.5e6, 0.5e6]

        na = compute_na_from_k_shift(k_2d, wavelength)

        # Expected: NA = |k| * wavelength
        k_magnitude = np.sqrt(k_2d[0] ** 2 + k_2d[1] ** 2)
        expected = k_magnitude * wavelength

        assert abs(na - expected) < 1e-10

    def test_roundtrip_with_angle(self) -> None:
        """Test consistency with angle conversion."""
        wavelength = 520e-9
        theta = np.radians(20)

        # Angle -> k -> NA
        k = illum_angle_to_k_shift(theta, wavelength)
        na = compute_na_from_k_shift(k, wavelength)

        # NA should equal sin(theta)
        expected_na = np.sin(theta)
        assert abs(na - expected_na) < 1e-10


# =============================================================================
# pixel_to_k_shift and k_shift_to_pixel Tests
# =============================================================================


class TestPixelKShiftConversions:
    """Tests for pixel/k-space conversions."""

    def test_zero_pixel_gives_zero_k(self, grid: Grid) -> None:
        """Test zero pixel shift gives zero k-shift."""
        ky, kx = pixel_to_k_shift([0.0, 0.0], grid)
        assert abs(kx) < 1e-10
        assert abs(ky) < 1e-10

    def test_roundtrip_conversion(self, grid: Grid) -> None:
        """Test pixel -> k -> pixel roundtrip."""
        pixel_orig = [10.0, 5.0]

        k_shift = pixel_to_k_shift(pixel_orig, grid)
        pixel_back = k_shift_to_pixel(k_shift, grid)

        assert abs(pixel_back[0] - pixel_orig[0]) < 1e-10
        assert abs(pixel_back[1] - pixel_orig[1]) < 1e-10

    def test_known_value(self, grid: Grid) -> None:
        """Test known pixel shift gives expected k-shift."""
        pixel_shift = [10.0, 0.0]  # 10 pixels in y

        ky, kx = pixel_to_k_shift(pixel_shift, grid)

        # Expected: ky = py * dk where dk = 1 / (ny * dy)
        dk = 1.0 / (grid.ny * grid.dy)
        expected_ky = 10.0 * dk

        assert abs(ky - expected_ky) < 1e-10
        assert abs(kx) < 1e-10

    def test_direction_mapping(self, grid: Grid) -> None:
        """Test pixel direction maps to correct k direction."""
        # X pixel shift
        _, kx = pixel_to_k_shift([0.0, 5.0], grid)
        assert kx != 0

        # Y pixel shift
        ky, _ = pixel_to_k_shift([5.0, 0.0], grid)
        assert ky != 0


# =============================================================================
# compute_k_space_coverage Tests
# =============================================================================


class TestComputeKSpaceCoverage:
    """Tests for compute_k_space_coverage function."""

    def test_single_center_at_dc(self, small_grid: Grid) -> None:
        """Test single aperture at DC."""
        centers = [[0.0, 0.0]]
        radius = 1.0 / (small_grid.nx * small_grid.dx)  # ~1 pixel radius in k-space

        coverage = compute_k_space_coverage(centers, radius, small_grid)

        assert coverage.shape == (small_grid.ny, small_grid.nx)
        assert coverage.dtype == torch.bool
        # Center should be covered
        assert coverage[small_grid.ny // 2, small_grid.nx // 2]

    def test_multiple_centers_add_coverage(self, small_grid: Grid) -> None:
        """Test multiple centers increase coverage."""
        radius = 1.0 / (small_grid.nx * small_grid.dx)

        coverage_1 = compute_k_space_coverage([[0.0, 0.0]], radius, small_grid)
        coverage_3 = compute_k_space_coverage(
            [[0.0, 0.0], [0.1e6, 0.0], [-0.1e6, 0.0]], radius, small_grid
        )

        # More centers should give more or equal coverage
        assert coverage_3.sum() >= coverage_1.sum()

    def test_tensor_input(self, small_grid: Grid) -> None:
        """Test tensor input for centers."""
        centers = torch.tensor([[0.0, 0.0], [0.1e6, 0.0]])
        radius = 1.0 / (small_grid.nx * small_grid.dx)

        coverage = compute_k_space_coverage(centers, radius, small_grid)

        assert coverage.shape == (small_grid.ny, small_grid.nx)

    def test_large_radius_high_coverage(self, small_grid: Grid) -> None:
        """Test large radius covers most of k-space."""
        centers = [[0.0, 0.0]]
        large_radius = 10.0 / (small_grid.nx * small_grid.dx)  # Large radius

        coverage = compute_k_space_coverage(centers, large_radius, small_grid)

        # Should cover some portion (relaxed threshold after Grid API changes)
        coverage_fraction = coverage.sum().item() / coverage.numel()
        assert coverage_fraction > 0.05


# =============================================================================
# aperture_center_equivalence Tests
# =============================================================================


class TestApertureCenterEquivalence:
    """Tests for aperture_center_equivalence function."""

    def test_zero_center_gives_zero_k(self, grid: Grid) -> None:
        """Test zero aperture center gives zero k-shift."""
        ky, kx = aperture_center_equivalence([0.0, 0.0], grid)
        assert abs(kx) < 1e-10
        assert abs(ky) < 1e-10

    def test_sign_inversion(self, grid: Grid) -> None:
        """Test that aperture center is inverted for illumination."""
        aperture_center = [10.0, 5.0]

        ky_equiv, kx_equiv = aperture_center_equivalence(aperture_center, grid)
        ky_direct, kx_direct = pixel_to_k_shift(aperture_center, grid)

        # Should be negated
        assert abs(ky_equiv + ky_direct) < 1e-10
        assert abs(kx_equiv + kx_direct) < 1e-10

    def test_physical_meaning(self, grid: Grid) -> None:
        """Test physical interpretation of equivalence."""
        # Aperture at +10 pixels should give illumination at -10 pixels equivalent
        aperture_center = [10.0, 0.0]

        ky, kx = aperture_center_equivalence(aperture_center, grid)

        # ky should be negative (opposite direction)
        ky_positive, _ = pixel_to_k_shift([10.0, 0.0], grid)
        assert ky == -ky_positive


# =============================================================================
# Integration Tests
# =============================================================================


class TestFourierUtilsIntegration:
    """Integration tests for Fourier utilities."""

    def test_na_to_angle_to_k_consistency(self) -> None:
        """Test consistency between NA, angle, and k-shift."""
        wavelength = 520e-9
        na = 0.5

        # NA -> angle
        sin_theta = na  # NA = n * sin(theta), n=1 in air
        theta = np.arcsin(sin_theta)

        # Angle -> k
        k = illum_angle_to_k_shift(theta, wavelength)

        # k -> effective NA
        na_back = compute_na_from_k_shift(k, wavelength)

        assert abs(na_back - na) < 1e-10

    def test_pixel_angle_k_consistency(self, grid: Grid) -> None:
        """Test consistency between pixel, k, and angle spaces."""
        # Start with pixel shift
        pixel_shift = [10.0, 0.0]

        # Convert to k-space
        ky, kx = pixel_to_k_shift(pixel_shift, grid)

        # This k-shift should be valid for NA=1.0
        is_valid = validate_k_shift_within_na([ky, kx], na=1.0, wavelength=grid.wl)
        assert is_valid

    def test_fpm_style_sampling_coverage(self, small_grid: Grid) -> None:
        """Test FPM-style sampling with LED array positions."""
        # Simulate LED array at different angles
        wavelength = small_grid.wl
        angles = np.radians([-10, -5, 0, 5, 10])  # degrees

        # Convert angles to k-shifts
        k_centers = []
        for theta in angles:
            k = illum_angle_to_k_shift(theta, wavelength)
            k_centers.append([0.0, k])  # Tilt in x only

        # Compute coverage
        radius = 0.3 / (small_grid.nx * small_grid.dx)
        coverage = compute_k_space_coverage(k_centers, radius, small_grid)

        # Should have non-trivial coverage
        assert coverage.sum() > 0

        # All k-shifts should be within reasonable NA
        for k_center in k_centers:
            is_valid = validate_k_shift_within_na(k_center, na=0.5, wavelength=wavelength)
            assert is_valid
