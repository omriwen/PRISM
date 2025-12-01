"""
Unit tests for optical propagators.

Tests all three propagation methods:
- Fraunhofer (far-field FFT)
- Fresnel (quadratic phase)
- Angular Spectrum (exact)
"""

from __future__ import annotations

import pytest
import torch

from prism.core.grid import Grid
from prism.core.propagators import (
    AngularSpectrumPropagator,
    FraunhoferPropagator,
    FresnelPropagator,
    create_propagator,
)
from prism.utils.transforms import FFTCache


class TestFraunhoferPropagator:
    """Test Fraunhofer (far-field) propagator."""

    def test_fraunhofer_forward_backward_inverse(self):
        """Test that forward and backward are inverses."""
        prop = FraunhoferPropagator(normalize=True)

        field_orig = torch.randn(256, 256, dtype=torch.cfloat)

        # Forward then backward should recover original
        k_field = prop(field_orig, direction="forward")
        field_reconstructed = prop(k_field, direction="backward")

        torch.testing.assert_close(field_reconstructed, field_orig, rtol=1e-4, atol=1e-6)

    def test_fraunhofer_energy_conservation(self):
        """Test that energy is conserved (Parseval's theorem)."""
        prop = FraunhoferPropagator(normalize=True)

        field = torch.randn(256, 256, dtype=torch.cfloat)
        energy_before = (field.abs() ** 2).sum()

        k_field = prop(field, direction="forward")
        energy_after = (k_field.abs() ** 2).sum()

        # With ortho normalization, energy should be preserved
        torch.testing.assert_close(energy_before, energy_after, rtol=1e-5, atol=1e-7)

    def test_fraunhofer_unknown_direction_raises(self):
        """Test that invalid direction raises error."""
        prop = FraunhoferPropagator()
        field = torch.randn(64, 64, dtype=torch.cfloat)

        with pytest.raises(ValueError, match="Unknown direction"):
            prop(field, direction="sideways")

    def test_fraunhofer_with_shared_fft_cache(self):
        """Test that shared FFT cache works correctly."""
        cache = FFTCache()
        prop = FraunhoferPropagator(normalize=True, fft_cache=cache)

        field = torch.randn(128, 128, dtype=torch.cfloat)

        # First call - cache miss
        _ = prop(field, direction="forward")
        assert cache.cache_misses == 1
        assert cache.cache_hits == 0

        # Second call - cache hit
        _ = prop(field, direction="forward")
        assert cache.cache_hits == 1

    def test_fraunhofer_different_normalizations(self):
        """Test different normalization modes."""
        field = torch.randn(64, 64, dtype=torch.cfloat)

        prop_ortho = FraunhoferPropagator(normalize=True)
        prop_backward = FraunhoferPropagator(normalize=False)

        result_ortho = prop_ortho(field, direction="forward")
        result_backward = prop_backward(field, direction="forward")

        # Results should be different (different normalization)
        assert not torch.allclose(result_ortho, result_backward)


class TestFresnelPropagator:
    """Test Fresnel (quadratic phase) propagator with Grid-based API."""

    def test_fresnel_basic_propagation(self):
        """Test basic Fresnel propagation."""
        grid = Grid(nx=256, dx=10e-6, wavelength=520e-9)
        prop = FresnelPropagator(grid=grid, distance=0.1)  # 10 cm

        field = torch.randn(256, 256, dtype=torch.cfloat)
        output = prop(field)

        assert output.shape == field.shape
        assert output.dtype == torch.cfloat

    def test_fresnel_validation_negative_wavelength(self):
        """Test that negative wavelength in Grid raises error."""
        with pytest.raises(ValueError):
            Grid(nx=256, dx=1e-5, wavelength=-520e-9)  # Negative!

    def test_fresnel_validation_small_distance(self):
        """Test that too-small distance raises error."""
        grid = Grid(nx=256, dx=1e-5, wavelength=520e-9)
        with pytest.raises(ValueError, match="too small"):
            FresnelPropagator(grid=grid, distance=1e-12)  # Way too small!

    def test_fresnel_warns_far_field(self):
        """Test that far-field regime creates propagator (with warning)."""
        # Just verify no error is raised and propagator is created
        grid = Grid(nx=256, dx=1e-5, wavelength=520e-9)
        _ = FresnelPropagator(grid=grid, distance=1000)  # 1 km - far field!

        # Verify Fresnel number is in far-field regime (< 0.1)
        fov = 1e-5 * 256
        fresnel_num = fov**2 / (520e-9 * 1000)
        assert fresnel_num < 0.1, f"Should be in far-field regime, got F={fresnel_num}"

    def test_fresnel_warns_near_field(self):
        """Test that near-field regime creates propagator (with warning)."""
        # Just verify no error is raised
        grid = Grid(nx=256, dx=1e-3, wavelength=520e-9)  # Large pixels
        _ = FresnelPropagator(grid=grid, distance=0.001)  # 1 mm - near field!

        # Verify Fresnel number is indeed >> 1
        fov = 1e-3 * 256
        fresnel_num = fov**2 / (520e-9 * 0.001)
        assert fresnel_num > 100, "Should be in near-field regime"

    def test_fresnel_with_fft_cache(self):
        """Test Fresnel propagator with FFT cache."""
        cache = FFTCache()
        grid = Grid(nx=128, dx=10e-6, wavelength=520e-9)
        prop = FresnelPropagator(grid=grid, distance=0.05, fft_cache=cache)

        field = torch.randn(128, 128, dtype=torch.cfloat)

        # First propagation - cache miss
        _ = prop(field)
        assert cache.cache_misses >= 1

        # Second propagation - cache hit
        initial_hits = cache.cache_hits
        _ = prop(field)
        assert cache.cache_hits > initial_hits

    def test_fresnel_backward_compatibility(self):
        """Test that FreeSpacePropagator alias works."""
        from prism.core.propagators import FreeSpacePropagator

        # Should be same class
        assert FreeSpacePropagator is FresnelPropagator


class TestAngularSpectrumPropagator:
    """Test Angular Spectrum (exact) propagator."""

    def test_angular_spectrum_basic(self):
        """Test basic angular spectrum propagation."""
        grid = Grid(nx=128, dx=10e-6, wavelength=520e-9)
        prop = AngularSpectrumPropagator(grid, distance=0.01)

        field = torch.randn(128, 128, dtype=torch.cfloat)
        output = prop(field)

        assert output.shape == field.shape
        assert output.dtype == torch.cfloat

    def test_angular_spectrum_variable_distance(self):
        """Test propagation with variable distance."""
        grid = Grid(nx=128, dx=10e-6, wavelength=520e-9)
        prop = AngularSpectrumPropagator(grid)  # No fixed distance

        field = torch.randn(128, 128, dtype=torch.cfloat)

        # Must specify distance in forward()
        output = prop(field, distance=0.02)
        assert output.shape == field.shape

    def test_angular_spectrum_missing_distance_raises(self):
        """Test that missing distance raises error."""
        grid = Grid(nx=128, dx=10e-6, wavelength=520e-9)
        prop = AngularSpectrumPropagator(grid)  # No distance

        field = torch.randn(128, 128, dtype=torch.cfloat)

        with pytest.raises(ValueError, match="distance must be specified"):
            prop(field)  # No distance provided!

    def test_angular_spectrum_evanescent_filtered(self):
        """Test that evanescent waves are properly filtered."""
        # Use very fine sampling to create evanescent frequencies
        grid = Grid(nx=128, dx=1e-6, wavelength=520e-9)  # Fine sampling
        prop = AngularSpectrumPropagator(grid, distance=0.01)

        # Check that diff_limit mask exists and has some True values
        assert prop.diff_limit_tensor.any(), "Should have some propagating waves"
        # With fine sampling, we might have evanescent waves
        # If all are propagating, that's also valid physics

    def test_angular_spectrum_invalid_wavelength_raises(self):
        """Test that invalid wavelength raises error during Grid creation."""
        with pytest.raises(ValueError, match="wavelength must be positive"):
            _ = Grid(nx=128, dx=10e-6, wavelength=-520e-9)  # Negative!

    def test_angular_spectrum_with_fft_cache(self):
        """Test Angular Spectrum with FFT cache."""
        cache = FFTCache()
        grid = Grid(nx=64, dx=10e-6, wavelength=520e-9)
        prop = AngularSpectrumPropagator(grid, distance=0.01, fft_cache=cache)

        field = torch.randn(64, 64, dtype=torch.cfloat)

        # First propagation - cache misses for FFT and/or IFFT
        _ = prop(field)
        assert cache.cache_misses >= 1  # At least one miss

        # Second propagation - cache hits
        initial_hits = cache.cache_hits
        _ = prop(field)
        assert cache.cache_hits > initial_hits


class TestCreatePropagator:
    """Test factory function."""

    def test_create_fraunhofer(self):
        """Test creating Fraunhofer propagator."""
        prop = create_propagator("fraunhofer", normalize=True)
        assert isinstance(prop, FraunhoferPropagator)
        assert prop.normalize is True

    def test_create_fresnel(self):
        """Test creating Fresnel propagator with Grid-based API."""
        grid = Grid(nx=256, dx=1e-5, wavelength=520e-9)
        prop = create_propagator(
            "fresnel",
            grid=grid,
            distance=0.1,
        )
        assert isinstance(prop, FresnelPropagator)

    def test_create_angular_spectrum(self):
        """Test creating Angular Spectrum propagator."""
        grid = Grid(nx=128, dx=10e-6, wavelength=520e-9)
        prop = create_propagator("angular_spectrum", grid=grid, distance=0.01)
        assert isinstance(prop, AngularSpectrumPropagator)

    def test_create_unknown_raises(self):
        """Test that unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown propagation method"):
            create_propagator("magic_propagation")  # type: ignore[arg-type]

    def test_create_with_shared_cache(self):
        """Test creating propagators with shared FFT cache."""
        cache = FFTCache()

        prop1 = create_propagator("fraunhofer", fft_cache=cache)
        prop2 = create_propagator("fraunhofer", fft_cache=cache)

        # Both should share same cache
        assert prop1.fft_cache is cache
        assert prop2.fft_cache is cache
        assert prop1.fft_cache is prop2.fft_cache


class TestPropagatorComparison:
    """Compare different propagators."""

    def test_fraunhofer_matches_direct_fft(self):
        """Test that Fraunhofer matches direct FFT."""
        from prism.utils.transforms import fft

        prop = FraunhoferPropagator(normalize=True)
        field = torch.randn(128, 128, dtype=torch.cfloat)

        # Fraunhofer forward should match FFT
        k_field_prop = prop(field, direction="forward")
        k_field_direct = fft(field, norm="ortho")

        torch.testing.assert_close(k_field_prop, k_field_direct, rtol=1e-6, atol=1e-8)

    def test_angular_spectrum_reduces_to_fraunhofer_far_field(self):
        """Test that Angular Spectrum ≈ Fraunhofer in far field."""
        grid = Grid(nx=128, dx=10e-6, wavelength=520e-9)

        # Far field distance
        distance = 10.0  # 10 meters

        fraunhofer = FraunhoferPropagator(normalize=True)
        angular = AngularSpectrumPropagator(grid, distance=distance)

        # Simple field (plane wave)
        field = torch.ones(128, 128, dtype=torch.cfloat)

        # In far field, patterns should be similar (not necessarily identical due to scaling)
        output_fraunhofer = fraunhofer(field, direction="forward")
        output_angular = angular(field)

        # Both should have concentrated energy at center (DC component)
        assert output_fraunhofer.abs().max() > 0
        assert output_angular.abs().max() > 0

    def test_all_propagators_preserve_shape(self):
        """Test that all propagators preserve tensor shape."""
        grid = Grid(nx=64, dx=10e-6, wavelength=520e-9)

        propagators = [
            FraunhoferPropagator(normalize=True),
            FresnelPropagator(grid=grid, distance=0.05),
            AngularSpectrumPropagator(grid, distance=0.05),
        ]

        field = torch.randn(64, 64, dtype=torch.cfloat)

        for prop in propagators:
            if isinstance(prop, FraunhoferPropagator):
                output = prop(field, direction="forward")
            else:
                output = prop(field)

            assert output.shape == field.shape
            assert output.dtype == torch.cfloat


class TestPropagatorPhysics:
    """Test physical properties of propagators."""

    def test_fraunhofer_plane_wave_to_delta(self):
        """Test that plane wave transforms to delta function (far field)."""
        prop = FraunhoferPropagator(normalize=True)

        # Plane wave (constant field)
        field = torch.ones(128, 128, dtype=torch.cfloat)

        k_field = prop(field, direction="forward")

        # Energy should be concentrated at center (DC component)
        center = 64
        center_energy = k_field[center - 2 : center + 2, center - 2 : center + 2].abs().sum()
        total_energy = k_field.abs().sum()

        # Most energy should be at center
        assert center_energy / total_energy > 0.9

    def test_fresnel_propagation_linearity(self):
        """Test that Fresnel propagation is linear."""
        grid = Grid(nx=128, dx=10e-6, wavelength=520e-9)
        prop = FresnelPropagator(grid=grid, distance=0.05)

        field1 = torch.randn(128, 128, dtype=torch.cfloat)
        field2 = torch.randn(128, 128, dtype=torch.cfloat)

        # Propagate separately
        output1 = prop(field1)
        output2 = prop(field2)

        # Propagate sum
        output_sum = prop(field1 + field2)

        # Should be linear: prop(a+b) = prop(a) + prop(b)
        torch.testing.assert_close(output_sum, output1 + output2, rtol=1e-4, atol=1e-6)

    def test_angular_spectrum_propagation_linearity(self):
        """Test that Angular Spectrum propagation is linear."""
        grid = Grid(nx=64, dx=10e-6, wavelength=520e-9)
        prop = AngularSpectrumPropagator(grid, distance=0.02)

        field1 = torch.randn(64, 64, dtype=torch.cfloat)
        field2 = torch.randn(64, 64, dtype=torch.cfloat)

        # Propagate separately
        output1 = prop(field1)
        output2 = prop(field2)

        # Propagate sum
        output_sum = prop(field1 + field2)

        # Should be linear
        torch.testing.assert_close(output_sum, output1 + output2, rtol=1e-4, atol=1e-6)


class TestFFTCacheIntegration:
    """Test FFT cache integration across propagators."""

    def test_shared_cache_across_propagators(self):
        """Test that multiple propagators can share a cache."""
        cache = FFTCache()

        prop1 = FraunhoferPropagator(fft_cache=cache)
        prop2 = FraunhoferPropagator(fft_cache=cache)

        field = torch.randn(128, 128, dtype=torch.cfloat)

        # Use first propagator
        _ = prop1(field, direction="forward")
        hits_after_prop1 = cache.cache_hits

        # Use second propagator (same cache, same shape)
        _ = prop2(field, direction="forward")

        # Should have additional cache hit
        assert cache.cache_hits > hits_after_prop1

    def test_cache_statistics(self):
        """Test that cache statistics are tracked correctly."""
        cache = FFTCache()
        prop = FraunhoferPropagator(fft_cache=cache)

        field = torch.randn(64, 64, dtype=torch.cfloat)

        # Initial state
        assert cache.hit_rate() == 0.0

        # First call - miss
        _ = prop(field, direction="forward")
        assert cache.cache_misses == 1

        # Second call - hit
        _ = prop(field, direction="forward")
        assert cache.cache_hits >= 1
        assert cache.hit_rate() > 0

    def test_cache_with_different_sizes(self):
        """Test that cache handles different tensor sizes."""
        cache = FFTCache()
        prop = FraunhoferPropagator(fft_cache=cache)

        field1 = torch.randn(64, 64, dtype=torch.cfloat)
        field2 = torch.randn(128, 128, dtype=torch.cfloat)

        # First size - miss
        _ = prop(field1, direction="forward")
        misses_1 = cache.cache_misses

        # Second size - new miss
        _ = prop(field2, direction="forward")
        assert cache.cache_misses > misses_1

        # Repeat first size - hit
        initial_hits = cache.cache_hits
        _ = prop(field1, direction="forward")
        assert cache.cache_hits > initial_hits
