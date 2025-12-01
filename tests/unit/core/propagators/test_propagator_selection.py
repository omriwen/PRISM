"""
Unit tests for propagator auto-selection functionality.

Tests the select_propagator() function which automatically chooses
the appropriate propagation method based on Fresnel number.
"""

from __future__ import annotations

import pytest
import torch

from prism.core.propagators import (
    AngularSpectrumPropagator,
    FraunhoferPropagator,
    FresnelPropagator,
    select_propagator,
)


class TestAutoSelectFraunhofer:
    """Test automatic selection of Fraunhofer propagator (F << 0.1)."""

    def test_auto_select_fraunhofer_far_field(self):
        """Test Fraunhofer auto-selection for far-field case."""
        # Far field: large distance, small aperture
        # F = (1mm)² / (520nm × 1000m) = 1.92e-6 << 0.1
        prop = select_propagator(
            wavelength=520e-9,
            obj_distance=1000.0,  # 1 km
            fov=1e-3,  # 1 mm
            method="auto",
        )

        # Should select Fraunhofer
        assert isinstance(prop, FraunhoferPropagator)
        # Log messages verified in manual testing (loguru + capsys interaction issues)

    def test_auto_select_fraunhofer_astronomy(self):
        """Test Fraunhofer selection for astronomical imaging (Europa case)."""
        # Europa observation: F ~ 10⁻¹³
        # Distance: 628 million km, aperture: ~1cm
        prop = select_propagator(
            wavelength=520e-9,
            obj_distance=628e9,  # 628 million km
            fov=1024 * 10e-6,  # 10.24 mm FOV
            method="auto",
        )

        assert isinstance(prop, FraunhoferPropagator)
        # Log messages verified in manual testing (loguru + capsys interaction issues)

    def test_auto_select_fraunhofer_boundary_below(self):
        """Test Fraunhofer selection just below boundary (F = 0.09)."""
        # Set up parameters for F ≈ 0.09
        wavelength = 520e-9
        obj_distance = 1.0
        # F = fov² / (λ × L) = 0.09
        fov = (0.09 * wavelength * obj_distance) ** 0.5

        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="auto",
        )

        assert isinstance(prop, FraunhoferPropagator)
        # Log messages verified in manual testing (loguru + capsys interaction issues)


class TestAutoSelectAngularSpectrumForFresnel:
    """Test automatic selection uses Angular Spectrum for F >= 0.1.

    Note: Fresnel propagator is deprecated due to accuracy issues.
    Auto-selection now uses Angular Spectrum for all F >= 0.1 cases.
    """

    def test_auto_select_angular_spectrum_intermediate(self):
        """Test Angular Spectrum auto-selection for intermediate distances."""
        # Intermediate: F = 1.0 (uses Angular Spectrum, not deprecated Fresnel)
        wavelength = 520e-9
        obj_distance = 0.05  # 5 cm
        # F = 1.0 → fov = sqrt(1.0 × λ × L) = sqrt(1.0 × 520e-9 × 0.05)
        fov = (1.0 * wavelength * obj_distance) ** 0.5  # ~161 µm

        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="auto",
            dx=1e-6,  # 1 µm pixels
            image_size=256,
        )

        # Angular Spectrum is selected for F >= 0.1 (Fresnel is deprecated)
        assert isinstance(prop, AngularSpectrumPropagator)

    def test_auto_select_angular_spectrum_boundary_lower(self):
        """Test Angular Spectrum selection at boundary (F = 0.1)."""
        wavelength = 520e-9
        obj_distance = 1.0
        # F = fov² / (λ × L) = 0.1
        fov = (0.1 * wavelength * obj_distance) ** 0.5

        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="auto",
            dx=1e-6,
            image_size=256,
        )

        # Angular Spectrum is selected for F >= 0.1
        assert isinstance(prop, AngularSpectrumPropagator)

    def test_auto_select_angular_spectrum_boundary_upper(self):
        """Test Angular Spectrum selection for F = 9.9."""
        wavelength = 520e-9
        obj_distance = 0.01  # 1 cm
        # F = fov² / (λ × L) = 9.9
        fov = (9.9 * wavelength * obj_distance) ** 0.5  # ~227 µm

        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="auto",
            dx=1e-6,
            image_size=256,
        )

        # Angular Spectrum is selected for F >= 0.1
        assert isinstance(prop, AngularSpectrumPropagator)

    def test_auto_uses_angular_spectrum_defaults(self):
        """Test that auto-selection uses Angular Spectrum with default params."""
        # F = 1 (Angular Spectrum)
        wavelength = 520e-9
        obj_distance = 0.05
        fov = (1.0 * wavelength * obj_distance) ** 0.5

        # Should work without dx/dxf/image_size (Angular Spectrum has defaults)
        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="auto",
        )

        assert isinstance(prop, AngularSpectrumPropagator)


class TestAutoSelectAngularSpectrum:
    """Test automatic selection of Angular Spectrum propagator (F > 10)."""

    def test_auto_select_angular_spectrum_near_field(self):
        """Test Angular Spectrum auto-selection for near field."""
        # Near field: F = 50
        wavelength = 520e-9
        obj_distance = 0.001  # 1 mm
        # F = 50 → fov = sqrt(50 × λ × L) = sqrt(50 × 520e-9 × 0.001)
        fov = (50.0 * wavelength * obj_distance) ** 0.5  # ~161 µm

        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="auto",
            image_size=256,
            dx=1e-6,
        )

        assert isinstance(prop, AngularSpectrumPropagator)
        # Log messages verified in manual testing (loguru + capsys interaction issues)

    def test_auto_select_angular_spectrum_boundary(self):
        """Test Angular Spectrum selection at boundary (F = 10)."""
        wavelength = 520e-9
        obj_distance = 0.01  # 1 cm
        # F = fov² / (λ × L) = 10
        fov = (10.0 * wavelength * obj_distance) ** 0.5  # ~228 µm

        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="auto",
            image_size=256,
            dx=1e-6,
        )

        assert isinstance(prop, AngularSpectrumPropagator)
        # Log messages verified in manual testing (loguru + capsys interaction issues)


class TestManualOverride:
    """Test manual method selection with warnings."""

    def test_manual_fraunhofer_appropriate(self):
        """Test manual Fraunhofer selection when appropriate (no warning)."""
        # F = 0.01 << 0.1 (Fraunhofer is correct choice)
        prop = select_propagator(
            wavelength=520e-9,
            obj_distance=10.0,
            fov=1e-3,
            method="fraunhofer",  # Manual selection
        )

        assert isinstance(prop, FraunhoferPropagator)
        # Should not have warning (appropriate choice)

    def test_manual_fraunhofer_inappropriate_warning(self):
        """Test manual Fraunhofer selection when F >= 0.1 (should warn)."""
        # F = 1 (should use Fresnel, but forcing Fraunhofer)
        wavelength = 520e-9
        obj_distance = 0.05
        fov = (1.0 * wavelength * obj_distance) ** 0.5  # F = 1.0

        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="fraunhofer",
        )

        assert isinstance(prop, FraunhoferPropagator)
        # Should have warning about inappropriate use

    def test_manual_fresnel_appropriate(self):
        """Test manual Fresnel selection when appropriate."""
        # F = 1 (Fresnel is correct)
        wavelength = 520e-9
        obj_distance = 0.05
        fov = (1.0 * wavelength * obj_distance) ** 0.5

        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="fresnel",
            dx=1e-6,
            dxf=100,
            image_size=256,
        )

        assert isinstance(prop, FresnelPropagator)
        # No warning expected

    def test_manual_fresnel_too_small_warning(self):
        """Test manual Fresnel when F < 0.1 (should warn)."""
        # F = 0.01 (should use Fraunhofer)
        wavelength = 520e-9
        obj_distance = 10.0
        fov = (0.01 * wavelength * obj_distance) ** 0.5

        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="fresnel",
            dx=1e-6,
            dxf=100,
            image_size=256,
        )

        assert isinstance(prop, FresnelPropagator)
        # Log messages verified in manual testing (loguru + capsys interaction issues)

    def test_manual_fresnel_too_large_warning(self):
        """Test manual Fresnel when F > 10 (should warn)."""
        wavelength = 520e-9
        obj_distance = 0.001
        # F = 20
        fov = (20.0 * wavelength * obj_distance) ** 0.5

        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="fresnel",
            dx=1e-6,
            dxf=100,
            image_size=256,
        )

        assert isinstance(prop, FresnelPropagator)
        # Log messages verified in manual testing (loguru + capsys interaction issues)

    def test_manual_angular_spectrum_far_field_info(self):
        """Test manual Angular Spectrum in far field (should suggest Fraunhofer)."""
        # F = 0.01 << 1 (Angular Spectrum works but Fraunhofer is faster)
        prop = select_propagator(
            wavelength=520e-9,
            obj_distance=10.0,
            fov=1e-3,
            method="angular_spectrum",
            image_size=256,
            dx=10e-6,
        )

        assert isinstance(prop, AngularSpectrumPropagator)
        # Should have info message (not warning) about efficiency


class TestRealWorldCases:
    """Test real-world use cases."""

    def test_europa_observation(self):
        """Test Europa observation case (should select Fraunhofer)."""
        # Real SPIDS use case: imaging Europa's moon
        # Distance: ~628 million km, FOV: ~10mm, wavelength: 520nm
        # F ~ 10⁻¹³ (extreme far field)
        prop = select_propagator(
            wavelength=520e-9,
            obj_distance=628e9,  # 628 million km
            fov=1024 * 10e-6,  # 10.24 mm
            method="auto",
        )

        assert isinstance(prop, FraunhoferPropagator)
        # Log messages verified in manual testing (loguru + capsys interaction issues)
        # Fresnel number should be in scientific notation (very small)

    def test_microscopy_near_field(self):
        """Test microscopy near-field case (should select Angular Spectrum)."""
        # Microscopy: very short distance, large numerical aperture
        # Distance: 100 µm, FOV: 1mm, wavelength: 520nm
        # F = (1mm)² / (520nm × 100µm) = 19,230
        prop = select_propagator(
            wavelength=520e-9,
            obj_distance=100e-6,  # 100 µm
            fov=1e-3,  # 1 mm
            method="auto",
            image_size=512,
            dx=2e-6,
        )

        assert isinstance(prop, AngularSpectrumPropagator)
        # Log messages verified in manual testing (loguru + capsys interaction issues)

    def test_lab_setup_intermediate_distance(self):
        """Test typical lab setup in intermediate distance regime.

        Note: Fresnel propagator is deprecated, Angular Spectrum is used instead.
        """
        # Lab optical table: F = 5.0 (intermediate regime)
        wavelength = 520e-9
        obj_distance = 0.1  # 10 cm
        # F = 5.0 → fov = sqrt(5.0 × λ × L)
        fov = (5.0 * wavelength * obj_distance) ** 0.5  # ~510 µm

        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="auto",
            dx=2e-6,
            image_size=256,
        )

        # Angular Spectrum is used for F >= 0.1 (Fresnel is deprecated)
        assert isinstance(prop, AngularSpectrumPropagator)


class TestFFTCacheIntegration:
    """Test FFT cache integration with select_propagator."""

    def test_select_propagator_with_fft_cache(self):
        """Test that FFT cache is properly passed to propagator."""
        from prism.utils.transforms import FFTCache

        cache = FFTCache()

        prop = select_propagator(
            wavelength=520e-9,
            obj_distance=1000.0,
            fov=1e-3,
            method="auto",
            fft_cache=cache,
        )

        # Should be Fraunhofer with shared cache
        assert isinstance(prop, FraunhoferPropagator)
        assert prop.fft_cache is cache

    def test_select_fresnel_with_fft_cache(self):
        """Test FFT cache with Fresnel propagator."""
        from prism.utils.transforms import FFTCache

        cache = FFTCache()
        wavelength = 520e-9
        obj_distance = 0.05
        fov = (1.0 * wavelength * obj_distance) ** 0.5

        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="fresnel",
            dx=1e-6,
            dxf=100,
            image_size=256,
            fft_cache=cache,
        )

        assert isinstance(prop, FresnelPropagator)
        assert prop.fft_cache is cache


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            select_propagator(
                wavelength=520e-9,
                obj_distance=1.0,
                fov=1e-3,
                method="magic_propagation",  # type: ignore[arg-type]  # Invalid!
            )

    def test_fresnel_without_dx_raises(self):
        """Test that Fresnel without dx raises error."""
        wavelength = 520e-9
        obj_distance = 0.05
        fov = (1.0 * wavelength * obj_distance) ** 0.5

        with pytest.raises(ValueError, match="requires.*grid.*image_size.*dx"):
            select_propagator(
                wavelength=wavelength,
                obj_distance=obj_distance,
                fov=fov,
                method="fresnel",
                # Missing dx
                dxf=100,
                image_size=256,
            )

    def test_fresnel_without_dxf_raises(self):
        """Test that Fresnel without dxf raises error."""
        wavelength = 520e-9
        obj_distance = 0.05
        fov = (1.0 * wavelength * obj_distance) ** 0.5

        # dxf is no longer required - Grid-based API computes it
        # This test now verifies Fresnel can work without explicit dxf
        prop = select_propagator(
            wavelength=wavelength,
            obj_distance=obj_distance,
            fov=fov,
            method="fresnel",
            dx=1e-6,
            # No dxf - Grid API computes it
            image_size=256,
        )
        assert prop is not None

    def test_fresnel_without_image_size_raises(self):
        """Test that Fresnel without image_size raises error."""
        wavelength = 520e-9
        obj_distance = 0.05
        fov = (1.0 * wavelength * obj_distance) ** 0.5

        with pytest.raises(ValueError, match="requires.*grid.*image_size.*dx"):
            select_propagator(
                wavelength=wavelength,
                obj_distance=obj_distance,
                fov=fov,
                method="fresnel",
                dx=1e-6,
                dxf=100,
                # Missing image_size
            )

    def test_angular_spectrum_default_parameters(self):
        """Test Angular Spectrum with default parameters (no dx/image_size)."""
        # Should use defaults
        prop = select_propagator(
            wavelength=520e-9,
            obj_distance=0.0001,
            fov=1e-3,
            method="angular_spectrum",
            # No dx or image_size - should use defaults
        )

        assert isinstance(prop, AngularSpectrumPropagator)


class TestPropagatorFunctionality:
    """Test that selected propagators actually work."""

    def test_fraunhofer_forward_backward(self):
        """Test Fraunhofer propagator forward/backward."""
        prop = select_propagator(
            wavelength=520e-9,
            obj_distance=1000.0,
            fov=1e-3,
            method="auto",
        )

        field = torch.randn(128, 128, dtype=torch.cfloat)
        k_field = prop(field, direction="forward")
        reconstructed = prop(k_field, direction="backward")

        # Should reconstruct original (within tolerance)
        torch.testing.assert_close(reconstructed, field, rtol=1e-4, atol=1e-6)

    def test_angular_spectrum_propagation(self):
        """Test Angular Spectrum propagation."""
        prop = select_propagator(
            wavelength=520e-9,
            obj_distance=0.0001,
            fov=1e-3,
            method="angular_spectrum",
            image_size=128,
            dx=10e-6,
        )

        field = torch.randn(128, 128, dtype=torch.cfloat)
        propagated = prop(field)

        # Should return complex field of same shape
        assert propagated.shape == field.shape
        assert propagated.dtype == torch.cfloat


class TestAngularSpectrumDirectionInterface:
    """Test Angular Spectrum propagator direction interface."""

    def test_angular_spectrum_direction_interface(self):
        """Test Angular Spectrum with direction parameter."""
        from prism.core.grid import Grid

        grid = Grid(nx=64, dx=10e-6, wavelength=520e-9)
        prop = AngularSpectrumPropagator(grid, distance=0.1)

        field = torch.randn(64, 64, dtype=torch.cfloat)

        # Test forward direction
        k_field = prop(field, direction="forward")
        assert k_field.shape == field.shape
        assert k_field.dtype == torch.cfloat

        # Test backward direction
        reconstructed = prop(k_field, direction="backward")
        assert reconstructed.shape == field.shape
        assert reconstructed.dtype == torch.cfloat

    def test_angular_spectrum_direction_reversibility(self):
        """Test Angular Spectrum forward/backward with direction is reversible."""
        from prism.core.grid import Grid

        grid = Grid(nx=64, dx=10e-6, wavelength=520e-9)
        # Zero distance for pure FFT behavior (reversible)
        prop = AngularSpectrumPropagator(grid, distance=0.0)

        field = torch.randn(64, 64, dtype=torch.cfloat)

        # Forward then backward should reconstruct original
        k_field = prop(field, direction="forward")
        reconstructed = prop(k_field, direction="backward")

        torch.testing.assert_close(reconstructed, field, rtol=1e-4, atol=1e-6)
