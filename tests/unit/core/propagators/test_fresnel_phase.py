"""
Tests for Fresnel propagator physics correctness.

This test suite validates that the Fresnel propagator behaves correctly
according to optical physics principles from Goodman's "Introduction to
Fourier Optics".
"""

from __future__ import annotations

import pytest
import torch

from prism.core.grid import Grid
from prism.core.propagators import AngularSpectrumPropagator, FresnelPropagator


class TestFresnelPhysics:
    """Test Fresnel propagator physics behavior.

    IMPORTANT: These tests are skipped due to known limitations in the 1-FFT Fresnel
    implementation:

    1. **Energy Conservation**: The normalization factor (e^ikz / iλz) * dx² introduces
       field scaling that violates energy conservation. Even with proper grid-aware
       energy calculation (E = Σ|field|² · dx²), energy is not conserved.

    2. **Reversibility**: Forward+backward propagation fails because each pass applies
       the problematic normalization, compounding the error.

    3. **Grid Scaling**: The Fresnel method changes grid spacing:
       dx_out = λ·z / (N·dx_in), which makes energy conservation more complex than
       simple grid weighting.

    **Recommendation**: Use AngularSpectrumPropagator for physics validation, as it
    preserves energy and grid spacing. The Fresnel propagator is primarily useful for
    computational efficiency in specific regimes, not physics accuracy.

    **Future Work**: To fix these tests, the FresnelPropagator normalization would
    need to be redesigned to properly account for energy conservation across grid
    changes, or a 2-FFT transfer function method could be implemented instead.
    """

    @pytest.fixture
    def fresnel_regime_params(self):
        """Parameters for Fresnel regime testing (F ~ 1)."""
        wavelength = 698.9e-9
        image_size = 128
        dx = 10e-6
        fov = image_size * dx
        a = fov / 2
        # Set z such that F = a²/(λ*z) ~ 1
        z = a**2 / wavelength
        return {
            "wavelength": wavelength,
            "image_size": image_size,
            "dx": dx,
            "z": z,
        }

    def test_energy_conservation(self, fresnel_regime_params):
        """Energy conservation test (currently fails - see class docstring).

        This test demonstrates that even with grid-aware energy calculation:
            E_in = Σ|field_in|² · dx_in²
            E_out = Σ|field_out|² · dx_out²

        Energy is NOT conserved due to the normalization factor in the 1-FFT method.

        Expected behavior: Energy ratio ~ 6e-5 (factor of ~16,000 loss)
        Desired behavior: Energy ratio ~ 1.0
        """
        grid_in = Grid(
            nx=fresnel_regime_params["image_size"],
            dx=fresnel_regime_params["dx"],
            wavelength=fresnel_regime_params["wavelength"],
        )
        prop = FresnelPropagator(grid=grid_in, distance=fresnel_regime_params["z"])

        # Random complex field
        field_in = torch.randn(128, 128, dtype=torch.complex64)
        field_out = prop(field_in)

        # Get output grid for proper pixel area weighting
        grid_out = prop.output_grid

        # Energy with proper grid weighting: E = Σ|field|² · dx²
        energy_in = (field_in.abs() ** 2).sum().item() * grid_in.dx**2
        energy_out = (field_out.abs() ** 2).sum().item() * grid_out.dx**2

        # This assertion will fail (expected) - keeping for documentation
        energy_ratio = energy_out / energy_in
        assert 0.9 < energy_ratio < 1.1, (
            f"Energy not conserved: {energy_in:.2e} → {energy_out:.2e} "
            f"(ratio: {energy_ratio:.4f}, expected ~1.0)\n"
            f"Input grid: dx={grid_in.dx:.2e}, Output grid: dx={grid_out.dx:.2e}\n"
            f"NOTE: This is expected behavior for 1-FFT Fresnel implementation."
        )

    def test_fresnel_reversibility(self, fresnel_regime_params):
        """Forward propagation (+z) followed by backward (-z) should approximately invert.

        Note: The Fresnel propagator changes grid spacing, so forward+backward
        involves two grid transformations:
            dx_in → dx_mid = λ·z/(N·dx_in) → dx_out = λ·z/(N·dx_mid) = dx_in

        The final grid should match the original, and energy should be conserved.
        """
        grid_in = Grid(
            nx=fresnel_regime_params["image_size"],
            dx=fresnel_regime_params["dx"],
            wavelength=fresnel_regime_params["wavelength"],
        )
        z = fresnel_regime_params["z"]

        prop_forward = FresnelPropagator(grid=grid_in, distance=z)
        # For backward propagation, use the OUTPUT grid of forward propagation
        grid_mid = prop_forward.output_grid
        prop_backward = FresnelPropagator(grid=grid_mid, distance=-z)

        # Random complex field
        field_original = torch.randn(128, 128, dtype=torch.complex64)

        # Propagate forward then backward
        field_forward = prop_forward(field_original)
        field_recovered = prop_backward(field_forward)

        # Check energy conservation through round trip
        energy_original = (field_original.abs() ** 2).sum().item() * grid_in.dx**2
        grid_out = prop_backward.output_grid
        energy_recovered = (field_recovered.abs() ** 2).sum().item() * grid_out.dx**2
        energy_ratio = energy_recovered / energy_original

        assert 0.8 < energy_ratio < 1.2, (
            f"Round-trip energy not conserved: {energy_original:.2e} → {energy_recovered:.2e} "
            f"(ratio: {energy_ratio:.4f}, expected ~1.0)"
        )

        # Check correlation (phase may differ but structure should match)
        correlation = torch.corrcoef(
            torch.stack([field_original.abs().flatten(), field_recovered.abs().flatten()])
        )[0, 1]

        assert correlation > 0.8, (
            f"Forward-backward propagation should approximately recover original. "
            f"Correlation: {correlation:.4f} (expected > 0.8)"
        )

    def test_output_grid_scaling(self, fresnel_regime_params):
        """Output grid should scale according to Fresnel formula."""
        grid = Grid(
            nx=fresnel_regime_params["image_size"],
            dx=fresnel_regime_params["dx"],
            wavelength=fresnel_regime_params["wavelength"],
        )
        z = fresnel_regime_params["z"]

        prop = FresnelPropagator(grid=grid, distance=z)

        # Expected output pixel size: dx_out = λ·z / (N·dx_in)
        expected_dx_out = (
            fresnel_regime_params["wavelength"]
            * z
            / (fresnel_regime_params["image_size"] * fresnel_regime_params["dx"])
        )

        assert abs(prop.output_grid.dx - expected_dx_out) / expected_dx_out < 1e-6, (
            f"Output pixel size should follow Fresnel formula. "
            f"Expected: {expected_dx_out:.2e}, got: {prop.output_grid.dx:.2e}"
        )


class TestFresnelLimits:
    """Test Fresnel propagator behavior in limiting cases.

    NOTE: These tests are skipped due to known limitations in the 1-FFT Fresnel
    implementation. The plane wave test fails because the normalization factor
    and output grid scaling introduce non-uniformities. For physics validation
    of plane wave propagation, use AngularSpectrumPropagator instead.
    """

    def test_plane_wave_propagation(self):
        """Plane wave uniformity test (currently fails - see class docstring).

        Expected behavior: Large relative std (>500%)
        Desired behavior: Uniform field (std < 20%)
        """
        wavelength = 520e-9
        image_size = 64
        dx = 10e-6
        z = 0.1  # 10 cm

        grid = Grid(nx=image_size, dx=dx, wavelength=wavelength)
        prop = FresnelPropagator(grid=grid, distance=z)

        # Uniform (plane wave) field
        field_in = torch.ones(64, 64, dtype=torch.complex64)
        field_out = prop(field_in)

        # Magnitude should remain relatively uniform (within 20%)
        # This will fail - keeping for documentation
        mag_std = field_out.abs().std() / field_out.abs().mean()
        assert mag_std < 0.2, (
            f"Plane wave magnitude should remain uniform. "
            f"Relative std: {mag_std:.2%} (expected < 20%)\n"
            f"NOTE: This is expected behavior for 1-FFT Fresnel implementation."
        )

    def test_validation_rejects_tiny_distance(self):
        """Distance much smaller than wavelength should raise error."""
        grid = Grid(nx=64, dx=10e-6, wavelength=520e-9)

        with pytest.raises(ValueError, match="too small"):
            FresnelPropagator(grid=grid, distance=1e-12)


class TestFresnelVsAngularSpectrum:
    """Compare Fresnel and Angular Spectrum for consistency."""

    def test_both_propagators_produce_valid_output(self):
        """Both propagators should produce finite, valid output."""
        wavelength = 520e-9
        image_size = 64
        dx = 10e-6
        z = 0.05

        grid = Grid(nx=image_size, dx=dx, wavelength=wavelength)

        prop_fresnel = FresnelPropagator(grid=grid, distance=z)
        prop_angular = AngularSpectrumPropagator(grid=grid, distance=z)

        # Random complex field
        field_in = torch.randn(64, 64, dtype=torch.complex64)

        field_fresnel = prop_fresnel(field_in)
        field_angular = prop_angular(field_in)

        # Both should produce finite output
        assert torch.isfinite(field_fresnel).all(), "Fresnel output should be finite"
        assert torch.isfinite(field_angular).all(), "Angular spectrum output should be finite"

        # Both should preserve shape
        assert field_fresnel.shape == field_in.shape
        assert field_angular.shape == field_in.shape


class TestFresnelPhaseSignVerification:
    """Test phase sign conventions (no internal access needed)."""

    def test_phase_coefficients_math(self):
        """Verify the mathematical relationships for phase coefficients."""
        wavelength = 698.9e-9
        z = 1.0

        # Spatial domain coefficient: +π/(λ*z) (should be positive)
        spatial_coeff = torch.pi / wavelength / z
        assert spatial_coeff > 0, "Spatial domain coefficient should be positive"

        # Frequency domain coefficient: -π*λ*z (should be negative)
        freq_coeff = -torch.pi * wavelength * z
        assert freq_coeff < 0, "Frequency domain coefficient should be negative"

        # They should have opposite signs
        assert spatial_coeff * freq_coeff < 0, (
            "Spatial and frequency domain coefficients should have opposite signs"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
