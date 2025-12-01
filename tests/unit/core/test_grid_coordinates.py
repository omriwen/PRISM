"""
Tests for grid coordinate system validation (Phase 3).

This test suite validates that grid coordinates have correct units and
are properly used in propagators, particularly the AngularSpectrumPropagator.
"""

from __future__ import annotations

import pytest
import torch

from prism.core.grid import Grid
from prism.core.propagators import AngularSpectrumPropagator


class TestGridCoordinateUnits:
    """Verify grid coordinate units are correct for propagators."""

    def test_kx_ky_are_spatial_frequencies(self):
        """Verify kx, ky have units of [1/m] (spatial frequency)."""
        grid = Grid(nx=128, ny=128, dx=10e-6, dy=10e-6, wavelength=698.9e-9)

        # kx should equal fx = n / (N * dx) where n is pixel index
        # For centered grid: n ranges from -N/2 to N/2
        # Maximum spatial frequency: f_max = 1/(2*dx) (Nyquist)
        f_max_expected = 1 / (2 * grid.dx)
        kx_max = grid.kx.abs().max().item()

        # Check that max spatial frequency matches Nyquist limit
        assert torch.isclose(torch.tensor(kx_max), torch.tensor(f_max_expected), rtol=1e-5), (
            f"kx_max={kx_max:.2e} should equal Nyquist frequency={f_max_expected:.2e}"
        )

    def test_angular_spectrum_uses_correct_formula(self):
        """
        Verify AngularSpectrum correctly compensates for spatial frequency units.

        The angular spectrum transfer function should be:
            H = exp(i * kz * z)
        where:
            kz = sqrt(k² - kx² - ky²)
            k = 2π/λ [rad/m]
            kx, ky should be angular frequencies [rad/m]

        However, grid.kx, grid.ky are spatial frequencies [1/m], so the
        propagator must include the 2π factor in the phase calculation.
        """
        grid = Grid(nx=128, ny=128, dx=10e-6, dy=10e-6, wavelength=698.9e-9)
        wavelength = grid.wl

        # Create propagator
        prop = AngularSpectrumPropagator(grid, distance=1.0)

        # Verify the k_sqrt formula
        # Code computes: k_sqrt = sqrt(1/λ² - fx² - fy²)
        # where fx, fy are spatial frequencies
        k_sqrt_expected = torch.sqrt(
            torch.clamp(1 / wavelength**2 - grid.kx**2 - grid.ky**2, min=0)
        )

        assert torch.allclose(prop.k_sqrt_tensor, k_sqrt_expected, rtol=1e-6), (
            "k_sqrt should be sqrt(1/λ² - fx² - fy²) where fx, fy are spatial frequencies"
        )

        # The phase in forward() includes the 2π factor:
        # phase = 2π * z * k_sqrt = 2π * z * sqrt(1/λ² - fx² - fy²)
        # This equals: z * sqrt((2π/λ)² - (2π·fx)² - (2π·fy)²)
        # which is the correct kz * z formula with angular frequencies

    def test_angular_spectrum_no_evanescent_waves_for_typical_params(self):
        """
        Verify kz² > 0 for typical parameters (no evanescent waves).

        For paraxial conditions, kx, ky << k, so kz should be real.
        """
        grid = Grid(nx=128, ny=128, dx=10e-6, dy=10e-6, wavelength=698.9e-9)
        prop = AngularSpectrumPropagator(grid, distance=1.0)

        # All values should be in propagating regime (diff_limit = True)
        # For small apertures and visible light, we don't expect evanescent waves
        num_evanescent = (~prop.diff_limit_tensor).sum().item()
        total_pixels = grid.nx * grid.ny

        # Allow some evanescent waves at edges due to Nyquist sampling
        # but should be < 5% of total
        evanescent_fraction = num_evanescent / total_pixels

        assert evanescent_fraction < 0.05, (
            f"Found {evanescent_fraction:.1%} evanescent waves, expected < 5% "
            f"for typical paraxial parameters"
        )

    def test_kx_ky_satisfy_fft_frequency_formula(self):
        """
        Verify kx, ky follow the standard FFT frequency formula.

        For FFT with fftshift (DC at center):
            f[n] = n / (N * dx)  for n in [-N/2, N/2-1]
        """
        grid = Grid(nx=128, ny=128, dx=10e-6, dy=10e-6, wavelength=698.9e-9)

        # Manually compute expected frequencies
        n_x = torch.arange(-(grid.nx // 2), (grid.nx + 1) // 2, dtype=torch.float32)
        fx_expected = n_x / (grid.nx * grid.dx)

        # grid.kx has shape (1, nx), fx_expected has shape (nx,)
        fx_from_grid = grid.kx.squeeze(0)

        assert torch.allclose(fx_from_grid, fx_expected, rtol=1e-6), (
            "kx should follow FFT frequency formula: f = n / (N * dx)"
        )

    def test_grid_coordinates_match_documentation(self):
        """Verify grid coordinates match the documented behavior in grid.py."""
        grid = Grid(nx=128, ny=128, dx=10e-6, dy=10e-6, wavelength=698.9e-9)

        # From grid.py documentation:
        # - Spatial: Range [-FOV/2, FOV/2] where FOV = nx * dx
        # - Frequency: kx = x / (nx * dx²)
        # - Frequency range: [-1/(2*dx), 1/(2*dx)] (Nyquist)

        # Check spatial range
        fov_x = grid.nx * grid.dx
        x_min, x_max = grid.x.min().item(), grid.x.max().item()
        assert abs(x_min) < fov_x / 2 + 1e-9, "x should be in range [-FOV/2, FOV/2]"
        assert abs(x_max) < fov_x / 2 + 1e-9, "x should be in range [-FOV/2, FOV/2]"

        # Check frequency range (Nyquist)
        nyquist_freq = 1 / (2 * grid.dx)
        kx_max = grid.kx.abs().max().item()
        assert abs(kx_max - nyquist_freq) < 1e-6, (
            f"Maximum kx frequency should be Nyquist limit: {nyquist_freq:.2e}"
        )

        # Check the relation kx = x / (nx * dx²)
        kx_from_formula = grid.x / (grid.nx * grid.dx**2)
        assert torch.allclose(grid.kx, kx_from_formula, rtol=1e-6), "kx should equal x / (nx * dx²)"


class TestAngularSpectrumImplementation:
    """Verify Angular Spectrum propagator implementation correctness."""

    def test_transfer_function_physics(self):
        """
        Verify the transfer function matches Goodman's formula.

        H(fx, fy, z) = exp(i * 2π * z * sqrt(1/λ² - fx² - fy²))

        This is equivalent to H(kx, ky, z) = exp(i * kz * z) where:
            kz = sqrt(k² - kx² - ky²)
            k = 2π/λ
            kx = 2π * fx, ky = 2π * fy
        """
        grid = Grid(nx=128, ny=128, dx=10e-6, dy=10e-6, wavelength=698.9e-9)
        z = 0.05  # 5 cm propagation
        prop = AngularSpectrumPropagator(grid, distance=z)

        # Compute expected transfer function
        # H = exp(i * 2π * z * sqrt(1/λ² - fx² - fy²))
        wavelength = grid.wl
        k_sqrt_expected = torch.sqrt(
            torch.clamp(1 / wavelength**2 - grid.kx**2 - grid.ky**2, min=0)
        )
        phase_expected = 2 * torch.pi * z * k_sqrt_expected
        h_expected = torch.exp(1j * phase_expected)

        # Zero out evanescent waves
        diff_limit = (1 / wavelength**2 - grid.kx**2 - grid.ky**2) > 0
        h_expected = torch.where(diff_limit, h_expected, torch.zeros_like(h_expected))

        # Compare with propagator's internal calculation
        # (We need to reconstruct it since it's computed in forward())
        phase_actual = 2 * torch.pi * z * prop.k_sqrt_tensor
        h_actual = torch.exp(1j * phase_actual)
        h_actual = torch.where(prop.diff_limit_tensor, h_actual, torch.zeros_like(h_actual))

        assert torch.allclose(h_actual, h_expected, rtol=1e-6), (
            "Transfer function should match Goodman's formula"
        )

    def test_energy_conservation(self):
        """Angular spectrum propagation should conserve energy."""
        grid = Grid(nx=128, ny=128, dx=10e-6, dy=10e-6, wavelength=698.9e-9)
        prop = AngularSpectrumPropagator(grid, distance=0.05)

        # Random complex field
        field_in = torch.randn(128, 128, dtype=torch.complex64)
        field_out = prop(field_in)

        energy_in = (field_in.abs() ** 2).sum().item()
        energy_out = (field_out.abs() ** 2).sum().item()

        # Energy should be conserved (within numerical error)
        # Note: Some energy loss occurs due to evanescent wave filtering
        assert abs(energy_in - energy_out) / energy_in < 0.01, (
            f"Energy not conserved: {energy_in:.2e} → {energy_out:.2e} "
            f"(loss: {100 * (1 - energy_out / energy_in):.1f}%)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
