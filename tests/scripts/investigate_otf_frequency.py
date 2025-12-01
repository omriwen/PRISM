"""
Investigation script for OTF frequency calibration issue.

This script helps diagnose the frequency coordinate mismatch between
test expectations and OTFPropagator implementation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from prism.core.grid import Grid
from prism.core.propagators import OTFPropagator


def create_circular_aperture(grid: Grid, radius: float) -> torch.Tensor:
    """Create circular aperture mask."""
    r = torch.sqrt(grid.x**2 + grid.y**2)
    aperture = (r <= radius).float()
    return aperture


def analytical_otf_circular(rho: np.ndarray) -> np.ndarray:
    """Analytical OTF for circular aperture."""
    rho = np.clip(rho, 0, 1)
    with np.errstate(invalid="ignore"):
        result = (2 / np.pi) * (np.arccos(rho) - rho * np.sqrt(1 - rho**2))
    result = np.where(np.isnan(result), 0.0, result)
    return result


def main():
    # Test parameters (same as in the test)
    wavelength = 550e-9
    aperture_radius_pixels = 64  # Aperture spans 128 pixels (diameter)
    n_pixels = 512
    pixel_size = 10e-6  # 10 micron pixels

    # Physical aperture diameter
    aperture_diameter = 2 * aperture_radius_pixels * pixel_size

    # Create grid and aperture
    grid = Grid(nx=n_pixels, dx=pixel_size, wavelength=wavelength)
    aperture = create_circular_aperture(grid, radius=aperture_radius_pixels * pixel_size)

    # Create OTF propagator
    otf_prop = OTFPropagator(aperture.to(torch.cfloat), grid=grid, normalize=True)
    otf = otf_prop.get_otf()

    # Expected cutoff frequencies
    ctf_cutoff = aperture_diameter / wavelength  # Coherent cutoff
    otf_cutoff_expected = 2 * ctf_cutoff  # Incoherent cutoff (2x CTF)

    print(f"=== OTF Frequency Investigation ===\n")
    print(f"Wavelength: {wavelength * 1e9:.1f} nm")
    print(f"Pixel size: {pixel_size * 1e6:.1f} µm")
    print(f"Grid size: {n_pixels} x {n_pixels}")
    print(f"Aperture diameter: {aperture_diameter * 1e3:.3f} mm")
    print(f"\nExpected cutoff frequencies:")
    print(f"  CTF cutoff: {ctf_cutoff:.2e} cycles/m")
    print(f"  OTF cutoff (2x CTF): {otf_cutoff_expected:.2e} cycles/m")

    # Compute frequency coordinates (test's approach)
    freq_1d = torch.fft.fftshift(torch.fft.fftfreq(grid.nx, d=grid.dx))
    print(f"\nFrequency coordinate range:")
    print(f"  Min: {freq_1d.min().item():.2e} cycles/m")
    print(f"  Max: {freq_1d.max().item():.2e} cycles/m")
    print(f"  Spacing: {(freq_1d[1] - freq_1d[0]).item():.2e} cycles/m")

    # Create 2D frequency grids
    kx_grid, ky_grid = torch.meshgrid(freq_1d, freq_1d, indexing="ij")
    k_radial = torch.sqrt(kx_grid**2 + ky_grid**2)

    # Analyze OTF
    center_idx = otf.shape[0] // 2
    print(f"\nOTF DC value (should be 1.0): {otf[center_idx, center_idx].item():.6f}")

    # Get radial profile along x-axis
    k_1d = freq_1d[center_idx:].cpu().numpy()
    measured_otf = otf[center_idx, center_idx:].cpu().numpy()

    # Find where OTF drops to 1% of DC
    otf_threshold = 0.01
    idx_1pct = np.where(measured_otf < otf_threshold)[0]
    if len(idx_1pct) > 0:
        k_at_1pct = k_1d[idx_1pct[0]]
        print(f"\nOTF drops to 1% at: {k_at_1pct:.2e} cycles/m")
        print(f"Expected OTF cutoff: {otf_cutoff_expected:.2e} cycles/m")
        print(f"Ratio (measured/expected): {k_at_1pct / otf_cutoff_expected:.3f}")

    # Compare with analytical OTF
    print(f"\n=== Analytical OTF Comparison ===")

    # Normalize frequency by expected cutoff
    rho = k_1d / otf_cutoff_expected
    analytical_otf = analytical_otf_circular(rho)

    # Compute error in valid range (rho <= 1)
    valid_mask = (rho >= 0) & (rho <= 1.0)
    if valid_mask.sum() > 10:
        l2_error = np.linalg.norm(
            measured_otf[valid_mask] - analytical_otf[valid_mask]
        ) / np.linalg.norm(analytical_otf[valid_mask])
        print(f"L2 error (rho in [0,1]): {l2_error * 100:.2f}%")

    # Try alternative frequency scaling (normalized by wavelength)
    freq_1d_alt = freq_1d / wavelength  # Dimensionless frequency
    k_1d_alt = freq_1d_alt[center_idx:].cpu().numpy()

    # Alternative cutoff (dimensionless)
    otf_cutoff_alt = aperture_diameter  # In meters (not divided by wavelength)
    rho_alt = k_1d_alt / (otf_cutoff_alt / wavelength)
    analytical_otf_alt = analytical_otf_circular(rho_alt)

    valid_mask_alt = (rho_alt >= 0) & (rho_alt <= 1.0)
    if valid_mask_alt.sum() > 10:
        l2_error_alt = np.linalg.norm(
            measured_otf[valid_mask_alt] - analytical_otf_alt[valid_mask_alt]
        ) / np.linalg.norm(analytical_otf_alt[valid_mask_alt])
        print(f"L2 error (alternative scaling): {l2_error_alt * 100:.2f}%")

    # Plot OTF profiles
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Measured OTF (2D)
    ax = axes[0, 0]
    im = ax.imshow(otf.cpu().numpy(), cmap='viridis', origin='lower')
    ax.set_title('Measured OTF (2D)')
    ax.set_xlabel('x pixels')
    ax.set_ylabel('y pixels')
    plt.colorbar(im, ax=ax)

    # Plot 2: Radial OTF vs frequency
    ax = axes[0, 1]
    ax.plot(k_1d, measured_otf, 'b-', linewidth=2, label='Measured OTF')
    ax.axvline(otf_cutoff_expected, color='r', linestyle='--', label=f'Expected cutoff: {otf_cutoff_expected:.2e}')
    if len(idx_1pct) > 0:
        ax.axvline(k_at_1pct, color='g', linestyle=':', label=f'Measured 1%: {k_at_1pct:.2e}')
    ax.set_xlabel('Spatial Frequency (cycles/m)')
    ax.set_ylabel('OTF')
    ax.set_title('Radial OTF Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Normalized OTF comparison
    ax = axes[1, 0]
    ax.plot(rho, measured_otf, 'b-', linewidth=2, label='Measured')
    ax.plot(rho, analytical_otf, 'r--', linewidth=2, label='Analytical')
    ax.set_xlabel('Normalized Frequency (ρ = f/f_cutoff)')
    ax.set_ylabel('OTF')
    ax.set_title(f'OTF Comparison (L2 error: {l2_error * 100:.2f}%)')
    ax.set_xlim(0, 1.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Error analysis
    ax = axes[1, 1]
    error = measured_otf - analytical_otf
    ax.plot(rho, error, 'r-', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Normalized Frequency (ρ)')
    ax.set_ylabel('Error (Measured - Analytical)')
    ax.set_title('OTF Error')
    ax.set_xlim(0, 1.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/otf_investigation.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to /tmp/otf_investigation.png")

    # Additional diagnostics
    print(f"\n=== Additional Diagnostics ===")
    print(f"OTF shape: {otf.shape}")
    print(f"OTF min: {otf.min().item():.6f}")
    print(f"OTF max: {otf.max().item():.6f}")
    print(f"OTF is real: {torch.all(otf.imag == 0).item() if torch.is_complex(otf) else True}")


if __name__ == "__main__":
    main()
