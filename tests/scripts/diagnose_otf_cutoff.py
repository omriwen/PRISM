"""
Diagnostic script to determine the correct OTF cutoff frequency formula.

The issue: The test expects OTF cutoff = 2 * (aperture_diameter / wavelength),
but the measured OTF cutoff is 5.16x larger.

This script investigates the relationship between aperture size, grid parameters,
and OTF support to find the correct formula.
"""

import torch
import numpy as np
from prism.core.grid import Grid
from prism.core.propagators import OTFPropagator


def create_circular_aperture(grid: Grid, radius: float) -> torch.Tensor:
    """Create circular aperture mask."""
    r = torch.sqrt(grid.x**2 + grid.y**2)
    aperture = (r <= radius).float()
    return aperture


def main():
    # Test parameters
    wavelength = 550e-9
    aperture_radius_pixels = 64  # Aperture radius in pixels
    n_pixels = 512
    pixel_size = 10e-6  # 10 micron pixels

    # Physical aperture dimensions
    aperture_radius_m = aperture_radius_pixels * pixel_size
    aperture_diameter_m = 2 * aperture_radius_m

    # Create grid and aperture
    grid = Grid(nx=n_pixels, dx=pixel_size, wavelength=wavelength)
    aperture = create_circular_aperture(grid, radius=aperture_radius_m)

    # Create OTF
    otf_prop = OTFPropagator(aperture.to(torch.cfloat), grid=grid, normalize=True)
    otf = otf_prop.get_otf()

    # Frequency coordinates
    freq_1d = torch.fft.fftshift(torch.fft.fftfreq(grid.nx, d=grid.dx))
    center_idx = otf.shape[0] // 2

    print("=== OTF Cutoff Frequency Analysis ===\n")
    print(f"Grid parameters:")
    print(f"  Pixels: {n_pixels} x {n_pixels}")
    print(f"  Pixel size: {pixel_size * 1e6:.1f} µm")
    print(f"  Physical size: {n_pixels * pixel_size * 1e3:.2f} mm x {n_pixels * pixel_size * 1e3:.2f} mm")
    print(f"\nAperture:")
    print(f"  Radius (pixels): {aperture_radius_pixels}")
    print(f"  Radius (physical): {aperture_radius_m * 1e3:.3f} mm")
    print(f"  Diameter (physical): {aperture_diameter_m * 1e3:.3f} mm")
    print(f"\nWavelength: {wavelength * 1e9:.1f} nm")

    # Frequency grid parameters
    df = freq_1d[1] - freq_1d[0]  # Frequency spacing
    f_max = freq_1d[-1]  # Maximum frequency
    print(f"\nFrequency grid:")
    print(f"  Spacing: {df.item():.3e} cycles/m")
    print(f"  Max frequency: {f_max.item():.3e} cycles/m")
    print(f"  Nyquist: {1/(2*pixel_size):.3e} cycles/m")

    # Find measured OTF cutoff (where OTF drops to 1%)
    k_1d = freq_1d[center_idx:].cpu().numpy()
    measured_otf = otf[center_idx, center_idx:].cpu().numpy()
    idx_1pct = np.where(measured_otf < 0.01)[0]
    if len(idx_1pct) > 0:
        f_cutoff_measured = k_1d[idx_1pct[0]]
        pixels_to_cutoff = idx_1pct[0]
    else:
        f_cutoff_measured = k_1d[-1]
        pixels_to_cutoff = len(k_1d) - 1

    print(f"\n=== Measured OTF Cutoff ===")
    print(f"  Frequency: {f_cutoff_measured:.3e} cycles/m")
    print(f"  Pixels from center: {pixels_to_cutoff}")

    # Test different cutoff formulas
    print(f"\n=== Cutoff Formula Comparison ===\n")

    # Formula 1: Test's current formula (WRONG)
    f_ctf_test = aperture_diameter_m / wavelength
    f_otf_test = 2 * f_ctf_test
    ratio1 = f_cutoff_measured / f_otf_test
    print(f"Formula 1 (Test's current - WRONG):")
    print(f"  CTF cutoff = D/λ = {f_ctf_test:.3e} cycles/m")
    print(f"  OTF cutoff = 2 * CTF = {f_otf_test:.3e} cycles/m")
    print(f"  Measured / Expected = {ratio1:.3f}")
    print(f"  ❌ ERROR: {abs(1 - ratio1) * 100:.1f}%\n")

    # Formula 2: Based on pixel count (autocorrelation width)
    # The aperture has diameter 2*64 = 128 pixels
    # The autocorrelation has support of 2*128 = 256 pixels
    # But in frequency space, this maps to...
    # The aperture support in pupil plane determines PSF width in image plane
    # PSF width ~= aperture_radius_pixels
    # OTF support ~= 2 * PSF support ~= 2 * aperture_radius_pixels
    # In frequency coordinates: (2 * aperture_radius_pixels) * df
    f_otf_pixels = 2 * aperture_radius_pixels * df.item()
    ratio2 = f_cutoff_measured / f_otf_pixels
    print(f"Formula 2 (Pixel-based):")
    print(f"  OTF cutoff = 2 * R_pixels * df = {f_otf_pixels:.3e} cycles/m")
    print(f"  Measured / Expected = {ratio2:.3f}")
    if abs(1 - ratio2) < 0.1:
        print(f"  ✅ MATCH! Error: {abs(1 - ratio2) * 100:.1f}%\n")
    else:
        print(f"  ❌ ERROR: {abs(1 - ratio2) * 100:.1f}%\n")

    # Formula 3: Based on autocorrelation diameter
    # Autocorrelation of a disk of radius R has radius 2R
    # In physical units: 2 * aperture_radius_m
    # In frequency space: (2 * aperture_radius_m) / (pixel_size)
    f_otf_autocorr = (2 * aperture_radius_m) / (pixel_size * wavelength)
    ratio3 = f_cutoff_measured / f_otf_autocorr
    print(f"Formula 3 (Autocorrelation - scaled by λ):")
    print(f"  OTF cutoff = 2R / (dx * λ) = {f_otf_autocorr:.3e} cycles/m")
    print(f"  Measured / Expected = {ratio3:.3f}")
    if abs(1 - ratio3) < 0.1:
        print(f"  ✅ MATCH! Error: {abs(1 - ratio3) * 100:.1f}%\n")
    else:
        print(f"  ❌ ERROR: {abs(1 - ratio3) * 100:.1f}%\n")

    # Formula 4: Direct mapping from pupil to frequency
    # The pupil plane sampling determines the frequency space support
    # Maximum frequency = 1 / (2 * pixel_size) = Nyquist
    # But OTF support is limited by aperture autocorrelation
    # Aperture diameter = 128 pixels, autocorrelation = 256 pixels
    # This maps to frequency: aperture_diameter_pixels * df
    f_otf_direct = (2 * aperture_radius_pixels) * df.item()
    ratio4 = f_cutoff_measured / f_otf_direct
    print(f"Formula 4 (Direct grid mapping):")
    print(f"  OTF cutoff = (2 * R_pixels) * df = {f_otf_direct:.3e} cycles/m")
    print(f"  Measured / Expected = {ratio4:.3f}")
    if abs(1 - ratio4) < 0.1:
        print(f"  ✅ MATCH! Error: {abs(1 - ratio4) * 100:.1f}%\n")
    else:
        print(f"  ❌ ERROR: {abs(1 - ratio4) * 100:.1f}%\n")

    # Formula 5: Correct optical formula
    # For an imaging system, OTF cutoff depends on NA and wavelength
    # For a telecentric system: f_cutoff = 2 * NA / λ
    # Where NA = aperture_radius / focal_length
    # But we don't have a focal length here, so this doesn't apply

    print(f"\n=== Conclusion ===")
    print(f"The correct formula appears to be:")
    print(f"  OTF_cutoff = 2 * aperture_radius_pixels * frequency_spacing")
    print(f"  OTF_cutoff = 2 * R_pixels * (1 / (N * dx))")
    print(f"  OTF_cutoff = 2 * R_pixels / (N * dx)")
    print(f"\nWhere:")
    print(f"  R_pixels = aperture radius in pixels")
    print(f"  N = grid size (number of pixels)")
    print(f"  dx = pixel size in meters")
    print(f"\nThe wavelength λ does NOT appear in the frequency coordinate formula!")
    print(f"This is because the OTF is computed in the pupil plane's frequency domain,")
    print(f"not in physical image plane coordinates.")


if __name__ == "__main__":
    main()
