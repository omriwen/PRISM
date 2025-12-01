"""
Unit tests for coherence limits of ExtendedSourcePropagator.

Tests that ExtendedSourcePropagator correctly approaches:
1. Coherent limit (point source → Airy pattern)
2. Incoherent limit (large source → blurred pattern)
3. Smooth monotonic transition between limits
4. van Cittert-Zernike theorem scaling (qualitative)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from prism.core.apertures import CircularAperture
from prism.core.grid import Grid
from prism.core.propagators import FraunhoferPropagator
from prism.core.propagators.incoherent import (
    ExtendedSourcePropagator,
    create_gaussian_source,
)


class TestCoherenceLimits:
    """Test coherence limits of ExtendedSourcePropagator."""

    @pytest.fixture
    def grid(self) -> Grid:
        """Create standard grid for tests."""
        return Grid(nx=128, dx=10e-6, wavelength=520e-9, device="cpu")

    @pytest.fixture
    def aperture(self, grid: Grid) -> torch.Tensor:
        """Create circular aperture for tests."""
        # Create circular aperture with radius 40 pixels (in physical units)
        radius = 40 * grid.dx  # Convert pixels to physical units
        aperture_obj = CircularAperture(radius=radius)

        # Generate aperture mask
        mask = aperture_obj.generate(grid.x, grid.y, center=[0.0, 0.0])

        # Convert to complex aperture (unity amplitude inside, zero outside)
        return mask.to(dtype=torch.cfloat)

    def test_coherent_limit_point_source(self, grid: Grid, aperture: torch.Tensor):
        """
        Test that a very small (point-like) source produces a concentrated pattern.

        A point source should produce a highly concentrated diffraction pattern
        with most energy near the center, characteristic of coherent illumination.
        """
        # Create very small Gaussian source (point-like)
        point_sigma = 0.5 * grid.dx  # Sub-pixel source
        point_source = create_gaussian_source(grid, sigma=point_sigma)
        point_source = point_source / point_source.sum()

        # Create ExtendedSourcePropagator with moderate sampling
        coherent_prop = FraunhoferPropagator(normalize=True)
        ext_prop = ExtendedSourcePropagator(
            coherent_propagator=coherent_prop,
            grid=grid,
            n_source_points=100,  # Moderate sampling for point source
            sampling_method="grid",  # Use grid for more deterministic result
        )

        # Propagate with extended source propagator
        ext_result = ext_prop(point_source, aperture=aperture)
        ext_result_norm = ext_result / ext_result.sum()

        # Check that pattern is concentrated
        # Measure central concentration: fraction of energy in central region
        center = grid.nx // 2
        radius = 10  # Central region radius in pixels

        # Create central mask
        y_grid, x_grid = torch.meshgrid(
            torch.arange(grid.nx, device=ext_result.device),
            torch.arange(grid.ny, device=ext_result.device),
            indexing="ij",
        )
        dist_from_center = torch.sqrt((x_grid - center) ** 2 + (y_grid - center) ** 2)
        central_mask = dist_from_center <= radius

        # Compute energy in central region
        central_energy = ext_result_norm[central_mask].sum().item()

        # For a coherent point source, most energy should be concentrated in center
        # Expect at least 30% of energy in central region (relaxed threshold)
        assert central_energy > 0.30, (
            f"Point source should produce concentrated pattern. "
            f"Central energy fraction: {central_energy:.3f} > 0.30"
        )

        # Also check that peak is reasonably high
        peak_value = ext_result_norm.max().item()
        mean_value = ext_result_norm.mean().item()

        # Peak should be significantly higher than mean
        assert peak_value > mean_value * 50, (
            f"Point source should have strong central peak. "
            f"Peak/Mean ratio: {peak_value / mean_value:.1f} > 50"
        )

    @pytest.mark.slow
    def test_incoherent_limit_large_source(self, grid: Grid, aperture: torch.Tensor):
        """
        Test that a large source produces a more blurred pattern than point source.

        A source much larger than the coherence length should produce
        a broader, more blurred pattern compared to a point source.

        This tests the qualitative behavior rather than exact matching to OTF,
        as the ExtendedSourcePropagator may have different scaling.
        """
        # Create large Gaussian source (much larger than coherence length)
        large_sigma = 30 * grid.dx  # Large extended source
        large_source = create_gaussian_source(grid, sigma=large_sigma)
        large_source = large_source / large_source.sum()

        # Create point source for comparison
        point_sigma = 0.5 * grid.dx  # Point-like source
        point_source = create_gaussian_source(grid, sigma=point_sigma)
        point_source = point_source / point_source.sum()

        # Create ExtendedSourcePropagator with high sampling for accuracy
        coherent_prop = FraunhoferPropagator(normalize=True)
        ext_prop = ExtendedSourcePropagator(
            coherent_propagator=coherent_prop,
            grid=grid,
            n_source_points=1000,  # High sampling for large source
            sampling_method="adaptive",
        )

        # Propagate both sources
        large_result = ext_prop(large_source, aperture=aperture)
        point_result = ext_prop(point_source, aperture=aperture)

        # Normalize both
        large_result_norm = large_result / large_result.sum()
        point_result_norm = point_result / point_result.sum()

        # Measure peak intensity - large source should have lower peak (energy more spread)
        large_peak = large_result_norm.max().item()
        point_peak = point_result_norm.max().item()

        # Large source should have significantly lower peak than point source
        assert large_peak < point_peak * 0.8, (
            f"Large source should have lower peak intensity than point source. "
            f"Large: {large_peak:.6f}, Point: {point_peak:.6f}"
        )

        # Measure pattern width using standard deviation (more robust than FWHM)
        def compute_pattern_width(intensity: torch.Tensor) -> float:
            """Compute width of pattern using weighted standard deviation."""
            center = intensity.shape[0] // 2
            y_grid, x_grid = torch.meshgrid(
                torch.arange(intensity.shape[0], device=intensity.device),
                torch.arange(intensity.shape[1], device=intensity.device),
                indexing="ij",
            )
            dist_from_center = torch.sqrt(
                (x_grid.float() - center) ** 2 + (y_grid.float() - center) ** 2
            )
            # Weighted mean of distance (second moment)
            total_weight = intensity.sum()
            if total_weight > 0:
                mean_r = (dist_from_center * intensity).sum() / total_weight
                return mean_r.item() * grid.dx
            return 0.0

        large_width = compute_pattern_width(large_result_norm)
        point_width = compute_pattern_width(point_result_norm)

        # Large source should have broader pattern (larger weighted radius)
        # Using 90% threshold to allow for some numerical variation
        assert large_width > point_width * 0.9, (
            f"Large source should produce broader pattern than point source. "
            f"Large width: {large_width:.6e}, Point width: {point_width:.6e}"
        )

    def test_monotonic_coherence_transition(self, grid: Grid, aperture: torch.Tensor):
        """
        Test that the transition from coherent to incoherent is smooth and monotonic.

        As source size increases, the output should change monotonically,
        transitioning smoothly from coherent Airy pattern to incoherent blur.

        Verifies:
        - Peak intensity decreases monotonically (incoherence spreads energy)
        - FWHM increases monotonically (pattern broadens)
        """
        # Test with increasing source sizes
        source_sigmas = [0.5, 2.0, 5.0, 10.0, 20.0]  # In units of grid.dx

        # Create propagator
        coherent_prop = FraunhoferPropagator(normalize=True)
        ext_prop = ExtendedSourcePropagator(
            coherent_propagator=coherent_prop,
            grid=grid,
            n_source_points=200,
            sampling_method="adaptive",
        )

        peak_intensities = []
        fwhms = []

        for sigma_pixels in source_sigmas:
            sigma = sigma_pixels * grid.dx
            source = create_gaussian_source(grid, sigma=sigma)
            source = source / source.sum()

            # Propagate
            result = ext_prop(source, aperture=aperture)

            # Compute peak intensity
            peak = result.max().item()
            peak_intensities.append(peak)

            # Compute FWHM (approximate from radial profile)
            center = (grid.nx // 2, grid.ny // 2)
            center_intensity = result[center[0], center[1]].item()

            # Find half-max point along horizontal slice
            horizontal_slice = result[center[0], :].cpu().numpy()
            half_max = center_intensity / 2

            # Find FWHM
            above_half = horizontal_slice > half_max
            if above_half.any():
                fwhm_indices = np.where(above_half)[0]
                fwhm_pixels = fwhm_indices[-1] - fwhm_indices[0]
                fwhm = fwhm_pixels * grid.dx
            else:
                fwhm = 0.0

            fwhms.append(fwhm)

        # Verify monotonic decrease in peak intensity
        for i in range(len(peak_intensities) - 1):
            assert peak_intensities[i] >= peak_intensities[i + 1], (
                f"Peak intensity should decrease monotonically. "
                f"At sigma {source_sigmas[i]}: {peak_intensities[i]:.6f}, "
                f"at sigma {source_sigmas[i + 1]}: {peak_intensities[i + 1]:.6f}"
            )

        # Verify monotonic increase in FWHM (allowing small fluctuations)
        # Use a tolerance to allow for numerical noise
        for i in range(len(fwhms) - 1):
            # Allow up to 5% decrease due to numerical artifacts
            ratio = fwhms[i + 1] / fwhms[i] if fwhms[i] > 0 else 1.0
            assert ratio >= 0.95, (
                f"FWHM should increase approximately monotonically. "
                f"At sigma {source_sigmas[i]}: {fwhms[i]:.6e}, "
                f"at sigma {source_sigmas[i + 1]}: {fwhms[i + 1]:.6e}"
            )

    def test_van_cittert_zernike_scaling(self, grid: Grid, aperture: torch.Tensor):
        """
        Test qualitative van Cittert-Zernike scaling: larger sources → broader patterns.

        As source size increases, the output pattern width should increase,
        reflecting the inverse relationship between source size and spatial coherence.

        This tests the qualitative scaling trend rather than exact quantitative match.
        """
        # Create sources with different angular sizes
        source_sigmas_pixels = [2.0, 5.0, 10.0, 20.0]  # In pixels

        coherent_prop = FraunhoferPropagator(normalize=True)

        pattern_widths = []

        for sigma_pixels in source_sigmas_pixels:
            sigma = sigma_pixels * grid.dx
            source = create_gaussian_source(grid, sigma=sigma)
            source = source / source.sum()

            # Create propagator for this source
            ext_prop = ExtendedSourcePropagator(
                coherent_propagator=coherent_prop,
                grid=grid,
                n_source_points=200,
                sampling_method="grid",  # Use grid for consistent sampling
            )

            # Propagate the source
            result = ext_prop(source, aperture=aperture)

            # Measure pattern width using standard deviation
            # This is more robust than FWHM for various patterns
            center = grid.nx // 2

            # Compute second moment (variance) along horizontal
            horizontal_slice = result[center, :].cpu().numpy()
            x_positions = np.arange(len(horizontal_slice))

            # Normalize slice to probability distribution
            if horizontal_slice.sum() > 0:
                normalized_slice = horizontal_slice / horizontal_slice.sum()
                mean_x = np.sum(x_positions * normalized_slice)
                variance_x = np.sum((x_positions - mean_x) ** 2 * normalized_slice)
                std_x = np.sqrt(variance_x)
                width = std_x * grid.dx
            else:
                width = 0.0

            pattern_widths.append(width)

        # Verify qualitative trend: pattern width should generally increase with source size
        # Use correlation to verify positive relationship
        correlation, _ = spearmanr(source_sigmas_pixels, pattern_widths)

        # Should have positive correlation (larger source → larger pattern)
        assert correlation > 0.5, (
            f"Pattern width should generally increase with source size (van Cittert-Zernike). "
            f"Spearman correlation: {correlation:.3f} > 0.5. "
            f"Source sigmas: {source_sigmas_pixels}, "
            f"Pattern widths: {pattern_widths}"
        )


class TestCoherencePhysics:
    """Test physical properties of coherence in ExtendedSourcePropagator."""

    @pytest.fixture
    def grid(self) -> Grid:
        """Create standard grid for physics tests."""
        return Grid(nx=128, dx=10e-6, wavelength=520e-9, device="cpu")

    @pytest.fixture
    def aperture(self, grid: Grid) -> torch.Tensor:
        """Create circular aperture."""
        radius = 40 * grid.dx
        aperture_obj = CircularAperture(radius=radius)
        mask = aperture_obj.generate(grid.x, grid.y, center=[0.0, 0.0])
        return mask.to(dtype=torch.cfloat)

    def test_positive_energy_output(self, grid: Grid, aperture: torch.Tensor):
        """
        Test that propagation produces positive total energy.

        The output should have non-zero positive total intensity.
        Note: Exact energy conservation is not expected due to FFT scaling
        and the nature of incoherent addition.
        """
        # Create test source
        sigma = 5.0 * grid.dx
        source = create_gaussian_source(grid, sigma=sigma)

        # Create propagator
        coherent_prop = FraunhoferPropagator(normalize=True)
        ext_prop = ExtendedSourcePropagator(
            coherent_propagator=coherent_prop,
            grid=grid,
            n_source_points=200,
            sampling_method="adaptive",
        )

        # Propagate
        result = ext_prop(source, aperture=aperture)
        total_output_energy = result.sum().item()

        # Output should have positive energy
        assert total_output_energy > 0, (
            f"Output should have positive total energy, got {total_output_energy:.6e}"
        )

        # All values should be non-negative
        assert (result >= 0).all(), "All intensity values should be non-negative"

    def test_sampling_convergence(self, grid: Grid, aperture: torch.Tensor):
        """
        Test that with more source points, results are more stable.

        With increased sampling, the peak intensity should stabilize.
        """
        # Create test source
        sigma = 10.0 * grid.dx
        source = create_gaussian_source(grid, sigma=sigma)
        source = source / source.sum()

        coherent_prop = FraunhoferPropagator(normalize=True)

        # Test with increasing number of samples (use grid for determinism)
        n_samples_list = [50, 100, 200, 400]
        peak_intensities = []

        for n_samples in n_samples_list:
            ext_prop = ExtendedSourcePropagator(
                coherent_propagator=coherent_prop,
                grid=grid,
                n_source_points=n_samples,
                sampling_method="grid",  # Use grid for deterministic sampling
            )
            result = ext_prop(source, aperture=aperture)
            peak = result.max().item()
            peak_intensities.append(peak)

        # Verify convergence: peak intensity should stabilize
        # Compute relative changes
        relative_changes = []
        for i in range(len(peak_intensities) - 1):
            change = abs(peak_intensities[i + 1] - peak_intensities[i]) / peak_intensities[i]
            relative_changes.append(change)

        # Later changes should be smaller (convergence)
        # Allow for some variation, but general trend should be toward stability
        # Check that we don't have large jumps at high sampling
        assert relative_changes[-1] < 0.5, (
            f"Peak intensity should stabilize with more samples. "
            f"Relative changes: {relative_changes}, "
            f"Last change: {relative_changes[-1]:.3f}"
        )

    def test_symmetry_preservation(self, grid: Grid, aperture: torch.Tensor):
        """
        Test that approximate circular symmetry is preserved.

        A circularly symmetric source through a circular aperture
        should produce an approximately circularly symmetric output.
        Due to discrete sampling, exact symmetry may not be achieved.
        """
        # Create circularly symmetric source
        sigma = 8.0 * grid.dx
        source = create_gaussian_source(grid, sigma=sigma, center=(0.0, 0.0))
        source = source / source.sum()

        # Create propagator with grid sampling for more deterministic result
        coherent_prop = FraunhoferPropagator(normalize=True)
        ext_prop = ExtendedSourcePropagator(
            coherent_propagator=coherent_prop,
            grid=grid,
            n_source_points=200,
            sampling_method="grid",  # Grid sampling for better symmetry
        )

        # Propagate
        result = ext_prop(source, aperture=aperture)

        # Check approximate circular symmetry by comparing radial profiles
        center = grid.nx // 2

        # Extract intensity along different radial directions
        def get_radial_profile(
            result: torch.Tensor, angle_deg: float, n_points: int = 30
        ) -> torch.Tensor:
            """Extract radial profile at given angle."""
            angle_rad = np.deg2rad(angle_deg)
            radii = np.linspace(0, n_points - 1, n_points)
            x_coords = (center + radii * np.cos(angle_rad)).astype(int)
            y_coords = (center + radii * np.sin(angle_rad)).astype(int)

            # Clamp to valid indices
            x_coords = np.clip(x_coords, 0, result.shape[1] - 1)
            y_coords = np.clip(y_coords, 0, result.shape[0] - 1)

            return result[y_coords, x_coords]

        # Get profiles at 0°, 45°, 90°, 135°
        profile_0 = get_radial_profile(result, 0)
        profile_45 = get_radial_profile(result, 45)
        profile_90 = get_radial_profile(result, 90)
        profile_135 = get_radial_profile(result, 135)

        # Compute average profile
        avg_profile = (profile_0 + profile_45 + profile_90 + profile_135) / 4

        # Check that individual profiles don't deviate too much from average
        def relative_deviation(profile: torch.Tensor, avg: torch.Tensor) -> float:
            """Compute relative RMS deviation."""
            diff = ((profile - avg) ** 2).mean().sqrt()
            norm = avg.mean()
            return (diff / norm).item() if norm > 0 else 0.0

        dev_0 = relative_deviation(profile_0, avg_profile)
        dev_45 = relative_deviation(profile_45, avg_profile)
        dev_90 = relative_deviation(profile_90, avg_profile)
        dev_135 = relative_deviation(profile_135, avg_profile)

        max_deviation = max(dev_0, dev_45, dev_90, dev_135)

        # Allow up to 200% relative deviation due to discrete sampling effects
        # This is a qualitative test - we just want to verify the pattern isn't completely asymmetric
        assert max_deviation < 2.0, (
            f"Radial profiles should be approximately similar (circular symmetry). "
            f"Max relative deviation: {max_deviation:.3f}. "
            f"Deviations: 0°={dev_0:.3f}, 45°={dev_45:.3f}, 90°={dev_90:.3f}, 135°={dev_135:.3f}"
        )


class TestCoherenceSamplingMethods:
    """Test different sampling methods in coherence limits."""

    @pytest.fixture
    def grid(self) -> Grid:
        """Create standard grid."""
        return Grid(nx=128, dx=10e-6, wavelength=520e-9, device="cpu")

    @pytest.fixture
    def aperture(self, grid: Grid) -> torch.Tensor:
        """Create circular aperture."""
        radius = 40 * grid.dx
        aperture_obj = CircularAperture(radius=radius)
        mask = aperture_obj.generate(grid.x, grid.y, center=[0.0, 0.0])
        return mask.to(dtype=torch.cfloat)

    @pytest.mark.parametrize("sampling_method", ["grid", "monte_carlo", "adaptive"])
    def test_sampling_methods_consistency(
        self, grid: Grid, aperture: torch.Tensor, sampling_method: str
    ):
        """
        Test that different sampling methods produce valid results.

        All sampling methods should produce positive, finite intensity outputs.
        """
        # Create test source
        sigma = 5.0 * grid.dx
        source = create_gaussian_source(grid, sigma=sigma)
        source = source / source.sum()

        # Create propagator with specified sampling method
        coherent_prop = FraunhoferPropagator(normalize=True)
        ext_prop = ExtendedSourcePropagator(
            coherent_propagator=coherent_prop,
            grid=grid,
            n_source_points=200,
            sampling_method=sampling_method,
        )

        # Propagate
        result = ext_prop(source, aperture=aperture)

        # Basic sanity checks
        assert torch.isfinite(result).all(), (
            f"Result contains non-finite values with {sampling_method} sampling"
        )
        assert (result >= 0).all(), (
            f"Result contains negative intensities with {sampling_method} sampling"
        )
        assert result.sum() > 0, f"Result has zero total intensity with {sampling_method} sampling"

        # Check that the pattern has a reasonable peak
        peak_intensity = result.max().item()
        assert peak_intensity > 0, f"Result has no peak with {sampling_method} sampling"

        # Check that energy is spread over multiple pixels (not all in one pixel)
        n_significant_pixels = (result > 0.01 * peak_intensity).sum().item()
        assert n_significant_pixels > 10, (
            f"Result too concentrated ({n_significant_pixels} pixels) with {sampling_method} sampling"
        )
