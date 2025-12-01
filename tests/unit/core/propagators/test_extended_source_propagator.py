"""Unit tests for ExtendedSourcePropagator."""

from __future__ import annotations

import pytest
import torch

from prism.core.grid import Grid
from prism.core.propagators import (
    ExtendedSourcePropagator,
    FraunhoferPropagator,
    create_binary_source,
    create_gaussian_source,
    create_propagator,
    create_ring_source,
    create_stellar_disk,
    estimate_required_samples,
    select_propagator,
)


class TestExtendedSourcePropagator:
    """Tests for ExtendedSourcePropagator."""

    @pytest.fixture
    def grid(self) -> Grid:
        """Create test grid."""
        return Grid(nx=64, dx=1e-5, wavelength=550e-9)

    @pytest.fixture
    def coherent_propagator(self) -> FraunhoferPropagator:
        """Create coherent propagator."""
        return FraunhoferPropagator()

    @pytest.fixture
    def propagator(
        self, coherent_propagator: FraunhoferPropagator, grid: Grid
    ) -> ExtendedSourcePropagator:
        """Create extended source propagator."""
        return ExtendedSourcePropagator(
            coherent_propagator=coherent_propagator,
            grid=grid,
            n_source_points=100,
            sampling_method="adaptive",
        )

    def test_initialization(self, propagator: ExtendedSourcePropagator) -> None:
        """Test propagator initializes correctly."""
        assert propagator.n_source_points == 100
        assert propagator.sampling_method == "adaptive"
        assert propagator.illumination_mode == "partially_coherent"

    def test_forward_with_uniform_disk(
        self, propagator: ExtendedSourcePropagator, grid: Grid
    ) -> None:
        """Test propagation of uniform disk source."""
        source = create_stellar_disk(grid, angular_diameter=10 * grid.dx)
        output = propagator(source)

        assert output.shape == source.shape
        assert not output.is_complex()
        assert (output >= 0).all()

    def test_forward_with_binary(self, propagator: ExtendedSourcePropagator, grid: Grid) -> None:
        """Test propagation of binary source."""
        source = create_binary_source(grid, separation=20 * grid.dx)
        output = propagator(source)

        assert output.shape == source.shape
        assert output.sum() > 0

    def test_forward_with_aperture(self, propagator: ExtendedSourcePropagator, grid: Grid) -> None:
        """Test propagation with aperture function."""
        source = create_stellar_disk(grid, angular_diameter=10 * grid.dx)

        # Create circular aperture
        x, y = grid.x, grid.y
        r = torch.sqrt(x**2 + y**2)
        aperture = (r <= 20 * grid.dx).to(torch.cfloat)

        output = propagator(source, aperture=aperture)
        assert output.shape == source.shape

    def test_sampling_methods(self, coherent_propagator: FraunhoferPropagator, grid: Grid) -> None:
        """Test all sampling methods work."""
        source = create_stellar_disk(grid, angular_diameter=10 * grid.dx)

        for method in ["grid", "monte_carlo", "adaptive"]:
            prop = ExtendedSourcePropagator(
                coherent_propagator=coherent_propagator,
                grid=grid,
                n_source_points=50,
                sampling_method=method,  # type: ignore[arg-type]
            )
            output = prop(source)
            assert output.shape == source.shape

    def test_diagnostics(self, propagator: ExtendedSourcePropagator, grid: Grid) -> None:
        """Test diagnostic output."""
        source = create_stellar_disk(grid, angular_diameter=10 * grid.dx)
        output, diagnostics = propagator(source, return_diagnostics=True)

        assert "n_samples" in diagnostics
        assert "positions" in diagnostics
        assert "weights" in diagnostics
        # n_samples may be slightly less than n_source_points due to grid rounding
        assert diagnostics["n_samples"] >= propagator.n_source_points - 10

    def test_batch_processing(self, coherent_propagator: FraunhoferPropagator, grid: Grid) -> None:
        """Test batch size affects memory but not result (with same seed)."""
        source = create_stellar_disk(grid, angular_diameter=10 * grid.dx)

        prop_small_batch = ExtendedSourcePropagator(
            coherent_propagator=coherent_propagator,
            grid=grid,
            n_source_points=100,
            batch_size=10,
            sampling_method="grid",  # Use grid for deterministic sampling
        )
        prop_large_batch = ExtendedSourcePropagator(
            coherent_propagator=coherent_propagator,
            grid=grid,
            n_source_points=100,
            batch_size=100,
            sampling_method="grid",
        )

        out1 = prop_small_batch(source)
        out2 = prop_large_batch(source)

        # Results should be identical for grid sampling
        assert torch.allclose(out1, out2, rtol=1e-4)

    def test_output_has_positive_energy(
        self, propagator: ExtendedSourcePropagator, grid: Grid
    ) -> None:
        """Test output has positive total energy (not conservation, due to FFT scaling)."""
        source = create_stellar_disk(grid, angular_diameter=10 * grid.dx)
        output = propagator(source)

        # Output should have positive energy (exact ratio depends on FFT scaling)
        assert output.sum() > 0

    def test_output_non_negative(self, propagator: ExtendedSourcePropagator, grid: Grid) -> None:
        """Test output is always non-negative."""
        source = create_stellar_disk(grid, angular_diameter=10 * grid.dx)
        output = propagator(source)

        assert (output >= 0).all()

    def test_rejects_complex_input(self, propagator: ExtendedSourcePropagator, grid: Grid) -> None:
        """Test propagator rejects complex input."""
        complex_field = torch.randn(grid.nx, grid.ny, dtype=torch.cfloat)

        with pytest.raises(ValueError, match="real"):
            propagator(complex_field)

    def test_sampling_diagnostics(self, propagator: ExtendedSourcePropagator, grid: Grid) -> None:
        """Test get_sampling_diagnostics method."""
        source = create_stellar_disk(grid, angular_diameter=10 * grid.dx)
        diagnostics = propagator.get_sampling_diagnostics(source)

        assert "n_samples" in diagnostics
        assert "positions" in diagnostics
        assert "weights" in diagnostics
        assert "coverage" in diagnostics
        assert "sampling_method" in diagnostics
        # n_samples may be slightly less than n_source_points due to grid rounding
        assert diagnostics["n_samples"] >= propagator.n_source_points - 10

    def test_set_coherence_from_source(self, propagator: ExtendedSourcePropagator) -> None:
        """Test van Cittert-Zernike coherence estimation."""
        # Set coherence from source
        propagator.set_coherence_from_source(source_angular_diameter=1e-6, propagation_distance=1e6)
        assert propagator.coherent_patch_size is not None
        assert propagator.coherent_patch_size > 0

        # Zero angular diameter should give None (fully coherent)
        propagator.set_coherence_from_source(source_angular_diameter=0, propagation_distance=1e6)
        assert propagator.coherent_patch_size is None

    def test_clear_cache(self, propagator: ExtendedSourcePropagator) -> None:
        """Test cache clearing."""
        # Just verify it doesn't raise
        propagator.clear_cache()
        assert propagator._psf_cache == {}


class TestSourceGeometryHelpers:
    """Tests for source geometry helper functions."""

    @pytest.fixture
    def grid(self) -> Grid:
        return Grid(nx=64, dx=1e-5, wavelength=550e-9)

    def test_stellar_disk_shape(self, grid: Grid) -> None:
        """Test stellar disk has correct shape."""
        disk = create_stellar_disk(grid, angular_diameter=10 * grid.dx)
        assert disk.shape == (grid.nx, grid.ny)

    def test_stellar_disk_normalized(self, grid: Grid) -> None:
        """Test stellar disk sums to 1."""
        disk = create_stellar_disk(grid, angular_diameter=10 * grid.dx)
        assert torch.isclose(disk.sum(), torch.tensor(1.0), atol=1e-5)

    def test_stellar_disk_limb_darkening(self, grid: Grid) -> None:
        """Test limb darkening reduces edge intensity."""
        disk_uniform = create_stellar_disk(grid, angular_diameter=20 * grid.dx, limb_darkening=0.0)
        disk_darkened = create_stellar_disk(grid, angular_diameter=20 * grid.dx, limb_darkening=0.6)

        # Center should be brighter relative to edge in darkened disk
        center = grid.nx // 2
        edge_offset = 8  # Near edge of disk

        uniform_ratio = disk_uniform[center, center] / disk_uniform[center, center + edge_offset]
        darkened_ratio = disk_darkened[center, center] / disk_darkened[center, center + edge_offset]

        assert darkened_ratio > uniform_ratio

    def test_gaussian_source_shape(self, grid: Grid) -> None:
        """Test Gaussian source has correct shape."""
        source = create_gaussian_source(grid, sigma=5 * grid.dx)
        assert source.shape == (grid.nx, grid.ny)

    def test_gaussian_source_centered(self, grid: Grid) -> None:
        """Test Gaussian source is centered by default."""
        source = create_gaussian_source(grid, sigma=5 * grid.dx)
        center = grid.nx // 2
        assert source[center, center] == source.max()

    def test_gaussian_source_custom_center(self, grid: Grid) -> None:
        """Test Gaussian source with custom center."""
        offset = 5 * grid.dx
        source = create_gaussian_source(grid, sigma=5 * grid.dx, center=(offset, offset))
        # Peak should not be at grid center
        center = grid.nx // 2
        assert source[center, center] < source.max()

    def test_binary_source_normalized(self, grid: Grid) -> None:
        """Test binary source sums to 1."""
        binary = create_binary_source(grid, separation=20 * grid.dx)
        assert torch.isclose(binary.sum(), torch.tensor(1.0), atol=1e-5)

    def test_binary_source_two_peaks(self, grid: Grid) -> None:
        """Test binary source has two distinct peaks."""
        binary = create_binary_source(grid, separation=20 * grid.dx)

        # Find peaks
        center = grid.nx // 2
        left_region = binary[center, : center - 5]
        right_region = binary[center, center + 5 :]

        # Both regions should have significant intensity
        assert left_region.max() > 0.01 * binary.max()
        assert right_region.max() > 0.01 * binary.max()

    def test_binary_source_flux_ratio(self, grid: Grid) -> None:
        """Test binary source respects flux ratio."""
        binary_equal = create_binary_source(grid, separation=20 * grid.dx, flux_ratio=1.0)
        binary_unequal = create_binary_source(grid, separation=20 * grid.dx, flux_ratio=0.5)

        # With flux_ratio=0.5, peaks should be less equal
        # Find the two peak values
        center = grid.nx // 2
        peak1_equal = binary_equal[center, center - 10]
        peak2_equal = binary_equal[center, center + 10]
        peak1_unequal = binary_unequal[center, center - 10]
        peak2_unequal = binary_unequal[center, center + 10]

        # Equal flux ratio should have similar peaks
        equal_ratio = peak1_equal / peak2_equal
        unequal_ratio = peak1_unequal / peak2_unequal

        # The difference should be visible
        assert abs(equal_ratio - 1.0) < abs(unequal_ratio - 1.0)

    def test_ring_source_shape(self, grid: Grid) -> None:
        """Test ring source has correct shape."""
        ring = create_ring_source(grid, inner_radius=5 * grid.dx, outer_radius=10 * grid.dx)
        assert ring.shape == (grid.nx, grid.ny)

    def test_ring_source_normalized(self, grid: Grid) -> None:
        """Test ring source sums to 1."""
        ring = create_ring_source(grid, inner_radius=5 * grid.dx, outer_radius=10 * grid.dx)
        assert torch.isclose(ring.sum(), torch.tensor(1.0), atol=1e-5)

    def test_ring_source_has_hole(self, grid: Grid) -> None:
        """Test ring source has zero intensity at center."""
        ring = create_ring_source(grid, inner_radius=5 * grid.dx, outer_radius=10 * grid.dx)
        center = grid.nx // 2
        # Center should be in the hole (near zero)
        assert ring[center, center] < 1e-6

    def test_estimate_required_samples_returns_int(self, grid: Grid) -> None:
        """Test estimate_required_samples returns integer."""
        source = create_stellar_disk(grid, angular_diameter=10 * grid.dx)
        n_samples = estimate_required_samples(source, grid)
        assert isinstance(n_samples, int)
        assert n_samples >= 100

    def test_estimate_required_samples_increases_with_snr(self, grid: Grid) -> None:
        """Test higher SNR requires more samples."""
        source = create_stellar_disk(grid, angular_diameter=10 * grid.dx)
        n_low = estimate_required_samples(source, grid, target_snr=50)
        n_high = estimate_required_samples(source, grid, target_snr=200)
        assert n_high >= n_low


class TestFactoryIntegration:
    """Test factory function integration."""

    @pytest.fixture
    def grid(self) -> Grid:
        return Grid(nx=64, dx=1e-5, wavelength=550e-9)

    def test_create_extended_source_propagator(self, grid: Grid) -> None:
        """Test creating via factory function."""
        prop = create_propagator(
            "extended_source",
            grid=grid,
            n_source_points=100,
        )

        assert isinstance(prop, ExtendedSourcePropagator)
        assert prop.n_source_points == 100

    def test_create_extended_source_requires_grid(self) -> None:
        """Test extended_source creation fails without grid."""
        with pytest.raises(ValueError, match="Grid required"):
            create_propagator("extended_source")

    def test_create_extended_source_with_custom_coherent_prop(self, grid: Grid) -> None:
        """Test creating with custom coherent propagator."""
        coherent_prop = FraunhoferPropagator(normalize=False)
        prop = create_propagator(
            "extended_source",
            grid=grid,
            coherent_propagator=coherent_prop,
        )

        assert isinstance(prop, ExtendedSourcePropagator)
        assert prop.coherent_propagator is coherent_prop

    def test_select_propagator_partially_coherent(self, grid: Grid) -> None:
        """Test select_propagator with partially coherent illumination."""
        prop = select_propagator(
            wavelength=550e-9,
            obj_distance=1e6,
            fov=64 * 1e-5,
            illumination="partially_coherent",
            grid=grid,
        )

        assert isinstance(prop, ExtendedSourcePropagator)

    def test_select_propagator_extended_source_method(self, grid: Grid) -> None:
        """Test select_propagator with extended_source method."""
        prop = select_propagator(
            wavelength=550e-9,
            obj_distance=1e6,
            fov=64 * 1e-5,
            method="extended_source",
            grid=grid,
        )

        assert isinstance(prop, ExtendedSourcePropagator)


class TestExtendedSourceIntegration:
    """Integration tests for extended source propagation."""

    @pytest.fixture
    def grid(self) -> Grid:
        return Grid(nx=64, dx=1e-5, wavelength=550e-9)

    def test_full_propagation_pipeline(self, grid: Grid) -> None:
        """Test complete propagation pipeline with stellar disk."""
        # Create source
        source = create_stellar_disk(grid, angular_diameter=10 * grid.dx, limb_darkening=0.3)

        # Create aperture
        x, y = grid.x, grid.y
        r = torch.sqrt(x**2 + y**2)
        aperture = (r <= 20 * grid.dx).to(torch.cfloat)

        # Create propagator
        prop = ExtendedSourcePropagator(
            coherent_propagator=FraunhoferPropagator(),
            grid=grid,
            n_source_points=100,
            sampling_method="adaptive",
        )

        # Propagate
        output, diagnostics = prop(source, aperture=aperture, return_diagnostics=True)

        # Verify output
        assert output.shape == source.shape
        assert (output >= 0).all()
        # n_samples may be slightly less due to grid rounding
        assert diagnostics["n_samples"] >= 90

    def test_binary_vs_single_source(self, grid: Grid) -> None:
        """Test binary source produces different result than single source."""
        single = create_gaussian_source(grid, sigma=3 * grid.dx)
        single = single / single.sum()

        binary = create_binary_source(grid, separation=15 * grid.dx)

        prop = ExtendedSourcePropagator(
            coherent_propagator=FraunhoferPropagator(),
            grid=grid,
            n_source_points=100,
            sampling_method="grid",
        )

        out_single = prop(single)
        out_binary = prop(binary)

        # Outputs should be different
        diff = (out_single - out_binary).abs().sum()
        assert diff > 0.01

    def test_limb_darkening_effect(self, grid: Grid) -> None:
        """Test limb darkening affects propagation result."""
        disk_uniform = create_stellar_disk(grid, angular_diameter=15 * grid.dx, limb_darkening=0.0)
        disk_darkened = create_stellar_disk(grid, angular_diameter=15 * grid.dx, limb_darkening=0.8)

        prop = ExtendedSourcePropagator(
            coherent_propagator=FraunhoferPropagator(),
            grid=grid,
            n_source_points=100,
            sampling_method="grid",
        )

        out_uniform = prop(disk_uniform)
        out_darkened = prop(disk_darkened)

        # Outputs should be different
        diff = (out_uniform - out_darkened).abs().sum()
        assert diff > 0.001
