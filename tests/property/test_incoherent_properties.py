"""
Property-based tests for incoherent propagators using hypothesis.

These tests verify physical properties that should hold for incoherent propagation:
- Energy conservation (intensity sum preserved)
- Linearity (prop(a + b) = prop(a) + prop(b))
- Non-negativity (output intensity >= 0)
- Scaling (prop(c * I) = c * prop(I) for c > 0)
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from prism.core.grid import Grid
from prism.core.propagators import (
    ExtendedSourcePropagator,
    FraunhoferPropagator,
    OTFPropagator,
    create_stellar_disk,
)


class TestOTFPropagatorProperties:
    """Property-based tests for OTF propagator."""

    @pytest.fixture
    def circular_aperture(self) -> torch.Tensor:
        """Create a fixed circular aperture for testing."""
        n = 64
        x = torch.linspace(-1, 1, n)
        y = torch.linspace(-1, 1, n)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        r = torch.sqrt(xx**2 + yy**2)
        return (r <= 0.5).to(torch.cfloat)

    @pytest.fixture
    def propagator(self, circular_aperture: torch.Tensor) -> OTFPropagator:
        """Create OTF propagator."""
        return OTFPropagator(circular_aperture, normalize=True)

    @given(scale=st.floats(min_value=0.1, max_value=10.0))
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_scaling_property(self, propagator: OTFPropagator, scale: float) -> None:
        """Property: prop(c * I) = c * prop(I) for positive c."""
        intensity = torch.rand(64, 64)

        scaled_prop = propagator(scale * intensity)
        prop_scaled = scale * propagator(intensity)

        assert torch.allclose(scaled_prop, prop_scaled, rtol=1e-4)

    @given(seed=st.integers(min_value=1, max_value=10000))
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_output_non_negative(self, propagator: OTFPropagator, seed: int) -> None:
        """Property: Output is always non-negative for any valid input."""
        torch.manual_seed(seed)
        intensity = torch.rand(64, 64)

        output = propagator(intensity)

        assert (output >= 0).all()

    @given(seed=st.integers(min_value=1, max_value=10000))
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_energy_conservation(self, propagator: OTFPropagator, seed: int) -> None:
        """Property: Total intensity (energy) is conserved."""
        torch.manual_seed(seed)
        intensity = torch.rand(64, 64)

        output = propagator(intensity)

        input_energy = intensity.sum()
        output_energy = output.sum()

        assert torch.isclose(input_energy, output_energy, rtol=0.01)

    @given(seed=st.integers(min_value=1, max_value=10000))
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_linearity(self, propagator: OTFPropagator, seed: int) -> None:
        """Property: prop(a + b) = prop(a) + prop(b) (linearity)."""
        torch.manual_seed(seed)
        a = torch.rand(64, 64)
        b = torch.rand(64, 64)

        prop_sum = propagator(a + b)
        sum_prop = propagator(a) + propagator(b)

        assert torch.allclose(prop_sum, sum_prop, rtol=1e-4)

    @given(seed=st.integers(min_value=1, max_value=10000))
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_gradient_flow(self, propagator: OTFPropagator, seed: int) -> None:
        """Property: Gradients should flow through the propagator."""
        torch.manual_seed(seed)
        intensity = torch.rand(64, 64, requires_grad=True)

        output = propagator(intensity)
        loss = output.sum()
        loss.backward()

        assert intensity.grad is not None
        assert not torch.isnan(intensity.grad).any()
        assert not torch.isinf(intensity.grad).any()


@given(
    n=st.integers(min_value=32, max_value=128),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
)
@settings(max_examples=20, deadline=None)
def test_otf_energy_conservation_parametric(n: int, wavelength: float) -> None:
    """Property: OTF propagation should conserve energy for various grid sizes."""
    # Create grid
    dx = 1e-5
    grid = Grid(nx=n, dx=dx, wavelength=wavelength)

    # Create circular aperture
    x, y = grid.x, grid.y
    r = torch.sqrt(x**2 + y**2)
    radius = 0.3 * n * dx
    aperture = (r <= radius).to(torch.cfloat)

    # Create propagator
    propagator = OTFPropagator(aperture, grid, normalize=True)

    # Create random intensity
    intensity = torch.rand(n, n)

    # Propagate
    output = propagator(intensity)

    # Check energy conservation
    input_energy = intensity.sum()
    output_energy = output.sum()

    assert torch.isclose(input_energy, output_energy, rtol=0.01)


@given(
    n=st.integers(min_value=32, max_value=128),
)
@settings(max_examples=20, deadline=None)
def test_otf_linearity_parametric(n: int) -> None:
    """Property: OTF propagation should be linear for various grid sizes."""
    # Create aperture
    aperture = torch.ones(n, n, dtype=torch.cfloat)

    # Create propagator
    propagator = OTFPropagator(aperture, normalize=True)

    # Create two random intensity distributions
    a = torch.rand(n, n)
    b = torch.rand(n, n)

    # Propagate sum
    prop_sum = propagator(a + b)

    # Propagate individually and sum
    sum_prop = propagator(a) + propagator(b)

    # Should be equal (linearity)
    assert torch.allclose(prop_sum, sum_prop, rtol=1e-4)


@given(
    n=st.integers(min_value=32, max_value=128),
    alpha=st.floats(min_value=0.1, max_value=5.0),  # Reduced range to avoid numerical issues
)
@settings(max_examples=20, deadline=None)
def test_otf_scaling_parametric(n: int, alpha: float) -> None:
    """Property: OTF propagation should scale linearly."""
    # Create aperture
    aperture = torch.ones(n, n, dtype=torch.cfloat)

    # Create propagator
    propagator = OTFPropagator(aperture, normalize=True)

    # Create random intensity
    intensity = torch.rand(n, n)

    # Propagate scaled
    prop_scaled = propagator(alpha * intensity)

    # Scale propagated
    scaled_prop = alpha * propagator(intensity)

    # Should be equal (allow tolerance for floating point operations)
    assert torch.allclose(prop_scaled, scaled_prop, rtol=1e-3, atol=1e-6)


class TestExtendedSourcePropagatorProperties:
    """Property-based tests for ExtendedSourcePropagator."""

    @pytest.fixture
    def grid(self) -> Grid:
        """Create test grid."""
        return Grid(nx=64, dx=1e-5, wavelength=550e-9)

    @pytest.fixture
    def propagator(self, grid: Grid) -> ExtendedSourcePropagator:
        """Create extended source propagator."""
        return ExtendedSourcePropagator(
            coherent_propagator=FraunhoferPropagator(),
            grid=grid,
            n_source_points=50,  # Lower for faster tests
            sampling_method="grid",  # Deterministic sampling
        )

    @given(seed=st.integers(min_value=1, max_value=10000))
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_output_non_negative(
        self, propagator: ExtendedSourcePropagator, grid: Grid, seed: int
    ) -> None:
        """Property: Output is always non-negative."""
        torch.manual_seed(seed)
        source = torch.rand(grid.nx, grid.ny)
        source = source / source.sum()

        output = propagator(source)

        assert (output >= 0).all()

    @given(seed=st.integers(min_value=1, max_value=10000))
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_positive_energy(
        self, propagator: ExtendedSourcePropagator, grid: Grid, seed: int
    ) -> None:
        """Property: Output has positive total energy."""
        torch.manual_seed(seed)
        source = torch.rand(grid.nx, grid.ny)
        source = source / source.sum()

        output = propagator(source)

        assert output.sum() > 0

    @given(scale=st.floats(min_value=0.5, max_value=2.0))
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_output_shape_invariant(
        self, propagator: ExtendedSourcePropagator, grid: Grid, scale: float
    ) -> None:
        """Property: Output shape should be same regardless of input scaling."""
        source = create_stellar_disk(grid, angular_diameter=10 * grid.dx)

        # Propagate original and scaled
        output_original = propagator(source)
        output_scaled = propagator(scale * source)

        # Shape should always be the same
        assert output_original.shape == output_scaled.shape
        # Both outputs should be non-negative
        assert (output_original >= 0).all()
        assert (output_scaled >= 0).all()
        # Both should have positive energy
        assert output_original.sum() > 0
        assert output_scaled.sum() > 0


@given(
    n_source_points=st.integers(min_value=20, max_value=200),
)
@settings(max_examples=10, deadline=None)
def test_extended_source_sample_count(n_source_points: int) -> None:
    """Property: Propagator should use approximately the requested number of samples."""
    grid = Grid(nx=64, dx=1e-5, wavelength=550e-9)

    propagator = ExtendedSourcePropagator(
        coherent_propagator=FraunhoferPropagator(),
        grid=grid,
        n_source_points=n_source_points,
        sampling_method="grid",
    )

    source = create_stellar_disk(grid, angular_diameter=10 * grid.dx)
    _, diagnostics = propagator(source, return_diagnostics=True)

    # For grid sampling, the actual count is approximately n_per_dim^2 where
    # n_per_dim = sqrt(n_source_points)
    # Allow some tolerance
    actual = diagnostics["n_samples"]
    expected_min = int(n_source_points * 0.5)
    expected_max = int(n_source_points * 1.5)

    assert expected_min <= actual <= expected_max


class TestIncoherentVsCoherentProperties:
    """Tests comparing properties of incoherent vs coherent propagation."""

    @pytest.fixture
    def grid(self) -> Grid:
        """Create test grid."""
        return Grid(nx=64, dx=1e-5, wavelength=550e-9)

    @pytest.fixture
    def aperture(self, grid: Grid) -> torch.Tensor:
        """Create circular aperture."""
        x, y = grid.x, grid.y
        r = torch.sqrt(x**2 + y**2)
        return (r <= 20 * grid.dx).to(torch.cfloat)

    def test_incoherent_mode_correct(self, aperture: torch.Tensor, grid: Grid) -> None:
        """Test that OTFPropagator reports incoherent mode."""
        prop = OTFPropagator(aperture, grid)
        assert prop.illumination_mode == "incoherent"

    def test_extended_source_mode_correct(self, grid: Grid) -> None:
        """Test that ExtendedSourcePropagator reports partially_coherent mode."""
        prop = ExtendedSourcePropagator(
            coherent_propagator=FraunhoferPropagator(),
            grid=grid,
        )
        assert prop.illumination_mode == "partially_coherent"

    def test_coherent_mode_correct(self) -> None:
        """Test that FraunhoferPropagator reports coherent mode."""
        prop = FraunhoferPropagator()
        assert prop.illumination_mode == "coherent"


class TestOTFSpecificProperties:
    """Tests for OTF-specific properties."""

    @given(
        n=st.integers(min_value=32, max_value=128),
    )
    @settings(max_examples=10, deadline=None)
    def test_mtf_non_negative(self, n: int) -> None:
        """Property: MTF (modulus of OTF) should always be non-negative."""
        aperture = torch.ones(n, n, dtype=torch.cfloat)
        propagator = OTFPropagator(aperture, normalize=True)

        mtf = propagator.get_mtf()

        assert (mtf >= 0).all()

    @given(
        n=st.integers(min_value=32, max_value=128),
    )
    @settings(max_examples=10, deadline=None)
    def test_otf_dc_normalized(self, n: int) -> None:
        """Property: OTF DC component should be 1 when normalized."""
        aperture = torch.ones(n, n, dtype=torch.cfloat)
        propagator = OTFPropagator(aperture, normalize=True)

        h, w = propagator.otf.shape
        dc = propagator.otf[h // 2, w // 2]

        assert torch.isclose(dc, torch.tensor(1.0), atol=1e-5)

    @given(
        n=st.integers(min_value=32, max_value=128),
    )
    @settings(max_examples=10, deadline=None)
    def test_otf_real_for_symmetric_aperture(self, n: int) -> None:
        """Property: OTF should be real for centrosymmetric aperture."""
        # Uniform aperture is centrosymmetric
        aperture = torch.ones(n, n, dtype=torch.cfloat)
        propagator = OTFPropagator(aperture, normalize=True)

        # OTF should be real (not complex)
        assert not propagator.otf.is_complex()


class TestBatchProcessingProperties:
    """Tests for batch processing properties."""

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        n=st.integers(min_value=32, max_value=64),
    )
    @settings(max_examples=10, deadline=None)
    def test_otf_batched_output_shape(self, batch_size: int, n: int) -> None:
        """Property: Batched input should produce batched output with same batch size."""
        aperture = torch.ones(n, n, dtype=torch.cfloat)
        propagator = OTFPropagator(aperture, normalize=True)

        # Batched input
        intensity = torch.rand(batch_size, n, n)
        output = propagator(intensity)

        assert output.shape == (batch_size, n, n)

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=10, deadline=None)
    def test_otf_batch_consistency(self, batch_size: int) -> None:
        """Property: Batched processing should give same result as individual processing."""
        n = 32
        aperture = torch.ones(n, n, dtype=torch.cfloat)
        propagator = OTFPropagator(aperture, normalize=True)

        # Process batch
        batch_intensity = torch.rand(batch_size, n, n)
        batch_output = propagator(batch_intensity)

        # Process individually
        individual_outputs = torch.stack(
            [propagator(batch_intensity[i]) for i in range(batch_size)]
        )

        assert torch.allclose(batch_output, individual_outputs, rtol=1e-5)
