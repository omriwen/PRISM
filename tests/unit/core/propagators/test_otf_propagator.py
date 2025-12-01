"""Unit tests for OTFPropagator (incoherent illumination)."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from prism.core.grid import Grid
from prism.core.propagators import OTFPropagator


class TestOTFPropagator:
    """Tests for OTFPropagator."""

    @pytest.fixture
    def grid(self) -> Grid:
        """Create test grid."""
        return Grid(nx=64, dx=1e-5, wavelength=550e-9)

    @pytest.fixture
    def circular_aperture(self, grid: Grid) -> Tensor:
        """Create circular aperture."""
        x, y = grid.x, grid.y
        r = torch.sqrt(x**2 + y**2)
        radius = 20 * grid.dx  # 20 pixel radius
        return (r <= radius).to(torch.cfloat)

    @pytest.fixture
    def propagator(self, circular_aperture: Tensor, grid: Grid) -> OTFPropagator:
        """Create OTF propagator."""
        return OTFPropagator(circular_aperture, grid)

    def test_initialization(self, propagator: OTFPropagator):
        """Test propagator initializes correctly."""
        assert propagator.otf is not None
        assert propagator.illumination_mode == "incoherent"

    def test_otf_shape(self, propagator: OTFPropagator, grid: Grid):
        """Test OTF has correct shape."""
        assert propagator.otf.shape == (grid.nx, grid.ny)

    def test_otf_is_real(self, propagator: OTFPropagator):
        """Test OTF is real-valued (for symmetric aperture)."""
        assert not propagator.otf.is_complex()

    def test_otf_dc_normalized(self, propagator: OTFPropagator):
        """Test OTF DC component is 1."""
        h, w = propagator.otf.shape
        dc = propagator.otf[h // 2, w // 2]
        assert torch.isclose(dc, torch.tensor(1.0), atol=1e-5)

    def test_forward_real_input(self, propagator: OTFPropagator, grid: Grid):
        """Test forward pass with real intensity input."""
        intensity = torch.rand(grid.nx, grid.ny)
        output = propagator(intensity)

        assert output.shape == intensity.shape
        assert not output.is_complex()

    def test_forward_rejects_complex(self, propagator: OTFPropagator, grid: Grid):
        """Test forward pass rejects complex input."""
        complex_field = torch.rand(grid.nx, grid.ny, dtype=torch.cfloat)

        with pytest.raises(ValueError, match="real intensity input"):
            propagator(complex_field)

    def test_output_non_negative(self, propagator: OTFPropagator, grid: Grid):
        """Test output is non-negative."""
        intensity = torch.rand(grid.nx, grid.ny)
        output = propagator(intensity)

        assert (output >= 0).all()

    def test_energy_conservation(self, propagator: OTFPropagator, grid: Grid):
        """Test total intensity is approximately conserved."""
        intensity = torch.rand(grid.nx, grid.ny)
        output = propagator(intensity)

        input_energy = intensity.sum()
        output_energy = output.sum()

        # Energy should be conserved within tolerance
        assert torch.isclose(input_energy, output_energy, rtol=0.01)

    def test_linearity(self, propagator: OTFPropagator, grid: Grid):
        """Test linearity: prop(a + b) = prop(a) + prop(b)."""
        a = torch.rand(grid.nx, grid.ny)
        b = torch.rand(grid.nx, grid.ny)

        prop_sum = propagator(a + b)
        sum_prop = propagator(a) + propagator(b)

        assert torch.allclose(prop_sum, sum_prop, rtol=1e-4)

    def test_gradient_flow(self, propagator: OTFPropagator, grid: Grid):
        """Test gradients flow through propagator."""
        intensity = torch.rand(grid.nx, grid.ny, requires_grad=True)
        output = propagator(intensity)
        loss = output.sum()
        loss.backward()

        assert intensity.grad is not None
        assert not torch.isnan(intensity.grad).any()

    def test_mtf_is_non_negative(self, propagator: OTFPropagator):
        """Test MTF (magnitude of OTF) is non-negative."""
        mtf = propagator.get_mtf()
        assert (mtf >= 0).all()

    def test_delta_function_gives_psf(self, propagator: OTFPropagator, grid: Grid):
        """Test that propagating delta gives PSF-like response."""
        # Create delta function at center
        delta = torch.zeros(grid.nx, grid.ny)
        delta[grid.nx // 2, grid.ny // 2] = 1.0

        output = propagator(delta)

        # Output should be centered and spread out (PSF)
        assert output.max() < 1.0  # Spread reduces peak
        assert output.sum() > 0  # Energy preserved


class TestOTFPropagatorEdgeCases:
    """Edge case tests for OTFPropagator."""

    def test_uniform_aperture(self):
        """Test with uniform (all-ones) aperture."""
        grid = Grid(nx=32, dx=1e-5, wavelength=550e-9)
        aperture = torch.ones(32, 32, dtype=torch.cfloat)
        propagator = OTFPropagator(aperture, grid)

        intensity = torch.rand(32, 32)
        output = propagator(intensity)

        assert output.shape == intensity.shape
        assert not torch.isnan(output).any()

    def test_no_grid_provided(self):
        """Test initialization without grid."""
        aperture = torch.ones(32, 32, dtype=torch.cfloat)
        propagator = OTFPropagator(aperture, grid=None)

        assert propagator.otf is not None

    def test_normalize_false(self):
        """Test with normalize=False."""
        aperture = torch.ones(32, 32, dtype=torch.cfloat)
        propagator = OTFPropagator(aperture, normalize=False)

        # Should still work
        intensity = torch.rand(32, 32)
        output = propagator(intensity)
        assert not torch.isnan(output).any()

    def test_batched_input(self):
        """Test with batched input."""
        grid = Grid(nx=32, dx=1e-5, wavelength=550e-9)
        aperture = torch.ones(32, 32, dtype=torch.cfloat)
        propagator = OTFPropagator(aperture, grid)

        # Batched input
        batch_intensity = torch.rand(4, 32, 32)
        output = propagator(batch_intensity)

        assert output.shape == batch_intensity.shape


class TestIlluminationModeProperty:
    """Test illumination_mode property across propagators."""

    def test_otf_is_incoherent(self):
        """Test OTFPropagator reports incoherent mode."""
        aperture = torch.ones(32, 32, dtype=torch.cfloat)
        propagator = OTFPropagator(aperture)
        assert propagator.illumination_mode == "incoherent"

    def test_fraunhofer_is_coherent(self):
        """Test FraunhoferPropagator reports coherent mode."""
        from prism.core.propagators import FraunhoferPropagator

        propagator = FraunhoferPropagator()
        assert propagator.illumination_mode == "coherent"

    def test_angular_spectrum_is_coherent(self):
        """Test AngularSpectrumPropagator reports coherent mode."""
        from prism.core.propagators import AngularSpectrumPropagator

        grid = Grid(nx=32, dx=1e-5, wavelength=550e-9)
        propagator = AngularSpectrumPropagator(grid, distance=1.0)
        assert propagator.illumination_mode == "coherent"


class TestFactoryFunctions:
    """Test factory function integration with OTF propagator."""

    def test_create_propagator_otf(self):
        """Test creating OTF propagator via factory."""
        from prism.core.propagators import create_propagator

        grid = Grid(nx=64, dx=1e-5, wavelength=550e-9)
        aperture = torch.ones(64, 64, dtype=torch.cfloat)

        prop = create_propagator("otf", aperture=aperture, grid=grid)

        assert isinstance(prop, OTFPropagator)
        assert prop.illumination_mode == "incoherent"

    def test_create_propagator_otf_requires_aperture(self):
        """Test OTF creation fails without aperture."""
        from prism.core.propagators import create_propagator

        with pytest.raises(ValueError, match="Aperture required"):
            create_propagator("otf")

    def test_create_propagator_incoherent_auto(self):
        """Test creating propagator via incoherent_auto method."""
        from prism.core.propagators import create_propagator

        grid = Grid(nx=64, dx=1e-5, wavelength=550e-9)
        aperture = torch.ones(64, 64, dtype=torch.cfloat)

        prop = create_propagator("incoherent_auto", aperture=aperture, grid=grid)

        assert isinstance(prop, OTFPropagator)

    def test_create_propagator_extended_source_requires_grid(self):
        """Test extended_source method requires grid parameter."""
        from prism.core.propagators import create_propagator

        with pytest.raises(ValueError, match="Grid required"):
            create_propagator("extended_source")

    def test_select_propagator_incoherent_illumination(self):
        """Test select_propagator with incoherent illumination."""
        from prism.core.propagators import select_propagator

        grid = Grid(nx=64, dx=1e-5, wavelength=550e-9)
        aperture = torch.ones(64, 64, dtype=torch.cfloat)

        prop = select_propagator(
            wavelength=550e-9,
            obj_distance=1e6,
            fov=64 * 1e-5,
            illumination="incoherent",
            aperture=aperture,
            grid=grid,
        )

        assert isinstance(prop, OTFPropagator)
        assert prop.illumination_mode == "incoherent"

    def test_select_propagator_incoherent_requires_aperture(self):
        """Test select_propagator with incoherent requires aperture."""
        from prism.core.propagators import select_propagator

        with pytest.raises(ValueError, match="Aperture required"):
            select_propagator(
                wavelength=550e-9,
                obj_distance=1e6,
                fov=64 * 1e-5,
                illumination="incoherent",
                # Missing aperture
            )

    def test_select_propagator_otf_method(self):
        """Test select_propagator with method='otf'."""
        from prism.core.propagators import select_propagator

        aperture = torch.ones(64, 64, dtype=torch.cfloat)

        prop = select_propagator(
            wavelength=550e-9,
            obj_distance=1e6,
            fov=64 * 1e-5,
            method="otf",
            aperture=aperture,
        )

        assert isinstance(prop, OTFPropagator)

    def test_select_propagator_creates_grid_from_params(self):
        """Test select_propagator creates grid when parameters provided."""
        from prism.core.propagators import select_propagator

        aperture = torch.ones(128, 128, dtype=torch.cfloat)

        prop = select_propagator(
            wavelength=550e-9,
            obj_distance=1e6,
            fov=128 * 1e-5,
            illumination="incoherent",
            aperture=aperture,
            image_size=128,
            dx=1e-5,
        )

        assert isinstance(prop, OTFPropagator)

    def test_factory_otf_propagator_works(self):
        """Test that factory-created OTF propagator actually works."""
        from prism.core.propagators import create_propagator

        grid = Grid(nx=64, dx=1e-5, wavelength=550e-9)
        x, y = grid.x, grid.y
        r = torch.sqrt(x**2 + y**2)
        radius = 20 * grid.dx
        aperture = (r <= radius).to(torch.cfloat)

        prop = create_propagator("otf", aperture=aperture, grid=grid)

        # Test propagation
        intensity = torch.rand(64, 64)
        output = prop(intensity)

        assert output.shape == intensity.shape
        assert not output.is_complex()
        assert (output >= 0).all()
