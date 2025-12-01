"""Unit tests for MicroscopeForwardModel and regime selection utilities."""

import pytest
import torch

from prism.core.grid import Grid
from prism.core.optics import (
    ForwardModelRegime,
    MicroscopeForwardModel,
    compute_defocus_parameter,
    select_forward_regime,
)


class TestDefocusParameter:
    """Tests for defocus parameter computation."""

    def test_at_focal_plane(self):
        """Object at focal plane should give delta = 0."""
        delta = compute_defocus_parameter(object_distance=0.01, focal_length=0.01)
        assert delta == pytest.approx(0.0)

    def test_one_percent_defocus(self):
        """1% deviation should give delta = 0.01."""
        delta = compute_defocus_parameter(object_distance=0.0101, focal_length=0.01)
        assert delta == pytest.approx(0.01, rel=1e-3)

    def test_fifty_percent_defocus(self):
        """50% deviation should give delta = 0.5."""
        delta = compute_defocus_parameter(object_distance=0.015, focal_length=0.01)
        assert delta == pytest.approx(0.5, rel=1e-3)

    def test_negative_defocus(self):
        """Object closer than focal plane should still give positive delta."""
        delta = compute_defocus_parameter(object_distance=0.005, focal_length=0.01)
        assert delta == pytest.approx(0.5, rel=1e-3)

    def test_negative_focal_length(self):
        """Should work with negative focal length (diverging lens)."""
        delta = compute_defocus_parameter(object_distance=-0.0101, focal_length=-0.01)
        assert delta == pytest.approx(0.01, rel=1e-3)

    def test_zero_focal_length_raises(self):
        """Zero focal length should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be zero"):
            compute_defocus_parameter(object_distance=0.01, focal_length=0)

    def test_double_focal_length(self):
        """Object at 2f should give delta = 1.0."""
        delta = compute_defocus_parameter(object_distance=0.02, focal_length=0.01)
        assert delta == pytest.approx(1.0, rel=1e-3)


class TestRegimeSelection:
    """Tests for forward model regime selection."""

    def test_auto_selects_simplified_at_focus(self):
        """At focal plane, should select SIMPLIFIED."""
        regime = select_forward_regime(
            object_distance=0.01,
            focal_length=0.01,
            threshold=0.01,
        )
        assert regime == ForwardModelRegime.SIMPLIFIED

    def test_auto_selects_simplified_within_threshold(self):
        """Within threshold, should select SIMPLIFIED."""
        regime = select_forward_regime(
            object_distance=0.01005,  # 0.5% defocus
            focal_length=0.01,
            threshold=0.01,  # 1% threshold
        )
        assert regime == ForwardModelRegime.SIMPLIFIED

    def test_auto_selects_full_when_defocused(self):
        """When defocused beyond threshold, should select FULL."""
        regime = select_forward_regime(
            object_distance=0.02,  # 100% defocus
            focal_length=0.01,
            threshold=0.01,
        )
        assert regime == ForwardModelRegime.FULL

    def test_auto_selects_full_at_threshold(self):
        """At or slightly beyond threshold, should select FULL."""
        # Use 1.5% defocus to clearly exceed 1% threshold (avoids float precision issues)
        regime = select_forward_regime(
            object_distance=0.01015,  # 1.5% defocus
            focal_length=0.01,
            threshold=0.01,
        )
        assert regime == ForwardModelRegime.FULL

    def test_manual_override_simplified(self):
        """Manual SIMPLIFIED override should be respected."""
        regime = select_forward_regime(
            object_distance=0.02,  # Would be FULL with auto
            focal_length=0.01,
            method=ForwardModelRegime.SIMPLIFIED,
        )
        assert regime == ForwardModelRegime.SIMPLIFIED

    def test_manual_override_full(self):
        """Manual FULL override should be respected."""
        regime = select_forward_regime(
            object_distance=0.01,  # Would be SIMPLIFIED with auto
            focal_length=0.01,
            method=ForwardModelRegime.FULL,
        )
        assert regime == ForwardModelRegime.FULL

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        # With 5% threshold, 3% defocus should be SIMPLIFIED
        regime = select_forward_regime(
            object_distance=0.0103,  # 3% defocus
            focal_length=0.01,
            threshold=0.05,  # 5% threshold
        )
        assert regime == ForwardModelRegime.SIMPLIFIED


class TestMicroscopeForwardModel:
    """Tests for MicroscopeForwardModel."""

    @pytest.fixture
    def grid(self):
        """Standard test grid."""
        return Grid(nx=128, dx=1e-6, wavelength=532e-9)

    @pytest.fixture
    def pupils(self, grid):
        """Create illumination and detection pupils."""
        # Simple circular pupils
        fx, fy = grid.kx, grid.ky
        cutoff = 0.9 / (1.0 * grid.wl)  # NA=0.9 in air
        r_norm = torch.sqrt(fx**2 + fy**2) / cutoff
        pupil = (r_norm <= 1.0).to(torch.complex64)
        return pupil, pupil  # Same for illum and detect

    def test_simplified_at_focal_plane(self, grid):
        """At focal plane, SIMPLIFIED regime should be selected."""
        f_obj = 0.005  # 5mm objective focal

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj,  # Exactly at focal plane
            na=0.9,
            medium_index=1.0,
        )

        assert model.selected_regime == ForwardModelRegime.SIMPLIFIED

    def test_full_when_defocused(self, grid):
        """When defocused, FULL regime should be selected."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj * 1.5,  # 50% defocus
            na=0.9,
            medium_index=1.0,
        )

        assert model.selected_regime == ForwardModelRegime.FULL

    def test_manual_regime_simplified(self, grid):
        """Manual SIMPLIFIED regime should be respected."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj * 2.0,  # Would be FULL with auto
            na=0.9,
            medium_index=1.0,
            regime=ForwardModelRegime.SIMPLIFIED,
        )

        assert model.selected_regime == ForwardModelRegime.SIMPLIFIED

    def test_manual_regime_full(self, grid):
        """Manual FULL regime should be respected."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj,  # Would be SIMPLIFIED with auto
            na=0.9,
            medium_index=1.0,
            regime=ForwardModelRegime.FULL,
        )

        assert model.selected_regime == ForwardModelRegime.FULL

    def test_forward_output_shape(self, grid, pupils):
        """Forward pass should preserve input shape."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj,
            na=0.9,
            medium_index=1.0,
        )

        field = torch.ones(grid.nx, grid.ny, dtype=torch.complex64)
        illum_pupil, detect_pupil = pupils

        output = model(field, illum_pupil, detect_pupil)

        assert output.shape == field.shape

    def test_forward_output_dtype(self, grid, pupils):
        """Forward pass should return complex tensor."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj,
            na=0.9,
            medium_index=1.0,
        )

        field = torch.ones(grid.nx, grid.ny, dtype=torch.complex64)
        illum_pupil, detect_pupil = pupils

        output = model(field, illum_pupil, detect_pupil)

        assert output.dtype == torch.complex64

    def test_simplified_forward_produces_finite_output(self, grid, pupils):
        """Simplified forward should produce finite values."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj,
            na=0.9,
            medium_index=1.0,
            regime=ForwardModelRegime.SIMPLIFIED,
        )

        field = torch.ones(grid.nx, grid.ny, dtype=torch.complex64)
        illum_pupil, detect_pupil = pupils

        output = model(field, illum_pupil, detect_pupil)

        assert torch.isfinite(output).all()

    def test_full_forward_produces_finite_output(self, grid, pupils):
        """Full forward should produce finite values."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj * 1.1,  # 10% defocus to trigger FULL
            na=0.9,
            medium_index=1.0,
            regime=ForwardModelRegime.FULL,
        )

        field = torch.ones(grid.nx, grid.ny, dtype=torch.complex64)
        illum_pupil, detect_pupil = pupils

        output = model(field, illum_pupil, detect_pupil)

        assert torch.isfinite(output).all()

    def test_defocus_parameter_stored(self, grid):
        """Model should store computed defocus parameter."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj * 1.1,
            na=0.9,
            medium_index=1.0,
        )

        expected_delta = abs(f_obj * 1.1 - f_obj) / f_obj
        assert model.defocus_parameter == pytest.approx(expected_delta, rel=1e-6)

    def test_get_info_returns_dict(self, grid):
        """get_info() should return configuration dictionary."""
        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=0.005,
            tube_lens_focal=0.2,
            working_distance=0.005,
            na=0.9,
            medium_index=1.0,
        )

        info = model.get_info()

        assert isinstance(info, dict)
        assert "regime" in info
        assert "defocus_parameter" in info
        assert "working_distance_mm" in info
        assert "objective_focal_mm" in info
        assert "tube_lens_focal_mm" in info
        assert "na" in info
        assert "medium_index" in info

    def test_get_info_values(self, grid):
        """get_info() should return correct values."""
        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=0.005,
            tube_lens_focal=0.2,
            working_distance=0.005,
            na=0.9,
            medium_index=1.0,
        )

        info = model.get_info()

        assert info["regime"] == "simplified"
        assert info["defocus_parameter"] == pytest.approx(0.0)
        assert info["working_distance_mm"] == pytest.approx(5.0)
        assert info["objective_focal_mm"] == pytest.approx(5.0)
        assert info["tube_lens_focal_mm"] == pytest.approx(200.0)
        assert info["na"] == pytest.approx(0.9)
        assert info["medium_index"] == pytest.approx(1.0)

    def test_full_model_initializes_lenses(self, grid):
        """FULL model should initialize lens components."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj * 1.5,  # Force FULL
            na=0.9,
            medium_index=1.0,
            regime=ForwardModelRegime.FULL,
        )

        assert model.objective_lens is not None
        assert model.tube_lens is not None

    def test_simplified_model_no_lenses(self, grid):
        """SIMPLIFIED model should not initialize lens components."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj,
            na=0.9,
            medium_index=1.0,
            regime=ForwardModelRegime.SIMPLIFIED,
        )

        assert model.objective_lens is None
        assert model.tube_lens is None

    def test_full_model_with_defocus_propagator(self, grid):
        """FULL model with defocus should initialize propagator."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj * 1.5,  # 50% defocus
            na=0.9,
            medium_index=1.0,
            regime=ForwardModelRegime.FULL,
        )

        assert model.defocus_propagator is not None

    def test_nn_module_interface(self, grid):
        """MicroscopeForwardModel should properly implement nn.Module interface."""
        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=0.005,
            tube_lens_focal=0.2,
            working_distance=0.005,
            na=0.9,
            medium_index=1.0,
        )

        # Should be a nn.Module subclass
        assert isinstance(model, torch.nn.Module)

        # Should be callable
        assert callable(model)

    def test_custom_defocus_threshold(self, grid):
        """Custom defocus threshold should be respected."""
        f_obj = 0.005

        # With 0.5 threshold, 10% defocus should still be SIMPLIFIED
        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj * 1.1,  # 10% defocus
            na=0.9,
            medium_index=1.0,
            defocus_threshold=0.5,  # 50% threshold
        )

        assert model.selected_regime == ForwardModelRegime.SIMPLIFIED

    def test_point_source_response(self, grid, pupils):
        """Point source should produce a PSF-like response."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj,
            na=0.9,
            medium_index=1.0,
        )

        # Create point source at center
        field = torch.zeros(grid.nx, grid.ny, dtype=torch.complex64)
        field[grid.nx // 2, grid.ny // 2] = 1.0

        illum_pupil, detect_pupil = pupils
        output = model(field, illum_pupil, detect_pupil)

        # Convert to intensity
        intensity = torch.abs(output) ** 2

        # Peak should be at or near center
        peak_idx = torch.argmax(intensity)
        peak_row, peak_col = peak_idx // grid.ny, peak_idx % grid.ny

        # Allow some tolerance for peak location
        assert abs(peak_row - grid.nx // 2) <= 2
        assert abs(peak_col - grid.ny // 2) <= 2

    def test_oil_immersion_config(self, grid, pupils):
        """Should work with oil immersion (medium_index > 1) configuration."""
        f_obj = 0.005

        model = MicroscopeForwardModel(
            grid=grid,
            objective_focal=f_obj,
            tube_lens_focal=0.2,
            working_distance=f_obj,
            na=1.4,  # High NA for oil immersion
            medium_index=1.515,  # Oil
        )

        field = torch.ones(grid.nx, grid.ny, dtype=torch.complex64)
        illum_pupil, detect_pupil = pupils

        output = model(field, illum_pupil, detect_pupil)

        assert torch.isfinite(output).all()
        assert output.shape == field.shape
