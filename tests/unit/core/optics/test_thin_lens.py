"""Unit tests for ThinLens class."""

import pytest
import torch

from prism.core.grid import Grid
from prism.core.optics import ThinLens


class TestThinLens:
    """Tests for ThinLens optical element."""

    @pytest.fixture
    def grid(self):
        """Standard test grid."""
        return Grid(nx=128, dx=1e-6, wavelength=532e-9)

    def test_converging_lens_phase_center(self, grid):
        """Phase at center should be zero (or very small)."""
        lens = ThinLens(focal_length=0.01, grid=grid)

        center_idx = grid.nx // 2
        center_phase = torch.angle(lens.lens_phase[center_idx, center_idx])

        assert torch.abs(center_phase) < 1e-10

    def test_converging_lens_phase_sign(self, grid):
        """Converging lens (f > 0) should have negative phase near center."""
        lens = ThinLens(focal_length=0.01, grid=grid)

        # Test at a small offset from center to avoid phase wrapping
        # Phase = -k*r²/(2f), so for f > 0, phase should be negative at r > 0
        center = grid.nx // 2
        offset = 5  # Small offset to stay within one phase wrap

        near_center_phase = torch.angle(lens.lens_phase[center, center + offset])

        # Phase should be negative (delayed) for converging lens
        assert near_center_phase < 0

    def test_diverging_lens_phase_sign(self, grid):
        """Diverging lens (f < 0) should have positive phase near center."""
        lens = ThinLens(focal_length=-0.01, grid=grid)

        # Test at a small offset from center to avoid phase wrapping
        # Phase = -k*r²/(2f), so for f < 0, phase should be positive at r > 0
        center = grid.nx // 2
        offset = 5  # Small offset to stay within one phase wrap

        near_center_phase = torch.angle(lens.lens_phase[center, center + offset])

        # Phase should be positive (advanced) for diverging lens
        assert near_center_phase > 0

    def test_aperture_clipping(self, grid):
        """Aperture should clip field outside diameter."""
        aperture_dia = 50e-6  # 50 microns
        lens = ThinLens(focal_length=0.01, grid=grid, aperture_diameter=aperture_dia)

        # Check center is inside aperture
        center_idx = grid.nx // 2
        assert lens.pupil[center_idx, center_idx].abs() > 0

        # Check corners are outside aperture (for small enough aperture)
        assert lens.pupil[0, 0].abs() == 0

    def test_no_aperture_full_transmission(self, grid):
        """Without aperture, pupil should be all ones."""
        lens = ThinLens(focal_length=0.01, grid=grid, aperture_diameter=None)

        # All points should be inside pupil
        assert torch.allclose(lens.pupil.abs(), torch.ones_like(lens.pupil.abs()))

    def test_output_grid_scaling(self, grid):
        """Output grid should have scaled pixel size."""
        f = 0.01  # 10mm focal length
        lens = ThinLens(focal_length=f, grid=grid)

        out_grid = lens.output_grid

        # dx_new = λ * f / (N * dx)
        expected_dx = grid.wl * f / (grid.nx * grid.dx)

        assert out_grid.dx == pytest.approx(expected_dx, rel=1e-6)

    def test_forward_preserves_dtype(self, grid):
        """Forward pass should preserve complex dtype."""
        lens = ThinLens(focal_length=0.01, grid=grid)

        field = torch.ones(grid.nx, grid.ny, dtype=torch.complex64)
        output = lens(field)

        assert output.dtype == torch.complex64

    def test_forward_preserves_shape(self, grid):
        """Forward pass should preserve input shape."""
        lens = ThinLens(focal_length=0.01, grid=grid)

        field = torch.ones(grid.nx, grid.ny, dtype=torch.complex64)
        output = lens(field)

        assert output.shape == field.shape

    def test_zero_focal_length_raises(self, grid):
        """Zero focal length should raise ValueError."""
        with pytest.raises(ValueError, match="Focal length cannot be zero"):
            ThinLens(focal_length=0, grid=grid)

    def test_phase_quadratic_dependence(self, grid):
        """Phase should have quadratic radial dependence."""
        f = 0.01  # 10mm focal length
        lens = ThinLens(focal_length=f, grid=grid)

        # Get phase at center and at a known offset
        center = grid.nx // 2
        offset = 10

        phase_center = torch.angle(lens.lens_phase[center, center])
        phase_offset = torch.angle(lens.lens_phase[center, center + offset])

        # Phase difference should scale as r^2
        # At offset dx pixels, r = offset * dx
        r = offset * grid.dx
        k = 2 * torch.pi / grid.wl

        expected_phase_diff = -k / (2 * f) * r**2

        # The phase at center should be 0 (or near 0), so phase_offset ~ expected
        actual_phase_diff = phase_offset - phase_center

        assert actual_phase_diff == pytest.approx(expected_phase_diff, rel=0.01)

    def test_lens_unit_magnitude(self, grid):
        """Lens phase factor should have unit magnitude."""
        lens = ThinLens(focal_length=0.01, grid=grid)

        # The lens_phase buffer should be complex exponential with magnitude 1
        magnitudes = torch.abs(lens.lens_phase)

        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)

    def test_batch_input(self, grid):
        """Forward should work with batched input."""
        lens = ThinLens(focal_length=0.01, grid=grid)

        # Batched input (B, H, W)
        field = torch.ones(4, grid.nx, grid.ny, dtype=torch.complex64)
        output = lens(field)

        assert output.shape == (4, grid.nx, grid.ny)

    def test_aperture_diameter_matches_grid_extent(self, grid):
        """Large aperture should not clip within grid."""
        # Aperture larger than grid diagonal
        aperture_dia = 2 * grid.nx * grid.dx  # Much larger than grid
        lens = ThinLens(focal_length=0.01, grid=grid, aperture_diameter=aperture_dia)

        # All points should be inside aperture
        assert (lens.pupil.abs() > 0).all()

    def test_small_aperture_clips_most(self, grid):
        """Very small aperture should clip most of the field."""
        aperture_dia = 1e-6  # 1 micron - very small
        lens = ThinLens(focal_length=0.01, grid=grid, aperture_diameter=aperture_dia)

        # Most of the pupil should be zero
        nonzero_fraction = (lens.pupil.abs() > 0).float().mean()
        assert nonzero_fraction < 0.01  # Less than 1% transmission

    def test_nn_module_interface(self, grid):
        """ThinLens should properly implement nn.Module interface."""
        lens = ThinLens(focal_length=0.01, grid=grid)

        # Check it has the expected registered buffers
        assert hasattr(lens, "lens_phase")
        assert hasattr(lens, "pupil")

        # Buffers should be tensors
        assert isinstance(lens.lens_phase, torch.Tensor)
        assert isinstance(lens.pupil, torch.Tensor)

        # Should be able to move to different devices (if available)
        # This just verifies the interface, not actual GPU movement
        lens_dict = dict(lens.named_buffers())
        assert "lens_phase" in lens_dict
        assert "pupil" in lens_dict
