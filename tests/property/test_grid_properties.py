"""
Property-based tests for Grid using hypothesis.

These tests verify mathematical properties that should hold for all valid inputs,
not just specific test cases.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from prism.core.grid import Grid


@given(
    n=st.integers(min_value=32, max_value=512),
    dx=st.floats(min_value=1e-6, max_value=1e-3),
)
@settings(max_examples=50, deadline=None)
def test_grid_coordinate_caching(n, dx):
    """Property: Grid coordinates should be identical when cached."""
    grid = Grid(nx=n, dx=dx)

    # First access (populates cache)
    coords1 = grid.x
    coords2 = grid.y

    # Second access (uses cache)
    coords1_cached = grid.x
    coords2_cached = grid.y

    # Should be identical objects (cached)
    assert coords1 is coords1_cached
    assert coords2 is coords2_cached


@given(
    n=st.integers(min_value=32, max_value=256),
    dx=st.floats(min_value=1e-6, max_value=1e-3),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
)
@settings(max_examples=50, deadline=None)
def test_grid_coordinate_symmetry(n, dx, wavelength):
    """Property: Grid coordinates should be centered around zero."""
    grid = Grid(nx=n, dx=dx, wavelength=wavelength)

    x = grid.x.flatten()
    y = grid.y.flatten()

    # Coordinates should be centered around 0
    # The range might be asymmetric for even n due to FFT convention
    # but the center should be at or near 0

    # Check that the mean is close to 0 (centered)
    assert torch.allclose(x.mean(), torch.tensor(0.0), atol=dx)
    assert torch.allclose(y.mean(), torch.tensor(0.0), atol=dx)

    # For odd n, min and max should be exactly symmetric
    if n % 2 == 1:
        assert torch.allclose(x.min(), -x.max(), rtol=1e-4)
        assert torch.allclose(y.min(), -y.max(), rtol=1e-4)


@given(
    n=st.integers(min_value=32, max_value=256),
    dx=st.floats(min_value=1e-6, max_value=1e-3),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
)
@settings(max_examples=50, deadline=None)
def test_grid_fov_property(n, dx, wavelength):
    """Property: Field of view should equal n * dx."""
    grid = Grid(nx=n, dx=dx, wavelength=wavelength)

    fov_x, fov_y = grid.fov

    # FOV should be exactly n * dx
    assert torch.isclose(torch.tensor(fov_x), torch.tensor(n * dx), rtol=1e-6)
    assert torch.isclose(torch.tensor(fov_y), torch.tensor(n * dx), rtol=1e-6)


@given(
    n=st.integers(min_value=32, max_value=256),
    dx=st.floats(min_value=1e-6, max_value=1e-3),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
)
@settings(max_examples=50, deadline=None)
def test_grid_frequency_range(n, dx, wavelength):
    """Property: Frequency coordinates should respect Nyquist limit."""
    grid = Grid(nx=n, dx=dx, wavelength=wavelength)

    kx = grid.kx.flatten()
    ky = grid.ky.flatten()

    # Nyquist frequency is 1/(2*dx)
    nyquist = 1.0 / (2.0 * dx)

    # All frequencies should be within Nyquist limit
    assert kx.max() <= nyquist * 1.01  # Small tolerance for numerical error
    assert ky.max() <= nyquist * 1.01
    assert kx.min() >= -nyquist * 1.01
    assert ky.min() >= -nyquist * 1.01


@given(
    n=st.integers(min_value=32, max_value=256),
    dx=st.floats(min_value=1e-6, max_value=1e-3),
    scale=st.integers(min_value=2, max_value=4),
)
@settings(max_examples=50, deadline=None)
def test_grid_padding_preserves_spacing(n, dx, scale):
    """Property: Padding should preserve pixel spacing."""
    grid = Grid(nx=n, dx=dx)

    padded = grid.pad(padding_scale=scale)

    # Padded grid should have same dx but larger nx
    assert padded.nx == n * scale
    assert torch.isclose(torch.tensor(padded.dx), torch.tensor(dx), rtol=1e-6)


@given(
    n=st.integers(min_value=32, max_value=128),
    dx=st.floats(min_value=1e-6, max_value=1e-3),
    scale=st.integers(min_value=2, max_value=4),
)
@settings(max_examples=50, deadline=None)
def test_grid_upsampling_preserves_fov(n, dx, scale):
    """Property: Upsampling should preserve field of view."""
    grid = Grid(nx=n, dx=dx)

    upsampled = grid.upsample(scale=scale)

    # Upsampled grid should have same FOV
    fov_original = grid.fov[0]
    fov_upsampled = upsampled.fov[0]

    assert torch.isclose(torch.tensor(fov_original), torch.tensor(fov_upsampled), rtol=1e-6)

    # But finer spacing
    assert upsampled.nx == n * scale
    assert torch.isclose(torch.tensor(upsampled.dx), torch.tensor(dx / scale), rtol=1e-6)


@given(
    n=st.integers(min_value=32, max_value=256),
    dx=st.floats(min_value=1e-6, max_value=1e-3),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
)
@settings(max_examples=50, deadline=None)
def test_grid_clone_equivalence(n, dx, wavelength):
    """Property: Cloned grid should be identical to original."""
    grid = Grid(nx=n, dx=dx, wavelength=wavelength)

    cloned = grid.clone()

    # All parameters should match
    assert cloned.nx == grid.nx
    assert cloned.ny == grid.ny
    assert cloned.dx == grid.dx
    assert cloned.dy == grid.dy
    assert cloned.wl == grid.wl

    # Coordinates should be equal (but not same object)
    assert torch.allclose(cloned.x, grid.x)
    assert torch.allclose(cloned.y, grid.y)


@given(
    n=st.integers(min_value=32, max_value=256),
    dx=st.floats(min_value=1e-6, max_value=1e-3),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
    focal_length=st.floats(min_value=0.01, max_value=1.0),
)
@settings(max_examples=50, deadline=None)
def test_grid_lens_ft_consistency(n, dx, wavelength, focal_length):
    """Property: Lens Fourier transform should be consistent."""
    grid = Grid(nx=n, dx=dx, wavelength=wavelength)

    # Get lens FT coordinates
    x_f, y_f = grid.lens_ft(f=focal_length)

    # Should also match grid created via lens_ft_grid
    f_grid = grid.lens_ft_grid(f=focal_length)

    # Coordinates should match
    assert torch.allclose(x_f, f_grid.x, rtol=1e-5)
    assert torch.allclose(y_f, f_grid.y, rtol=1e-5)


@given(
    nx=st.integers(min_value=32, max_value=128),
    ny=st.integers(min_value=32, max_value=128),
    dx=st.floats(min_value=1e-6, max_value=1e-3),
    dy=st.floats(min_value=1e-6, max_value=1e-3),
)
@settings(max_examples=50, deadline=None)
def test_grid_rectangular_grids(nx, ny, dx, dy):
    """Property: Rectangular grids (nx != ny) should work correctly."""
    grid = Grid(nx=nx, dx=dx, ny=ny, dy=dy)

    # Check dimensions
    assert grid.nx == nx
    assert grid.ny == ny
    assert grid.x.shape[-1] == nx
    assert grid.y.shape[-2] == ny

    # Check FOV
    fov_x, fov_y = grid.fov
    assert torch.isclose(torch.tensor(fov_x), torch.tensor(nx * dx), rtol=1e-6)
    assert torch.isclose(torch.tensor(fov_y), torch.tensor(ny * dy), rtol=1e-6)


@given(
    n=st.integers(min_value=32, max_value=256),
    dx=st.floats(min_value=1e-6, max_value=1e-3),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
)
@settings(max_examples=50, deadline=None)
def test_grid_device_transfer(n, dx, wavelength):
    """Property: Grid should transfer to different devices correctly."""
    grid = Grid(nx=n, dx=dx, wavelength=wavelength, device="cpu")

    # Access coordinates to populate cache
    x_cpu = grid.x
    y_cpu = grid.y

    assert x_cpu.device.type == "cpu"
    assert y_cpu.device.type == "cpu"

    # If CUDA available, test device transfer
    if torch.cuda.is_available():
        grid.to("cuda")
        x_cuda = grid.x
        y_cuda = grid.y

        assert x_cuda.device.type == "cuda"
        assert y_cuda.device.type == "cuda"

        # Values should be the same
        assert torch.allclose(x_cuda.cpu(), x_cpu, rtol=1e-6)
        assert torch.allclose(y_cuda.cpu(), y_cpu, rtol=1e-6)

        # Move back to CPU
        grid.to("cpu")
        x_cpu2 = grid.x
        assert x_cpu2.device.type == "cpu"


# Validation tests
def test_grid_invalid_parameters():
    """Test that invalid parameters raise appropriate errors."""
    # Negative nx
    with pytest.raises(ValueError, match="nx must be positive"):
        Grid(nx=-10, dx=1e-5)

    # Zero nx
    with pytest.raises(ValueError, match="nx must be positive"):
        Grid(nx=0, dx=1e-5)

    # Negative dx
    with pytest.raises(ValueError, match="dx must be positive"):
        Grid(nx=256, dx=-1e-5)

    # Zero dx
    with pytest.raises(ValueError, match="dx must be positive"):
        Grid(nx=256, dx=0)

    # Negative wavelength
    with pytest.raises(ValueError, match="wavelength must be positive"):
        Grid(nx=256, dx=1e-5, wavelength=-520e-9)

    # Zero wavelength
    with pytest.raises(ValueError, match="wavelength must be positive"):
        Grid(nx=256, dx=1e-5, wavelength=0)

    # Negative ny
    with pytest.raises(ValueError, match="ny must be positive"):
        Grid(nx=256, dx=1e-5, ny=-128)

    # Negative dy
    with pytest.raises(ValueError, match="dy must be positive"):
        Grid(nx=256, dx=1e-5, dy=-1e-5)
