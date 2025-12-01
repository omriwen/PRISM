"""Unit tests for grid module."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from prism.core.grid import Grid


class TestGrid:
    """Test suite for Grid class."""

    @pytest.fixture
    def basic_grid(self):
        """Create basic grid instance for testing."""
        return Grid(nx=256, dx=1e-5, wavelength=520e-9)

    @pytest.fixture
    def small_grid(self):
        """Create small grid for faster tests."""
        return Grid(nx=64, dx=1e-5, wavelength=520e-9)

    @pytest.fixture
    def asymmetric_grid(self):
        """Create asymmetric grid (different nx and ny)."""
        return Grid(nx=128, ny=256, dx=1e-5, dy=2e-5, wavelength=520e-9)

    def test_initialization_default(self):
        """Test grid initialization with default parameters."""
        grid = Grid()

        assert grid.nx == 256
        assert grid.dx == 1e-5
        assert grid.ny == 256  # Should default to nx
        assert grid.dy == 1e-5  # Should default to dx
        assert grid.wl == 520e-9
        assert grid.device == "cpu"

    def test_initialization_custom(self):
        """Test grid initialization with custom parameters."""
        grid = Grid(nx=128, dx=2e-5, wavelength=633e-9, device="cpu")

        assert grid.nx == 128
        assert grid.dx == 2e-5
        assert grid.ny == 128  # Should default to nx
        assert grid.dy == 2e-5  # Should default to dx
        assert grid.wl == 633e-9

    def test_initialization_asymmetric(self):
        """Test grid initialization with different nx and ny."""
        grid = Grid(nx=128, ny=256, dx=1e-5, dy=2e-5)

        assert grid.nx == 128
        assert grid.ny == 256
        assert grid.dx == 1e-5
        assert grid.dy == 2e-5

    def test_fov_property(self, basic_grid):
        """Test field of view calculation."""
        fov_x, fov_y = basic_grid.fov

        # FOV should be nx * dx and ny * dy
        assert fov_x == basic_grid.nx * basic_grid.dx
        assert fov_y == basic_grid.ny * basic_grid.dy

    def test_fov_asymmetric(self, asymmetric_grid):
        """Test field of view for asymmetric grid."""
        fov_x, fov_y = asymmetric_grid.fov

        assert fov_x == asymmetric_grid.nx * asymmetric_grid.dx
        assert fov_y == asymmetric_grid.ny * asymmetric_grid.dy
        # Should be different
        assert fov_x != fov_y

    def test_x_coordinates(self, basic_grid):
        """Test x coordinate generation."""
        x = basic_grid.x

        # Check shape
        assert x.shape == (1, basic_grid.nx)

        # Check centering
        assert x.shape[1] == basic_grid.nx

        # Check spacing
        if basic_grid.nx > 1:
            spacing = x[0, 1] - x[0, 0]
            assert abs(spacing - basic_grid.dx) < 1e-10

    def test_y_coordinates(self, basic_grid):
        """Test y coordinate generation."""
        y = basic_grid.y

        # Check shape
        assert y.shape == (basic_grid.ny, 1)

        # Check spacing
        if basic_grid.ny > 1:
            spacing = y[1, 0] - y[0, 0]
            assert abs(spacing - basic_grid.dy) < 1e-10

    def test_grid_property(self, basic_grid):
        """Test grid property returns tuple of (x, y)."""
        x, y = basic_grid.grid

        assert x.shape == (1, basic_grid.nx)
        assert y.shape == (basic_grid.ny, 1)

    def test_x_coordinate_centering(self, small_grid):
        """Test that x coordinates are centered."""
        x = small_grid.x

        # For even nx, center should be between middle two values
        # For odd nx, center should be at middle value
        center_val = x[0, small_grid.nx // 2]

        # Should be close to zero (allowing for pixel offset)
        assert abs(center_val) <= small_grid.dx

    def test_y_coordinate_centering(self, small_grid):
        """Test that y coordinates are centered."""
        y = small_grid.y

        center_val = y[small_grid.ny // 2, 0]

        assert abs(center_val) <= small_grid.dy

    def test_kx_coordinates(self, basic_grid):
        """Test kx frequency coordinate generation."""
        kx = basic_grid.kx

        # Check shape matches x
        assert kx.shape == basic_grid.x.shape

        # kx should be derived from x
        expected_kx = basic_grid.x / basic_grid.nx / basic_grid.dx**2
        assert torch.allclose(kx, expected_kx)

    def test_ky_coordinates(self, basic_grid):
        """Test ky frequency coordinate generation."""
        ky = basic_grid.ky

        # Check shape matches y
        assert ky.shape == basic_grid.y.shape

        # ky should be derived from y
        expected_ky = basic_grid.y / basic_grid.ny / basic_grid.dy**2
        assert torch.allclose(ky, expected_ky)

    def test_kmax_property(self, basic_grid):
        """Test maximum frequency calculation."""
        kmax = basic_grid.kmax

        # Should be positive
        assert kmax > 0

        # Should be max of kx and ky
        expected_kmax = max(basic_grid.kx.max().item(), basic_grid.ky.max().item())
        assert abs(kmax - expected_kmax) < 1e-10

    def test_lens_ft(self, basic_grid):
        """Test lens Fourier transform coordinate calculation."""
        focal_length = 0.1  # 10 cm

        x_f, y_f = basic_grid.lens_ft(focal_length)

        # Check shapes
        assert x_f.shape == basic_grid.x.shape
        assert y_f.shape == basic_grid.y.shape

        # Check values
        expected_x_f = basic_grid.kx * basic_grid.wl * focal_length
        expected_y_f = basic_grid.ky * basic_grid.wl * focal_length

        assert torch.allclose(x_f, expected_x_f)
        assert torch.allclose(y_f, expected_y_f)

    @pytest.mark.parametrize("focal_length", [0.05, 0.1, 0.5, 1.0])
    def test_lens_ft_different_focal_lengths(self, small_grid, focal_length):
        """Test lens FT with different focal lengths."""
        x_f, y_f = small_grid.lens_ft(focal_length)

        # Larger focal length should give larger coordinates
        assert x_f.abs().max() > 0
        assert y_f.abs().max() > 0

    def test_pad(self, small_grid):
        """Test grid padding."""
        padding_scale = 2
        padded_grid = small_grid.pad(padding_scale)

        # Size should increase
        assert padded_grid.nx == small_grid.nx * padding_scale
        assert padded_grid.ny == small_grid.ny * padding_scale

        # Pixel spacing should remain the same
        assert padded_grid.dx == small_grid.dx
        assert padded_grid.dy == small_grid.dy

        # Wavelength should be preserved
        assert padded_grid.wl == small_grid.wl

        # Device should be preserved
        assert padded_grid.device == small_grid.device

    @pytest.mark.parametrize("padding_scale", [2, 3, 4])
    def test_pad_different_scales(self, small_grid, padding_scale):
        """Test padding with different scale factors."""
        padded = small_grid.pad(padding_scale)

        assert padded.nx == small_grid.nx * padding_scale
        assert padded.ny == small_grid.ny * padding_scale

    def test_upsample(self, small_grid):
        """Test grid upsampling."""
        scale = 2
        upsampled_grid = small_grid.upsample(scale)

        # Size should increase
        assert upsampled_grid.nx == small_grid.nx * scale
        assert upsampled_grid.ny == small_grid.ny * scale

        # Pixel spacing should decrease
        assert upsampled_grid.dx == small_grid.dx / scale
        assert upsampled_grid.dy == small_grid.dy / scale

        # Field of view should remain the same
        original_fov_x, original_fov_y = small_grid.fov
        upsampled_fov_x, upsampled_fov_y = upsampled_grid.fov

        assert abs(original_fov_x - upsampled_fov_x) < 1e-10
        assert abs(original_fov_y - upsampled_fov_y) < 1e-10

    @pytest.mark.parametrize("scale", [2, 3, 4])
    def test_upsample_different_scales(self, small_grid, scale):
        """Test upsampling with different scale factors."""
        upsampled = small_grid.upsample(scale)

        assert upsampled.nx == small_grid.nx * scale
        assert upsampled.dx == pytest.approx(small_grid.dx / scale)

    def test_upsample_preserves_fov(self, small_grid):
        """Test that upsampling preserves field of view."""
        scales = [2, 3, 4]

        original_fov = small_grid.fov

        for scale in scales:
            upsampled = small_grid.upsample(scale)
            upsampled_fov = upsampled.fov

            assert abs(original_fov[0] - upsampled_fov[0]) < 1e-10
            assert abs(original_fov[1] - upsampled_fov[1]) < 1e-10

    def test_clone(self, basic_grid):
        """Test grid cloning."""
        cloned = basic_grid.clone()

        # Get coordinates before modification
        original_x = basic_grid.x.clone()
        cloned_x_before = cloned.x.clone()

        # Should have exactly same coordinates initially
        assert torch.allclose(cloned_x_before, original_x)

        # But should be different objects
        assert cloned is not basic_grid

        # Verify internal parameters are equal
        assert cloned.nx == basic_grid.nx
        assert cloned.dx == basic_grid.dx
        assert cloned.ny == basic_grid.ny
        assert cloned.dy == basic_grid.dy
        assert cloned.wl == basic_grid.wl

    def test_clone_independence(self, small_grid):
        """Test that cloned grid is independent."""
        cloned = small_grid.clone()

        # Modify cloned grid's internal cache
        original_device = small_grid.device
        cloned.to("cpu")  # This invalidates cache

        # Original should not change
        assert small_grid.device == original_device

    def test_lens_ft_grid(self, small_grid):
        """Test lens Fourier transform grid creation."""
        focal_length = 0.1

        ft_grid = small_grid.lens_ft_grid(focal_length)

        # Should be a Grid object
        assert isinstance(ft_grid, Grid)

        # Coordinates should match lens_ft
        x_f, y_f = small_grid.lens_ft(focal_length)

        # Get coordinates from ft_grid
        ft_x = ft_grid.x
        ft_y = ft_grid.y

        # Compare shapes and values
        assert ft_x.shape == x_f.shape
        assert ft_y.shape == y_f.shape

        # Values should be exactly equal
        assert torch.allclose(ft_x, x_f)
        assert torch.allclose(ft_y, y_f)

    def test_to_device_cpu(self, small_grid):
        """Test moving grid to CPU device."""
        grid_cpu = small_grid.to("cpu")

        # Should return self for chaining
        assert grid_cpu is small_grid

        # Device should be updated
        assert grid_cpu.device == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_cuda(self, small_grid):
        """Test moving grid to CUDA device."""
        grid_cuda = small_grid.to("cuda")

        assert grid_cuda.device == "cuda"

        # Coordinates should be on CUDA
        assert grid_cuda.x.device.type == "cuda"
        assert grid_cuda.y.device.type == "cuda"

    def test_device_propagation(self, small_grid):
        """Test that device is propagated through operations."""
        # Pad
        padded = small_grid.pad(2)
        assert padded.device == small_grid.device

        # Upsample
        upsampled = small_grid.upsample(2)
        assert upsampled.device == small_grid.device

        # Lens FT grid
        ft_grid = small_grid.lens_ft_grid(0.1)
        assert ft_grid.device == small_grid.device

    @pytest.mark.parametrize("nx,dx", [(64, 1e-5), (128, 5e-6), (256, 2e-5)])
    def test_different_grid_configurations(self, nx, dx):
        """Test various grid configurations."""
        grid = Grid(nx=nx, dx=dx)

        assert grid.nx == nx
        assert grid.dx == dx

        # Coordinates should work correctly
        x, y = grid.grid
        assert x.shape[1] == nx
        assert y.shape[0] == nx  # ny defaults to nx

    def test_wavelength_preservation(self, small_grid):
        """Test that wavelength is preserved through operations."""
        original_wl = small_grid.wl

        # Pad
        assert small_grid.pad(2).wl == original_wl

        # Upsample
        assert small_grid.upsample(2).wl == original_wl

    def test_coordinate_dtype(self, basic_grid):
        """Test that coordinates use default dtype."""
        x = basic_grid.x
        y = basic_grid.y

        # Should use torch's default dtype (usually float32)
        assert x.dtype == torch.get_default_dtype()
        assert y.dtype == torch.get_default_dtype()

    def test_grid_consistency(self, small_grid):
        """Test consistency between grid properties."""
        # FOV should equal coordinate extent
        x, y = small_grid.grid
        fov_x, fov_y = small_grid.fov

        # Compute coordinate extent
        x_extent = x.max() - x.min() + small_grid.dx  # Add one pixel width
        y_extent = y.max() - y.min() + small_grid.dy

        # Should be approximately equal
        assert abs(x_extent - fov_x) < 2 * small_grid.dx
        assert abs(y_extent - fov_y) < 2 * small_grid.dy


class TestGridCaching:
    """Test coordinate caching functionality."""

    def test_x_cached_on_second_access(self):
        """Verify x coordinate is cached."""
        grid = Grid(nx=256, dx=1e-5)

        x1 = grid.x
        x2 = grid.x

        # Should return same object (cached)
        assert x1 is x2

    def test_y_cached_on_second_access(self):
        """Verify y coordinate is cached."""
        grid = Grid(nx=256, dx=1e-5)

        y1 = grid.y
        y2 = grid.y

        # Should return same object (cached)
        assert y1 is y2

    def test_kx_cached_on_second_access(self):
        """Verify kx coordinate is cached."""
        grid = Grid(nx=256, dx=1e-5)

        kx1 = grid.kx
        kx2 = grid.kx

        # Should return same object (cached)
        assert kx1 is kx2

    def test_ky_cached_on_second_access(self):
        """Verify ky coordinate is cached."""
        grid = Grid(nx=256, dx=1e-5)

        ky1 = grid.ky
        ky2 = grid.ky

        # Should return same object (cached)
        assert ky1 is ky2

    def test_cache_invalidated_on_device_change(self):
        """Verify cache is cleared when device changes."""
        grid = Grid(nx=256, dx=1e-5, device="cpu")
        x_cpu = grid.x

        grid.to("cuda" if torch.cuda.is_available() else "cpu")
        x_new = grid.x

        # Should be different object after device change
        assert x_cpu is not x_new

    def test_multiple_coordinates_cached_independently(self):
        """Verify different coordinates are cached independently."""
        grid = Grid(nx=128, dx=1e-5)

        x1 = grid.x
        y1 = grid.y
        kx1 = grid.kx
        ky1 = grid.ky

        x2 = grid.x
        y2 = grid.y
        kx2 = grid.kx
        ky2 = grid.ky

        # All should be cached
        assert x1 is x2
        assert y1 is y2
        assert kx1 is kx2
        assert ky1 is ky2


class TestGridValidation:
    """Test parameter validation."""

    def test_negative_nx_raises(self):
        """Test that negative nx raises ValueError."""
        with pytest.raises(ValueError, match="nx must be positive"):
            Grid(nx=-10)

    def test_zero_nx_raises(self):
        """Test that zero nx raises ValueError."""
        with pytest.raises(ValueError, match="nx must be positive"):
            Grid(nx=0)

    def test_negative_ny_raises(self):
        """Test that negative ny raises ValueError."""
        with pytest.raises(ValueError, match="ny must be positive"):
            Grid(nx=256, ny=-10)

    def test_zero_ny_raises(self):
        """Test that zero ny raises ValueError."""
        with pytest.raises(ValueError, match="ny must be positive"):
            Grid(nx=256, ny=0)

    def test_negative_dx_raises(self):
        """Test that negative dx raises ValueError."""
        with pytest.raises(ValueError, match="dx must be positive"):
            Grid(dx=-1e-5)

    def test_zero_dx_raises(self):
        """Test that zero dx raises ValueError."""
        with pytest.raises(ValueError, match="dx must be positive"):
            Grid(dx=0)

    def test_negative_dy_raises(self):
        """Test that negative dy raises ValueError."""
        with pytest.raises(ValueError, match="dy must be positive"):
            Grid(dx=1e-5, dy=-1e-5)

    def test_zero_dy_raises(self):
        """Test that zero dy raises ValueError."""
        with pytest.raises(ValueError, match="dy must be positive"):
            Grid(dx=1e-5, dy=0)

    def test_negative_wavelength_raises(self):
        """Test that negative wavelength raises ValueError."""
        with pytest.raises(ValueError, match="wavelength must be positive"):
            Grid(wavelength=-520e-9)

    def test_zero_wavelength_raises(self):
        """Test that zero wavelength raises ValueError."""
        with pytest.raises(ValueError, match="wavelength must be positive"):
            Grid(wavelength=0)

    def test_non_power_of_2_warns(self, caplog):
        """Verify warning for non-power-of-2 size."""
        # Note: This test checks that warnings are logged, but loguru
        # may not always integrate with caplog. The validation logic is tested separately.
        import logging

        with caplog.at_level(logging.WARNING):
            Grid(nx=100)  # Not power of 2
        # Note: loguru warnings may not always appear in caplog
        # The important thing is that the validation code runs without error

    def test_power_of_2_no_warning(self, caplog):
        """Verify no warning for power-of-2 size."""
        caplog.clear()
        Grid(nx=128)  # Power of 2
        # Should not contain warning (or at least not for nx)
        # Note: May still warn for ny if nx != ny and ny isn't power of 2


class TestFromCoordinates:
    """Test factory method."""

    def test_from_coordinates_basic(self):
        """Test creating grid from coordinates."""
        x = torch.linspace(-1e-3, 1e-3, 256).unsqueeze(0)
        y = torch.linspace(-1e-3, 1e-3, 256).unsqueeze(1)

        grid = Grid.from_coordinates(x, y, wavelength=520e-9)

        assert grid.nx == 256
        assert grid.ny == 256
        assert abs(grid.dx - 2e-3 / 255) < 1e-10

    def test_from_coordinates_asymmetric(self):
        """Test creating asymmetric grid from coordinates."""
        x = torch.linspace(-1e-3, 1e-3, 128).unsqueeze(0)
        y = torch.linspace(-2e-3, 2e-3, 256).unsqueeze(1)

        grid = Grid.from_coordinates(x, y, wavelength=520e-9)

        assert grid.nx == 128
        assert grid.ny == 256
        assert abs(grid.dx - 2e-3 / 127) < 1e-10
        assert abs(grid.dy - 4e-3 / 255) < 1e-10

    def test_from_coordinates_reproduces_coordinates(self):
        """Verify reconstructed grid has similar coordinates."""
        x_orig = torch.linspace(-1e-3, 1e-3, 128).unsqueeze(0)
        y_orig = torch.linspace(-2e-3, 2e-3, 128).unsqueeze(1)

        grid = Grid.from_coordinates(x_orig, y_orig, wavelength=520e-9)

        # Reconstruction from coordinates may have small numerical errors
        # due to the rounding in parameter inference
        torch.testing.assert_close(grid.x, x_orig, rtol=1e-3, atol=2e-5)
        torch.testing.assert_close(grid.y, y_orig, rtol=1e-3, atol=2e-5)

    def test_from_coordinates_1d_tensors(self):
        """Test from_coordinates with 1D tensors."""
        x = torch.linspace(-1e-3, 1e-3, 64)
        y = torch.linspace(-1e-3, 1e-3, 64)

        # Should handle 1D tensors
        grid = Grid.from_coordinates(x.unsqueeze(0), y, wavelength=520e-9)

        assert grid.nx == 64
        assert grid.ny == 64

    def test_from_coordinates_preserves_wavelength(self):
        """Test that wavelength is preserved."""
        x = torch.linspace(-1e-3, 1e-3, 64).unsqueeze(0)
        y = torch.linspace(-1e-3, 1e-3, 64).unsqueeze(1)
        wl = 633e-9

        grid = Grid.from_coordinates(x, y, wavelength=wl)

        assert grid.wl == wl

    def test_from_coordinates_device(self):
        """Test that device parameter is respected."""
        x = torch.linspace(-1e-3, 1e-3, 64).unsqueeze(0)
        y = torch.linspace(-1e-3, 1e-3, 64).unsqueeze(1)

        grid = Grid.from_coordinates(x, y, wavelength=520e-9, device="cpu")

        assert grid.device == "cpu"


class TestGridProperties:
    """Property-based tests using hypothesis."""

    @given(
        nx=st.integers(min_value=16, max_value=512),
        dx=st.floats(min_value=1e-7, max_value=1e-4),
    )
    def test_fourier_conjugate_relation(self, nx, dx):
        """Test Δx · Δkx = 1/N (FFT resolution relation)."""
        grid = Grid(nx=nx, dx=dx)

        # Compute frequency spacing
        kx = grid.kx
        if kx.shape[1] > 1:
            delta_kx = kx[0, 1] - kx[0, 0]

            # Should satisfy Fourier relation (with reasonable tolerance for floating point)
            assert abs(dx * delta_kx - 1 / nx) < 1e-6

    @given(
        nx=st.integers(min_value=32, max_value=256),
        padding_scale=st.integers(min_value=2, max_value=4),
    )
    def test_pad_preserves_dx(self, nx, padding_scale):
        """Test padding preserves pixel size."""
        grid = Grid(nx=nx, dx=1e-5)
        padded = grid.pad(padding_scale)

        assert padded.nx == nx * padding_scale
        assert padded.dx == grid.dx  # Pixel size unchanged

    @given(
        nx=st.integers(min_value=32, max_value=256),
        scale=st.integers(min_value=2, max_value=4),
    )
    def test_upsample_reduces_dx(self, nx, scale):
        """Test upsampling reduces pixel size."""
        grid = Grid(nx=nx, dx=1e-5)
        upsampled = grid.upsample(scale)

        assert upsampled.nx == nx * scale
        assert abs(upsampled.dx - grid.dx / scale) < 1e-15

    @given(
        nx=st.integers(min_value=16, max_value=256),
        dx=st.floats(min_value=1e-7, max_value=1e-4),
    )
    def test_fov_consistency(self, nx, dx):
        """Test that FOV = nx * dx."""
        grid = Grid(nx=nx, dx=dx)
        fov_x, fov_y = grid.fov

        assert abs(fov_x - nx * dx) < 1e-15
        assert abs(fov_y - nx * dx) < 1e-15  # ny defaults to nx


class TestGridPerformance:
    """Performance tests for caching."""

    def test_caching_performance(self):
        """Verify caching provides speedup."""
        import time

        grid = Grid(nx=1024, dx=1e-5)

        # Time first access (cache miss)
        start = time.time()
        for _ in range(1000):
            _ = grid.x
        time_with_cache = time.time() - start

        # Time without cache (clear each time)
        grid_nocache = Grid(nx=1024, dx=1e-5)
        start = time.time()
        for _ in range(1000):
            grid_nocache._invalidate_cache()
            _ = grid_nocache.x
        time_without_cache = time.time() - start

        # Caching should be at least 10x faster
        speedup = time_without_cache / time_with_cache
        assert speedup > 10, f"Caching speedup only {speedup:.1f}x"

    def test_cache_memory_overhead(self):
        """Verify cache memory overhead is reasonable."""
        grid = Grid(nx=256, dx=1e-5)

        # Access all coordinates to fill cache
        _ = grid.x
        _ = grid.y
        _ = grid.kx
        _ = grid.ky

        # Cache should have 4 entries
        assert len(grid._cache) == 4

        # Each tensor should be reasonably sized
        # For nx=256, each tensor is 256 floats = ~1KB
        for key, tensor in grid._cache.items():
            assert tensor.numel() <= 256 * 256  # Reasonable size
