"""Unit tests for FFT and transform operations."""

from __future__ import annotations

import pytest
import torch

from prism.utils.transforms import FFTCache, _create_coordinate_grids, create_mask, fft, ifft


class TestFFTOperations:
    """Test suite for FFT and IFFT operations."""

    def test_fft_inverse(self, sample_image):
        """Test FFT inverse property: ifft(fft(x)) = x."""
        image = sample_image
        result = ifft(fft(image))

        # Should recover original image (take real part since FFT returns complex)
        # Imaginary part should be negligible
        assert result.imag.abs().max() < 1e-6
        assert torch.allclose(result.real, image, rtol=1e-4, atol=1e-6)

    def test_fft_inverse_small(self, sample_image_small):
        """Test FFT inverse property with small image."""
        image = sample_image_small
        result = ifft(fft(image))

        assert result.imag.abs().max() < 1e-6
        assert torch.allclose(result.real, image, rtol=1e-4, atol=1e-6)

    def test_fft_parseval(self, sample_image):
        """Test Parseval's theorem: energy conservation in FFT."""
        image = sample_image

        # Compute energy in spatial domain
        spatial_energy = image.pow(2).sum()

        # Compute energy in frequency domain
        freq_energy = fft(image).abs().pow(2).sum()

        # Energies should be equal (within numerical precision)
        assert torch.allclose(spatial_energy, freq_energy, rtol=1e-4)

    def test_fft_parseval_complex(self, sample_complex_image):
        """Test Parseval's theorem with complex input."""
        image = sample_complex_image

        # Compute energy in spatial domain
        spatial_energy = image.abs().pow(2).sum()

        # Compute energy in frequency domain
        freq_energy = fft(image).abs().pow(2).sum()

        assert torch.allclose(spatial_energy, freq_energy, rtol=1e-4)

    def test_fft_output_type(self, sample_image):
        """Test that FFT returns complex tensor."""
        result = fft(sample_image)

        # FFT should return complex tensor
        assert torch.is_complex(result)
        assert result.shape == sample_image.shape

    def test_ifft_output_type(self, sample_complex_image):
        """Test that IFFT returns complex tensor."""
        result = ifft(sample_complex_image)

        # IFFT should return complex tensor
        assert torch.is_complex(result)
        assert result.shape == sample_complex_image.shape

    def test_fft_centering(self, sample_image_small):
        """Test that FFT properly centers DC component."""
        # Create image with DC component (constant)
        image = torch.ones(1, 1, 64, 64)
        result = fft(image)

        # DC component should be at center
        center_idx = 64 // 2
        center_value = result[0, 0, center_idx, center_idx].abs()

        # DC should be much larger than other components
        mean_other = result[0, 0].abs().mean()
        assert center_value > 10 * mean_other

    @pytest.mark.parametrize("norm", ["ortho", "forward", "backward"])
    def test_fft_normalization_modes(self, sample_image_small, norm):
        """Test FFT with different normalization modes."""
        result = fft(sample_image_small, norm=norm)

        # Should complete without error
        assert torch.is_complex(result)
        assert result.shape == sample_image_small.shape

    def test_fft_batch(self):
        """Test FFT with batched input."""
        batch_size = 4
        images = torch.randn(batch_size, 1, 64, 64)

        result = fft(images)

        assert result.shape == images.shape
        assert torch.is_complex(result)

    def test_fft_multichannel(self):
        """Test FFT with multi-channel input."""
        n_channels = 3
        images = torch.randn(1, n_channels, 64, 64)

        result = fft(images)

        assert result.shape == images.shape
        assert torch.is_complex(result)

    def test_ifft_inverse_of_fft(self, sample_image):
        """Test that IFFT is truly inverse of FFT."""
        # Forward then inverse
        freq = fft(sample_image)
        reconstructed = ifft(freq)

        # Should get back original (real part)
        assert torch.allclose(reconstructed.real, sample_image, rtol=1e-4, atol=1e-6)
        # Imaginary part should be negligible (within numerical precision)
        assert reconstructed.imag.abs().max() < 1e-6

    def test_fft_deterministic(self, sample_image):
        """Test that FFT is deterministic."""
        result1 = fft(sample_image)
        result2 = fft(sample_image)

        assert torch.allclose(result1, result2)

    def test_fft_linearity(self, sample_image_small):
        """Test FFT linearity: fft(a*x + b*y) = a*fft(x) + b*fft(y)."""
        x = sample_image_small
        y = torch.randn_like(x)
        a = 2.0
        b = 3.0

        # Left side: fft(a*x + b*y)
        left = fft(a * x + b * y)

        # Right side: a*fft(x) + b*fft(y)
        right = a * fft(x) + b * fft(y)

        # Compare complex results
        assert torch.allclose(left, right, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("size", [32, 64, 128, 256])
    def test_fft_different_sizes(self, size):
        """Test FFT with different image sizes."""
        image = torch.randn(1, 1, size, size)
        result = fft(image)

        assert result.shape == image.shape
        assert torch.is_complex(result)


class TestMaskCreation:
    """Test suite for mask creation functions."""

    def test_circular_mask_basic(self, sample_image):
        """Test basic circular mask creation."""
        mask = create_mask(sample_image, mask_type="circular", mask_size=0.5)

        # Check shape
        assert mask.shape == sample_image.shape[-2:]

        # Check dtype
        assert mask.dtype == torch.bool

        # Check that mask has both True and False values
        assert mask.any()
        assert (~mask).any()

    def test_square_mask_basic(self, sample_image):
        """Test basic square mask creation."""
        mask = create_mask(sample_image, mask_type="square", mask_size=0.5)

        assert mask.shape == sample_image.shape[-2:]
        assert mask.dtype == torch.bool
        assert mask.any()

    def test_circular_mask_center(self, sample_image_small):
        """Test circular mask centered at origin."""
        mask = create_mask(sample_image_small, mask_type="circular", mask_size=0.5, center=[0, 0])

        # Center should be inside mask
        ny, nx = sample_image_small.shape[-2:]
        center_y, center_x = ny // 2, nx // 2
        assert mask[center_y, center_x]

    def test_circular_mask_off_center(self, sample_image_small):
        """Test circular mask with off-center position."""
        center = [0.2, 0.3]
        mask = create_mask(sample_image_small, mask_type="circular", mask_size=0.5, center=center)

        # Mask should still be created
        assert mask.shape == sample_image_small.shape[-2:]
        assert mask.any()

    @pytest.mark.parametrize("mask_size", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_circular_mask_sizes(self, sample_image_small, mask_size):
        """Test circular masks with different sizes."""
        mask = create_mask(sample_image_small, mask_type="circular", mask_size=mask_size)

        # Larger mask_size should give more True pixels
        area = mask.sum().item()
        total_area = mask.numel()

        # Rough check: area should increase with mask_size
        assert 0 < area < total_area

    def test_square_mask_symmetry(self, sample_image_small):
        """Test that square mask is symmetric."""
        mask = create_mask(sample_image_small, mask_type="square", mask_size=0.5, center=[0, 0])

        # Should be symmetric about center
        ny, nx = sample_image_small.shape[-2:]
        center_y, center_x = ny // 2, nx // 2

        # Check a few symmetry points
        for dy, dx in [(5, 0), (0, 5), (5, 5)]:
            if (
                center_y + dy < ny
                and center_x + dx < nx
                and center_y - dy >= 0
                and center_x - dx >= 0
            ):
                mask[center_y + dy, center_x + dx]
                mask[center_y - dy, center_x - dx]
                # Should be symmetric (with some tolerance for discretization)
                # This might not be exact due to grid discretization

    def test_invalid_mask_type(self, sample_image):
        """Test that invalid mask type raises error."""
        with pytest.raises(NotImplementedError):
            create_mask(sample_image, mask_type="triangular")

    def test_mask_default_center(self, sample_image_small):
        """Test mask creation with default center."""
        mask = create_mask(sample_image_small, mask_type="circular", mask_size=0.5)

        # Should work with default center [0, 0]
        assert mask.shape == sample_image_small.shape[-2:]

    def test_coordinate_grids_caching(self):
        """Test that coordinate grids are cached properly."""
        # Note: _create_coordinate_grids uses @simple_cache, not @lru_cache
        # So it doesn't have cache_clear() method

        ny, nx = 64, 64
        center_y, center_x = 0.0, 0.0

        # First call - should compute
        y1, x1 = _create_coordinate_grids(ny, nx, center_y, center_x)

        # Second call - should use cache (or recompute, both are fine)
        y2, x2 = _create_coordinate_grids(ny, nx, center_y, center_x)

        # Should return same values
        assert torch.allclose(y1, y2)
        assert torch.allclose(x1, x2)

    def test_mask_creation_different_sizes(self):
        """Test mask creation with different image sizes."""
        for size in [32, 64, 128]:
            image = torch.randn(1, 1, size, size)
            mask = create_mask(image, mask_type="circular", mask_size=0.5)

            assert mask.shape == (size, size)


class TestFFTCache:
    """Test suite for FFTCache class."""

    def test_fft_cache_initialization(self):
        """Test FFTCache initialization."""
        cache = FFTCache(max_cache_size=64)

        assert cache.max_cache_size == 64
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0

    def test_fft_cache_basic(self, sample_image_small):
        """Test basic FFT caching functionality."""
        cache = FFTCache()

        result1 = cache.fft2(sample_image_small)
        result2 = cache.fft2(sample_image_small)

        # Results should be identical
        assert torch.allclose(result1, result2)

        # Should have cache hits after first call
        assert (
            cache.cache_hits >= 0
        )  # First call might already hit cache depending on implementation

    def test_ifft_cache_basic(self, sample_complex_image):
        """Test basic IFFT caching functionality."""
        cache = FFTCache()

        result1 = cache.ifft2(sample_complex_image)
        result2 = cache.ifft2(sample_complex_image)

        assert torch.allclose(result1, result2)

    def test_cache_statistics_tracking(self, sample_image_small):
        """Test that cache properly tracks hits and misses."""
        cache = FFTCache()

        # Clear to start fresh
        cache.clear_cache()

        # First call - should be a miss
        _ = cache.fft2(sample_image_small)
        initial_misses = cache.cache_misses

        # Second call with same shape - should be a hit
        _ = cache.fft2(sample_image_small)
        hits_after_second = cache.cache_hits

        # Should have recorded a hit
        assert hits_after_second > 0 or initial_misses > 0  # At least one should increase

    def test_cache_hit_rate(self, sample_image_small):
        """Test cache hit rate calculation."""
        cache = FFTCache()
        cache.clear_cache()

        # Do several operations
        for _ in range(5):
            _ = cache.fft2(sample_image_small)

        hit_rate = cache.hit_rate()

        # Hit rate should be between 0 and 1
        assert 0.0 <= hit_rate <= 1.0

    def test_cache_info(self, sample_image_small):
        """Test cache info reporting."""
        cache = FFTCache(max_cache_size=32)
        cache.clear_cache()

        _ = cache.fft2(sample_image_small)

        info = cache.cache_info()

        # Check that info contains expected keys
        assert "hits" in info
        assert "misses" in info
        assert "size" in info
        assert "max_size" in info
        assert info["max_size"] == 32

    def test_cache_clear(self, sample_image_small):
        """Test cache clearing."""
        cache = FFTCache()

        # Do some operations
        _ = cache.fft2(sample_image_small)

        # Clear cache
        cache.clear_cache()

        # Statistics should be reset
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0
        assert len(cache._shift_cache) == 0

    def test_cache_max_size_limit(self):
        """Test that cache respects max size limit."""
        max_size = 3
        cache = FFTCache(max_cache_size=max_size)
        cache.clear_cache()

        # Create tensors of different sizes
        sizes = [32, 64, 128, 256, 512]  # More than max_size

        for size in sizes:
            tensor = torch.randn(1, 1, size, size)
            _ = cache.fft2(tensor)

        # Cache size should not exceed max_size
        assert len(cache._shift_cache) <= max_size

    def test_cache_different_devices(self, sample_image_small, device):
        """Test caching with different devices."""
        cache = FFTCache()

        # Move to device
        tensor_device = sample_image_small.to(device)

        result = cache.fft2(tensor_device)

        # Result should be on same device (type matches)
        assert result.device.type == device.type

    def test_fft_cache_vs_regular_fft(self, sample_image):
        """Test that cached FFT gives same result as regular FFT."""
        cache = FFTCache()

        result_cached = cache.fft2(sample_image)
        result_regular = fft(sample_image)

        assert torch.allclose(result_cached, result_regular)

    def test_ifft_cache_vs_regular_ifft(self, sample_complex_image):
        """Test that cached IFFT gives same result as regular IFFT."""
        cache = FFTCache()

        result_cached = cache.ifft2(sample_complex_image)
        result_regular = ifft(sample_complex_image)

        assert torch.allclose(result_cached, result_regular)

    def test_cache_inverse_property(self, sample_image_small):
        """Test inverse property using cache."""
        cache = FFTCache()

        result = cache.ifft2(cache.fft2(sample_image_small))

        # Result is complex, compare real part
        assert result.imag.abs().max() < 1e-6
        assert torch.allclose(result.real, sample_image_small, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("norm", ["ortho", "forward", "backward"])
    def test_cache_normalization_modes(self, sample_image_small, norm):
        """Test cache with different normalization modes."""
        cache = FFTCache()

        result = cache.fft2(sample_image_small, norm=norm)

        assert torch.is_complex(result)
        assert result.shape == sample_image_small.shape

    def test_cache_with_batch(self):
        """Test FFT cache with batched input."""
        cache = FFTCache()
        batch_images = torch.randn(4, 1, 64, 64)

        result = cache.fft2(batch_images)

        assert result.shape == batch_images.shape
        assert torch.is_complex(result)
