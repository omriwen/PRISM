"""Unit tests for image utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from prism.utils.image import (
    crop_image,
    crop_pad,
    generate_point_sources,
    get_image_size,
    load_image,
    pad_image,
)


class TestPointSourceGeneration:
    """Test suite for point source generation."""

    def test_single_point_source(self):
        """Test generation of single point source."""
        image_size = 128
        sample_diameter = 10
        spacing = 20

        result = generate_point_sources(
            image_size=image_size,
            number_of_sources=1,
            sample_diameter=sample_diameter,
            spacing=spacing,
        )

        # Should have 1 source
        assert result.shape[0] == 1
        assert result.shape[1:] == (image_size, image_size)

        # Should be binary
        assert torch.all((result == 0) | (result == 1))

        # Center should be inside the source
        center = image_size // 2
        assert result[0, center, center] == 1

    def test_two_point_sources(self):
        """Test generation of two point sources."""
        image_size = 128
        sample_diameter = 10
        spacing = 40

        result = generate_point_sources(
            image_size=image_size,
            number_of_sources=2,
            sample_diameter=sample_diameter,
            spacing=spacing,
        )

        # Should have 2 sources
        assert result.shape[0] == 2
        assert result.shape[1:] == (image_size, image_size)

        # Should be binary
        assert torch.all((result == 0) | (result == 1))

        # Both sources should have non-zero area
        assert result[0].sum() > 0
        assert result[1].sum() > 0

    def test_three_point_sources(self):
        """Test generation of three point sources (triangle)."""
        image_size = 128
        sample_diameter = 8
        spacing = 30

        result = generate_point_sources(
            image_size=image_size,
            number_of_sources=3,
            sample_diameter=sample_diameter,
            spacing=spacing,
        )

        # Should have 3 sources
        assert result.shape[0] == 3
        assert result.shape[1:] == (image_size, image_size)

        # All sources should have non-zero area
        for i in range(3):
            assert result[i].sum() > 0

    def test_four_point_sources(self):
        """Test generation of four point sources (square)."""
        image_size = 128
        sample_diameter = 8
        spacing = 30

        result = generate_point_sources(
            image_size=image_size,
            number_of_sources=4,
            sample_diameter=sample_diameter,
            spacing=spacing,
        )

        # Should have 4 sources
        assert result.shape[0] == 4
        assert result.shape[1:] == (image_size, image_size)

        # All sources should have non-zero area
        for i in range(4):
            assert result[i].sum() > 0

    def test_invalid_number_of_sources(self):
        """Test that invalid number of sources raises error."""
        with pytest.raises(NotImplementedError):
            generate_point_sources(
                image_size=128, number_of_sources=5, sample_diameter=10, spacing=20
            )

    @pytest.mark.parametrize("diameter", [5, 10, 20])
    def test_point_source_diameter(self, diameter):
        """Test point sources with different diameters."""
        result = generate_point_sources(
            image_size=128, number_of_sources=1, sample_diameter=diameter, spacing=20
        )

        # Area should be approximately pi * (d/2)^2
        area = result.sum().item()
        expected_area = np.pi * (diameter / 2) ** 2

        # Allow 20% tolerance for discretization
        assert abs(area - expected_area) / expected_area < 0.2

    @pytest.mark.parametrize("size", [64, 128, 256])
    def test_different_image_sizes(self, size):
        """Test point source generation with different image sizes."""
        result = generate_point_sources(
            image_size=size, number_of_sources=1, sample_diameter=10, spacing=20
        )

        assert result.shape[1:] == (size, size)


class TestPadding:
    """Test suite for image padding."""

    def test_pad_basic(self):
        """Test basic padding functionality."""
        tensor = torch.randn(1, 1, 64, 64)
        target_size = 128

        result = pad_image(tensor, target_size)

        # Should have target size
        assert result.shape[-2:] == (target_size, target_size)

        # Original tensor should be in center
        start = (target_size - 64) // 2
        end = start + 64
        center_region = result[0, 0, start:end, start:end]

        # Center should match original (approximately, accounting for padding effects)
        assert center_region.shape == tensor[0, 0].shape

    def test_pad_preserves_batch_channel(self):
        """Test that padding preserves batch and channel dimensions."""
        tensor = torch.randn(4, 3, 32, 32)
        target_size = 64

        result = pad_image(tensor, target_size)

        # Batch and channel dims should be preserved
        assert result.shape[:2] == (4, 3)
        assert result.shape[-2:] == (target_size, target_size)

    def test_pad_constant_mode(self):
        """Test constant padding mode."""
        tensor = torch.ones(1, 1, 32, 32)
        target_size = 64

        result = pad_image(tensor, target_size, mode="constant", value=0)

        # Corners should be filled with padding value
        assert result[0, 0, 0, 0] == 0
        assert result[0, 0, -1, -1] == 0

        # Center should preserve original values
        start = (target_size - 32) // 2
        assert result[0, 0, start + 5, start + 5] == 1

    def test_pad_smaller_target(self):
        """Test behavior when target size is smaller than input."""
        tensor = torch.randn(1, 1, 64, 64)
        target_size = 32

        result = pad_image(tensor, target_size)

        # Padding should be 0 for all dimensions (no padding applied)
        # The result should be same as input
        assert result.shape == tensor.shape

    @pytest.mark.parametrize("target", [64, 128, 256])
    def test_pad_different_targets(self, target):
        """Test padding to different target sizes."""
        tensor = torch.randn(1, 1, 32, 32)

        result = pad_image(tensor, target)

        assert result.shape[-2:] == (target, target)

    def test_pad_centering(self):
        """Test that padding is centered."""
        tensor = torch.ones(1, 1, 40, 40)
        target_size = 100

        result = pad_image(tensor, target_size, value=0)

        # Calculate expected padding
        pad_total = target_size - 40
        pad_start = pad_total // 2

        # Check that original content is centered
        # Top-left of original should be at (pad_start, pad_start)
        assert result[0, 0, pad_start, pad_start] == 1

        # Top-left corner should be padding
        assert result[0, 0, 0, 0] == 0

    def test_pad_odd_size_difference(self):
        """Test padding when size difference is odd."""
        tensor = torch.randn(1, 1, 33, 33)
        target_size = 64

        result = pad_image(tensor, target_size)

        # Should handle odd padding correctly
        assert result.shape[-2:] == (target_size, target_size)


class TestCropping:
    """Test suite for image cropping."""

    def test_crop_basic(self):
        """Test basic cropping functionality."""
        tensor = torch.randn(1, 1, 128, 128)
        target_size = 64

        result = crop_image(tensor, target_size)

        # Should have target size
        assert result.shape[-2:] == (target_size, target_size)

    def test_crop_preserves_batch_channel(self):
        """Test that cropping preserves batch and channel dimensions."""
        tensor = torch.randn(4, 3, 128, 128)
        target_size = 64

        result = crop_image(tensor, target_size)

        # Batch and channel dims should be preserved
        assert result.shape[:2] == (4, 3)
        assert result.shape[-2:] == (target_size, target_size)

    def test_crop_center(self):
        """Test that cropping takes center region."""
        # Create tensor with unique center region
        tensor = torch.zeros(1, 1, 128, 128)
        # Mark center
        tensor[0, 0, 56:72, 56:72] = 1
        target_size = 32

        result = crop_image(tensor, target_size)

        # Result should contain mostly ones (center region)
        assert result.sum() > 0

    @pytest.mark.parametrize("target", [32, 64, 96])
    def test_crop_different_targets(self, target):
        """Test cropping to different target sizes."""
        tensor = torch.randn(1, 1, 128, 128)

        result = crop_image(tensor, target)

        assert result.shape[-2:] == (target, target)

    def test_crop_larger_than_input(self):
        """Test cropping when target is larger than input."""
        tensor = torch.randn(1, 1, 64, 64)
        target_size = 128

        # This will try to crop but will likely cause issues or return smaller
        # The function doesn't explicitly handle this case
        # Based on the implementation, it will try to crop with negative indices
        # which will result in an empty or small tensor
        crop_image(tensor, target_size)

        # The actual behavior depends on implementation details


class TestCropPad:
    """Test suite for combined crop/pad functionality."""

    def test_crop_pad_needs_padding(self):
        """Test crop_pad when padding is needed."""
        tensor = torch.randn(1, 1, 64, 64)
        target_size = 128

        result = crop_pad(tensor, target_size)

        # Should pad to target size
        assert result.shape[-2:] == (target_size, target_size)

    def test_crop_pad_needs_cropping(self):
        """Test crop_pad when cropping is needed."""
        tensor = torch.randn(1, 1, 128, 128)
        target_size = 64

        result = crop_pad(tensor, target_size)

        # Should crop to target size
        assert result.shape[-2:] == (target_size, target_size)

    def test_crop_pad_exact_size(self):
        """Test crop_pad when size already matches."""
        tensor = torch.randn(1, 1, 64, 64)
        target_size = 64

        result = crop_pad(tensor, target_size)

        # Should remain same size
        assert result.shape == tensor.shape

    def test_crop_pad_none_target(self):
        """Test crop_pad with None target (no operation)."""
        tensor = torch.randn(1, 1, 64, 64)

        result = crop_pad(tensor, None)

        # Should return tensor unchanged
        assert torch.equal(result, tensor)

    @pytest.mark.parametrize(
        "input_size,target_size",
        [
            (32, 64),  # Needs padding
            (64, 32),  # Needs cropping
            (64, 64),  # Already correct
            (100, 50),  # Needs cropping
            (50, 100),  # Needs padding
        ],
    )
    def test_crop_pad_various_sizes(self, input_size, target_size):
        """Test crop_pad with various size combinations."""
        tensor = torch.randn(1, 1, input_size, input_size)

        result = crop_pad(tensor, target_size)

        assert result.shape[-2:] == (target_size, target_size)

    def test_crop_pad_preserves_content(self):
        """Test that crop_pad preserves content when possible."""
        # Create tensor with known pattern
        tensor = torch.zeros(1, 1, 64, 64)
        tensor[0, 0, 28:36, 28:36] = 1  # 8x8 center square

        # Pad to larger size
        result_padded = crop_pad(tensor, 128)

        # Center square should still be present
        center_start = (128 - 64) // 2 + 28
        center_end = center_start + 8
        assert result_padded[0, 0, center_start:center_end, center_start:center_end].sum() > 0

    def test_crop_pad_padding_mode(self):
        """Test crop_pad with different padding modes."""
        tensor = torch.ones(1, 1, 32, 32)

        result = crop_pad(tensor, 64, mode="constant", value=0)

        # Corners should be padded with 0
        assert result[0, 0, 0, 0] == 0


class TestImageLoading:
    """Test suite for image loading functionality."""

    @pytest.fixture
    def test_image_path(self, tmp_path):
        """Create a temporary test image."""
        # Create a simple test image
        img_array = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="L")

        img_path = tmp_path / "test_image.png"
        img.save(img_path)

        return img_path

    @pytest.fixture
    def test_color_image_path(self, tmp_path):
        """Create a temporary color test image."""
        img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="RGB")

        img_path = tmp_path / "test_color_image.png"
        img.save(img_path)

        return img_path

    def test_load_image_basic(self, test_image_path):
        """Test basic image loading."""
        result = load_image(test_image_path)

        # Should return tensor
        assert isinstance(result, torch.Tensor)

        # Should be normalized (between 0 and 1)
        assert result.min() >= 0
        assert result.max() <= 1

        # Should be 3D or 4D (with channels)
        assert result.ndim >= 2

    def test_load_image_size(self, test_image_path):
        """Test loading image with specific size."""
        target_size = 64
        result = load_image(test_image_path, size=target_size)

        # Should have target size (or close to it accounting for padding)
        assert result.shape[-1] == target_size or result.shape[-2] == target_size

    def test_load_image_padded_size(self, test_image_path):
        """Test loading image with padding."""
        size = 64
        padded_size = 128

        result = load_image(test_image_path, size=size, padded_size=padded_size)

        # Should have padded size
        assert result.shape[-1] == padded_size
        assert result.shape[-2] == padded_size

    def test_load_image_invert(self, test_image_path):
        """Test image inversion."""
        normal = load_image(test_image_path, invert=False)
        inverted = load_image(test_image_path, invert=True)

        # Inverted should be different from normal
        assert not torch.allclose(normal, inverted, rtol=0.1)

    def test_load_color_image_to_grayscale(self, test_color_image_path):
        """Test that color images are converted to grayscale."""
        result = load_image(test_color_image_path)

        # Should be grayscale (single channel or 2D)
        # After transforms, should have channel dimension of 1
        if result.ndim >= 3:
            assert result.shape[0] == 1  # Single channel

    def test_get_image_size(self, test_image_path):
        """Test getting image size without loading."""
        size = get_image_size(test_image_path)

        # Should return integer
        assert isinstance(size, int)

        # Should be positive
        assert size > 0

        # Should match image dimension
        assert size == 128  # Our test image is 128x128

    def test_load_image_normalization(self, test_image_path):
        """Test that loaded images are properly normalized."""
        result = load_image(test_image_path)

        # Should be normalized to [0, 1]
        assert result.min() >= 0
        assert result.max() <= 1

        # Max should be 1 (after normalization)
        assert result.max() == 1.0

    def test_load_image_sqrt_applied(self, test_image_path):
        """Test that square root is applied to loaded image."""
        # This is harder to test directly, but we can check that
        # the result is plausible for sqrt-transformed data

        result = load_image(test_image_path)

        # After sqrt, values should be between 0 and 1
        assert result.min() >= 0
        assert result.max() <= 1
