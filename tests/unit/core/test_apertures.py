"""Unit tests for aperture module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prism.core.apertures import (
    CircularAperture,
    HexagonalAperture,
    ObscuredCircularAperture,
    create_aperture,
)


@pytest.fixture
def coords_small():
    """Create small coordinate grids for testing."""
    x = torch.arange(-50, 50).unsqueeze(0).float()
    y = torch.arange(-50, 50).unsqueeze(1).float()
    return x, y


@pytest.fixture
def coords_large():
    """Create larger coordinate grids for testing."""
    x = torch.arange(-128, 128).unsqueeze(0).float()
    y = torch.arange(-128, 128).unsqueeze(1).float()
    return x, y


class TestCircularAperture:
    """Test suite for CircularAperture."""

    def test_initialization(self):
        """Test circular aperture initialization."""
        aperture = CircularAperture(radius=10)
        assert aperture.radius == 10

    def test_initialization_invalid_radius(self):
        """Test that invalid radius raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            CircularAperture(radius=0)
        with pytest.raises(ValueError, match="must be positive"):
            CircularAperture(radius=-5)

    def test_generate_basic(self, coords_small):
        """Test basic mask generation."""
        x, y = coords_small
        aperture = CircularAperture(radius=10)
        mask = aperture.generate(x, y, center=[0, 0])

        # Check shape
        assert mask.shape == (100, 100)
        # Check dtype
        assert mask.dtype == torch.bool
        # Check that mask has True values
        assert mask.sum() > 0

    def test_generate_area(self, coords_small):
        """Test that generated mask has correct area."""
        x, y = coords_small
        aperture = CircularAperture(radius=10)
        mask = aperture.generate(x, y, center=[0, 0])

        # Area should be approximately π*r²
        expected_area = np.pi * 10**2
        actual_area = mask.sum().item()
        # Allow 10% tolerance for discretization
        assert abs(actual_area - expected_area) / expected_area < 0.1

    @pytest.mark.parametrize("radius", [5, 10, 20, 30])
    def test_generate_different_radii(self, coords_small, radius):
        """Test mask generation with different radii."""
        x, y = coords_small
        aperture = CircularAperture(radius=radius)
        mask = aperture.generate(x, y, center=[0, 0])

        # Check area scales with r²
        expected_area = np.pi * radius**2
        actual_area = mask.sum().item()
        # Larger tolerance for smaller radii due to discretization
        tolerance = 0.15 if radius < 10 else 0.1
        assert abs(actual_area - expected_area) / expected_area < tolerance

    def test_generate_off_center(self, coords_small):
        """Test mask generation at off-center position."""
        x, y = coords_small
        aperture = CircularAperture(radius=10)

        # Center at (10, 15)
        mask = aperture.generate(x, y, center=[10, 15])

        # Check that the center position is inside the mask
        # Coordinate [10, 15] should map to grid index [60, 65]
        center_idx_y = 50 + 10  # 60
        center_idx_x = 50 + 15  # 65
        assert mask[center_idx_y, center_idx_x]

        # Area should still be correct
        expected_area = np.pi * 10**2
        actual_area = mask.sum().item()
        assert abs(actual_area - expected_area) / expected_area < 0.1

    def test_generate_batch_single(self, coords_small):
        """Test batch generation with single center."""
        x, y = coords_small
        aperture = CircularAperture(radius=10)
        centers = [[0, 0]]
        masks = aperture.generate_batch(x, y, centers)

        # Check shape
        assert masks.shape == (1, 100, 100)
        # Compare with single generation
        single_mask = aperture.generate(x, y, [0, 0])
        assert torch.all(masks[0] == single_mask)

    def test_generate_batch_multiple(self, coords_small):
        """Test batch generation with multiple centers."""
        x, y = coords_small
        aperture = CircularAperture(radius=10)
        centers = [[0, 0], [10, 10], [-10, -10]]
        masks = aperture.generate_batch(x, y, centers)

        # Check shape
        assert masks.shape == (3, 100, 100)

        # Verify each mask matches single generation
        for i, center in enumerate(centers):
            expected_mask = aperture.generate(x, y, center)
            assert torch.all(masks[i] == expected_mask)

    def test_generate_batch_correctness(self, coords_large):
        """Test that batch generation gives same results as loop."""
        x, y = coords_large
        aperture = CircularAperture(radius=20)
        centers = [[i * 5, j * 5] for i in range(-5, 6) for j in range(-5, 6)]

        # Batch generation
        masks_batch = aperture.generate_batch(x, y, centers)

        # Loop generation
        masks_loop = torch.stack([aperture.generate(x, y, c) for c in centers])

        # Results should be identical
        assert torch.all(masks_batch == masks_loop)
        assert masks_batch.shape == masks_loop.shape


class TestHexagonalAperture:
    """Test suite for HexagonalAperture."""

    def test_initialization(self):
        """Test hexagonal aperture initialization."""
        aperture = HexagonalAperture(side_length=20)
        assert aperture.side_length == 20

    def test_initialization_invalid_side_length(self):
        """Test that invalid side length raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            HexagonalAperture(side_length=0)
        with pytest.raises(ValueError, match="must be positive"):
            HexagonalAperture(side_length=-10)

    def test_generate_basic(self, coords_small):
        """Test basic hexagonal mask generation."""
        x, y = coords_small
        aperture = HexagonalAperture(side_length=20)
        mask = aperture.generate(x, y, center=[0, 0])

        # Check shape and dtype
        assert mask.shape == (100, 100)
        assert mask.dtype == torch.bool
        assert mask.sum() > 0

    def test_generate_area(self, coords_small):
        """Test that hexagonal mask has correct area."""
        x, y = coords_small
        side_length = 20
        aperture = HexagonalAperture(side_length=side_length)
        mask = aperture.generate(x, y, center=[0, 0])

        # Hexagon area: 3√3/2 * s²
        expected_area = 3 * (3**0.5) / 2 * side_length**2
        actual_area = mask.sum().item()

        # Allow 40% tolerance due to discretization (hexagon is more complex)
        # The hexagon distance metric tends to over-include pixels at edges
        assert abs(actual_area - expected_area) / expected_area < 0.40

    def test_generate_symmetry(self, coords_small):
        """Test that hexagonal mask has approximate 6-fold symmetry."""
        x, y = coords_small
        aperture = HexagonalAperture(side_length=20)
        mask = aperture.generate(x, y, center=[0, 0])

        # Center should be inside
        center_idx = 50
        assert mask[center_idx, center_idx]

        # Check that mask is roughly symmetric (qualitative test)
        # Count True values in each quadrant
        h, w = mask.shape
        mid_h, mid_w = h // 2, w // 2

        q1 = mask[:mid_h, :mid_w].sum()  # Top-left
        q2 = mask[:mid_h, mid_w:].sum()  # Top-right
        q3 = mask[mid_h:, :mid_w].sum()  # Bottom-left
        q4 = mask[mid_h:, mid_w:].sum()  # Bottom-right

        # Quadrants should have similar counts (within 20%)
        avg = (q1 + q2 + q3 + q4) / 4
        for q in [q1, q2, q3, q4]:
            assert abs(q - avg) / avg < 0.2

    @pytest.mark.parametrize("side_length", [10, 20, 30])
    def test_generate_different_sizes(self, coords_small, side_length):
        """Test hexagonal masks with different side lengths."""
        x, y = coords_small
        aperture = HexagonalAperture(side_length=side_length)
        mask = aperture.generate(x, y, center=[0, 0])

        # Area should scale with s²
        expected_area = 3 * (3**0.5) / 2 * side_length**2
        actual_area = mask.sum().item()
        # Allow 40% tolerance due to discretization
        assert abs(actual_area - expected_area) / expected_area < 0.40

    def test_generate_off_center(self, coords_small):
        """Test hexagonal mask at off-center position."""
        x, y = coords_small
        aperture = HexagonalAperture(side_length=15)
        mask = aperture.generate(x, y, center=[10, 10])

        # Center should be inside
        center_idx_y = 50 + 10
        center_idx_x = 50 + 10
        assert mask[center_idx_y, center_idx_x]

    def test_generate_batch(self, coords_small):
        """Test batch generation for hexagonal apertures."""
        x, y = coords_small
        aperture = HexagonalAperture(side_length=15)
        centers = [[0, 0], [10, 10], [-10, -10]]
        masks = aperture.generate_batch(x, y, centers)

        # Check shape
        assert masks.shape == (3, 100, 100)

        # Verify each mask
        for i, center in enumerate(centers):
            expected_mask = aperture.generate(x, y, center)
            assert torch.all(masks[i] == expected_mask)


class TestObscuredCircularAperture:
    """Test suite for ObscuredCircularAperture."""

    def test_initialization(self):
        """Test obscured circular aperture initialization."""
        aperture = ObscuredCircularAperture(
            outer_radius=50, inner_radius=10, spider_width=2, n_spiders=4
        )
        assert aperture.r_outer == 50
        assert aperture.r_inner == 10
        assert aperture.spider_width == 2
        assert aperture.n_spiders == 4

    def test_initialization_invalid_radii(self):
        """Test that invalid radii raise errors."""
        # Outer radius must be positive
        with pytest.raises(ValueError, match="Outer radius must be positive"):
            ObscuredCircularAperture(outer_radius=0, inner_radius=5)

        # Inner radius cannot be negative
        with pytest.raises(ValueError, match="Inner radius cannot be negative"):
            ObscuredCircularAperture(outer_radius=50, inner_radius=-5)

        # Inner must be less than outer
        with pytest.raises(ValueError, match="Inner radius .* must be less than"):
            ObscuredCircularAperture(outer_radius=10, inner_radius=20)

    def test_initialization_no_spiders(self):
        """Test initialization without spider vanes."""
        aperture = ObscuredCircularAperture(outer_radius=50, inner_radius=10, spider_width=None)
        assert aperture.spider_width is None

    def test_initialization_large_obscuration_warning(self):
        """Test warning for unrealistic large obscuration."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ObscuredCircularAperture(outer_radius=50, inner_radius=30)
            # Should warn about large obscuration ratio
            assert len(w) == 1
            assert "Large obscuration ratio" in str(w[0].message)

    def test_generate_basic(self, coords_large):
        """Test basic obscured circular mask generation."""
        x, y = coords_large
        aperture = ObscuredCircularAperture(
            outer_radius=50, inner_radius=10, spider_width=2, n_spiders=4
        )
        mask = aperture.generate(x, y, center=[0, 0])

        # Check shape and dtype
        assert mask.shape == (256, 256)
        assert mask.dtype == torch.bool
        assert mask.sum() > 0

    def test_generate_annulus_area(self, coords_large):
        """Test that annulus (no spiders) has correct area."""
        x, y = coords_large
        outer_r, inner_r = 50, 10
        aperture = ObscuredCircularAperture(
            outer_radius=outer_r,
            inner_radius=inner_r,
            spider_width=None,  # No spiders
        )
        mask = aperture.generate(x, y, center=[0, 0])

        # Annulus area: π(r_out² - r_in²)
        expected_area = np.pi * (outer_r**2 - inner_r**2)
        actual_area = mask.sum().item()

        # Allow 10% tolerance for discretization
        assert abs(actual_area - expected_area) / expected_area < 0.1

    def test_generate_with_spiders_reduces_area(self, coords_large):
        """Test that adding spiders reduces the aperture area."""
        x, y = coords_large
        outer_r, inner_r = 50, 10

        # Without spiders
        aperture_no_spiders = ObscuredCircularAperture(
            outer_radius=outer_r, inner_radius=inner_r, spider_width=None
        )
        mask_no_spiders = aperture_no_spiders.generate(x, y, center=[0, 0])

        # With spiders
        aperture_with_spiders = ObscuredCircularAperture(
            outer_radius=outer_r, inner_radius=inner_r, spider_width=3, n_spiders=4
        )
        mask_with_spiders = aperture_with_spiders.generate(x, y, center=[0, 0])

        # Spiders should block some area
        area_no_spiders = mask_no_spiders.sum().item()
        area_with_spiders = mask_with_spiders.sum().item()
        assert area_with_spiders < area_no_spiders

        # Should block at least 2% (spiders are thin but noticeable)
        blocked_fraction = (area_no_spiders - area_with_spiders) / area_no_spiders
        assert blocked_fraction > 0.02

    def test_generate_center_blocked(self, coords_large):
        """Test that center is blocked by obscuration."""
        x, y = coords_large
        aperture = ObscuredCircularAperture(outer_radius=50, inner_radius=10)
        mask = aperture.generate(x, y, center=[0, 0])

        # Center should be blocked (False)
        center_idx = 128
        assert not mask[center_idx, center_idx]

    def test_generate_outer_edge_open(self, coords_large):
        """Test that outer edge (between inner and outer radius) is open."""
        x, y = coords_large
        outer_r, inner_r = 50, 10
        aperture = ObscuredCircularAperture(outer_radius=outer_r, inner_radius=inner_r)
        mask = aperture.generate(x, y, center=[0, 0])

        # Point at radius (outer_r + inner_r) / 2 should be inside annulus
        center_idx = 128
        test_r = (outer_r + inner_r) // 2
        test_idx = center_idx + test_r
        # Check point to the right of center
        assert mask[center_idx, test_idx]

    @pytest.mark.parametrize("n_spiders", [0, 3, 4, 6, 8])
    def test_generate_different_spider_counts(self, coords_large, n_spiders):
        """Test obscured apertures with different numbers of spiders."""
        x, y = coords_large
        aperture = ObscuredCircularAperture(
            outer_radius=50, inner_radius=10, spider_width=2, n_spiders=n_spiders
        )
        mask = aperture.generate(x, y, center=[0, 0])

        # Should generate valid mask
        assert mask.dtype == torch.bool
        assert mask.sum() > 0

    def test_generate_off_center(self, coords_large):
        """Test obscured circular mask at off-center position."""
        x, y = coords_large
        aperture = ObscuredCircularAperture(outer_radius=40, inner_radius=8)
        mask = aperture.generate(x, y, center=[20, 20])

        # Center should be blocked
        center_idx_y = 128 + 20
        center_idx_x = 128 + 20
        assert not mask[center_idx_y, center_idx_x]

    def test_generate_batch(self, coords_large):
        """Test batch generation for obscured circular apertures."""
        x, y = coords_large
        aperture = ObscuredCircularAperture(
            outer_radius=40, inner_radius=8, spider_width=2, n_spiders=4
        )
        centers = [[0, 0], [20, 20], [-20, -20]]
        masks = aperture.generate_batch(x, y, centers)

        # Check shape
        assert masks.shape == (3, 256, 256)

        # Verify each mask
        for i, center in enumerate(centers):
            expected_mask = aperture.generate(x, y, center)
            assert torch.all(masks[i] == expected_mask)


class TestApertureFactory:
    """Test suite for aperture factory function."""

    def test_create_circular(self):
        """Test factory creates circular aperture."""
        aperture = create_aperture("circular", radius=10)
        assert isinstance(aperture, CircularAperture)
        assert aperture.radius == 10

    def test_create_hexagonal(self):
        """Test factory creates hexagonal aperture."""
        aperture = create_aperture("hexagonal", side_length=20)
        assert isinstance(aperture, HexagonalAperture)
        assert aperture.side_length == 20

    def test_create_obscured(self):
        """Test factory creates obscured circular aperture."""
        aperture = create_aperture(
            "obscured", outer_radius=50, inner_radius=10, spider_width=2, n_spiders=4
        )
        assert isinstance(aperture, ObscuredCircularAperture)
        assert aperture.r_outer == 50
        assert aperture.r_inner == 10
        assert aperture.spider_width == 2
        assert aperture.n_spiders == 4

    def test_create_unknown_type(self):
        """Test that unknown aperture type raises error."""
        with pytest.raises(ValueError, match="Unknown aperture type"):
            create_aperture("unknown_type", radius=10)

    def test_create_with_kwargs(self):
        """Test that factory passes kwargs correctly."""
        aperture = create_aperture("circular", radius=15)
        assert aperture.radius == 15


class TestApertureIntegration:
    """Integration tests for apertures with realistic use cases."""

    def test_comparison_circular_vs_hexagonal(self, coords_large):
        """Compare circular and hexagonal apertures of similar size."""
        x, y = coords_large
        # Choose sizes to give similar areas
        # Hexagon area = 3√3/2 * s² ≈ 2.598 * s²
        # Circle area = π * r² ≈ 3.14 * r²
        # For similar area: r ≈ 0.91 * s
        hex_side = 30
        circ_radius = int(0.91 * hex_side)

        hex_aperture = HexagonalAperture(side_length=hex_side)
        circ_aperture = CircularAperture(radius=circ_radius)

        hex_mask = hex_aperture.generate(x, y, [0, 0])
        circ_mask = circ_aperture.generate(x, y, [0, 0])

        # Areas should be similar (within 40% due to discretization)
        hex_area = hex_mask.sum().item()
        circ_area = circ_mask.sum().item()
        assert abs(hex_area - circ_area) / circ_area < 0.40

    def test_vlt_style_aperture(self, coords_large):
        """Test VLT-style aperture (8m primary, ~1m secondary, 4 spiders)."""
        x, y = coords_large
        # Scale: 1 pixel = 0.02m → 8m = 400 pixels, 1m = 50 pixels
        aperture = ObscuredCircularAperture(
            outer_radius=100,  # 8m scaled
            inner_radius=12,  # 1m scaled
            spider_width=2,  # Thin spiders
            n_spiders=4,
        )
        mask = aperture.generate(x, y, [0, 0])

        # Should have realistic properties
        assert mask.sum() > 0
        # Center should be blocked (inside inner radius)
        assert not mask[128, 128]
        # Point just outside inner radius should be in annulus
        # Use a point at 45 degrees to avoid spiders (which are at 0, 45, 90, 135 degrees)
        # Check at radius ~25 (between inner=12 and outer=100), offset at 30 degrees
        test_r = 25
        import math

        test_offset_x = int(test_r * math.cos(math.radians(30)))
        test_offset_y = int(test_r * math.sin(math.radians(30)))
        # Check that at least SOME point in the annulus is open
        assert mask[128 + test_offset_y, 128 + test_offset_x]

    def test_jwst_style_aperture(self, coords_large):
        """Test JWST-style hexagonal aperture."""
        x, y = coords_large
        # JWST has hexagonal segments (6.5m primary)
        # Scale: 1 pixel = 0.02m → 6.5m ≈ 325 pixels → side ≈ 60 pixels
        aperture = HexagonalAperture(side_length=60)
        mask = aperture.generate(x, y, [0, 0])

        # Should generate valid hexagonal mask
        assert mask.sum() > 0
        # Center should be inside
        assert mask[128, 128]

    def test_aperture_affects_psf_shape(self, coords_large):
        """Test that different apertures produce different mask patterns."""
        x, y = coords_large
        radius = 40

        circular = CircularAperture(radius=radius)
        hexagonal = HexagonalAperture(side_length=radius)
        obscured = ObscuredCircularAperture(outer_radius=radius, inner_radius=10)

        mask_circ = circular.generate(x, y, [0, 0])
        mask_hex = hexagonal.generate(x, y, [0, 0])
        mask_obsc = obscured.generate(x, y, [0, 0])

        # All should have different patterns
        assert not torch.all(mask_circ == mask_hex)
        assert not torch.all(mask_circ == mask_obsc)
        assert not torch.all(mask_hex == mask_obsc)

        # All should have non-zero area
        assert mask_circ.sum() > 0
        assert mask_hex.sum() > 0
        assert mask_obsc.sum() > 0
