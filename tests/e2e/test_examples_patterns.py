"""End-to-end tests for pattern generator examples."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch


PATTERNS_DIR = Path(__file__).parent.parent.parent / "examples" / "patterns"

# Add patterns directory to path for imports
sys.path.insert(0, str(PATTERNS_DIR))


@dataclass
class PatternConfig:
    """Mock config for pattern generators."""

    n_samples: int = 100
    roi_diameter: float = 100.0


class TestPatternGenerators:
    """Test pattern generator scripts."""

    @pytest.fixture
    def config(self) -> PatternConfig:
        """Return a pattern configuration."""
        return PatternConfig(n_samples=100, roi_diameter=100.0)

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        "pattern_module",
        [
            "concentric_circles",
            "continuous_spiral",
            "jittered_grid",
            "logarithmic_spiral",
        ],
    )
    def test_pattern_generates_valid_points(self, pattern_module, config):
        """Test that pattern generator produces valid sample points."""
        try:
            module = __import__(pattern_module)
        except ImportError:
            pytest.skip(f"Could not import {pattern_module}")

        if not hasattr(module, "generate_pattern"):
            pytest.skip(f"{pattern_module} has no generate_pattern function")

        points = module.generate_pattern(config)

        # Validate output is a tensor
        assert isinstance(points, torch.Tensor), f"Expected torch.Tensor, got {type(points)}"

        # Validate output shape
        assert points.shape[0] == config.n_samples, (
            f"Expected {config.n_samples} points, got {points.shape[0]}"
        )
        assert points.shape[-1] == 2, "Points should have x, y coordinates"

        # Validate points are within bounds
        max_radius = config.roi_diameter / 2
        # Handle different shapes: (N, 2) or (N, 1, 2)
        if points.ndim == 3:
            points_2d = points.squeeze(1)
        else:
            points_2d = points

        distances = torch.sqrt(points_2d[..., 0] ** 2 + points_2d[..., 1] ** 2)
        assert distances.max() <= max_radius * 1.1, (
            f"Points exceed roi_diameter: max distance = {distances.max():.2f}, "
            f"allowed = {max_radius:.2f}"
        )

        # Validate no NaN or Inf
        assert not torch.isnan(points).any(), "Points contain NaN"
        assert not torch.isinf(points).any(), "Points contain Inf"

    @pytest.mark.e2e
    def test_concentric_circles_specific_structure(self, config):
        """Test concentric circles pattern has expected circular structure."""
        try:
            import concentric_circles
        except ImportError:
            pytest.skip("Could not import concentric_circles")

        points = concentric_circles.generate_pattern(config)

        # Squeeze to 2D if needed
        if points.ndim == 3:
            points_2d = points.squeeze(1)
        else:
            points_2d = points

        # Calculate distances from origin
        distances = torch.sqrt(points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2)

        # Should have multiple distinct radii (concentric circles)
        # Round to 2 decimal places to group by circle
        unique_radii = torch.unique(distances.round(decimals=2))

        # Should have around 5 circles based on the implementation
        assert len(unique_radii) >= 3, (
            f"Expected at least 3 distinct circles, found {len(unique_radii)}"
        )

    @pytest.mark.e2e
    def test_spiral_patterns_progressive_radius(self):
        """Test that spiral patterns have progressively increasing radius."""
        config = PatternConfig(n_samples=100, roi_diameter=100.0)

        for pattern_name in ["continuous_spiral", "logarithmic_spiral"]:
            try:
                module = __import__(pattern_name)
            except ImportError:
                pytest.skip(f"Could not import {pattern_name}")

            points = module.generate_pattern(config)

            # Squeeze to 2D if needed
            if points.ndim == 3:
                points_2d = points.squeeze(1)
            else:
                points_2d = points

            # Calculate distances from origin
            distances = torch.sqrt(points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2)

            # For spiral patterns, distances should generally increase
            # Check that at least 80% of points follow increasing trend
            increasing_count = (distances[1:] >= distances[:-1]).sum()
            increasing_ratio = increasing_count.float() / (len(distances) - 1)

            assert increasing_ratio > 0.7, (
                f"{pattern_name}: Expected spiral pattern with increasing radius, "
                f"but only {increasing_ratio:.1%} of points follow this trend"
            )

    @pytest.mark.e2e
    def test_jittered_grid_coverage(self):
        """Test that jittered grid maintains good spatial coverage."""
        config = PatternConfig(n_samples=100, roi_diameter=100.0)

        try:
            import jittered_grid
        except ImportError:
            pytest.skip("Could not import jittered_grid")

        points = jittered_grid.generate_pattern(config)

        # Squeeze to 2D if needed
        if points.ndim == 3:
            points_2d = points.squeeze(1)
        else:
            points_2d = points

        # Check that points are reasonably spread out
        # Calculate minimum distance between points
        # (should not have clustering)
        n_points = points_2d.shape[0]
        min_distances = []
        for i in range(min(50, n_points)):  # Sample subset for efficiency
            diffs = points_2d - points_2d[i]
            distances = torch.sqrt((diffs**2).sum(dim=-1))
            # Exclude self (distance = 0)
            distances[i] = float("inf")
            min_distances.append(distances.min().item())

        avg_min_distance = sum(min_distances) / len(min_distances)

        # Should have reasonable spacing (not all clustered)
        # For 100 points in diameter 100, expect average min distance > 2
        assert avg_min_distance > 1.0, (
            f"Points appear too clustered: avg minimum distance = {avg_min_distance:.2f}"
        )
