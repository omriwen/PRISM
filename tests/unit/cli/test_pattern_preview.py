"""Tests for pattern preview and visualization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from prism.core.pattern_preview import (
    compute_pattern_statistics,
    preview_pattern,
    visualize_pattern,
)


class MockConfig:
    """Mock config for testing."""

    n_samples = 50
    roi_diameter = 512
    sample_diameter = 32
    sample_length = 0
    line_angle = None
    samples_r_cutoff = None
    n_angs = 4
    obj_size = 1024


def test_compute_pattern_statistics_points():
    """Test statistics computation for point sampling."""
    # Create simple pattern
    n = 50
    theta = torch.linspace(0, 2 * torch.pi, n)
    r = torch.linspace(0, 200, n)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    sample_centers = torch.stack([x, y], dim=-1)[:, None, :]

    stats = compute_pattern_statistics(sample_centers, roi_diameter=512)

    assert stats["n_samples"] == n
    assert stats["is_line_sampling"] is False
    assert 0 <= stats["radial_mean"] <= 200
    assert stats["radial_min"] >= 0
    assert stats["radial_max"] <= 200


def test_compute_pattern_statistics_lines():
    """Test statistics computation for line sampling."""
    # Create line pattern
    n = 30
    centers = torch.randn(n, 2) * 100
    offsets = torch.randn(n, 2) * 10
    sample_centers = torch.stack([centers - offsets, centers + offsets], dim=1)

    stats = compute_pattern_statistics(sample_centers, roi_diameter=512)

    assert stats["n_samples"] == n
    assert stats["is_line_sampling"] is True


def test_visualize_pattern():
    """Test pattern visualization."""
    # Create pattern
    n = 50
    sample_centers = torch.randn(n, 1, 2) * 100

    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = Path(f.name)

    try:
        _fig = visualize_pattern(
            sample_centers,
            roi_diameter=512,
            save_path=temp_path,
            show_statistics=True,
        )

        # Check file was created
        assert temp_path.exists()
        assert temp_path.stat().st_size > 0
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_preview_pattern_builtin():
    """Test preview with builtin pattern."""
    config = MockConfig()

    result = preview_pattern("builtin:fermat", config, save_path=None)

    assert "sample_centers" in result
    assert "statistics" in result
    assert "metadata" in result
    assert result["sample_centers"].shape[0] == config.n_samples
    assert result["statistics"]["n_samples"] == config.n_samples
