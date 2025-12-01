"""Tests for builtin pattern wrappers."""

from __future__ import annotations

import torch

from prism.core.pattern_builtins import fermat_builtin, random_builtin, star_builtin


class MockConfig:
    """Mock config for testing."""

    n_samples = 100
    n_angs = 4
    roi_diameter = 512
    sample_diameter = 32
    sample_length = 0
    line_angle = None
    samples_r_cutoff = None
    obj_size = 1024


def test_fermat_builtin_point_sampling():
    """Test fermat builtin with point sampling."""
    config = MockConfig()
    result = fermat_builtin(config)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 1, 2)


def test_fermat_builtin_line_sampling():
    """Test fermat builtin with line sampling."""
    config = MockConfig()
    config.sample_length = 64
    result = fermat_builtin(config)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 2, 2)


def test_star_builtin():
    """Test star pattern builtin."""
    config = MockConfig()
    config.sample_length = 64
    result = star_builtin(config)

    assert isinstance(result, torch.Tensor)
    assert result.ndim == 3
    assert result.shape[2] == 2


def test_random_builtin():
    """Test random pattern builtin."""
    config = MockConfig()
    result = random_builtin(config)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 1, 2)


def test_patterns_are_different():
    """Test that different patterns generate different results."""
    config = MockConfig()

    fermat = fermat_builtin(config)
    random = random_builtin(config)

    # Should not be identical
    assert not torch.allclose(fermat, random)
