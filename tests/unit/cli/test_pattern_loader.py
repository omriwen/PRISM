"""Tests for pattern loading and execution."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from prism.core.pattern_loader import PatternLoader, load_and_generate_pattern


class MockConfig:
    """Mock config for testing."""

    n_samples = 100
    roi_diameter = 512
    sample_diameter = 32
    sample_length = 0
    line_angle = None
    obj_size = 1024
    n_angs = 4


def test_load_builtin_fermat():
    """Test loading builtin fermat pattern."""
    loader = PatternLoader()
    func, metadata = loader.load_pattern_function("builtin:fermat")

    assert callable(func)
    assert metadata["is_builtin"] is True
    assert metadata["source"] == "builtin:fermat"
    assert metadata["docstring"] is not None


def test_load_builtin_invalid():
    """Test loading invalid builtin raises error."""
    loader = PatternLoader()

    with pytest.raises(ValueError, match="Unknown builtin pattern"):
        loader.load_pattern_function("builtin:nonexistent")


def test_load_from_file_valid():
    """Test loading valid pattern from file."""
    # Create temporary pattern file
    pattern_code = '''
import torch
import numpy as np

def generate_pattern(config):
    """Test spiral pattern."""
    n = config.n_samples
    theta = np.linspace(0, 4*np.pi, n)
    r = np.linspace(0, config.roi_diameter/2, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return torch.stack([torch.tensor(x), torch.tensor(y)], dim=-1)[:, None, :]
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(pattern_code)
        temp_path = f.name

    try:
        loader = PatternLoader()
        func, metadata = loader.load_pattern_function(temp_path)

        assert callable(func)
        assert metadata["is_builtin"] is False
        assert metadata["hash"] is not None
        assert "Test spiral pattern" in metadata["docstring"]
        assert metadata["source"] == pattern_code
    finally:
        Path(temp_path).unlink()


def test_load_from_file_missing_function():
    """Test loading file without generate_pattern raises error."""
    pattern_code = """
def some_other_function(config):
    pass
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(pattern_code)
        temp_path = f.name

    try:
        loader = PatternLoader()
        with pytest.raises(AttributeError, match="must define 'generate_pattern'"):
            loader.load_pattern_function(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_from_file_wrong_signature():
    """Test loading function with wrong signature raises error."""
    pattern_code = """
def generate_pattern(config, extra_arg):
    pass
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(pattern_code)
        temp_path = f.name

    try:
        loader = PatternLoader()
        with pytest.raises(TypeError, match="must take exactly 1 argument"):
            loader.load_pattern_function(temp_path)
    finally:
        Path(temp_path).unlink()


def test_execute_pattern_function_valid():
    """Test executing pattern function with valid output."""

    def valid_pattern(config):
        return torch.zeros(config.n_samples, 1, 2)

    loader = PatternLoader()
    config = MockConfig()
    result = loader.execute_pattern_function(valid_pattern, config)

    assert result.shape == (100, 1, 2)


def test_execute_pattern_function_invalid_type():
    """Test executing pattern that returns non-tensor raises error."""

    def invalid_pattern(config):
        return [[0, 0]]

    loader = PatternLoader()
    config = MockConfig()

    with pytest.raises(TypeError, match="must return torch.Tensor"):
        loader.execute_pattern_function(invalid_pattern, config)


def test_execute_pattern_function_invalid_shape():
    """Test executing pattern with invalid shape raises error."""

    def invalid_pattern(config):
        return torch.zeros(100, 2)  # Missing middle dimension

    loader = PatternLoader()
    config = MockConfig()

    with pytest.raises(ValueError, match="must be 3D tensor"):
        loader.execute_pattern_function(invalid_pattern, config)


def test_load_and_generate_pattern_integration():
    """Test full workflow from load to generation."""
    config = MockConfig()

    # Test with builtin
    sample_centers, metadata = load_and_generate_pattern("builtin:fermat", config)

    assert isinstance(sample_centers, torch.Tensor)
    assert sample_centers.shape[0] == config.n_samples
    assert sample_centers.shape[2] == 2
    assert metadata["is_builtin"] is True
