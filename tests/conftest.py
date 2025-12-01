"""Shared test fixtures for SPIDS unit tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_image():
    """Generate sample test image (random noise)."""
    return torch.randn(1, 1, 256, 256)


@pytest.fixture
def sample_image_small():
    """Generate small sample test image for faster tests."""
    return torch.randn(1, 1, 64, 64)


@pytest.fixture
def sample_complex_image():
    """Generate sample complex-valued test image."""
    real = torch.randn(1, 1, 256, 256)
    imag = torch.randn(1, 1, 256, 256)
    return torch.complex(real, imag)


@pytest.fixture
def sample_real_positive_image():
    """Generate sample real positive image (like actual intensity data)."""
    return torch.rand(1, 1, 256, 256).abs()


@pytest.fixture
def device():
    """Get test device (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for test artifacts."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def sample_sampling_pattern():
    """Generate sample sampling pattern (list of 2D points)."""
    n_points = 10
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radius = 50
    points = [[radius * np.cos(t), radius * np.sin(t)] for t in theta]
    return points


@pytest.fixture
def small_grid_size():
    """Standard small grid size for faster tests."""
    return 64


@pytest.fixture
def standard_grid_size():
    """Standard grid size matching typical usage."""
    return 256


@pytest.fixture
def wavelength():
    """Standard test wavelength (500nm, green light)."""
    return 500e-9


@pytest.fixture
def telescope_params():
    """Standard telescope parameters for testing."""
    return {
        "n": 256,
        "r": 10.0,
        "wavelength": 500e-9,
        "distance": 1.0,
    }
