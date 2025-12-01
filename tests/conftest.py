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


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    """Get test device with CPU/GPU parametrization.

    Tests using this fixture will run on both CPU and CUDA (if available).
    """
    device_name = request.param
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(device_name)


@pytest.fixture
def gpu_device():
    """Get GPU device only (skip if unavailable).

    Use this for GPU-only tests.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(autouse=True, scope="function")
def cleanup_gpu_memory():
    """Auto-cleanup GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session", autouse=True)
def configure_gpu_for_testing():
    """Configure GPU settings for testing."""
    if torch.cuda.is_available():
        # Disable TF32 for deterministic results
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Enable cuDNN benchmarking for performance
        torch.backends.cudnn.benchmark = True
    yield


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
