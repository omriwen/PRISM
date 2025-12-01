"""Shared fixtures for end-to-end tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib


matplotlib.use("Agg")  # Must be before pyplot import

import pytest
import torch


if TYPE_CHECKING:
    from collections.abc import Callable

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Close all matplotlib figures after each test."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def examples_dir() -> Path:
    """Return the examples directory."""
    return EXAMPLES_DIR


@pytest.fixture
def run_subprocess() -> Callable[..., subprocess.CompletedProcess]:
    """Return a helper function to run subprocesses with proper environment."""

    def _run(
        cmd: list[str],
        timeout: int = 120,
        cwd: Path | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        env = {**os.environ, "MPLBACKEND": "Agg"}
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or PROJECT_ROOT,
            env=env,
            check=check,
        )

    return _run


@pytest.fixture
def minimal_prism_args() -> list[str]:
    """Return minimal arguments for fast PRISM execution."""
    return [
        "--obj_name",
        "europa",
        "--n_samples",
        "16",
        "--sample_diameter",
        "32",
        "--max_epochs",
        "1",
        "--n_epochs",
        "10",
        "--image_size",
        "128",
        "--debug",
    ]


@pytest.fixture
def minimal_mopie_args() -> list[str]:
    """Return minimal arguments for fast MoPIE execution."""
    return [
        "--obj_name",
        "europa",
        "--n_samples",
        "16",
        "--sample_diameter",
        "32",
        "--n_epochs",
        "5",
        "--image_size",
        "128",
        "--debug",
    ]


@pytest.fixture(scope="module")
def mock_checkpoint(tmp_path_factory) -> Path:
    """Generate a minimal mock checkpoint for CLI tool testing."""
    checkpoint_dir = tmp_path_factory.mktemp("mock_experiment")

    # Create minimal checkpoint structure
    checkpoint = {
        "current_rec": torch.randn(1, 1, 64, 64),
        "losses": [1.0, 0.5, 0.3, 0.2, 0.1],
        "ssims": [0.1, 0.3, 0.5, 0.7, 0.8],
        "psnrs": [10.0, 15.0, 20.0, 25.0, 30.0],
        "failed_samples": [],
        "sample_centers": torch.randn(16, 2),
    }

    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    # Create config.yaml
    config_path = checkpoint_dir / "config.yaml"
    config_path.write_text("""
physics:
  obj_name: europa
  wavelength: 5.5e-7
telescope:
  n_samples: 16
  sample_diameter: 32
training:
  max_epochs: 1
  lr: 0.001
""")

    # Create args.pt
    args_path = checkpoint_dir / "args.pt"
    torch.save({"obj_name": "europa", "n_samples": 16}, args_path)

    # Create sample_points.pt
    points_path = checkpoint_dir / "sample_points.pt"
    torch.save(torch.randn(16, 2), points_path)

    return checkpoint_dir


@pytest.fixture
def minimal_prism_line_args() -> list[str]:
    """Minimal args for PRISM with line sampling enabled."""
    return [
        "--obj_name",
        "europa",
        "--n_samples",
        "16",
        "--sample_length",
        "16",  # Enable line sampling
        "--sample_diameter",
        "32",
        "--max_epochs",
        "1",
        "--n_epochs",
        "10",
        "--image_size",
        "128",
        "--debug",
    ]


@pytest.fixture
def minimal_mopie_line_args() -> list[str]:
    """Minimal args for MoPIE with line sampling enabled."""
    return [
        "--obj_name",
        "europa",
        "--n_samples",
        "16",
        "--sample_length",
        "16",  # Enable line sampling
        "--sample_diameter",
        "32",
        "--n_epochs",
        "5",
        "--image_size",
        "128",
        "--debug",
    ]
