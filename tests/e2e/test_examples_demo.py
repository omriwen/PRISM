"""End-to-end tests for demo scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


class TestDemoScripts:
    """Test demo scripts execute without errors."""

    @pytest.mark.e2e
    def test_demo_optical_simulation(self, run_subprocess, tmp_path):
        """Test demo_optical_simulation.py runs successfully."""
        demo_path = EXAMPLES_DIR / "demo_optical_simulation.py"
        if not demo_path.exists():
            pytest.skip("demo_optical_simulation.py not found")

        result = run_subprocess(
            [sys.executable, str(demo_path)],
            timeout=120,
            cwd=tmp_path,
        )

        assert result.returncode == 0, (
            f"Demo failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
