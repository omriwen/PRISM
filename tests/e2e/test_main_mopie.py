"""End-to-end tests for main_mopie.py MoPIE entry point."""

from __future__ import annotations

import sys

import pytest


class TestMainMoPIEExecution:
    """Test main_mopie.py execution."""

    @pytest.mark.e2e
    def test_minimal_run_completes(
        self, run_subprocess, project_root, minimal_mopie_args, tmp_path
    ):
        """Test that main_mopie.py completes with minimal parameters."""
        cmd = [
            sys.executable,
            str(project_root / "main_mopie.py"),
            *minimal_mopie_args,
            "--name",
            f"e2e_mopie_test_{tmp_path.name}",
            "--log_dir",
            str(tmp_path),
        ]
        result = run_subprocess(cmd, timeout=180)

        assert result.returncode == 0, f"main_mopie.py failed:\n{result.stderr}"

    @pytest.mark.e2e
    @pytest.mark.slow_e2e
    def test_produces_expected_outputs(
        self, run_subprocess, project_root, minimal_mopie_args, tmp_path
    ):
        """Test that main_mopie.py produces expected output files."""
        run_name = f"e2e_mopie_output_{tmp_path.name}"
        cmd = [
            sys.executable,
            str(project_root / "main_mopie.py"),
            *minimal_mopie_args,
            "--name",
            run_name,
            "--log_dir",
            str(tmp_path),
        ]
        cmd = [arg for arg in cmd if arg != "--debug"]

        result = run_subprocess(cmd, timeout=180)
        assert result.returncode == 0

        run_dir = tmp_path / run_name
        assert run_dir.exists()
        assert (run_dir / "checkpoint.pt").exists()


class TestMainMoPIELineSampling:
    """Test main_mopie.py with line sampling enabled."""

    @pytest.mark.e2e
    def test_mopie_line_sampling_completes(
        self, run_subprocess, project_root, minimal_mopie_line_args, tmp_path
    ):
        """Test that main_mopie.py completes with line sampling enabled."""
        cmd = [
            sys.executable,
            str(project_root / "main_mopie.py"),
            *minimal_mopie_line_args,
            "--name",
            f"e2e_mopie_line_{tmp_path.name}",
            "--log_dir",
            str(tmp_path),
        ]
        result = run_subprocess(cmd, timeout=180)
        assert result.returncode == 0, f"main_mopie.py with line sampling failed:\n{result.stderr}"
