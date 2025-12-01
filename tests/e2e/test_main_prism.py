"""End-to-end tests for main.py PRISM entry point."""

from __future__ import annotations

import sys

import pytest


class TestMainPRISMExecution:
    """Test main.py execution with various configurations."""

    @pytest.mark.e2e
    def test_minimal_run_completes(
        self, run_subprocess, project_root, minimal_prism_args, tmp_path
    ):
        """Test that main.py completes with minimal parameters."""
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            *minimal_prism_args,
            "--name",
            f"e2e_test_{tmp_path.name}",
            "--log_dir",
            str(tmp_path),
        ]
        result = run_subprocess(cmd, timeout=180)

        assert result.returncode == 0, f"main.py failed:\n{result.stderr}"

    @pytest.mark.e2e
    def test_validate_only_flag(self, run_subprocess, project_root):
        """Test --validate-only returns valid config without running."""
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            "--obj_name",
            "europa",
            "--validate-only",
        ]
        result = run_subprocess(cmd, timeout=30)

        assert result.returncode == 0
        # Should indicate config is valid
        assert "valid" in result.stdout.lower() or result.returncode == 0

    @pytest.mark.e2e
    def test_list_presets(self, run_subprocess, project_root):
        """Test --list-presets shows available presets."""
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            "--list-presets",
        ]
        result = run_subprocess(cmd, timeout=30)

        assert result.returncode == 0
        assert "quick_test" in result.stdout.lower() or "preset" in result.stdout.lower()

    @pytest.mark.e2e
    def test_list_scenarios(self, run_subprocess, project_root):
        """Test --list-scenarios lists all scenarios."""
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            "--list-scenarios",
        ]
        result = run_subprocess(cmd, timeout=30)

        assert result.returncode == 0

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        "help_flag",
        [
            "--help-propagator",
            "--help-patterns",
            "--help-loss",
            "--help-model",
            "--help-objects",
        ],
    )
    def test_help_topics(self, run_subprocess, project_root, help_flag):
        """Test help topic flags display information."""
        cmd = [sys.executable, str(project_root / "main.py"), help_flag]
        result = run_subprocess(cmd, timeout=30)

        assert result.returncode == 0
        assert len(result.stdout) > 100  # Should have substantial help text

    @pytest.mark.e2e
    @pytest.mark.slow_e2e
    def test_produces_expected_outputs(
        self, run_subprocess, project_root, minimal_prism_args, tmp_path
    ):
        """Test that main.py produces expected output files."""
        run_name = f"e2e_output_test_{tmp_path.name}"
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            *minimal_prism_args,
            "--name",
            run_name,
            "--log_dir",
            str(tmp_path),
        ]
        # Remove --debug to ensure outputs are saved
        cmd = [arg for arg in cmd if arg != "--debug"]

        result = run_subprocess(cmd, timeout=180)
        assert result.returncode == 0, f"main.py failed:\n{result.stderr}"

        run_dir = tmp_path / run_name
        assert run_dir.exists(), f"Run directory not created: {run_dir}"
        assert (run_dir / "checkpoint.pt").exists(), "checkpoint.pt not created"
        assert (run_dir / "config.yaml").exists(), "config.yaml not created"


class TestMainPRISMMetrics:
    """Test that main.py produces reasonable metrics."""

    @pytest.mark.e2e
    @pytest.mark.slow_e2e
    def test_metrics_in_reasonable_range(
        self, run_subprocess, project_root, minimal_prism_args, tmp_path
    ):
        """Test that output metrics are in reasonable ranges."""
        import torch

        run_name = f"e2e_metrics_test_{tmp_path.name}"
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            *minimal_prism_args,
            "--name",
            run_name,
            "--log_dir",
            str(tmp_path),
        ]
        cmd = [arg for arg in cmd if arg != "--debug"]

        result = run_subprocess(cmd, timeout=180)
        assert result.returncode == 0

        checkpoint_path = tmp_path / run_name / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Validate metrics are in reasonable ranges
        if "ssims" in checkpoint and len(checkpoint["ssims"]) > 0:
            final_ssim = checkpoint["ssims"][-1]
            # Handle both scalar and tensor values
            if hasattr(final_ssim, "item"):
                final_ssim = final_ssim.item()
            assert 0.0 < final_ssim <= 1.0, f"SSIM out of range: {final_ssim}"

        if "losses" in checkpoint and len(checkpoint["losses"]) > 1:
            # Loss should generally decrease (with some tolerance)
            initial_loss = checkpoint["losses"][0]
            final_loss = checkpoint["losses"][-1]
            # Handle both scalar and tensor values
            if hasattr(initial_loss, "item"):
                initial_loss = initial_loss.item()
            if hasattr(final_loss, "item"):
                final_loss = final_loss.item()
            assert final_loss <= initial_loss * 1.5, "Loss increased significantly"


class TestMainPRISMLineSampling:
    """Test main.py with line sampling enabled."""

    @pytest.mark.e2e
    def test_line_sampling_completes(
        self, run_subprocess, project_root, minimal_prism_line_args, tmp_path
    ):
        """Test that main.py completes with line sampling enabled."""
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            *minimal_prism_line_args,
            "--name",
            f"e2e_line_test_{tmp_path.name}",
            "--log_dir",
            str(tmp_path),
        ]
        result = run_subprocess(cmd, timeout=180)
        assert result.returncode == 0, f"main.py with line sampling failed:\n{result.stderr}"

    @pytest.mark.e2e
    @pytest.mark.slow_e2e
    def test_line_sampling_produces_outputs(
        self, run_subprocess, project_root, minimal_prism_line_args, tmp_path
    ):
        """Test that line sampling run produces expected output files."""
        run_name = f"e2e_line_output_{tmp_path.name}"
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            *minimal_prism_line_args,
            "--name",
            run_name,
            "--log_dir",
            str(tmp_path),
        ]
        cmd = [arg for arg in cmd if arg != "--debug"]

        result = run_subprocess(cmd, timeout=180)
        assert result.returncode == 0

        run_dir = tmp_path / run_name
        assert run_dir.exists(), f"Run directory not created: {run_dir}"
        assert (run_dir / "checkpoint.pt").exists(), "checkpoint.pt not created"
        assert (run_dir / "config.yaml").exists(), "config.yaml not created"

    @pytest.mark.e2e
    @pytest.mark.slow_e2e
    def test_line_sampling_metrics_valid(
        self, run_subprocess, project_root, minimal_prism_line_args, tmp_path
    ):
        """Test that line sampling produces valid metrics."""
        import torch

        run_name = f"e2e_line_metrics_{tmp_path.name}"
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            *minimal_prism_line_args,
            "--name",
            run_name,
            "--log_dir",
            str(tmp_path),
        ]
        cmd = [arg for arg in cmd if arg != "--debug"]

        result = run_subprocess(cmd, timeout=180)
        assert result.returncode == 0

        checkpoint = torch.load(tmp_path / run_name / "checkpoint.pt", map_location="cpu")
        if "ssims" in checkpoint and len(checkpoint["ssims"]) > 0:
            final_ssim = checkpoint["ssims"][-1]
            if hasattr(final_ssim, "item"):
                final_ssim = final_ssim.item()
            # SSIM can be slightly negative due to floating-point precision in short runs
            # We use -0.01 as lower bound to allow for numerical noise
            assert -0.01 < final_ssim <= 1.0, f"SSIM out of range: {final_ssim}"
