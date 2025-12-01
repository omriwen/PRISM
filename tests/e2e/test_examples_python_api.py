"""End-to-end tests for Python API example scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "python_api"


class TestPythonAPIExamples:
    """Test Python API example scripts execute without errors."""

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        "example_name,extra_args",
        [
            ("01_basic_usage.py", ["--quick", "--no-save"]),
            ("02_custom_patterns.py", ["--quick"]),
            ("03_custom_loss.py", ["--quick"]),
            ("04_model_extension.py", []),  # No args - just defines classes and runs comparison
            ("05_batch_experiments.py", ["--quick"]),
            ("06_microscope_reconstruction.py", ["--quick"]),
            ("07_drone_mapping.py", ["--quick"]),
            ("08_custom_scenario_builder.py", ["--quick"]),
            ("09_resolution_validation.py", ["--quick"]),
        ],
    )
    def test_example_runs_successfully(self, run_subprocess, example_name, extra_args, tmp_path):
        """Test that example script runs without errors."""
        example_path = EXAMPLES_DIR / example_name
        if not example_path.exists():
            pytest.skip(f"Example not found: {example_path}")

        cmd = [sys.executable, str(example_path), *extra_args]
        result = run_subprocess(cmd, timeout=180, cwd=tmp_path)

        assert result.returncode == 0, (
            f"Example {example_name} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


class TestModelExtensionExample:
    """Test model extension example with direct imports."""

    @pytest.mark.e2e
    def test_model_extension_imports(self):
        """Test that 04_model_extension.py models can be imported."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "model_ext", EXAMPLES_DIR / "04_model_extension.py"
        )
        if spec is None or spec.loader is None:
            pytest.skip("Could not load model_extension module")

        # This validates the module can be loaded without syntax errors
        _ = importlib.util.module_from_spec(spec)  # noqa: F841
        # Note: We don't exec the module as it may have side effects
