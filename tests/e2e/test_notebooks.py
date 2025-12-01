"""End-to-end tests for Jupyter notebook execution."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Note: Notebook tests were previously skipped due to outdated API usage.
# As of 2025-12-01, all notebooks have been updated to use the current API.

NOTEBOOKS_DIR = Path(__file__).parent.parent.parent / "examples" / "notebooks"

# Notebook categories with corresponding test files
# These are the notebooks that actually exist in examples/notebooks/
TUTORIAL_NOTEBOOKS = [
    "tutorial_01_quickstart.ipynb",
    "tutorial_02_pattern_design.ipynb",
    "tutorial_03_result_analysis.ipynb",
    "tutorial_04_dashboard.ipynb",
    "tutorial_05_reporting.ipynb",
]

QUICKSTART_NOTEBOOKS = [
    "quickstart_01_microscopy_basic.ipynb",
    "quickstart_02_drone_basic.ipynb",
    "quickstart_03_validation_intro.ipynb",
]

LEARNING_NOTEBOOKS = [
    "learning_01_resolution_fundamentals.ipynb",
    "learning_02_resolution_validation.ipynb",
    "learning_03_illumination_modes.ipynb",
    "learning_04_gsd_basics.ipynb",
    "learning_05_drone_altitudes.ipynb",
    "learning_08_scenario_comparison.ipynb",
]

# Additional demonstration notebooks
DEMO_NOTEBOOKS = [
    "camera_simulation.ipynb",
    "microscope_simulation.ipynb",
    "scanning_illumination_demo.ipynb",
    "spids_microscopy_resolution_enhancement.ipynb",
]


def execute_notebook(notebook_path: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    """Execute a notebook using nbconvert.

    Parameters
    ----------
    notebook_path : Path
        Path to the notebook file to execute
    timeout : int, default=300
        Timeout in seconds for notebook execution

    Returns
    -------
    subprocess.CompletedProcess
        Result of the notebook execution command

    Notes
    -----
    Uses nbconvert with --execute flag to run notebooks.
    Output is written to a temporary directory to avoid permission issues.
    The command is wrapped to apply nest_asyncio for event loop compatibility.
    """
    import tempfile

    # Use a wrapper script to apply nest_asyncio before running nbconvert
    # This fixes "RuntimeError: no running event loop" in pytest environment
    wrapper_code = '''
import nest_asyncio
nest_asyncio.apply()

import sys
from nbconvert.nbconvertapp import main
sys.exit(main())
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        return subprocess.run(
            [
                sys.executable, "-c", wrapper_code,
                "--to", "notebook",
                "--execute",
                f"--ExecutePreprocessor.timeout={timeout}",
                "--ExecutePreprocessor.kernel_name=python3",
                "--output-dir", tmpdir,
                str(notebook_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 60,  # Extra buffer for nbconvert overhead
        )


class TestTutorialNotebooks:
    """Test tutorial notebooks execute without errors.

    Tutorial notebooks are designed for quick learning and should execute
    in under 5 minutes each.
    """

    @pytest.mark.e2e
    @pytest.mark.notebooks
    @pytest.mark.parametrize("notebook", TUTORIAL_NOTEBOOKS)
    def test_tutorial_notebook_executes(self, notebook):
        """Test tutorial notebook runs to completion.

        Parameters
        ----------
        notebook : str
            Name of the tutorial notebook to test
        """
        notebook_path = NOTEBOOKS_DIR / notebook
        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        result = execute_notebook(notebook_path, timeout=300)

        assert result.returncode == 0, (
            f"Notebook {notebook} failed with return code {result.returncode}:\n"
            f"STDERR:\n{result.stderr}"
        )


class TestQuickstartNotebooks:
    """Test quickstart notebooks (slower, more comprehensive).

    Quickstart notebooks contain more extensive examples and may take
    longer to execute. These are marked as slow_e2e tests.
    """

    @pytest.mark.e2e
    @pytest.mark.notebooks
    @pytest.mark.slow_e2e
    @pytest.mark.parametrize("notebook", QUICKSTART_NOTEBOOKS)
    def test_quickstart_notebook_executes(self, notebook):
        """Test quickstart notebook runs to completion.

        Parameters
        ----------
        notebook : str
            Name of the quickstart notebook to test
        """
        notebook_path = NOTEBOOKS_DIR / notebook
        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        result = execute_notebook(notebook_path, timeout=600)

        assert result.returncode == 0, (
            f"Notebook {notebook} failed with return code {result.returncode}:\n"
            f"STDERR:\n{result.stderr}"
        )


class TestLearningNotebooks:
    """Test learning track notebooks.

    Learning notebooks provide in-depth educational content on optical
    simulation concepts.
    """

    @pytest.mark.e2e
    @pytest.mark.notebooks
    @pytest.mark.slow_e2e
    @pytest.mark.parametrize("notebook", LEARNING_NOTEBOOKS)
    def test_learning_notebook_executes(self, notebook):
        """Test learning notebook runs to completion.

        Parameters
        ----------
        notebook : str
            Name of the learning notebook to test
        """
        notebook_path = NOTEBOOKS_DIR / notebook
        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        result = execute_notebook(notebook_path, timeout=300)

        assert result.returncode == 0, (
            f"Notebook {notebook} failed with return code {result.returncode}:\n"
            f"STDERR:\n{result.stderr}"
        )


class TestDemoNotebooks:
    """Test demonstration notebooks.

    These are specialized notebooks that demonstrate specific features or
    advanced use cases of the PRISM framework.
    """

    @pytest.mark.e2e
    @pytest.mark.notebooks
    @pytest.mark.slow_e2e
    @pytest.mark.parametrize("notebook", DEMO_NOTEBOOKS)
    def test_demo_notebook_executes(self, notebook):
        """Test demonstration notebook runs to completion.

        Parameters
        ----------
        notebook : str
            Name of the demo notebook to test
        """
        notebook_path = NOTEBOOKS_DIR / notebook
        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        # Demo notebooks may be more computationally intensive
        result = execute_notebook(notebook_path, timeout=600)

        assert result.returncode == 0, (
            f"Notebook {notebook} failed with return code {result.returncode}:\n"
            f"STDERR:\n{result.stderr}"
        )
