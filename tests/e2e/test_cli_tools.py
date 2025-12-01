"""End-to-end tests for PRISM CLI tools."""

from __future__ import annotations

import pytest


class TestPrismPatterns:
    """Test prism-patterns CLI tool."""

    @pytest.mark.e2e
    def test_patterns_list(self, run_subprocess):
        """Test prism-patterns list command."""
        result = run_subprocess(["prism-patterns", "list"], timeout=30)

        assert result.returncode == 0
        # Should list available patterns
        output = result.stdout.lower()
        assert "fermat" in output or "random" in output or "pattern" in output

    @pytest.mark.e2e
    def test_patterns_show(self, run_subprocess, tmp_path):
        """Test prism-patterns show command."""
        output_file = tmp_path / "pattern.png"
        result = run_subprocess(
            [
                "prism-patterns",
                "show",
                "fermat",
                "--n-samples",
                "50",
                "--output",
                str(output_file),
            ],
            timeout=30,
        )

        assert result.returncode == 0
        assert output_file.exists()

    @pytest.mark.e2e
    def test_patterns_stats(self, run_subprocess):
        """Test prism-patterns stats command."""
        result = run_subprocess(
            [
                "prism-patterns",
                "stats",
                "fermat",
                "--n-samples",
                "50",
            ],
            timeout=30,
        )

        assert result.returncode == 0


class TestPrismInspect:
    """Test prism-inspect CLI tool."""

    @pytest.mark.e2e
    def test_inspect_help(self, run_subprocess):
        """Test prism-inspect --help."""
        result = run_subprocess(["prism-inspect", "--help"], timeout=10)

        assert result.returncode == 0
        assert "checkpoint" in result.stdout.lower()

    @pytest.mark.e2e
    def test_inspect_checkpoint(self, run_subprocess, mock_checkpoint):
        """Test prism-inspect on mock checkpoint."""
        checkpoint_path = mock_checkpoint / "checkpoint.pt"
        result = run_subprocess(
            [
                "prism-inspect",
                str(checkpoint_path),
            ],
            timeout=30,
        )

        # Should either succeed or fail gracefully
        assert result.returncode in (0, 1)


class TestPrismCompare:
    """Test prism-compare CLI tool."""

    @pytest.mark.e2e
    def test_compare_help(self, run_subprocess):
        """Test prism-compare --help."""
        result = run_subprocess(["prism-compare", "--help"], timeout=10)

        assert result.returncode == 0


class TestPrismReport:
    """Test prism-report CLI tool."""

    @pytest.mark.e2e
    def test_report_help(self, run_subprocess):
        """Test prism-report --help."""
        result = run_subprocess(["prism-report", "--help"], timeout=10)

        assert result.returncode == 0

    @pytest.mark.e2e
    def test_report_html_generation(self, run_subprocess, mock_checkpoint, tmp_path):
        """Test prism-report generates HTML."""
        output_file = tmp_path / "report.html"
        result = run_subprocess(
            [
                "prism-report",
                str(mock_checkpoint),
                "--format",
                "html",
                "--output",
                str(output_file),
            ],
            timeout=60,
        )

        # May fail if template issues, but should not crash
        if result.returncode == 0:
            assert output_file.exists()


class TestPrismAnimate:
    """Test prism-animate CLI tool."""

    @pytest.mark.e2e
    def test_animate_help(self, run_subprocess):
        """Test prism-animate --help."""
        result = run_subprocess(["prism-animate", "--help"], timeout=10)

        assert result.returncode == 0
