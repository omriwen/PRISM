"""
Tests for SPIDS reporting module.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from prism.analysis.comparison import ExperimentResult
from prism.reporting import ReportGenerator


@pytest.fixture
def mock_experiment():
    """Create mock experiment result for testing."""
    config = {
        "object_name": "europa",
        "n_samples": 100,
        "pattern": "fermat",
        "snr": 40,
        "propagator_method": "fraunhofer",
        "learning_rate": 0.001,
        "max_epochs": 1000,
        "timestamp": "2025-11-20 18:00:00",
    }

    final_metrics = {
        "loss": 0.0034,
        "ssim": 0.9234,
        "psnr": 35.2,
    }

    training_history = {  # type: ignore[var-annotated]
        "loss": [0.1, 0.05, 0.02, 0.01, 0.0034],
        "ssim": [0.7, 0.8, 0.85, 0.9, 0.9234],
        "psnr": [20.0, 25.0, 30.0, 33.0, 35.2],
        "duration": 2723.5,  # ~45 minutes
        "converged": True,
    }

    # Create simple synthetic images
    ground_truth = np.random.rand(256, 256)
    reconstruction = ground_truth + np.random.randn(256, 256) * 0.01

    checkpoint = {
        "ground_truth": torch.from_numpy(ground_truth),
        "reconstruction": torch.from_numpy(reconstruction),
        "aperture_mask": torch.rand(256, 256),
    }

    return ExperimentResult(
        name="test_experiment",
        path=Path("/tmp/test_experiment"),
        config=config,
        final_metrics=final_metrics,
        training_history=training_history,  # type: ignore[arg-type]
        checkpoint=checkpoint,
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestReportGenerator:
    """Test suite for ReportGenerator."""

    def test_init_default_templates(self):
        """Test initialization with default templates."""
        generator = ReportGenerator()
        assert generator.template_dir.exists()
        assert (generator.template_dir / "report.html").exists()
        assert (generator.template_dir / "styles.css").exists()

    def test_init_custom_templates(self, temp_output_dir):
        """Test initialization with custom template directory."""
        custom_template_dir = temp_output_dir / "custom_templates"
        custom_template_dir.mkdir()
        (custom_template_dir / "report.html").touch()

        generator = ReportGenerator(template_dir=custom_template_dir)
        assert generator.template_dir == custom_template_dir

    def test_format_time(self):
        """Test time formatting helper."""
        assert ReportGenerator._format_time(45.5) == "45.5s"
        assert ReportGenerator._format_time(125.0) == "2.1m"
        assert ReportGenerator._format_time(7200.0) == "2.0h"

    def test_format_size(self):
        """Test file size formatting helper."""
        assert ReportGenerator._format_size(512) == "512.0 B"
        assert ReportGenerator._format_size(1024) == "1.0 KB"
        assert ReportGenerator._format_size(1024 * 1024) == "1.0 MB"
        assert ReportGenerator._format_size(1024 * 1024 * 1024) == "1.0 GB"

    def test_compute_summary_single_experiment(self, mock_experiment):
        """Test summary computation for single experiment."""
        generator = ReportGenerator()
        summary = generator._compute_summary([mock_experiment])

        assert "best_experiments" in summary
        assert "loss" in summary["best_experiments"]
        assert "ssim" in summary["best_experiments"]
        assert "psnr" in summary["best_experiments"]
        assert summary["best_experiments"]["loss"]["name"] == "test_experiment"
        assert summary["total_duration"] == 2723.5
        assert summary["avg_epochs"] == 5

    def test_compute_summary_multiple_experiments(self, mock_experiment):
        """Test summary computation for multiple experiments."""
        # Create second experiment with different metrics
        exp2 = ExperimentResult(
            name="test_experiment_2",
            path=Path("/tmp/test_experiment_2"),
            config=mock_experiment.config,
            final_metrics={"loss": 0.0050, "ssim": 0.9500, "psnr": 32.0},
            training_history={"loss": [0.1, 0.05], "duration": 1800, "converged": False},  # type: ignore[arg-type, dict-item]
            checkpoint={},
        )

        generator = ReportGenerator()
        summary = generator._compute_summary([mock_experiment, exp2])

        # Exp1 has best loss and PSNR, Exp2 has best SSIM
        assert summary["best_experiments"]["loss"]["name"] == "test_experiment"
        assert summary["best_experiments"]["ssim"]["name"] == "test_experiment_2"
        assert summary["best_experiments"]["psnr"]["name"] == "test_experiment"

    def test_prepare_report_data(self, mock_experiment):
        """Test report data preparation."""
        generator = ReportGenerator()
        data = generator._prepare_report_data([mock_experiment], include_appendix=True)

        assert data["n_experiments"] == 1
        assert len(data["experiments"]) == 1
        assert data["include_appendix"] is True
        assert "timestamp" in data
        assert "summary" in data

        exp_data = data["experiments"][0]
        assert exp_data["name"] == "test_experiment"
        assert exp_data["duration"] == 2723.5
        assert exp_data["converged"] is True
        assert exp_data["n_epochs"] == 5

    def test_generate_html_single_experiment(self, mock_experiment, temp_output_dir):
        """Test HTML report generation for single experiment."""
        generator = ReportGenerator()
        output_path = temp_output_dir / "report.html"

        with patch("prism.reporting.generator.ExperimentComparator") as mock_comparator:
            mock_comparator.return_value.load_experiments.return_value = [mock_experiment]

            generator.generate_html([mock_experiment.path], output_path, include_appendix=True)

        assert output_path.exists()
        content = output_path.read_text()

        # Check key content is present
        assert "SPIDS Experiment Report" in content
        assert "test_experiment" in content
        assert "Executive Summary" in content
        assert "Configuration Details" in content

    def test_generate_html_no_experiments(self, temp_output_dir):
        """Test HTML generation with no valid experiments."""
        generator = ReportGenerator()
        output_path = temp_output_dir / "report.html"

        with patch("prism.reporting.generator.ExperimentComparator") as mock_comparator:
            mock_comparator.return_value.load_experiments.return_value = []

            with pytest.raises(ValueError, match="No valid experiments"):
                generator.generate_html([], output_path)

    def test_generate_figures(self, mock_experiment):
        """Test figure generation."""
        generator = ReportGenerator()
        figures = generator._generate_figures([mock_experiment])

        # Should generate at least training curves and reconstruction comparison
        assert "training_curves" in figures
        assert "reconstruction_comparison" in figures
        assert "kspace_coverage" in figures

        # Each figure should be a base64-encoded string
        for fig_name, fig_data in figures.items():
            assert isinstance(fig_data, str)
            assert len(fig_data) > 0

    def test_generate_pdf_without_weasyprint(self, mock_experiment, temp_output_dir):
        """Test PDF generation when WeasyPrint is not installed."""
        generator = ReportGenerator()
        output_path = temp_output_dir / "report.pdf"

        with patch("prism.reporting.generator.ExperimentComparator") as mock_comparator:
            mock_comparator.return_value.load_experiments.return_value = [mock_experiment]

            with patch.dict("sys.modules", {"weasyprint": None}):
                with pytest.raises(ImportError, match="WeasyPrint"):
                    generator.generate_pdf([mock_experiment.path], output_path)


class TestReportCLI:
    """Test suite for report CLI command."""

    def test_report_command_invalid_path(self, capsys):
        """Test report command with invalid experiment path."""
        from argparse import Namespace

        from prism.cli.report import report_command

        args = Namespace(
            experiments=["/nonexistent/path"],
            format="html",
            output=None,
            template=None,
            no_appendix=False,
            template_dir=None,
        )

        report_command(args)
        captured = capsys.readouterr()
        assert "do not exist" in captured.out or "Error" in captured.out

    def test_report_command_missing_checkpoint(self, temp_output_dir, capsys):
        """Test report command with missing checkpoint."""
        from argparse import Namespace

        from prism.cli.report import report_command

        # Create experiment directory without checkpoint
        exp_dir = temp_output_dir / "experiment"
        exp_dir.mkdir()

        args = Namespace(
            experiments=[str(exp_dir)],
            format="html",
            output=None,
            template=None,
            no_appendix=False,
            template_dir=None,
        )

        report_command(args)
        captured = capsys.readouterr()
        assert "No checkpoint.pt found" in captured.out or "Warning" in captured.out


class TestReportIntegration:
    """Integration tests for reporting module."""

    @pytest.mark.slow
    def test_full_report_generation_workflow(self, mock_experiment, temp_output_dir):
        """Test complete report generation workflow."""
        generator = ReportGenerator()
        output_path = temp_output_dir / "integration_report.html"

        with patch("prism.reporting.generator.ExperimentComparator") as mock_comparator:
            mock_comparator.return_value.load_experiments.return_value = [
                mock_experiment,
            ]

            # Generate HTML report
            generator.generate_html([mock_experiment.path], output_path, include_appendix=True)

            assert output_path.exists()
            content = output_path.read_text()

            # Verify all sections are present
            assert "Executive Summary" in content
            assert "Results" in content
            assert "Configuration Details" in content
            assert "Training History" in content
            assert "Appendix" in content

            # Verify metrics are included
            assert "0.0034" in content  # loss
            assert "0.9234" in content  # ssim
            assert "35.2" in content  # psnr

            # Verify figures are embedded
            assert "data:image/png;base64," in content

    @pytest.mark.slow
    def test_comparison_report_generation(self, mock_experiment, temp_output_dir):
        """Test comparison report with multiple experiments."""
        # Create second experiment
        exp2 = ExperimentResult(
            name="experiment_2",
            path=Path("/tmp/experiment_2"),
            config=mock_experiment.config.copy(),
            final_metrics={"loss": 0.0040, "ssim": 0.9100, "psnr": 34.0},
            training_history={  # type: ignore[arg-type, dict-item]
                "loss": [0.1, 0.04],
                "ssim": [0.7, 0.91],
                "psnr": [20.0, 34.0],
                "duration": 2000,  # type: ignore[dict-item]
                "converged": True,  # type: ignore[dict-item]
            },
            checkpoint=mock_experiment.checkpoint,
        )

        generator = ReportGenerator()
        output_path = temp_output_dir / "comparison_report.html"

        with patch("prism.reporting.generator.ExperimentComparator") as mock_comparator:
            mock_comparator.return_value.load_experiments.return_value = [
                mock_experiment,
                exp2,
            ]

            generator.generate_html(
                [mock_experiment.path, exp2.path], output_path, include_appendix=False
            )

            assert output_path.exists()
            content = output_path.read_text()

            # Both experiments should be in the report
            assert "test_experiment" in content
            assert "experiment_2" in content

            # Metric comparison should be present
            assert "metric_comparison" in content or "Metric Comparison" in content
