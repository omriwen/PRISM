"""
Report Generator Module
=======================

Generates comprehensive HTML and PDF reports from experiment results.
"""

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib


matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger

from prism.analysis.comparison import ExperimentComparator, ExperimentResult


class ReportGenerator:
    """Generate HTML and PDF reports from experiment results."""

    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize report generator.

        Parameters
        ----------
        template_dir : Path, optional
            Directory containing Jinja2 templates. If None, uses default templates.
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"

        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

        # Add custom filters
        self.env.filters["format_float"] = lambda x: f"{x:.4f}" if x is not None else "N/A"
        self.env.filters["format_time"] = self._format_time
        self.env.filters["format_size"] = self._format_size

        logger.info(f"ReportGenerator initialized with templates from {template_dir}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    @staticmethod
    def _format_size(bytes_size: float) -> str:
        """Format file size in human-readable form."""
        size = float(bytes_size)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def generate_html(
        self,
        experiment_paths: List[Path],
        output_path: Path,
        template_name: str = "report.html",
        include_appendix: bool = True,
    ) -> None:
        """
        Generate HTML report from experiments.

        Parameters
        ----------
        experiment_paths : List[Path]
            Paths to experiment directories
        output_path : Path
            Output path for HTML report
        template_name : str
            Template file name to use
        include_appendix : bool
            Whether to include appendix section
        """
        logger.info(f"Generating HTML report for {len(experiment_paths)} experiment(s)")

        # Load experiment data
        comparator = ExperimentComparator()
        experiments = comparator.load_experiments(experiment_paths)

        if not experiments:
            logger.error("No valid experiments found")
            raise ValueError("No valid experiments to report on")

        # Prepare report data
        data = self._prepare_report_data(experiments, include_appendix)

        # Generate figures
        figures = self._generate_figures(experiments)
        data["figures"] = figures

        # Render template
        template = self.env.get_template(template_name)
        html = template.render(**data)

        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        logger.success(f"HTML report saved to {output_path}")

    def generate_pdf(
        self, experiment_paths: List[Path], output_path: Path, include_appendix: bool = True
    ) -> None:
        """
        Generate PDF report from experiments.

        Parameters
        ----------
        experiment_paths : List[Path]
            Paths to experiment directories
        output_path : Path
            Output path for PDF report
        include_appendix : bool
            Whether to include appendix section
        """
        logger.info(f"Generating PDF report for {len(experiment_paths)} experiment(s)")

        try:
            from weasyprint import HTML
        except ImportError:
            logger.error("WeasyPrint not installed. Install with: uv add weasyprint")
            raise ImportError(
                "WeasyPrint is required for PDF generation. Install with: uv add weasyprint"
            )

        # Generate HTML first
        html_path = output_path.with_suffix(".html")
        self.generate_html(experiment_paths, html_path, include_appendix=include_appendix)

        # Convert to PDF
        logger.info("Converting HTML to PDF...")
        HTML(str(html_path)).write_pdf(str(output_path))

        # Optionally remove temporary HTML
        # html_path.unlink()

        logger.success(f"PDF report saved to {output_path}")

    def _prepare_report_data(
        self, experiments: List[ExperimentResult], include_appendix: bool
    ) -> Dict[str, Any]:
        """Prepare data dictionary for template rendering."""

        # Compute summary statistics
        summary = self._compute_summary(experiments)

        # Prepare experiment details
        exp_details = []
        for exp in experiments:
            details = {
                "name": exp.name,
                "timestamp": exp.config.get("timestamp", "Unknown"),
                "duration": exp.training_history.get("duration", 0),
                "final_metrics": exp.final_metrics,
                "config": exp.config,
                "converged": exp.training_history.get("converged", False),
                "n_epochs": len(exp.training_history.get("loss", [])),
            }
            exp_details.append(details)

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_experiments": len(experiments),
            "experiments": exp_details,
            "summary": summary,
            "include_appendix": include_appendix,
        }

    def _compute_summary(self, experiments: List[ExperimentResult]) -> Dict[str, Any]:
        """Compute summary statistics across experiments."""

        if not experiments:
            return {}

        # Find best experiment for each metric
        metrics = ["loss", "ssim", "psnr"]
        best_experiments = {}

        for metric in metrics:
            values = [exp.final_metrics.get(metric, float("inf")) for exp in experiments]

            if metric == "loss":
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)

            best_experiments[metric] = {
                "name": experiments[best_idx].name,
                "value": values[best_idx],
            }

        # Compute statistics
        total_duration = sum(
            float(exp.training_history.get("duration", 0))  # type: ignore[arg-type, misc]
            for exp in experiments
        )

        summary = {
            "best_experiments": best_experiments,
            "total_duration": total_duration,
            "avg_epochs": np.mean(
                [len(exp.training_history.get("loss", [])) for exp in experiments]
            ),
        }

        return summary

    def _generate_figures(self, experiments: List[ExperimentResult]) -> Dict[str, str]:
        """Generate all figures as base64-encoded PNG."""

        figures = {}

        try:
            # Training curves
            fig = self._create_training_curves(experiments)
            figures["training_curves"] = self._fig_to_base64(fig)
            plt.close(fig)

            # Reconstruction comparison
            fig = self._create_reconstruction_comparison(experiments)
            figures["reconstruction_comparison"] = self._fig_to_base64(fig)
            plt.close(fig)

            # K-space coverage
            if len(experiments) > 0 and "aperture_mask" in experiments[0].checkpoint:
                fig = self._create_kspace_coverage(experiments)
                figures["kspace_coverage"] = self._fig_to_base64(fig)
                plt.close(fig)

            # Metric comparison (if multiple experiments)
            if len(experiments) > 1:
                fig = self._create_metric_comparison(experiments)
                figures["metric_comparison"] = self._fig_to_base64(fig)
                plt.close(fig)

        except Exception as e:  # noqa: BLE001 - Figure generation failure is non-fatal
            logger.warning(f"Error generating figures: {e}")

        return figures

    def _create_training_curves(self, experiments: List[ExperimentResult]) -> plt.Figure:
        """Create training curves plot."""

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        for exp in experiments:
            history = exp.training_history
            epochs = list(range(len(history.get("loss", []))))

            if "loss" in history:
                axes[0].plot(epochs, history["loss"], label=exp.name, linewidth=2)
            if "ssim" in history:
                axes[1].plot(epochs, history["ssim"], label=exp.name, linewidth=2)
            if "psnr" in history:
                axes[2].plot(epochs, history["psnr"], label=exp.name, linewidth=2)

        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_yscale("log")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel("SSIM", fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].set_ylabel("PSNR (dB)", fontsize=12)
        axes[2].set_xlabel("Epoch", fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_reconstruction_comparison(self, experiments: List[ExperimentResult]) -> plt.Figure:
        """Create reconstruction comparison plot."""

        n_exp = min(len(experiments), 4)  # Limit to 4 experiments
        fig, axes = plt.subplots(2, n_exp, figsize=(5 * n_exp, 10))

        if n_exp == 1:
            axes = axes.reshape(-1, 1)

        for i, exp in enumerate(experiments[:n_exp]):
            checkpoint = exp.checkpoint

            # Ground truth
            if "ground_truth" in checkpoint:
                gt = checkpoint["ground_truth"]
                if isinstance(gt, torch.Tensor):
                    gt = gt.cpu().numpy()
                axes[0, i].imshow(gt, cmap="gray")
                axes[0, i].set_title(f"{exp.name}\nGround Truth", fontsize=10)
                axes[0, i].axis("off")

            # Reconstruction
            if "reconstruction" in checkpoint:
                recon = checkpoint["reconstruction"]
                if isinstance(recon, torch.Tensor):
                    recon = recon.cpu().numpy()
                axes[1, i].imshow(recon, cmap="gray")
                ssim = exp.final_metrics.get("ssim", 0)
                axes[1, i].set_title(f"Reconstruction\nSSIM: {ssim:.4f}", fontsize=10)
                axes[1, i].axis("off")

        plt.tight_layout()
        return fig

    def _create_kspace_coverage(self, experiments: List[ExperimentResult]) -> plt.Figure:
        """Create k-space coverage plot."""

        n_exp = min(len(experiments), 4)
        fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 5))

        if n_exp == 1:
            axes = [axes]

        for i, exp in enumerate(experiments[:n_exp]):
            checkpoint = exp.checkpoint

            if "aperture_mask" in checkpoint:
                mask = checkpoint["aperture_mask"]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()

                im = axes[i].imshow(mask, cmap="viridis")
                axes[i].set_title(f"{exp.name}\nK-Space Coverage", fontsize=10)
                axes[i].axis("off")
                plt.colorbar(im, ax=axes[i], fraction=0.046)

        plt.tight_layout()
        return fig

    def _create_metric_comparison(self, experiments: List[ExperimentResult]) -> plt.Figure:
        """Create metric comparison bar chart."""

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        names = [exp.name for exp in experiments]
        losses = [exp.final_metrics.get("loss", 0) for exp in experiments]
        ssims = [exp.final_metrics.get("ssim", 0) for exp in experiments]
        psnrs = [exp.final_metrics.get("psnr", 0) for exp in experiments]

        x = np.arange(len(names))
        width = 0.6

        axes[0].bar(x, losses, width, color="steelblue")
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("Final Loss Comparison", fontsize=12)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha="right")
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3, axis="y")

        axes[1].bar(x, ssims, width, color="seagreen")
        axes[1].set_ylabel("SSIM", fontsize=12)
        axes[1].set_title("Final SSIM Comparison", fontsize=12)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=45, ha="right")
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis="y")

        axes[2].bar(x, psnrs, width, color="coral")
        axes[2].set_ylabel("PSNR (dB)", fontsize=12)
        axes[2].set_title("Final PSNR Comparison", fontsize=12)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, rotation=45, ha="right")
        axes[2].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    @staticmethod
    def _fig_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
        """Convert matplotlib figure to base64-encoded PNG."""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return img_base64
