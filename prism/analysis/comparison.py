"""
Experiment comparison module for SPIDS.

This module provides functionality to compare multiple SPIDS experiments,
including metrics comparison, configuration diffing, and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from rich.console import Console
from rich.table import Table


@dataclass
class ExperimentResult:
    """Container for experiment results.

    Attributes
    ----------
    name : str
        Experiment identifier (directory name)
    path : Path
        Path to experiment directory
    config : dict
        Experiment configuration (from args.pt)
    final_metrics : Dict[str, float]
        Final metric values
    training_history : Dict[str, List[float]]
        Training curves (losses, ssims, psnrs, etc.)
    checkpoint : dict
        Full checkpoint data
    """

    name: str
    path: Path
    config: dict
    final_metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    checkpoint: dict


class ExperimentComparator:
    """Compare multiple SPIDS experiments.

    This class provides methods to load experiment data from checkpoint files,
    compare metrics, find configuration differences, and generate comparison
    visualizations.

    Examples
    --------
    >>> comparator = ExperimentComparator()
    >>> experiments = comparator.load_experiments([
    ...     Path("runs/exp1"),
    ...     Path("runs/exp2")
    ... ])
    >>> comparator.print_metrics_table(experiments)
    >>> fig = comparator.plot_comparison(experiments)
    >>> fig.savefig("comparison.png")
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize comparator.

        Parameters
        ----------
        console : Console, optional
            Rich console for output. If None, creates a new console.
        """
        self.console = console or Console()

    def load_experiments(self, paths: List[Path]) -> List[ExperimentResult]:
        """Load experiment data from checkpoint files.

        Parameters
        ----------
        paths : List[Path]
            List of experiment directory paths

        Returns
        -------
        List[ExperimentResult]
            Loaded experiment results

        Raises
        ------
        FileNotFoundError
            If checkpoint or args files are missing
        """
        experiments = []

        for path in paths:
            path = Path(path)

            if not path.exists():
                self.console.print(f"[yellow]Warning: Path does not exist: {path}[/yellow]")
                continue

            checkpoint_path = path / "checkpoint.pt"
            args_path = path / "args.pt"

            if not checkpoint_path.exists():
                self.console.print(f"[yellow]Warning: No checkpoint found in {path}[/yellow]")
                continue

            try:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

                # Load configuration
                if args_path.exists():
                    config = torch.load(args_path, map_location="cpu", weights_only=False)
                else:
                    # Fallback to checkpoint metadata if args.pt missing
                    config = checkpoint.get("freq_pattern_args", {})

                # Extract training history
                training_history = {
                    "losses": checkpoint.get("losses", []),
                    "ssims": checkpoint.get("ssims", []),
                    "psnrs": checkpoint.get("psnrs", []),
                    "rmses": checkpoint.get("rmses", []),
                }

                # Compute final metrics
                final_metrics = {}
                if len(training_history["losses"]) > 0:
                    final_metrics["loss"] = float(training_history["losses"][-1])
                if len(training_history["ssims"]) > 0:
                    final_metrics["ssim"] = float(training_history["ssims"][-1])
                if len(training_history["psnrs"]) > 0:
                    final_metrics["psnr"] = float(training_history["psnrs"][-1])
                if len(training_history["rmses"]) > 0:
                    final_metrics["rmse"] = float(training_history["rmses"][-1])

                # Add epoch count
                if len(training_history["losses"]) > 0:
                    final_metrics["epochs"] = len(training_history["losses"])

                # Create result object
                result = ExperimentResult(
                    name=path.name,
                    path=path,
                    config=config,
                    final_metrics=final_metrics,
                    training_history=training_history,
                    checkpoint=checkpoint,
                )

                experiments.append(result)

            except Exception as e:  # noqa: BLE001 - Continue loading other experiments on failure
                self.console.print(f"[red]Error loading experiment from {path}: {e}[/red]")
                continue

        return experiments

    def compare_metrics(self, experiments: List[ExperimentResult]) -> Dict[str, Dict[str, Any]]:
        """Compare final metrics across experiments.

        Parameters
        ----------
        experiments : List[ExperimentResult]
            Experiments to compare

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Comparison results with best values and statistics
        """
        if not experiments:
            return {}

        # Collect all metric names
        all_metrics: set[str] = set()
        for exp in experiments:
            all_metrics.update(exp.final_metrics.keys())

        comparison = {}

        for metric in all_metrics:
            values = []
            exp_names = []

            for exp in experiments:
                if metric in exp.final_metrics:
                    values.append(exp.final_metrics[metric])
                    exp_names.append(exp.name)

            if values:
                # Determine if lower is better (for loss, rmse) or higher (for ssim, psnr)
                lower_is_better = metric in ["loss", "rmse"]

                if lower_is_better:
                    best_idx = int(np.argmin(values))
                else:
                    best_idx = int(np.argmax(values))

                comparison[metric] = {
                    "values": values,
                    "experiments": exp_names,
                    "best_value": values[best_idx],
                    "best_experiment": exp_names[best_idx],
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "lower_is_better": lower_is_better,
                }

        return comparison

    def compare_configs(self, experiments: List[ExperimentResult]) -> Dict[str, Dict[str, Any]]:
        """Find configuration differences across experiments.

        Parameters
        ----------
        experiments : List[ExperimentResult]
            Experiments to compare

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping parameter names to their values across experiments
        """
        if not experiments:
            return {}

        # Collect all config keys
        all_keys: set[str] = set()
        for exp in experiments:
            all_keys.update(exp.config.keys())

        differences = {}

        for key in all_keys:
            values: dict[str, list[str]] = {}

            for exp in experiments:
                value = exp.config.get(key, None)
                # Convert to string for comparison (handles various types)
                value_str = str(value)

                if value_str not in values:
                    values[value_str] = []
                values[value_str].append(exp.name)

            # Only include if there are differences
            if len(values) > 1:
                differences[key] = values

        return differences

    def print_metrics_table(
        self,
        experiments: List[ExperimentResult],
        metrics: Optional[List[str]] = None,
    ) -> None:
        """Print formatted metrics comparison table.

        Parameters
        ----------
        experiments : List[ExperimentResult]
            Experiments to display
        metrics : List[str], optional
            Specific metrics to display. If None, shows all available metrics.
        """
        if not experiments:
            self.console.print("[yellow]No experiments to display[/yellow]")
            return

        comparison = self.compare_metrics(experiments)

        # Filter metrics if specified
        if metrics:
            comparison = {k: v for k, v in comparison.items() if k in metrics}

        # Create table
        table = Table(title="Experiment Comparison", show_header=True)
        table.add_column("Metric", style="cyan", no_wrap=True)

        for exp in experiments:
            table.add_column(exp.name, style="white")

        table.add_column("Best", style="green")

        # Add rows for each metric
        metric_order = ["loss", "ssim", "psnr", "rmse", "epochs"]
        sorted_metrics = sorted(
            comparison.keys(), key=lambda x: metric_order.index(x) if x in metric_order else 999
        )

        for metric in sorted_metrics:
            data = comparison[metric]
            row = [metric.upper()]

            best_exp = data["best_experiment"]

            # Add values for each experiment
            for exp in experiments:
                if exp.name in data["experiments"]:
                    idx = data["experiments"].index(exp.name)
                    value = data["values"][idx]

                    # Format based on metric type
                    if metric == "loss":
                        value_str = f"{value:.4f}"
                    elif metric in ["ssim"]:
                        value_str = f"{value:.4f}"
                    elif metric == "psnr":
                        value_str = f"{value:.2f} dB"
                    elif metric == "rmse":
                        value_str = f"{value:.4f}"
                    elif metric == "epochs":
                        value_str = f"{int(value)}"
                    else:
                        value_str = f"{value:.4f}"

                    # Highlight best value
                    if exp.name == best_exp:
                        value_str += " ✓"

                    row.append(value_str)
                else:
                    row.append("N/A")

            row.append(best_exp)
            table.add_row(*row)

        # Display table
        self.console.print()
        self.console.print(table)
        self.console.print()

    def print_config_diff(self, experiments: List[ExperimentResult]) -> None:
        """Print configuration differences.

        Parameters
        ----------
        experiments : List[ExperimentResult]
            Experiments to compare
        """
        differences = self.compare_configs(experiments)

        if not differences:
            self.console.print("[green]✓ All configurations are identical[/green]")
            return

        self.console.print()
        self.console.print("[bold]Configuration Differences:[/bold]")
        self.console.print()

        for param, values in sorted(differences.items()):
            lines = [f"[cyan]{param}:[/cyan]"]

            for value_str, exp_names in values.items():
                exp_list = ", ".join(exp_names)
                lines.append(f"  • {value_str} → {exp_list}")

            self.console.print("\n".join(lines))
            self.console.print()

    def plot_comparison(
        self,
        experiments: List[ExperimentResult],
        output_path: Optional[Path] = None,
        dpi: int = 150,
    ) -> Figure:
        """Generate comparison visualization.

        Creates a multi-panel figure with:
        - Training curves (loss, SSIM, PSNR)
        - Metric comparison bar chart

        Parameters
        ----------
        experiments : List[ExperimentResult]
            Experiments to visualize
        output_path : Path, optional
            If provided, save figure to this path
        dpi : int, optional
            Figure DPI (default: 150)

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        if not experiments:
            raise ValueError("No experiments to plot")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))

        # Create grid: 3 rows for training curves, 1 row for bars
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # Training curves
        ax_loss = fig.add_subplot(gs[0, :])
        ax_ssim = fig.add_subplot(gs[1, :])
        ax_psnr = fig.add_subplot(gs[2, :])

        # Plot training curves
        colors = plt.colormaps["tab10"](np.linspace(0, 1, len(experiments)))

        for i, exp in enumerate(experiments):
            history = exp.training_history
            color = colors[i]
            label = exp.name

            # Loss curve
            if len(history["losses"]) > 0:
                epochs = range(1, len(history["losses"]) + 1)
                ax_loss.plot(epochs, history["losses"], label=label, color=color, linewidth=2)

            # SSIM curve
            if len(history["ssims"]) > 0:
                epochs = range(1, len(history["ssims"]) + 1)
                ax_ssim.plot(epochs, history["ssims"], label=label, color=color, linewidth=2)

            # PSNR curve
            if len(history["psnrs"]) > 0:
                epochs = range(1, len(history["psnrs"]) + 1)
                ax_psnr.plot(epochs, history["psnrs"], label=label, color=color, linewidth=2)

        # Configure loss axis
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training Loss", fontsize=14, fontweight="bold")
        ax_loss.set_yscale("log")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc="best")

        # Configure SSIM axis
        ax_ssim.set_xlabel("Epoch")
        ax_ssim.set_ylabel("SSIM")
        ax_ssim.set_title("Structural Similarity Index (SSIM)", fontsize=14, fontweight="bold")
        ax_ssim.grid(True, alpha=0.3)
        ax_ssim.legend(loc="best")
        ax_ssim.set_ylim((0, 1))

        # Configure PSNR axis
        ax_psnr.set_xlabel("Epoch")
        ax_psnr.set_ylabel("PSNR (dB)")
        ax_psnr.set_title("Peak Signal-to-Noise Ratio (PSNR)", fontsize=14, fontweight="bold")
        ax_psnr.grid(True, alpha=0.3)
        ax_psnr.legend(loc="best")

        # Metric comparison bars
        ax_bars = fig.add_subplot(gs[3, :])

        comparison = self.compare_metrics(experiments)

        # Prepare data for bar chart
        metrics_to_plot = ["loss", "ssim", "psnr"]
        available_metrics = [m for m in metrics_to_plot if m in comparison]

        if available_metrics:
            x = np.arange(len(available_metrics))
            width = 0.8 / len(experiments)

            for i, exp in enumerate(experiments):
                values = []
                for metric in available_metrics:
                    if exp.name in comparison[metric]["experiments"]:
                        idx = comparison[metric]["experiments"].index(exp.name)
                        value = comparison[metric]["values"][idx]

                        # Normalize for visualization
                        if metric == "loss":
                            value = -np.log10(value + 1e-10)  # Log scale for loss
                        elif metric == "ssim":
                            value = value * 100  # Convert to percentage
                        # PSNR stays as is

                        values.append(value)
                    else:
                        values.append(0)

                offset = (i - len(experiments) / 2) * width + width / 2
                ax_bars.bar(
                    x + offset,
                    values,
                    width,
                    label=exp.name,
                    color=colors[i],
                )

            ax_bars.set_xlabel("Metric")
            ax_bars.set_ylabel("Value (normalized)")
            ax_bars.set_title("Final Metrics Comparison", fontsize=14, fontweight="bold")
            ax_bars.set_xticks(x)
            ax_bars.set_xticklabels([m.upper() for m in available_metrics])
            ax_bars.legend(loc="best")
            ax_bars.grid(True, alpha=0.3, axis="y")

        # Overall title
        fig.suptitle(
            f"SPIDS Experiment Comparison ({len(experiments)} experiments)",
            fontsize=16,
            fontweight="bold",
        )

        # Save if path provided
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            self.console.print(f"[green]✓ Comparison figure saved: {output_path}[/green]")

        return fig
