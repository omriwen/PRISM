"""Checkpoint inspector class for SPIDS experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import questionary
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class CheckpointInspector:
    """Inspect SPIDS experiment checkpoints.

    This class provides methods to load and inspect checkpoint files,
    displaying metadata, metrics, configuration, and exporting visualizations.

    Examples
    --------
    >>> inspector = CheckpointInspector("runs/experiment/checkpoint.pt")
    >>> inspector.show_summary()
    >>> inspector.export_reconstruction("reconstruction.png")
    """

    def __init__(self, checkpoint_path: Path, console: Optional[Console] = None):
        """Initialize checkpoint inspector.

        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file
        console : Console, optional
            Rich console for output. If None, creates a new console.

        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist
        RuntimeError
            If checkpoint is corrupted or cannot be loaded
        """
        self.path = Path(checkpoint_path)
        self.console = console or Console()

        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.path}")

        try:
            self.checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e

        # Get experiment directory and load additional files
        self.exp_dir = self.path.parent
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load additional metadata files (args.pt, args.txt)."""
        self.args = None
        self.args_dict = None

        # Try loading args.pt
        args_pt_path = self.exp_dir / "args.pt"
        if args_pt_path.exists():
            try:
                self.args = torch.load(args_pt_path, map_location="cpu", weights_only=False)
                if isinstance(self.args, dict):
                    self.args_dict = self.args
                elif hasattr(self.args, "__dict__"):
                    self.args_dict = vars(self.args)
            except Exception:  # noqa: BLE001 - Silent fallback for args loading
                pass

        # Try loading freq_pattern_args from checkpoint if args not found
        if self.args_dict is None and "freq_pattern_args" in self.checkpoint:
            freq_args = self.checkpoint["freq_pattern_args"]
            if isinstance(freq_args, dict):
                self.args_dict = freq_args
            elif hasattr(freq_args, "__dict__"):
                self.args_dict = vars(freq_args)

    def _get_config_value(self, key: str, default: Any = "unknown") -> Any:
        """Get configuration value from args or checkpoint."""
        if self.args_dict and key in self.args_dict:
            return self.args_dict[key]
        if key in self.checkpoint:
            return self.checkpoint[key]
        return default

    def _get_experiment_name(self) -> str:
        """Get experiment name from directory or config."""
        name = self._get_config_value("exp_name")
        if name == "unknown":
            name = self.exp_dir.name
        return str(name)

    def _get_timestamp(self) -> str:
        """Get experiment timestamp."""
        # Try to get from checkpoint metadata
        if "timestamp" in self.checkpoint:
            return str(self.checkpoint["timestamp"])

        # Fall back to checkpoint file modification time
        mtime = self.path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

    def _get_prism_version(self) -> str:
        """Get SPIDS version from checkpoint."""
        return str(self._get_config_value("version", "unknown"))

    def _get_final_metrics(self) -> Dict[str, float]:
        """Extract final metrics from checkpoint."""
        metrics = {}

        # Get final values from training history
        if "losses" in self.checkpoint and len(self.checkpoint["losses"]) > 0:
            metrics["loss"] = float(self.checkpoint["losses"][-1])

        if "ssims" in self.checkpoint and len(self.checkpoint["ssims"]) > 0:
            metrics["ssim"] = float(self.checkpoint["ssims"][-1])

        if "psnrs" in self.checkpoint and len(self.checkpoint["psnrs"]) > 0:
            metrics["psnr"] = float(self.checkpoint["psnrs"][-1])

        if "rmses" in self.checkpoint and len(self.checkpoint["rmses"]) > 0:
            metrics["rmse"] = float(self.checkpoint["rmses"][-1])

        return metrics

    def _get_training_info(self) -> Dict[str, Any]:
        """Get training information."""
        info = {}

        # Number of epochs/samples
        if "losses" in self.checkpoint:
            info["total_samples"] = len(self.checkpoint["losses"])

        if "last_center_idx" in self.checkpoint:
            info["last_sample_idx"] = int(self.checkpoint["last_center_idx"])

        # Failed samples
        if "failed_samples" in self.checkpoint:
            failed = self.checkpoint["failed_samples"]
            if isinstance(failed, list):
                info["failed_samples"] = len(failed)
            elif isinstance(failed, torch.Tensor):
                info["failed_samples"] = int(failed.sum())

        return info

    def show_summary(self) -> None:
        """Display checkpoint summary with metadata, metrics, and configuration."""
        # Metadata panel
        metadata_table = Table(show_header=False, box=None, padding=(0, 2))
        metadata_table.add_column("Key", style="cyan")
        metadata_table.add_column("Value", style="white")

        metadata_table.add_row("Experiment", self._get_experiment_name())
        metadata_table.add_row("Date", self._get_timestamp())
        metadata_table.add_row("SPIDS Version", self._get_prism_version())
        metadata_table.add_row("Checkpoint", str(self.path.name))

        self.console.print(Panel(metadata_table, title="[bold cyan]Metadata[/bold cyan]"))
        self.console.print()

        # Metrics panel
        metrics = self._get_final_metrics()
        if metrics:
            metrics_table = Table(show_header=False, box=None, padding=(0, 2))
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="white")

            if "loss" in metrics:
                metrics_table.add_row("Loss", f"{metrics['loss']:.6f}")
            if "ssim" in metrics:
                metrics_table.add_row(
                    "SSIM", f"{metrics['ssim']:.4f}  ({metrics['ssim'] * 100:.1f}% similarity)"
                )
            if "psnr" in metrics:
                metrics_table.add_row("PSNR", f"{metrics['psnr']:.2f} dB")
            if "rmse" in metrics:
                metrics_table.add_row("RMSE", f"{metrics['rmse']:.6f}")

            self.console.print(Panel(metrics_table, title="[bold cyan]Final Metrics[/bold cyan]"))
            self.console.print()

        # Training info panel
        training_info = self._get_training_info()
        if training_info:
            training_table = Table(show_header=False, box=None, padding=(0, 2))
            training_table.add_column("Key", style="cyan")
            training_table.add_column("Value", style="white")

            if "total_samples" in training_info:
                training_table.add_row("Total Samples", str(training_info["total_samples"]))
            if "last_sample_idx" in training_info:
                training_table.add_row("Last Sample Index", str(training_info["last_sample_idx"]))
            if "failed_samples" in training_info:
                failed_count = training_info["failed_samples"]
                if failed_count > 0:
                    training_table.add_row("Failed Samples", f"[yellow]{failed_count}[/yellow]")
                else:
                    training_table.add_row("Failed Samples", f"[green]{failed_count}[/green]")

            self.console.print(Panel(training_table, title="[bold cyan]Training Info[/bold cyan]"))
            self.console.print()

        # Configuration panel
        config_table = self._create_config_table()
        if config_table:
            self.console.print(Panel(config_table, title="[bold cyan]Configuration[/bold cyan]"))

    def _create_config_table(self) -> Optional[Table]:
        """Create configuration table from args."""
        if not self.args_dict:
            return None

        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="white")

        # Key parameters to show
        key_params = [
            ("obj_name", "Object"),
            ("n_samples", "Samples"),
            ("fermat", "Fermat Pattern"),
            ("sample_diameter", "Sample Diameter"),
            ("snr", "SNR"),
            ("propagator_method", "Propagator"),
            ("loss_type", "Loss Type"),
            ("lr", "Learning Rate"),
            ("max_epochs", "Max Epochs"),
        ]

        for key, label in key_params:
            if key in self.args_dict:
                value = self.args_dict[key]
                if isinstance(value, bool):
                    value = "Yes" if value else "No"
                config_table.add_row(label, str(value))

        return config_table if config_table.row_count > 0 else None

    def show_metrics_only(self) -> None:
        """Display only metrics (for --metrics-only flag)."""
        metrics = self._get_final_metrics()
        if not metrics:
            self.console.print("[yellow]No metrics found in checkpoint[/yellow]")
            return

        # Create simple metrics table
        table = Table(title="Final Metrics", box=None)
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Value", style="white", justify="right")

        if "loss" in metrics:
            table.add_row("Loss", f"{metrics['loss']:.6f}")
        if "ssim" in metrics:
            table.add_row("SSIM", f"{metrics['ssim']:.4f}")
        if "psnr" in metrics:
            table.add_row("PSNR", f"{metrics['psnr']:.2f} dB")
        if "rmse" in metrics:
            table.add_row("RMSE", f"{metrics['rmse']:.6f}")

        self.console.print(table)

    def show_training_history(self) -> None:
        """Display training history as ASCII plot."""
        if "losses" not in self.checkpoint:
            self.console.print("[yellow]No training history found[/yellow]")
            return

        losses = self.checkpoint["losses"]
        ssims = self.checkpoint.get("ssims", [])
        psnrs = self.checkpoint.get("psnrs", [])

        self.console.print(f"\n[cyan]Training History ({len(losses)} samples)[/cyan]")
        self.console.print(f"Loss range: {min(losses):.6f} - {max(losses):.6f}")

        if ssims:
            self.console.print(f"SSIM range: {min(ssims):.4f} - {max(ssims):.4f}")
        if psnrs:
            self.console.print(f"PSNR range: {min(psnrs):.2f} - {max(psnrs):.2f} dB")

    def export_reconstruction(self, output_path: Path, dpi: int = 300) -> None:
        """Save reconstruction comparison as PNG.

        Parameters
        ----------
        output_path : Path
            Output file path
        dpi : int, optional
            Resolution in DPI (default: 300)

        Raises
        ------
        ValueError
            If reconstruction data is not found in checkpoint
        """
        if "current_rec" not in self.checkpoint:
            raise ValueError("No reconstruction found in checkpoint")

        # Get reconstruction
        rec = self.checkpoint["current_rec"]
        if isinstance(rec, torch.Tensor):
            rec = rec.detach().cpu().numpy()
        # Squeeze to 2D if needed
        rec = np.squeeze(rec)
        if rec.ndim != 2:
            raise ValueError(f"Expected 2D reconstruction, got shape {rec.shape}")

        # Try to load ground truth from telescope_agg
        ground_truth = None
        if "telescope_agg" in self.checkpoint:
            telescope_agg = self.checkpoint["telescope_agg"]
            if hasattr(telescope_agg, "init_im"):
                ground_truth = telescope_agg.init_im
                if isinstance(ground_truth, torch.Tensor):
                    ground_truth = ground_truth.detach().cpu().numpy()
                # Squeeze to 2D if needed
                ground_truth = np.squeeze(ground_truth)
                if ground_truth.ndim != 2:
                    ground_truth = None  # Invalid shape, skip

        # Create figure
        if ground_truth is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Ground truth
            axes[0].imshow(np.abs(ground_truth), cmap="gray")
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")

            # Reconstruction
            axes[1].imshow(np.abs(rec), cmap="gray")
            axes[1].set_title("Reconstruction")
            axes[1].axis("off")

            # Difference
            diff = np.abs(ground_truth - rec)
            im = axes[2].imshow(diff, cmap="hot")
            axes[2].set_title("Absolute Difference")
            axes[2].axis("off")
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        else:
            # Only reconstruction available
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(np.abs(rec), cmap="gray")
            ax.set_title("Reconstruction")
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        self.console.print(f"[green]✓ Reconstruction exported to {output_path}[/green]")

    def show_full_config(self) -> None:
        """Display full configuration details."""
        if not self.args_dict:
            self.console.print("[yellow]No configuration found[/yellow]")
            return

        self.console.print("[cyan]Full Configuration:[/cyan]")
        for key, value in sorted(self.args_dict.items()):
            if not key.startswith("_"):  # Skip private attributes
                self.console.print(f"  [cyan]{key}:[/cyan] {value}")

    def visualize_sample_pattern(self) -> None:
        """Display sample pattern if available."""
        if "sample_centers" not in self.checkpoint:
            self.console.print("[yellow]No sample pattern data found[/yellow]")
            return

        sample_centers = self.checkpoint["sample_centers"]
        if isinstance(sample_centers, torch.Tensor):
            sample_centers = sample_centers.detach().cpu().numpy()

        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(sample_centers[:, 0], sample_centers[:, 1], s=20, alpha=0.6)
        ax.set_title(f"Sample Pattern ({len(sample_centers)} samples)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Show plot
        plt.tight_layout()
        plt.show()
        plt.close(fig)

        self.console.print("[green]✓ Sample pattern displayed[/green]")

    def run_interactive(self) -> None:
        """Run interactive mode with menu-driven navigation."""
        self.console.print()
        self.console.print("[bold cyan]Interactive Checkpoint Inspector[/bold cyan]")
        self.console.print()

        # Show summary first
        self.show_summary()
        self.console.print()

        while True:
            # Create menu choices
            choices = [
                "Show training history",
                "Visualize sample pattern",
                "Export reconstruction image",
                "Show full configuration",
                "Show metrics only",
                "Quit",
            ]

            # Display menu
            action = questionary.select(
                "What would you like to do?",
                choices=choices,
                style=questionary.Style(
                    [
                        ("qmark", "fg:#673ab7 bold"),
                        ("question", "bold"),
                        ("answer", "fg:#f44336 bold"),
                        ("pointer", "fg:#673ab7 bold"),
                        ("highlighted", "fg:#673ab7 bold"),
                        ("selected", "fg:#cc5454"),
                    ]
                ),
            ).ask()

            if action is None or action == "Quit":
                self.console.print("[cyan]Exiting interactive mode[/cyan]")
                break

            self.console.print()

            if action == "Show training history":
                self.show_training_history()

            elif action == "Visualize sample pattern":
                self.visualize_sample_pattern()

            elif action == "Export reconstruction image":
                # Ask for output path
                default_path = f"{self._get_experiment_name()}_reconstruction.png"
                output_path_str = questionary.text(
                    "Enter output path:",
                    default=default_path,
                ).ask()

                if output_path_str:
                    try:
                        output_path = Path(output_path_str)
                        dpi = questionary.text(
                            "Enter DPI (default: 300):",
                            default="300",
                            validate=lambda x: x.isdigit() and int(x) > 0,
                        ).ask()

                        self.export_reconstruction(output_path, dpi=int(dpi))
                    except Exception as e:  # noqa: BLE001 - Interactive menu catch-all
                        self.console.print(f"[red]Error: {e}[/red]")

            elif action == "Show full configuration":
                self.show_full_config()

            elif action == "Show metrics only":
                self.show_metrics_only()

            self.console.print()
            # Ask if they want to continue
            continue_choice = questionary.confirm("Continue exploring?", default=True).ask()

            if not continue_choice:
                self.console.print("[cyan]Exiting interactive mode[/cyan]")
                break

            self.console.print()
