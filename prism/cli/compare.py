"""
CLI command for comparing SPIDS experiments.

Usage:
    python -m spids.cli.compare runs/exp1 runs/exp2 [options]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from prism.analysis.comparison import ExperimentComparator


def create_compare_parser() -> argparse.ArgumentParser:
    """Create argument parser for compare command.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for experiment comparison
    """
    parser = argparse.ArgumentParser(
        description="Compare SPIDS experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "experiments",
        type=str,
        nargs="+",
        help="Paths to experiment directories (e.g., runs/exp1 runs/exp2)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for comparison visualization (PNG/PDF)",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics to compare (e.g., loss ssim psnr)",
    )

    parser.add_argument(
        "--show-config-diff",
        action="store_true",
        help="Show configuration differences between experiments",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output figure (higher = better quality, larger file)",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating comparison visualization",
    )

    return parser


def compare_command(args: argparse.Namespace) -> int:
    """Execute comparison command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    console = Console()

    # Convert experiment paths to Path objects
    experiment_paths = [Path(p) for p in args.experiments]

    # Validate that we have at least 2 experiments
    if len(experiment_paths) < 2:
        console.print("[red]Error: At least 2 experiments are required for comparison[/red]")
        return 1

    # Create comparator
    comparator = ExperimentComparator(console=console)

    # Load experiments
    console.print(f"[cyan]Loading {len(experiment_paths)} experiments...[/cyan]")
    experiments = comparator.load_experiments(experiment_paths)

    if not experiments:
        console.print("[red]Error: No experiments could be loaded[/red]")
        return 1

    if len(experiments) < 2:
        console.print(
            f"[yellow]Warning: Only {len(experiments)} experiment(s) loaded. "
            "Comparison requires at least 2 experiments.[/yellow]"
        )
        return 1

    console.print(f"[green]✓ Successfully loaded {len(experiments)} experiments[/green]")
    console.print()

    # Print metrics table
    comparator.print_metrics_table(experiments, metrics=args.metrics)

    # Print configuration differences if requested
    if args.show_config_diff:
        comparator.print_config_diff(experiments)

    # Generate visualization unless --no-plot is set
    if not args.no_plot:
        try:
            output_path = Path(args.output) if args.output else None

            if output_path is None:
                # Default output path
                output_path = Path("comparison_report.png")

            console.print("[cyan]Generating comparison visualization...[/cyan]")

            fig = comparator.plot_comparison(
                experiments,
                output_path=output_path,
                dpi=args.dpi,
            )

            # Close figure to free memory
            import matplotlib.pyplot as plt

            plt.close(fig)

        except Exception as e:  # noqa: BLE001 - CLI catch-all for user-friendly error display
            console.print(f"[red]Error generating visualization: {e}[/red]")
            return 1

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for compare command.

    Parameters
    ----------
    argv : List[str], optional
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code
    """
    parser = create_compare_parser()
    args = parser.parse_args(argv)
    return compare_command(args)


if __name__ == "__main__":
    sys.exit(main())
