"""Command handlers and parser for inspect CLI."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from prism.cli.inspect_pkg.inspector import CheckpointInspector


def create_inspect_parser() -> argparse.ArgumentParser:
    """Create argument parser for inspect command.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for checkpoint inspection
    """
    parser = argparse.ArgumentParser(
        description="Inspect SPIDS experiment checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "checkpoint", type=str, help="Path to checkpoint file (e.g., runs/exp1/checkpoint.pt)"
    )

    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Show only metrics (minimal output)",
    )

    parser.add_argument(
        "--export-reconstruction",
        type=str,
        default=None,
        metavar="PATH",
        help="Export reconstruction as image (PNG/PDF)",
    )

    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Show training history summary",
    )

    parser.add_argument(
        "--show-full-config",
        action="store_true",
        help="Show full configuration details",
    )

    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for exported images (default: 300)"
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode with menu-driven navigation",
    )

    return parser


def inspect_command(args: argparse.Namespace) -> int:
    """Execute inspect command.

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

    try:
        # Create inspector
        checkpoint_path = Path(args.checkpoint)
        inspector = CheckpointInspector(checkpoint_path, console=console)

        # Run interactive mode if requested
        if args.interactive:
            inspector.run_interactive()
            return 0

        # Show appropriate output based on flags
        if args.metrics_only:
            inspector.show_metrics_only()
        else:
            inspector.show_summary()

        if args.show_history:
            inspector.show_training_history()

        if args.show_full_config:
            console.print()
            inspector.show_full_config()

        if args.export_reconstruction:
            output_path = Path(args.export_reconstruction)
            inspector.export_reconstruction(output_path, dpi=args.dpi)

        return 0

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    except RuntimeError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]The checkpoint file may be corrupted.[/yellow]")
        return 1
    except Exception as e:  # noqa: BLE001 - CLI catch-all for user-friendly error display
        console.print(f"[red]Unexpected error: {e}[/red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for inspect command.

    Parameters
    ----------
    argv : List[str], optional
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code
    """
    parser = create_inspect_parser()
    args = parser.parse_args(argv)
    return inspect_command(args)
