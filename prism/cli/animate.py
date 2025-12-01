"""
CLI command for generating training animations from SPIDS experiments.

Usage:
    python -m spids.cli.animate runs/experiment --output training.mp4 [options]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from prism.visualization.animation import TrainingAnimator


def create_animate_parser() -> argparse.ArgumentParser:
    """Create argument parser for animate command.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for animation generation
    """
    parser = argparse.ArgumentParser(
        description="Generate training progression animations from SPIDS experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "experiments",
        type=str,
        nargs="+",
        help="Path(s) to experiment directory/directories (e.g., runs/exp1 runs/exp2)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (e.g., training.mp4 or training.gif)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["mp4", "gif", "auto"],
        default="auto",
        help="Output format (auto detects from extension)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second (for MP4 only)",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=100,
        help="Duration per frame in milliseconds (for GIF only)",
    )

    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Number of frames to generate (default: auto-determined from training history)",
    )

    parser.add_argument(
        "--layout",
        type=str,
        choices=["grid", "horizontal", "side_by_side"],
        default="grid",
        help="Layout for multi-experiment comparison (grid or horizontal)",
    )

    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable metric overlays",
    )

    parser.add_argument(
        "--no-difference",
        action="store_true",
        help="Disable difference map (for single experiment)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint.pt",
        help="Name of checkpoint file to load",
    )

    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Number of loops for GIF (0 = infinite)",
    )

    return parser


def animate_command(args: argparse.Namespace) -> int:
    """Execute animate command.

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
        # Validate experiment paths
        exp_paths = [Path(p) for p in args.experiments]
        for path in exp_paths:
            if not path.exists():
                console.print(f"[red]Error: Experiment path not found: {path}[/red]")
                return 1

        # Determine output format
        output_path = Path(args.output)
        if args.format == "auto":
            if output_path.suffix.lower() == ".gif":
                output_format = "gif"
            elif output_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
                output_format = "mp4"
            else:
                console.print(
                    f"[yellow]Warning: Unknown extension '{output_path.suffix}', "
                    f"defaulting to MP4[/yellow]"
                )
                output_format = "mp4"
        else:
            output_format = args.format

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Single or multiple experiments?
        if len(exp_paths) == 1:
            # Single experiment animation
            console.print(f"[cyan]Generating animation for: {exp_paths[0].name}[/cyan]")

            animator = TrainingAnimator(exp_paths[0], checkpoint_file=args.checkpoint)

            if output_format == "mp4":
                animator.generate_video(
                    output_path,
                    fps=args.fps,
                    n_frames=args.n_frames,
                    show_metrics=not args.no_metrics,
                    show_difference=not args.no_difference,
                )
            else:  # gif
                animator.generate_gif(
                    output_path,
                    duration=args.duration,
                    n_frames=args.n_frames,
                    show_metrics=not args.no_metrics,
                    show_difference=not args.no_difference,
                    loop=args.loop,
                )

        else:
            # Multi-experiment comparison
            console.print(
                f"[cyan]Generating comparison animation for {len(exp_paths)} experiments:[/cyan]"
            )
            for path in exp_paths:
                console.print(f"  - {path.name}")

            # Normalize layout names
            layout = args.layout
            if layout == "side_by_side":
                layout = "horizontal"

            multi_animator = TrainingAnimator.from_multiple(
                exp_paths, checkpoint_file=args.checkpoint
            )

            if output_format == "mp4":
                multi_animator.generate_video(
                    output_path,
                    fps=args.fps,
                    n_frames=args.n_frames,
                    layout=layout,
                )
            else:  # gif
                multi_animator.generate_gif(
                    output_path,
                    duration=args.duration,
                    n_frames=args.n_frames,
                    layout=layout,
                    loop=args.loop,
                )

        console.print(f"[green]✓ Animation complete: {output_path}[/green]")
        return 0

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print(
            "[yellow]Hint: Install required dependencies with: uv add opencv-python pillow[/yellow]"
        )
        return 1
    except Exception as e:  # noqa: BLE001 - CLI catch-all for user-friendly error display
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for animate command.

    Parameters
    ----------
    argv : List[str], optional
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code
    """
    parser = create_animate_parser()
    args = parser.parse_args(argv)
    return animate_command(args)


if __name__ == "__main__":
    sys.exit(main())
