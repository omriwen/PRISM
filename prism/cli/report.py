"""
Report Generation CLI
=====================

Command-line interface for generating experiment reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from prism.reporting import ReportGenerator


console = Console()


def add_report_parser(subparsers: Any) -> None:
    """
    Add report command to CLI parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser action to add report command to
    """
    parser = subparsers.add_parser(
        "report",
        help="Generate experiment reports",
        description="Generate comprehensive HTML or PDF reports from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate HTML report for single experiment
  spids report runs/experiment --format html

  # Generate PDF report
  spids report runs/experiment --format pdf

  # Multi-experiment comparison report
  spids report runs/exp1 runs/exp2 runs/exp3 --format pdf --output comparison.pdf

  # Custom template
  spids report runs/experiment --template my_template.html --format html

  # Without appendix
  spids report runs/experiment --format pdf --no-appendix
        """,
    )

    parser.add_argument("experiments", type=str, nargs="+", help="Paths to experiment directories")

    parser.add_argument(
        "--format",
        type=str,
        choices=["html", "pdf"],
        default="html",
        help="Report format (default: html)",
    )

    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output file path (default: auto-generated)"
    )

    parser.add_argument("--template", type=str, default=None, help="Custom template file path")

    parser.add_argument(
        "--no-appendix", action="store_true", help="Exclude appendix section from report"
    )

    parser.add_argument(
        "--template-dir", type=str, default=None, help="Directory containing custom templates"
    )

    parser.set_defaults(func=report_command)


def report_command(args: argparse.Namespace) -> None:
    """
    Execute report generation command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    """
    console.print("\n[bold cyan]SPIDS Report Generator[/bold cyan]\n")

    # Convert experiment paths to Path objects
    experiment_paths = [Path(exp).resolve() for exp in args.experiments]

    # Validate experiment paths
    invalid_paths = [p for p in experiment_paths if not p.exists()]
    if invalid_paths:
        console.print("[bold red]Error:[/bold red] The following paths do not exist:")
        for p in invalid_paths:
            console.print(f"  - {p}")
        return

    # Check for checkpoint files
    missing_checkpoints = []
    for path in experiment_paths:
        checkpoint = path / "checkpoint.pt"
        if not checkpoint.exists():
            missing_checkpoints.append(path)

    if missing_checkpoints:
        console.print("[bold yellow]Warning:[/bold yellow] No checkpoint.pt found in:")
        for p in missing_checkpoints:
            console.print(f"  - {p}")
        console.print("\nContinuing with available experiments...\n")
        experiment_paths = [p for p in experiment_paths if p not in missing_checkpoints]

    if not experiment_paths:
        console.print("[bold red]Error:[/bold red] No valid experiments found")
        return

    console.print(f"[green]✓[/green] Found {len(experiment_paths)} valid experiment(s)")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if len(experiment_paths) == 1:
            output_name = f"{experiment_paths[0].name}_report.{args.format}"
        else:
            output_name = f"comparison_report.{args.format}"
        output_path = Path.cwd() / output_name

    console.print(f"[green]✓[/green] Output path: {output_path}\n")

    # Initialize report generator
    try:
        if args.template_dir:
            template_dir = Path(args.template_dir)
            if not template_dir.exists():
                console.print(
                    f"[bold red]Error:[/bold red] Template directory not found: {template_dir}"
                )
                return
            generator = ReportGenerator(template_dir=template_dir)
        else:
            generator = ReportGenerator()

        console.print("[green]✓[/green] Report generator initialized\n")

    except Exception as e:  # noqa: BLE001 - CLI catch-all for user-friendly error display
        console.print(f"[bold red]Error:[/bold red] Failed to initialize report generator: {e}")
        logger.exception(e)
        return

    # Generate report
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if args.format == "html":
            task = progress.add_task("Generating HTML report...", total=None)
            try:
                template_name = Path(args.template).name if args.template else "report.html"
                generator.generate_html(
                    experiment_paths,
                    output_path,
                    template_name=template_name,
                    include_appendix=not args.no_appendix,
                )
                progress.update(task, completed=True)
            except Exception as e:  # noqa: BLE001 - CLI catch-all for user-friendly error display
                console.print(f"\n[bold red]Error:[/bold red] Failed to generate HTML report: {e}")
                logger.exception(e)
                return

        elif args.format == "pdf":
            task = progress.add_task("Generating PDF report...", total=None)
            try:
                generator.generate_pdf(
                    experiment_paths, output_path, include_appendix=not args.no_appendix
                )
                progress.update(task, completed=True)
            except ImportError:
                console.print(
                    "\n[bold red]Error:[/bold red] WeasyPrint not installed.\n"
                    "Install with: [cyan]uv add weasyprint[/cyan]"
                )
                return
            except Exception as e:  # noqa: BLE001 - CLI catch-all for user-friendly error display
                console.print(f"\n[bold red]Error:[/bold red] Failed to generate PDF report: {e}")
                logger.exception(e)
                return

    # Success message
    console.print("\n[bold green]✓ Report generated successfully![/bold green]")
    console.print(f"[cyan]Output:[/cyan] {output_path}")

    # Display file size
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        console.print(f"[cyan]Size:[/cyan] {size_mb:.2f} MB")

    console.print()


def main() -> None:
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="SPIDS Report Generator", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    add_report_parser(subparsers)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
