"""Command handlers for patterns CLI."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from prism.cli.patterns.config import PatternConfig
from prism.cli.patterns.gallery import generate_gallery_html
from prism.cli.patterns.visualizers import plot_coverage_heatmap, plot_sample_positions
from prism.core.pattern_library import PatternLibrary
from prism.core.pattern_loader import load_and_generate_pattern
from prism.core.pattern_preview import (
    compute_pattern_statistics,
    visualize_pattern,
)


def list_command(args: argparse.Namespace, console: Console) -> int:
    """
    List all available patterns.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    console : Console
        Rich console for formatted output

    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Get patterns based on filters
    if args.recommended:
        patterns = PatternLibrary.get_recommended_patterns()
        title = "Recommended Patterns"
    elif args.property:
        patterns = PatternLibrary.get_pattern_by_property(args.property)
        title = f"Patterns with Property: {args.property}"
    else:
        patterns = PatternLibrary.list_patterns()
        title = "Available Patterns"

    if not patterns:
        console.print("[yellow]No patterns found matching criteria[/yellow]")
        return 0

    # Create table
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green", width=20)
    table.add_column("Properties", width=25)

    if args.detailed:
        table.add_column("Description", width=40)
        table.add_column("Parameters", width=30)
        table.add_column("Reference", width=30)

    # Add rows
    for pattern in patterns:
        properties_str = ", ".join(pattern.properties)

        if args.detailed:
            params_str = ", ".join(pattern.parameters[:3])
            if len(pattern.parameters) > 3:
                params_str += ", ..."
            table.add_row(
                f"{pattern.name}{'*' if pattern.recommended else ''}",
                properties_str,
                (
                    pattern.description[:100] + "..."
                    if len(pattern.description) > 100
                    else pattern.description
                ),
                params_str,
                (
                    pattern.reference[:50] + "..."
                    if len(pattern.reference) > 50
                    else pattern.reference
                ),
            )
        else:
            table.add_row(
                f"{pattern.name}{'*' if pattern.recommended else ''}",
                properties_str,
            )

    console.print(table)

    if not args.detailed:
        console.print("\n[dim]Use --detailed to see full information[/dim]")

    console.print("\n[dim]* = Recommended for general use[/dim]")

    return 0


def show_command(args: argparse.Namespace, console: Console) -> int:
    """
    Visualize a specific pattern.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    console : Console
        Rich console for formatted output

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    try:
        # Get pattern info
        pattern_info = PatternLibrary.get_pattern_info(args.pattern)
        console.print(f"[cyan]Visualizing pattern:[/cyan] {pattern_info.name}")

        # Create config
        config = PatternConfig(args)

        # Generate pattern
        sample_centers, metadata = load_and_generate_pattern(pattern_info.spec, config)

        # Visualize
        output_path = Path(args.output)
        fig = visualize_pattern(
            sample_centers,
            config.roi_diameter,
            save_path=output_path,
            show_statistics=True,
        )
        plt.close(fig)

        console.print(f"[green]✓[/green] Visualization saved to: {output_path}")

        # Print pattern info
        console.print(f"\n[bold]Pattern:[/bold] {pattern_info.name}")
        console.print(f"[bold]Description:[/bold] {pattern_info.description}")
        console.print(f"[bold]Properties:[/bold] {', '.join(pattern_info.properties)}")

        return 0

    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print(
            f"[yellow]Available patterns:[/yellow] {', '.join(PatternLibrary.get_pattern_names())}"
        )
        return 1
    except Exception as e:  # noqa: BLE001 - CLI catch-all for user-friendly error display
        console.print(f"[red]Error:[/red] {e}")
        return 1


def compare_command(args: argparse.Namespace, console: Console) -> int:
    """
    Compare multiple patterns side-by-side.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    console : Console
        Rich console for formatted output

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    try:
        console.print(f"[cyan]Comparing {len(args.patterns)} patterns...[/cyan]")

        # Create config
        config = PatternConfig(args)

        # Generate all patterns
        patterns_data = []
        for pattern_name in args.patterns:
            try:
                pattern_info = PatternLibrary.get_pattern_info(pattern_name)
                sample_centers, _ = load_and_generate_pattern(pattern_info.spec, config)
                stats = compute_pattern_statistics(sample_centers, config.roi_diameter)
                patterns_data.append(
                    {
                        "name": pattern_info.name,
                        "sample_centers": sample_centers,
                        "stats": stats,
                    }
                )
            except KeyError:
                console.print(
                    f"[yellow]Warning:[/yellow] Pattern '{pattern_name}' not found, skipping"
                )
                continue

        if len(patterns_data) < 2:
            console.print("[red]Error:[/red] Need at least 2 valid patterns to compare")
            return 1

        # Create comparison visualization
        n_patterns = len(patterns_data)
        fig, axes = plt.subplots(2, n_patterns, figsize=(5 * n_patterns, 10))

        if n_patterns == 1:
            axes = axes.reshape(-1, 1)

        for i, pattern_data in enumerate(patterns_data):
            # Plot sample positions
            ax = axes[0, i]
            centers = pattern_data["sample_centers"]
            name = pattern_data["name"]
            plot_sample_positions(
                ax,
                centers,  # type: ignore[arg-type]
                config.roi_diameter,
                str(name),
            )

            # Plot coverage heatmap
            ax = axes[1, i]
            plot_coverage_heatmap(
                ax,
                centers,  # type: ignore[arg-type]
                config.roi_diameter,
                str(name),
            )

        plt.tight_layout()
        output_path = Path(args.output)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        console.print(f"[green]✓[/green] Comparison saved to: {output_path}")

        # Print comparison table
        table = Table(title="Pattern Comparison", show_header=True, header_style="bold cyan")
        table.add_column("Pattern", style="green")
        table.add_column("Samples", justify="right")
        table.add_column("Mean Radius", justify="right")
        table.add_column("Coverage %", justify="right")

        for pattern_data in patterns_data:
            pattern_stats: dict[str, Any] = pattern_data["stats"]  # type: ignore[assignment]
            table.add_row(
                str(pattern_data["name"]),
                str(pattern_stats["n_samples"]),
                f"{pattern_stats['radial_mean']:.1f}",
                f"{pattern_stats['coverage_percentage']:.1f}%",
            )

        console.print()
        console.print(table)

        return 0

    except Exception as e:  # noqa: BLE001 - CLI catch-all for user-friendly error display
        console.print(f"[red]Error:[/red] {e}")
        import traceback

        traceback.print_exc()
        return 1


def stats_command(args: argparse.Namespace, console: Console) -> int:
    """
    Show statistics for a pattern.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    console : Console
        Rich console for formatted output

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    try:
        # Get pattern info
        pattern_info = PatternLibrary.get_pattern_info(args.pattern)

        # Create config
        config = PatternConfig(args)

        # Generate pattern
        console.print(f"[cyan]Generating pattern:[/cyan] {pattern_info.name}")
        sample_centers, _ = load_and_generate_pattern(pattern_info.spec, config)

        # Compute statistics
        stats = compute_pattern_statistics(sample_centers, config.roi_diameter)

        # Display statistics
        console.print()
        console.print(f"[bold cyan]Pattern: {pattern_info.name}[/bold cyan]")
        console.print(f"[dim]{pattern_info.description}[/dim]")
        console.print()

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Number of Samples", str(stats["n_samples"]))
        table.add_row(
            "Sampling Type",
            "Line" if stats["is_line_sampling"] else "Point",
        )
        table.add_row("Mean Radius", f"{stats['radial_mean']:.2f} pixels")
        table.add_row("Radial Std Dev", f"{stats['radial_std']:.2f} pixels")
        table.add_row("Min Radius", f"{stats['radial_min']:.2f} pixels")
        table.add_row("Max Radius", f"{stats['radial_max']:.2f} pixels")
        table.add_row("Coverage", f"{stats['coverage_percentage']:.1f}%")
        table.add_row(
            "Inter-Sample Distance (mean)",
            f"{stats['inter_sample_dist_mean']:.2f} pixels",
        )
        table.add_row(
            "Inter-Sample Distance (std)",
            f"{stats['inter_sample_dist_std']:.2f} pixels",
        )

        console.print(table)

        return 0

    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print(
            f"[yellow]Available patterns:[/yellow] {', '.join(PatternLibrary.get_pattern_names())}"
        )
        return 1
    except Exception as e:  # noqa: BLE001 - CLI catch-all for user-friendly error display
        console.print(f"[red]Error:[/red] {e}")
        return 1


def gallery_command(args: argparse.Namespace, console: Console) -> int:
    """
    Generate HTML gallery of all patterns.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    console : Console
        Rich console for formatted output

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    try:
        console.print("[cyan]Generating pattern gallery...[/cyan]")

        # Create config
        config = PatternConfig(args)

        # Get all patterns
        patterns = PatternLibrary.list_patterns()

        # Generate visualizations
        pattern_images = []
        for pattern in patterns:
            console.print(f"  Processing: {pattern.name}")

            try:
                sample_centers, _ = load_and_generate_pattern(pattern.spec, config)
                stats = compute_pattern_statistics(sample_centers, config.roi_diameter)

                # Create compact visualization
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                plot_sample_positions(
                    ax,
                    sample_centers,
                    config.roi_diameter,
                    pattern.name,
                )

                # Convert to base64
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode()
                plt.close(fig)

                pattern_images.append(
                    {
                        "info": pattern,
                        "image": img_base64,
                        "stats": stats,
                    }
                )

            except Exception as e:  # noqa: BLE001 - Continue with other patterns on failure
                console.print(
                    f"    [yellow]Warning:[/yellow] Failed to generate {pattern.name}: {e}"
                )
                continue

        # Generate HTML
        html = generate_gallery_html(pattern_images)

        # Save
        output_path = Path(args.output)
        output_path.write_text(html)

        console.print(f"[green]✓[/green] Gallery saved to: {output_path}")
        console.print(f"[dim]Open in browser:[/dim] file://{output_path.absolute()}")

        return 0

    except Exception as e:  # noqa: BLE001 - CLI catch-all for user-friendly error display
        console.print(f"[red]Error:[/red] {e}")
        import traceback

        traceback.print_exc()
        return 1


def patterns_command(args: argparse.Namespace) -> int:
    """
    Execute patterns command based on subcommand.

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

    if args.subcommand is None:
        console.print("[red]Error:[/red] No subcommand specified")
        console.print("Use --help to see available subcommands")
        return 1

    if args.subcommand == "list":
        return list_command(args, console)
    elif args.subcommand == "show":
        return show_command(args, console)
    elif args.subcommand == "compare":
        return compare_command(args, console)
    elif args.subcommand == "stats":
        return stats_command(args, console)
    elif args.subcommand == "gallery":
        return gallery_command(args, console)
    else:
        console.print(f"[red]Error:[/red] Unknown subcommand: {args.subcommand}")
        return 1
