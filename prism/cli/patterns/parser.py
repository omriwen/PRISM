"""Argument parser for patterns CLI command."""

from __future__ import annotations

import argparse


def create_patterns_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for patterns command.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with subcommands
    """
    parser = argparse.ArgumentParser(
        description="Browse and visualize SPIDS sampling patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # List subcommand
    list_parser = subparsers.add_parser(
        "list",
        help="List all available patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    list_parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed information including parameters and references",
    )
    list_parser.add_argument(
        "--recommended",
        "-r",
        action="store_true",
        help="Show only recommended patterns",
    )
    list_parser.add_argument(
        "--property",
        "-p",
        type=str,
        default=None,
        help="Filter by property (e.g., 'uniform', 'incoherent', 'radial')",
    )

    # Show subcommand
    show_parser = subparsers.add_parser(
        "show",
        help="Visualize a specific pattern",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    show_parser.add_argument("pattern", type=str, help="Pattern name (e.g., fermat, star, random)")
    show_parser.add_argument(
        "--n-samples", type=int, default=100, help="Number of samples to generate"
    )
    show_parser.add_argument(
        "--roi-diameter", type=float, default=512, help="ROI diameter in pixels"
    )
    show_parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output file path (PNG)"
    )
    show_parser.add_argument(
        "--sample-length", type=float, default=0, help="Line length (0 for point sampling)"
    )
    show_parser.add_argument(
        "--line-angle",
        type=float,
        default=None,
        help="Fixed line angle in degrees (None for random)",
    )

    # Compare subcommand
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple patterns side-by-side",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    compare_parser.add_argument(
        "patterns",
        type=str,
        nargs="+",
        help="Pattern names to compare (e.g., fermat random star)",
    )
    compare_parser.add_argument(
        "--n-samples", type=int, default=100, help="Number of samples for each pattern"
    )
    compare_parser.add_argument(
        "--roi-diameter", type=float, default=512, help="ROI diameter in pixels"
    )
    compare_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="pattern_comparison.png",
        help="Output file path (PNG)",
    )
    compare_parser.add_argument(
        "--sample-length", type=float, default=0, help="Line length (0 for point sampling)"
    )

    # Stats subcommand
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show statistics for a pattern",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    stats_parser.add_argument("pattern", type=str, help="Pattern name (e.g., fermat, star, random)")
    stats_parser.add_argument(
        "--n-samples", type=int, default=100, help="Number of samples to generate"
    )
    stats_parser.add_argument(
        "--roi-diameter", type=float, default=512, help="ROI diameter in pixels"
    )
    stats_parser.add_argument(
        "--sample-length", type=float, default=0, help="Line length (0 for point sampling)"
    )

    # Gallery subcommand
    gallery_parser = subparsers.add_parser(
        "gallery",
        help="Generate HTML gallery of all patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    gallery_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="pattern_gallery.html",
        help="Output HTML file path",
    )
    gallery_parser.add_argument(
        "--n-samples", type=int, default=100, help="Number of samples per pattern"
    )
    gallery_parser.add_argument(
        "--roi-diameter", type=float, default=512, help="ROI diameter in pixels"
    )

    return parser
