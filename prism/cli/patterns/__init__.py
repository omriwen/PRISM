"""
CLI command for browsing and visualizing SPIDS sampling patterns.

Usage:
    python -m spids.cli.patterns list
    python -m spids.cli.patterns show fermat --n-samples 100 --output fermat.png
    python -m spids.cli.patterns compare fermat random star --n-samples 100
    python -m spids.cli.patterns stats fermat --n-samples 100
    python -m spids.cli.patterns gallery --output pattern_gallery.html
"""

from __future__ import annotations

import sys

import matplotlib

from prism.cli.patterns.commands import (
    compare_command,
    gallery_command,
    list_command,
    patterns_command,
    show_command,
    stats_command,
)
from prism.cli.patterns.config import PatternConfig
from prism.cli.patterns.gallery import generate_gallery_html
from prism.cli.patterns.parser import create_patterns_parser
from prism.cli.patterns.visualizers import plot_coverage_heatmap, plot_sample_positions


matplotlib.use("Agg")  # Non-interactive backend

__all__ = [
    # Config
    "PatternConfig",
    # Parser
    "create_patterns_parser",
    # Commands
    "list_command",
    "show_command",
    "compare_command",
    "stats_command",
    "gallery_command",
    "patterns_command",
    # Visualization helpers
    "plot_sample_positions",
    "plot_coverage_heatmap",
    # Gallery generation
    "generate_gallery_html",
    # Main entry point
    "main",
]


def main() -> int:
    """Main entry point for patterns CLI."""
    parser = create_patterns_parser()
    args = parser.parse_args()
    return patterns_command(args)


if __name__ == "__main__":
    sys.exit(main())
