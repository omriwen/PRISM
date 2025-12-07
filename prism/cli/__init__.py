"""Command-line interface utilities for SPIDS."""

from __future__ import annotations

from prism.cli.animate import animate_command, create_animate_parser
from prism.cli.dashboard import create_dashboard_parser, dashboard_command
from prism.cli.entry_points import (
    create_unified_parser,
    main,
    main_mopie,
    main_prism,
)
from prism.cli.parser import create_main_parser, create_mopie_parser
from prism.cli.report import add_report_parser, report_command


__all__ = [
    # Parsers
    "create_main_parser",
    "create_mopie_parser",
    "create_unified_parser",
    # Unified entry points
    "main",
    "main_prism",
    "main_mopie",
    # Dashboard
    "create_dashboard_parser",
    "dashboard_command",
    # Animate
    "create_animate_parser",
    "animate_command",
    # Report
    "add_report_parser",
    "report_command",
    # Profile (lazy loaded)
    "create_profile_parser",
    "profile_command",
]


def __getattr__(name: str):
    """Lazy import for profile module to avoid sys.modules conflict."""
    if name in ("create_profile_parser", "profile_command"):
        from prism.cli.profile import create_profile_parser, profile_command
        globals()[name] = locals()[name]  # Cache for subsequent access
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
