"""CLI command for launching SPIDS dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def create_dashboard_parser() -> argparse.ArgumentParser:
    """Create argument parser for dashboard command.

    Returns:
        Configured argument parser for dashboard
    """
    parser = argparse.ArgumentParser(
        description="PRISM Dashboard - Interactive web interface for experiment monitoring and comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="Directory containing experiment runs"
    )

    parser.add_argument(
        "--port", type=int, default=8050, help="Port number for the dashboard server"
    )

    parser.add_argument("--debug", action="store_true", help="Run in debug mode with auto-reload")

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")

    return parser


def dashboard_command(args: argparse.Namespace) -> int:
    """Execute dashboard command.

    Args:
        args: Parsed command-line arguments
    """
    try:
        from prism.web import run_dashboard

        logger.info("Starting PRISM Dashboard...")
        logger.info(f"  Runs directory: {args.runs_dir.absolute()}")
        logger.info(f"  Port: {args.port}")
        logger.info(f"  Host: {args.host}")
        logger.info(f"  Debug mode: {args.debug}")
        logger.info("")
        logger.info(f"Dashboard will be available at: http://localhost:{args.port}")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("")

        run_dashboard(runs_dir=args.runs_dir, port=args.port, debug=args.debug)

    except ImportError as e:
        logger.error("Dashboard dependencies not installed!")
        logger.error("Please install required packages: dash, dash-bootstrap-components")
        logger.error(f"Error: {e}")
        return 1

    except KeyboardInterrupt:
        logger.info("\nDashboard stopped by user")
        return 0

    except Exception as e:  # noqa: BLE001 - CLI catch-all for user-friendly error display
        logger.error(f"Error running dashboard: {e}")
        return 1

    return 0


def main() -> int:
    """Entry point for standalone dashboard script."""
    parser = create_dashboard_parser()
    args = parser.parse_args()
    return dashboard_command(args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
