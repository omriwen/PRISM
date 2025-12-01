"""CLI command for profiling analysis and visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def create_profile_parser() -> argparse.ArgumentParser:
    """Create argument parser for profile commands.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for profile commands
    """
    parser = argparse.ArgumentParser(
        prog="prism profile",
        description="Analyze and visualize training performance profiles",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a profile and print summary",
    )
    analyze_parser.add_argument("profile", type=Path, help="Path to profile file")
    analyze_parser.add_argument(
        "--output", "-o", type=Path, help="Output report path"
    )
    analyze_parser.add_argument(
        "--format",
        choices=["txt", "html", "json"],
        default="txt",
        help="Output format",
    )

    # view subcommand
    view_parser = subparsers.add_parser(
        "view",
        help="Launch interactive profile viewer",
    )
    view_parser.add_argument("profile", type=Path, help="Path to profile file")
    view_parser.add_argument(
        "--port", type=int, default=8051, help="Dashboard port"
    )

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two profiles",
    )
    compare_parser.add_argument(
        "profiles", type=Path, nargs=2, help="Profile files to compare"
    )
    compare_parser.add_argument(
        "--output", "-o", type=Path, help="Output comparison report"
    )

    # export subcommand
    export_parser = subparsers.add_parser(
        "export",
        help="Export profile to different format",
    )
    export_parser.add_argument("profile", type=Path, help="Path to profile file")
    export_parser.add_argument(
        "--format",
        choices=["chrome-trace", "csv", "tensorboard"],
        required=True,
        help="Export format",
    )
    export_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output path"
    )

    return parser


def profile_command(args: argparse.Namespace) -> int:
    """Execute profile command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    from prism.profiling.analyzer import ProfileAnalyzer
    from prism.profiling.storage import export_chrome_trace, load_profile

    try:
        if args.subcommand == "analyze":
            analyzer = ProfileAnalyzer(args.profile)
            report = analyzer.get_efficiency_report()

            if args.output:
                args.output.write_text(report)
                logger.info(f"Report saved to {args.output}")
            else:
                print(report)

        elif args.subcommand == "view":
            # Launch interactive viewer
            from prism.profiling.visualization.interactive import launch_viewer
            launch_viewer(args.profile, port=args.port)

        elif args.subcommand == "compare":
            # Compare two profiles
            analyzer1 = ProfileAnalyzer(args.profiles[0])
            analyzer2 = ProfileAnalyzer(args.profiles[1])

            summary1 = analyzer1.get_summary()
            summary2 = analyzer2.get_summary()

            print("Profile Comparison")
            print("=" * 60)
            print(f"{'Metric':<25} {'Profile 1':>15} {'Profile 2':>15}")
            print("-" * 60)
            for key in summary1:
                v1 = summary1[key]
                v2 = summary2[key]
                if isinstance(v1, float):
                    print(f"{key:<25} {v1:>15.2f} {v2:>15.2f}")
                else:
                    print(f"{key:<25} {str(v1):>15} {str(v2):>15}")

            if args.output:
                # Save comparison report to file
                comparison_text = [
                    "Profile Comparison",
                    "=" * 60,
                    f"{'Metric':<25} {'Profile 1':>15} {'Profile 2':>15}",
                    "-" * 60,
                ]
                for key in summary1:
                    v1 = summary1[key]
                    v2 = summary2[key]
                    if isinstance(v1, float):
                        comparison_text.append(f"{key:<25} {v1:>15.2f} {v2:>15.2f}")
                    else:
                        comparison_text.append(f"{key:<25} {str(v1):>15} {str(v2):>15}")

                args.output.write_text("\n".join(comparison_text))
                logger.info(f"Comparison saved to {args.output}")

        elif args.subcommand == "export":
            data = load_profile(args.profile)

            if args.format == "chrome-trace":
                export_chrome_trace(data, args.output)
                logger.info(f"Chrome trace exported to {args.output}")
            else:
                logger.warning(f"Export format '{args.format}' not yet implemented")
                return 1

        return 0

    except FileNotFoundError as e:
        logger.error(f"Profile file not found: {e}")
        return 1

    except ImportError as e:
        logger.error(f"Required dependencies not installed: {e}")
        logger.error("Please ensure profiling dependencies are installed")
        return 1

    except Exception as e:  # noqa: BLE001 - CLI catch-all for user-friendly error display
        logger.error(f"Error executing profile command: {e}")
        logger.exception(e)
        return 1


def main() -> int:
    """Entry point for standalone profile script."""
    parser = create_profile_parser()
    args = parser.parse_args()
    return profile_command(args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
