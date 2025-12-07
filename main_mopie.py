"""
DEPRECATED: Use `python main.py --algorithm mopie` instead.

This file is deprecated and will be removed in a future version.
Please update your scripts to use the unified entry point.

Migration Guide
---------------
Old command:
    python main_mopie.py --obj_name europa --n_epochs 100

New command:
    python main.py --algorithm mopie --obj_name europa --n_epochs 100

The unified entry point provides the same functionality with additional
features like preset loading, scenario support, and AI configuration.
"""

import sys
import warnings


# Emit deprecation warning
warnings.warn(
    "main_mopie.py is deprecated. Use 'python main.py --algorithm mopie' instead. "
    "This file will be removed in version 2.0.",
    DeprecationWarning,
    stacklevel=2,
)


def main() -> None:
    """Deprecated entry point for Mo-PIE algorithm.

    This function forwards to the unified entry point with --algorithm=mopie.
    """
    # Insert --algorithm=mopie as the first argument
    sys.argv.insert(1, "--algorithm=mopie")

    # Import and call the unified entry point
    from prism.cli.entry_points import main as unified_main

    unified_main()


if __name__ == "__main__":
    main()
