"""
CLI command for inspecting SPIDS experiment checkpoints.

Usage:
    python -m spids.cli.inspect_pkg runs/experiment/checkpoint.pt [options]
"""

from __future__ import annotations

import sys

from prism.cli.inspect_pkg.commands import (
    create_inspect_parser,
    inspect_command,
    main,
)
from prism.cli.inspect_pkg.inspector import CheckpointInspector


__all__ = [
    "CheckpointInspector",
    "create_inspect_parser",
    "inspect_command",
    "main",
]

if __name__ == "__main__":
    sys.exit(main())
