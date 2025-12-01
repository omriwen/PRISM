"""Entry point for python -m spids.cli.inspect_pkg."""

from __future__ import annotations

import sys

from prism.cli.inspect_pkg import main


if __name__ == "__main__":
    sys.exit(main())
