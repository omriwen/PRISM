"""Entry point for python -m spids.cli.patterns."""

from __future__ import annotations

import sys

from prism.cli.patterns import main


if __name__ == "__main__":
    sys.exit(main())
