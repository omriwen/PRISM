"""PRISM: Progressive Reconstruction from Imaging with Sparse Measurements"""

from __future__ import annotations


__version__ = "0.3.0"

# Import submodules for easier access
from prism import config, core, models, utils
from prism.utils.logging_config import setup_logging


__all__ = [
    "config",
    "core",
    "models",
    "utils",
    "setup_logging",
    "__version__",
]
