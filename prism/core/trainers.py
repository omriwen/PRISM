"""
DEPRECATED: Import from prism.core.trainers instead.

This module is maintained for backward compatibility.
"""

import warnings

warnings.warn(
    "Importing from prism.core.trainers (module file) is deprecated. "
    "Use 'from prism.core.trainers import PRISMTrainer, create_scheduler' instead.",
    DeprecationWarning,
    stacklevel=2
)

from prism.core.trainers.progressive import PRISMTrainer, create_scheduler

__all__ = ["PRISMTrainer", "create_scheduler"]
