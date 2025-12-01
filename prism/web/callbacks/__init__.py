"""Callback handlers for SPIDS dashboard."""

from __future__ import annotations

from .comparison import register_comparison_callbacks
from .profiling import register_profiling_callbacks
from .realtime import register_realtime_callbacks
from .training import register_callbacks


__all__ = [
    "register_callbacks",
    "register_realtime_callbacks",
    "register_comparison_callbacks",
    "register_profiling_callbacks",
]
