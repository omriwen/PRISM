"""
Analysis module for SPIDS experiments.

This module provides tools for analyzing and comparing experimental results,
including metrics comparison, configuration diffing, and visualization.
"""

from __future__ import annotations

from prism.analysis.comparison import ExperimentComparator, ExperimentResult


__all__ = ["ExperimentComparator", "ExperimentResult"]
