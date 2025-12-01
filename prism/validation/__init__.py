"""
Validation module for SPIDS.

This module provides theoretical baselines and validation metrics for
verifying SPIDS optical physics correctness.

Submodules
----------
baselines : Theoretical baseline calculations
    - Resolution limits (Abbe, Rayleigh)
    - Diffraction patterns (Airy disk, sinc)
    - Ground sampling distance (GSD)
"""

from __future__ import annotations

from prism.validation.baselines import (
    DiffractionPatterns,
    FresnelBaseline,
    GSDBaseline,
    ResolutionBaseline,
)


__all__ = [
    "ResolutionBaseline",
    "DiffractionPatterns",
    "GSDBaseline",
    "FresnelBaseline",
]
