"""
Builtin pattern functions.

Wraps existing pattern generators in the pattern function interface.
"""

from __future__ import annotations

from typing import Any

import torch

from prism.core import patterns


def fermat_builtin(config: Any) -> torch.Tensor:
    """
    Fermat spiral pattern (golden angle spiral).

    Provides optimal k-space coverage with logarithmic spiral sampling.
    This is the recommended pattern for most SPIDS applications.

    Uses parameters from config:
    - n_samples: Number of sampling positions
    - roi_diameter: K-space region diameter
    - sample_length: Line length (0 for point sampling)
    - samples_r_cutoff: Maximum radius for sample centers
    - line_angle: Fixed line angle (None for random)
    """
    return patterns.generate_fermat_spiral(
        n_points=config.n_samples,
        r_max=config.roi_diameter / 2,
        length=config.sample_length,
        r_cutoff=getattr(config, "samples_r_cutoff", None),
        line_angle=config.line_angle,
    )


def star_builtin(config: Any) -> torch.Tensor:
    """
    Star pattern (radial lines from center).

    Generates evenly-spaced radial lines, useful for testing
    rotational symmetry and basic reconstruction capabilities.

    Uses parameters from config:
    - n_angs: Number of radial angles
    - sample_length: Line length
    - roi_diameter: K-space region diameter
    """
    return patterns.generate_star_pattern(
        n_angles=config.n_angs,
        length=config.sample_length,
        size=config.roi_diameter,
        shape="circle",
    )


def random_builtin(config: Any) -> torch.Tensor:
    """
    Random uniform sampling pattern.

    Samples positions uniformly at random within k-space region.
    Generally less optimal than Fermat spiral but useful for comparison.

    Uses parameters from config:
    - n_samples: Number of sampling positions
    - sample_length: Line length (0 for point sampling)
    - roi_diameter: K-space region diameter
    """
    return patterns.generate_samples(
        n_points=config.n_samples,
        length=config.sample_length,
        size=config.roi_diameter,
        shape="circle",
    )
