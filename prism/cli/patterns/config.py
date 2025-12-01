"""Configuration class for pattern generation."""

from __future__ import annotations

import argparse


class PatternConfig:
    """Minimal configuration for pattern generation."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.n_samples = args.n_samples
        self.n_angs = getattr(args, "n_angs", 4)
        self.roi_diameter = args.roi_diameter
        self.sample_diameter = getattr(args, "sample_diameter", 32)
        self.sample_length = getattr(args, "sample_length", 0)
        self.line_angle = getattr(args, "line_angle", None)
        self.samples_r_cutoff = getattr(args, "samples_r_cutoff", None)
        self.obj_size = args.roi_diameter
