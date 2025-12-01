"""
Continuous Archimedean spiral pattern.

Creates a smooth spiral with adjustable number of turns and density.
Good for testing resolution vs. sampling density tradeoffs.
"""

import numpy as np
import torch


def generate_pattern(config):
    """
    Continuous Archimedean spiral with uniform spacing.

    The spiral maintains constant angular velocity while radius increases
    linearly from center to edge of ROI.
    """
    n = config.n_samples
    n_turns = 8.5  # Number of complete rotations

    # Angular progression
    theta = np.linspace(0, n_turns * 2 * np.pi, n)

    # Linear radial progression
    r_max = config.roi_diameter / 2
    r = np.linspace(0, r_max, n)

    # Convert to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return torch.stack([torch.tensor(x), torch.tensor(y)], dim=-1)[:, None, :]
