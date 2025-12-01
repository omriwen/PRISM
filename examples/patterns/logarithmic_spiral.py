"""
Logarithmic spiral pattern.

Similar to Fermat but with exponential radial growth for different
k-space coverage characteristics.
"""

import numpy as np
import torch


def generate_pattern(config):
    """
    Logarithmic (equiangular) spiral.

    Unlike Archimedean spiral (linear r), this uses exponential radial
    growth: r = a * exp(b * theta)
    """
    n = config.n_samples
    r_max = config.roi_diameter / 2

    # Solve for b to reach r_max at desired angle
    n_turns = 5
    theta_max = n_turns * 2 * np.pi
    b = np.log(r_max / 1) / theta_max  # Start at r=1 to avoid log(0)
    a = 1

    # Generate spiral
    theta = np.linspace(0, theta_max, n)
    r = a * np.exp(b * theta)

    # Convert to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return torch.stack([torch.tensor(x), torch.tensor(y)], dim=-1)[:, None, :]
