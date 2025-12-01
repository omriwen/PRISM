"""
Concentric circles pattern.

Samples along concentric circles with controllable density distribution.
"""

import numpy as np
import torch


def generate_pattern(config):
    """
    Concentric circles with center-weighted density.

    Creates multiple concentric circles with more samples in inner circles
    for better low-frequency coverage.
    """
    n_circles = 5

    # Denser sampling in center
    samples_per_circle = [30, 25, 20, 15, 10]
    assert sum(samples_per_circle) == config.n_samples, (
        f"samples_per_circle must sum to n_samples ({config.n_samples})"
    )

    r_max = config.roi_diameter / 2
    positions = []

    for i, n_samples in enumerate(samples_per_circle):
        # Radius for this circle
        r = r_max * (i + 1) / n_circles

        # Angular positions
        theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

        # Convert to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        positions.append(np.stack([x, y], axis=-1))

    # Concatenate all circles
    all_positions = np.concatenate(positions, axis=0)

    return torch.tensor(all_positions, dtype=torch.float32)[:, None, :]
