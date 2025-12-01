"""
Jittered grid pattern.

Regular grid with random perturbations to avoid aliasing artifacts.
"""

import numpy as np
import torch


def generate_pattern(config):
    """
    Jittered grid with Gaussian perturbations.

    Creates a regular grid then adds random jitter to each point
    to break up regular sampling artifacts while maintaining coverage.
    """
    n = config.n_samples

    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(n)))

    # Create regular grid
    r_max = config.roi_diameter / 2
    coords_1d = np.linspace(-r_max, r_max, grid_size)
    xx, yy = np.meshgrid(coords_1d, coords_1d)

    # Flatten and clip to n_samples
    x_grid = xx.flatten()[:n]
    y_grid = yy.flatten()[:n]

    # Add jitter (10% of grid spacing)
    jitter_sigma = (coords_1d[1] - coords_1d[0]) * 0.1
    x_jitter = np.random.randn(n) * jitter_sigma
    y_jitter = np.random.randn(n) * jitter_sigma

    x = x_grid + x_jitter
    y = y_grid + y_jitter

    # Clip to ROI
    r = np.sqrt(x**2 + y**2)
    valid = r <= r_max
    x = x[valid]
    y = y[valid]

    # Pad if needed (lost points outside circle)
    if len(x) < n:
        # Add random points to fill
        n_missing = n - len(x)
        theta = np.random.rand(n_missing) * 2 * np.pi
        r = np.sqrt(np.random.rand(n_missing)) * r_max
        x = np.concatenate([x, r * np.cos(theta)])
        y = np.concatenate([y, r * np.sin(theta)])

    return torch.stack([torch.tensor(x[:n]), torch.tensor(y[:n])], dim=-1)[:, None, :]
