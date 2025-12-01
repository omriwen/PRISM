"""
Custom sampling patterns example.

Shows how to define and use custom pattern functions for telescope sampling.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# TODO: The aggregator classes (LossAgg, TelescopeAgg) don't exist in current codebase
# TODO: The PRISMTrainer API has changed - it now expects MeasurementSystem, not separate aggregators
# TODO: This example needs to be rewritten to use the current API (see prism.core.runner.PRISMRunner)
# from prism.core.aggregator import LossAgg, TelescopeAgg  # Module doesn't exist
# from prism.core.telescope import Telescope
# from prism.core.trainers import PRISMTrainer
# from prism.models.networks import ProgressiveDecoder


def circular_pattern(n: int, radius: float) -> np.ndarray:
    """
    Generate points in concentric circles.

    Parameters
    ----------
    n : int
        Number of points
    radius : float
        Maximum radius

    Returns
    -------
    np.ndarray
        Points array of shape (n, 2)
    """
    points = []

    # Distribute points across 5 concentric circles
    n_circles = 5
    points_per_circle = n // n_circles

    for i in range(n_circles):
        circle_radius = radius * (i + 1) / n_circles
        n_points = points_per_circle if i < n_circles - 1 else (n - len(points))

        # Generate evenly spaced points on circle
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        for angle in angles:
            x = circle_radius * np.cos(angle)
            y = circle_radius * np.sin(angle)
            points.append([x, y])

    return np.array(points)


def grid_pattern(n: int, radius: float) -> np.ndarray:
    """
    Generate points in a square grid.

    Parameters
    ----------
    n : int
        Number of points (will be rounded to perfect square)
    radius : float
        Half-width of grid

    Returns
    -------
    np.ndarray
        Points array of shape (n, 2)
    """
    # Find nearest perfect square
    n_side = int(np.sqrt(n))

    # Generate grid
    x = np.linspace(-radius, radius, n_side)
    y = np.linspace(-radius, radius, n_side)

    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Trim to exact n
    return points[:n]


def random_clustered_pattern(n: int, radius: float, n_clusters: int = 5) -> np.ndarray:
    """
    Generate random points clustered around random centers.

    Parameters
    ----------
    n : int
        Number of points
    radius : float
        Maximum radius
    n_clusters : int, default=5
        Number of clusters

    Returns
    -------
    np.ndarray
        Points array of shape (n, 2)
    """
    points = []
    points_per_cluster = n // n_clusters

    # Generate cluster centers
    cluster_centers = np.random.uniform(-radius * 0.7, radius * 0.7, (n_clusters, 2))

    for i, center in enumerate(cluster_centers):
        # Number of points in this cluster
        n_points = points_per_cluster if i < n_clusters - 1 else (n - len(points))

        # Generate points around cluster center
        cluster_radius = radius * 0.2  # Cluster spread
        cluster_points = np.random.normal(0, cluster_radius, (n_points, 2)) + center

        points.extend(cluster_points)

    return np.array(points)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare different sampling patterns for telescope imaging"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick mode with reduced parameters for faster testing",
    )
    return parser.parse_args()


def main():
    """Compare different sampling patterns."""
    args = parse_args()

    # TODO: This example uses outdated API - needs complete rewrite
    logger.error("This example is not functional - API has changed")
    logger.info("Please use the CLI interface (main.py) with --pattern argument")
    logger.info("For pattern examples, see prism/core/patterns.py and examples/patterns/")
    return

    patterns = {
        "circular": circular_pattern,
        "grid": grid_pattern,
        "clustered": random_clustered_pattern,
    }

    # Configuration - adjust based on quick mode
    if args.quick:
        image_size = 128
        obj_size = 64
        n_samples = 16
        max_epochs = 2
        logger.info("Running in QUICK mode with reduced parameters")
    else:
        image_size = 512
        obj_size = 128
        n_samples = 50
        max_epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for pattern_name, pattern_func in patterns.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing pattern: {pattern_name}")
        logger.info(f"{'=' * 60}\n")

        # Generate sample points
        sample_points = pattern_func(n=n_samples, radius=image_size // 4)
        logger.info(f"Generated {len(sample_points)} points")

        # Set up model and trainer (simplified)
        model = ProgressiveDecoder(input_size=image_size, output_size=obj_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        telescope = Telescope(diameter=10.0, wavelength=500e-9, distance=6e8, image_size=image_size)

        tel_agg = TelescopeAgg(image_size=image_size)
        loss_agg = LossAgg()

        trainer = PRISMTrainer(
            model=model,
            optimizer=optimizer,
            telescope=telescope,
            tel_agg=tel_agg,
            loss_agg=loss_agg,
            max_epochs=max_epochs,
        )

        # Train
        results = trainer.train_progressive(
            sample_points=sample_points,
            aperture_diameter=64,
            save_dir=Path(f"runs/pattern_comparison_{pattern_name}"),
        )

        # Report
        logger.info(f"Coverage: {results['coverage']:.1f}%")
        logger.info(f"Avg loss: {sum(results['losses']) / len(results['losses']):.4f}")

    logger.success("Pattern comparison complete!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    main()
