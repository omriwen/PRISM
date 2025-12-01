"""
Basic SPIDS usage example - Python API.

This example shows how to run a basic SPIDS reconstruction using the Python API
instead of the command-line interface.
"""

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

from prism.config.constants import um
from prism.config.objects import get_obj_params


# TODO: The aggregator classes (LossAgg, TelescopeAgg) don't exist in current codebase
# TODO: The PRISMTrainer API has changed - it now expects MeasurementSystem, not separate aggregators
# TODO: This example needs to be rewritten to use the current API (see prism.core.runner.PRISMRunner)
# from prism.core.aggregator import LossAgg, TelescopeAgg  # Module doesn't exist
# from prism.core.telescope import Telescope
# from prism.core.trainers import PRISMTrainer
# from prism.models.networks import ProgressiveDecoder
# from prism.utils.sampling import fermat_spiral_points


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Basic PRISM usage example")
    parser.add_argument("--quick", action="store_true", help="Fast mode for testing")
    parser.add_argument("--no-save", action="store_true", help="Skip saving outputs")
    return parser.parse_args()


def main(args=None):
    """Run basic SPIDS reconstruction."""
    if args is None:
        args = parse_args()

    # TODO: This example uses outdated API - needs complete rewrite
    logger.error("This example is not functional - API has changed")
    logger.info("Please use the CLI interface (main.py) or see prism.core.runner.PRISMRunner")
    logger.info("For working examples, see tests/integration/ directory")
    return None

    # Configuration
    obj_name = "europa"

    # Configure based on quick mode
    if args.quick:
        n_samples = 10
        max_epochs = 2
        image_size = 256
        logger.info("Running in QUICK mode for testing")
    else:
        n_samples = 50
        max_epochs = 10
        image_size = 512  # Smaller for faster demo

    logger.info(f"Starting PRISM reconstruction: {obj_name}")

    # 1. Get object parameters
    obj_params = get_obj_params(obj_name)
    obj_size = obj_params["size"]
    wavelength = obj_params["wavelength"]
    distance = obj_params["distance"]

    logger.info(f"Object size: {obj_size}px, wavelength: {wavelength / um:.2f}μm")

    # 2. Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 3. Create model
    model = ProgressiveDecoder(
        input_size=image_size,
        output_size=obj_size,
        latent_channels=512,
    ).to(device)

    logger.info(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # 4. Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 5. Create telescope simulator
    telescope = Telescope(
        diameter=10.0,  # 10m aperture
        wavelength=wavelength,
        distance=distance,
        image_size=image_size,
    )

    logger.info(f"Telescope: {telescope.diameter}m aperture")

    # 6. Create aggregators
    tel_agg = TelescopeAgg(image_size=image_size)
    loss_agg = LossAgg()

    # 7. Generate sampling points
    sample_points = fermat_spiral_points(n=n_samples, radius=image_size // 4)

    logger.info(f"Generated {len(sample_points)} Fermat spiral sample points")

    # 8. Create trainer
    trainer = PRISMTrainer(
        model=model,
        optimizer=optimizer,
        telescope=telescope,
        tel_agg=tel_agg,
        loss_agg=loss_agg,
        max_epochs=max_epochs,
        loss_threshold=0.001,
    )

    # 9. Run progressive training
    logger.info("Starting progressive training...")

    # Configure save directory based on --no-save flag
    save_dir = None if args.no_save else Path("runs/api_example_basic")

    results = trainer.train_progressive(
        sample_points=sample_points,
        aperture_diameter=64,
        save_dir=save_dir,
    )

    # 10. Report results
    logger.info("Training complete!")
    logger.info(f"Final coverage: {results['coverage']:.1f}%")
    logger.info(f"Average loss: {sum(results['losses']) / len(results['losses']):.4f}")
    logger.info(f"Failed samples: {len(results['failed_samples'])}/{results['n_samples']}")

    # 11. Access final reconstruction
    final_reconstruction = results["final_reconstruction"]
    logger.info(f"Reconstruction shape: {final_reconstruction.shape}")

    # Can now save, visualize, or further process the reconstruction
    if args.no_save:
        logger.info("Output saving skipped (--no-save flag)")
    else:
        logger.info("Results saved to: runs/api_example_basic/")

    return results


if __name__ == "__main__":
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    results = main()
    logger.success("Example completed successfully!")
