"""
Batch experiments example.

Shows how to run multiple experiments programmatically with
different configurations.
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from loguru import logger

from prism.config.objects import get_object_params
from prism.core.aggregator import LossAgg, TelescopeAgg
from prism.core.telescope import Telescope
from prism.core.trainers import PRISMTrainer
from prism.models.networks import ProgressiveDecoder
from prism.utils.sampling import fermat_spiral_points


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    obj_name: str
    n_samples: int
    max_epochs: int
    learning_rate: float
    aperture_diameter: int


def run_experiment(config: ExperimentConfig) -> dict:
    """
    Run a single experiment with given configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration

    Returns
    -------
    dict
        Results including losses, coverage, etc.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running experiment: {config.name}")
    logger.info(f"{'=' * 60}\n")
    logger.info(f"Config: {asdict(config)}")

    # Setup
    image_size = 512
    obj_params = get_object_params(config.obj_name)
    obj_size = obj_params["size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = ProgressiveDecoder(input_size=image_size, output_size=obj_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Telescope
    telescope = Telescope(
        diameter=10.0,
        wavelength=obj_params["wavelength"],
        distance=obj_params["distance"],
        image_size=image_size,
    )

    # Aggregators
    tel_agg = TelescopeAgg(image_size=image_size)
    loss_agg = LossAgg()

    # Sampling
    sample_points = fermat_spiral_points(n=config.n_samples, radius=image_size // 4)

    # Trainer
    trainer = PRISMTrainer(
        model=model,
        optimizer=optimizer,
        telescope=telescope,
        tel_agg=tel_agg,
        loss_agg=loss_agg,
        max_epochs=config.max_epochs,
    )

    # Train
    results = trainer.train_progressive(
        sample_points=sample_points,
        aperture_diameter=config.aperture_diameter,
        save_dir=Path(f"runs/batch_{config.name}"),
    )

    # Add config to results
    results["config"] = asdict(config)

    # Log summary
    logger.info(f"\nExperiment {config.name} complete:")
    logger.info(f"  Coverage: {results['coverage']:.1f}%")
    logger.info(f"  Avg loss: {sum(results['losses']) / len(results['losses']):.4f}")
    logger.info(f"  Failed: {len(results['failed_samples'])}/{results['n_samples']}")

    return results


def main():
    """Run batch of experiments."""

    # Define experiments
    experiments = [
        ExperimentConfig(
            name="baseline",
            obj_name="europa",
            n_samples=50,
            max_epochs=10,
            learning_rate=0.001,
            aperture_diameter=64,
        ),
        ExperimentConfig(
            name="more_samples",
            obj_name="europa",
            n_samples=100,
            max_epochs=10,
            learning_rate=0.001,
            aperture_diameter=64,
        ),
        ExperimentConfig(
            name="higher_lr",
            obj_name="europa",
            n_samples=50,
            max_epochs=10,
            learning_rate=0.01,  # 10x higher
            aperture_diameter=64,
        ),
        ExperimentConfig(
            name="larger_aperture",
            obj_name="europa",
            n_samples=50,
            max_epochs=10,
            learning_rate=0.001,
            aperture_diameter=128,  # 2x larger
        ),
    ]

    # Run all experiments
    all_results = []

    for config in experiments:
        try:
            results = run_experiment(config)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Experiment {config.name} failed: {e}")
            continue

    # Save comparison
    comparison_dir = Path("runs/batch_comparison")
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary = {
        exp["config"]["name"]: {
            "coverage": exp["coverage"],
            "avg_loss": sum(exp["losses"]) / len(exp["losses"]),
            "failed_samples": len(exp["failed_samples"]),
            "config": exp["config"],
        }
        for exp in all_results
    }

    with open(comparison_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.success(f"\nBatch experiments complete! Summary saved to {comparison_dir}/summary.json")

    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)

    for name, data in summary.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Coverage: {data['coverage']:.1f}%")
        logger.info(f"  Avg Loss: {data['avg_loss']:.4f}")
        logger.info(f"  Failed: {data['failed_samples']}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    main()
