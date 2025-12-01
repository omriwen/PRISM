"""
Custom loss functions example.

Shows how to use different loss functions for SPIDS training, including:
- Built-in losses (L1, L2, SSIM, MS-SSIM)
- Custom perceptual loss
- Combined loss functions

SSIM Loss Functions:
-------------------
SPIDS supports Structural Similarity Index (SSIM) as a loss function:
- SSIM: Single-scale structural similarity (fast, perceptual quality)
- MS-SSIM: Multi-scale SSIM (slower, better perceptual quality)

Both SSIM losses:
- Operate in measurement space (same as L1/L2)
- Use DSSIM formulation: (1 - SSIM) / 2, range [0, 0.5]
- Are fully differentiable for gradient-based optimization
- Match scikit-image SSIM for consistency with evaluation metrics

Limitations:
-----------
- MS-SSIM requires minimum image size of 176×176 for 5 scales
- SSIM is ~5-10× slower than L1 loss
- MS-SSIM is ~20-30× slower than L1 loss
- Gradient magnitude can vary near flat regions
"""

import sys

import torch
import torch.nn as nn
from loguru import logger

from prism.config.base import PRISMConfig
from prism.core.aggregator import TelescopeAgg
from prism.models.losses import LossAgg
from prism.models.networks import ProgressiveDecoder


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.

    Useful for maintaining high-frequency details in reconstruction.
    """

    def __init__(self):
        super().__init__()
        # Simplified - in practice, use pretrained VGG
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )

        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        # MSE in feature space
        return nn.functional.mse_loss(pred_features, target_features)


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss terms.

    Combines measurement fidelity, perceptual loss, and regularization.
    """

    def __init__(
        self,
        measurement_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        tv_weight: float = 0.01,
    ):
        super().__init__()
        self.measurement_weight = measurement_weight
        self.perceptual_weight = perceptual_weight
        self.tv_weight = tv_weight

        self.perceptual_loss = PerceptualLoss()

    def total_variation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation for smoothness."""
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()

    def forward(
        self,
        reconstruction: torch.Tensor,
        ground_truth: torch.Tensor,
        measurement_pred: torch.Tensor,
        measurement_true: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with 'total' and individual loss components
        """
        # Measurement fidelity
        meas_loss = nn.functional.mse_loss(measurement_pred.abs(), measurement_true.abs())

        # Perceptual loss (if ground truth available)
        if ground_truth is not None:
            perc_loss = self.perceptual_loss(reconstruction, ground_truth)
        else:
            perc_loss = torch.tensor(0.0, device=reconstruction.device)

        # Total variation regularization
        tv_loss = self.total_variation(reconstruction)

        # Combine
        total_loss = (
            self.measurement_weight * meas_loss
            + self.perceptual_weight * perc_loss
            + self.tv_weight * tv_loss
        )

        return {
            "total": total_loss,
            "measurement": meas_loss,
            "perceptual": perc_loss,
            "tv": tv_loss,
        }


def train_with_custom_loss():
    """Train SPIDS with custom loss function."""

    logger.info("Training with custom combined loss")

    # Setup
    image_size = 256
    obj_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = ProgressiveDecoder(obj_size=obj_size, image_size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Custom loss
    criterion = CombinedLoss(
        measurement_weight=1.0,
        perceptual_weight=0.1,
        tv_weight=0.01,
    ).to(device)

    # Dummy training loop (simplified)
    for epoch in range(10):
        optimizer.zero_grad()

        # Forward pass
        reconstruction = model()

        # Dummy measurements and ground truth
        measurement_pred = torch.randn(64, 64, dtype=torch.complex64, device=device)
        measurement_true = torch.randn(64, 64, dtype=torch.complex64, device=device)
        ground_truth = torch.randn_like(reconstruction)

        # Compute loss
        loss_dict = criterion(
            reconstruction=reconstruction,
            ground_truth=ground_truth,
            measurement_pred=measurement_pred,
            measurement_true=measurement_true,
        )

        # Backward
        loss_dict["total"].backward()
        optimizer.step()

        # Log
        if epoch % 2 == 0:
            logger.info(
                f"Epoch {epoch}: "
                f"Total={loss_dict['total'].item():.4f}, "
                f"Meas={loss_dict['measurement'].item():.4f}, "
                f"Perc={loss_dict['perceptual'].item():.4f}, "
                f"TV={loss_dict['tv'].item():.4f}"
            )

    logger.success("Training with custom loss complete!")


def compare_loss_functions():
    """
    Compare different loss functions: L1, L2, SSIM, MS-SSIM.

    This example demonstrates how to use SPIDS built-in loss functions
    and shows the performance characteristics of each.
    """
    logger.info("Comparing different loss functions")

    # Setup
    image_size = 256
    obj_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create telescope for measurement generation
    telescope = TelescopeAgg(n=image_size, r=50, device=device)

    # Model
    model = ProgressiveDecoder(obj_size=obj_size, image_size=image_size).to(device)

    # Test different loss types
    loss_types = ["l1", "l2", "ssim", "ms-ssim"]

    for loss_type in loss_types:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing {loss_type.upper()} loss")
        logger.info(f"{'=' * 60}")

        # Create loss function
        criterion = LossAgg(loss_type=loss_type).to(device)

        # Create dummy reconstruction and target
        with torch.no_grad():
            reconstruction = model()

        # Dummy target measurements [2, C, H, W]
        target = torch.rand(2, 2, image_size, image_size, device=device)

        # Dummy center for measurement
        center = [image_size // 2, image_size // 2]

        # Compute loss
        import time

        start = time.time()
        loss_old, loss_new = criterion(reconstruction, target, telescope, center)
        elapsed = time.time() - start

        logger.info(f"Loss old: {loss_old.item():.6f}")
        logger.info(f"Loss new: {loss_new.item():.6f}")
        logger.info(f"Total loss: {(loss_old + loss_new).item():.6f}")
        logger.info(f"Computation time: {elapsed * 1000:.2f} ms")

        # Expected ranges
        if loss_type in ["l1", "l2"]:
            logger.info("Range: [0, ∞) (normalized by zero-loss)")
        elif loss_type in ["ssim", "ms-ssim"]:
            logger.info("Range: [0, 0.5] (DSSIM formulation)")


def train_with_ssim_loss():
    """
    Example: Training SPIDS with SSIM loss.

    Shows how to use SSIM as a loss function in a progressive training loop.
    """
    logger.info("Training with SSIM loss")

    # Setup
    image_size = 256
    obj_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and optimizer
    model = ProgressiveDecoder(obj_size=obj_size, image_size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # SSIM loss
    criterion = LossAgg(loss_type="ssim").to(device)

    # Create telescope for measurement generation
    telescope = TelescopeAgg(n=image_size, r=50, device=device)

    # Convergence thresholds
    threshold_old = 0.1  # DSSIM threshold for accumulated measurements
    threshold_new = 0.1  # DSSIM threshold for new measurement

    # Dummy progressive training loop
    sample_centers = [[64, 64], [128, 128], [192, 192]]

    for i, center in enumerate(sample_centers):
        logger.info(f"\nSample {i + 1}/{len(sample_centers)} at center {center}")

        # Generate target measurement
        with torch.no_grad():
            # In real training, this would be actual measured data
            target = torch.rand(2, 2, image_size, image_size, device=device)

        # Progressive training for this sample
        for epoch in range(10):
            optimizer.zero_grad()

            # Forward pass (decoder-only model)
            reconstruction = model()

            # Compute dual SSIM loss
            loss_old, loss_new = criterion(reconstruction, target, telescope, center)
            total_loss = loss_old + loss_new

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Log progress
            if epoch % 3 == 0:
                logger.info(
                    f"Epoch {epoch}: "
                    f"Loss old={loss_old.item():.4f}, "
                    f"Loss new={loss_new.item():.4f}, "
                    f"Total={total_loss.item():.4f}"
                )

            # Check convergence
            if loss_old < threshold_old and loss_new < threshold_new:
                logger.success(f"Converged after {epoch + 1} epochs!")
                break

        # Add measurement to accumulator (in real training)
        # telescope.add_measurement(center)

    logger.success("SSIM training complete!")


def train_with_ms_ssim_loss():
    """
    Example: Training SPIDS with Multi-Scale SSIM loss.

    MS-SSIM provides better perceptual quality but is slower than single-scale SSIM.
    """
    logger.info("Training with MS-SSIM loss")

    # Setup
    image_size = 256  # MS-SSIM needs larger images (min 176×176 for 5 scales)
    obj_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and optimizer
    model = ProgressiveDecoder(obj_size=obj_size, image_size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # MS-SSIM loss
    criterion = LossAgg(loss_type="ms-ssim").to(device)

    # Create telescope
    telescope = TelescopeAgg(n=image_size, r=50, device=device)

    logger.info("MS-SSIM uses 5 scales with weights [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]")
    logger.info("This provides better perceptual quality than single-scale SSIM")

    # Training loop (simplified)
    for epoch in range(5):
        optimizer.zero_grad()

        # Forward pass
        reconstruction = model()

        # Dummy target
        target = torch.rand(2, 2, image_size, image_size, device=device)
        center = [image_size // 2, image_size // 2]

        # Compute MS-SSIM loss
        import time

        start = time.time()
        loss_old, loss_new = criterion(reconstruction, target, telescope, center)
        elapsed = time.time() - start

        total_loss = loss_old + loss_new

        # Backward pass
        total_loss.backward()
        optimizer.step()

        logger.info(f"Epoch {epoch}: Loss={total_loss.item():.4f}, Time={elapsed * 1000:.1f}ms")

    logger.success("MS-SSIM training complete!")


def load_config_with_ssim():
    """
    Example: Load configuration file with SSIM loss.

    Shows how to configure SSIM loss in YAML config files.
    """
    logger.info("Loading configuration with SSIM loss")

    # Option 1: Load from YAML and modify
    try:
        config = PRISMConfig.from_yaml("configs/default.yaml")
        config.training.loss_type = "ssim"
        config.validate()
        logger.success("Loaded config from YAML, set loss_type='ssim'")
    except FileNotFoundError:
        logger.warning("Default config not found, creating new config")
        config = PRISMConfig()
        config.training.loss_type = "ssim"

    # Option 2: Create config programmatically (example dict structure)
    logger.info("Example config dict structure:")
    logger.info("  {'training': {'loss_type': 'ms-ssim', 'learning_rate': 0.001, ...}}")
    logger.info(f"Current loss type: {config.training.loss_type}")
    logger.info("Valid loss types: ['l1', 'l2', 'ssim', 'ms-ssim']")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Run examples
    logger.info("SPIDS Loss Functions Examples\n")

    # Example 1: Compare different loss functions
    logger.info("\n" + "=" * 70)
    logger.info("Example 1: Comparing Loss Functions")
    logger.info("=" * 70)
    compare_loss_functions()

    # Example 2: Train with SSIM
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: Training with SSIM Loss")
    logger.info("=" * 70)
    train_with_ssim_loss()

    # Example 3: Train with MS-SSIM
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: Training with MS-SSIM Loss")
    logger.info("=" * 70)
    train_with_ms_ssim_loss()

    # Example 4: Load config with SSIM
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Configuration with SSIM")
    logger.info("=" * 70)
    load_config_with_ssim()

    # Example 5: Custom combined loss (original example)
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: Custom Combined Loss")
    logger.info("=" * 70)
    train_with_custom_loss()
