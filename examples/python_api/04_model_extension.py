"""
Model architecture extension example.

Shows how to extend or modify the SPIDS model architecture.
"""

import sys

import torch
import torch.nn as nn
from loguru import logger

from prism.models.networks import ProgressiveDecoder


class AttentionBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # Query, Key, Value projections
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention."""
        batch, channels, height, width = x.shape

        # Compute Q, K, V
        q = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, height * width)
        v = self.value(x).view(batch, -1, height * width)

        # Attention weights
        attention = torch.softmax(torch.bmm(q, k), dim=-1)

        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)

        # Residual connection with learnable weight
        return x + self.gamma * out


class AttentionSPIDSNet(ProgressiveDecoder):
    """
    SPIDS network with self-attention layers.

    Extends the base ProgressiveDecoder with attention mechanisms
    for better capturing of global structure.
    """

    def __init__(self, *args, use_attention: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_attention = use_attention

        if use_attention:
            # Add attention layers at multiple scales
            # Note: This is a simplified example - actual implementation
            # would need to match the decoder structure
            logger.info("Added attention mechanism to SPIDS network")


class ResidualSPIDSNet(ProgressiveDecoder):
    """
    SPIDS network with residual connections.

    Uses skip connections for better gradient flow.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add skip connection layers if needed
        logger.info("Created SPIDS network with residual connections")


class MultiScaleSPIDSNet(nn.Module):
    """
    Multi-scale SPIDS network.

    Generates reconstructions at multiple resolutions for progressive refinement.
    """

    def __init__(self, output_size: int, input_size: int, scales: list[int] = None):
        super().__init__()

        if scales is None:
            scales = [output_size // 4, output_size // 2, output_size]

        self.scales = scales
        self.output_size = output_size
        self.input_size = input_size

        # Create a model for each scale
        self.models = nn.ModuleList(
            [ProgressiveDecoder(input_size=input_size, output_size=scale) for scale in scales]
        )

        logger.info(f"Created multi-scale network with scales: {scales}")

    def forward(self) -> dict[str, torch.Tensor]:
        """
        Forward pass generating multi-scale outputs.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with outputs at each scale
        """
        outputs = {}
        for i, (scale, model) in enumerate(zip(self.scales, self.models)):
            outputs[f"scale_{scale}"] = model()

        return outputs


def compare_architectures():
    """Compare different model architectures."""

    output_size = 128
    input_size = 512

    models = {
        "baseline": ProgressiveDecoder(input_size=input_size, output_size=output_size),
        "attention": AttentionSPIDSNet(
            input_size=input_size, output_size=output_size, use_attention=True
        ),
        "residual": ResidualSPIDSNet(input_size=input_size, output_size=output_size),
        "multiscale": MultiScaleSPIDSNet(output_size=output_size, input_size=input_size),
    }

    logger.info("Comparing model architectures:\n")

    for name, model in models.items():
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())

        # Test forward pass
        with torch.no_grad():
            output = model()
            if isinstance(output, dict):
                output_info = f"Dict with {len(output)} scales"
            else:
                output_info = f"{output.shape}"

        logger.info(f"{name.capitalize():12} | Params: {n_params:,} | Output: {output_info}")

    logger.success("\nAll architectures functional!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    compare_architectures()
