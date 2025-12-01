"""
Builder pattern for constructing neural networks from configuration.

This module provides a builder class that constructs ProgressiveDecoder
instances from NetworkConfig dataclasses, enabling flexible and
configurable network architectures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from prism.models.layers import (
    init_weights_kaiming,
    init_weights_orthogonal,
    init_weights_xavier,
)
from prism.models.network_config import NetworkConfig


if TYPE_CHECKING:
    from prism.models.networks import ProgressiveDecoder


class NetworkBuilder:
    """
    Builder for ProgressiveDecoder with flexible configuration.

    Constructs networks from NetworkConfig dataclasses, applying
    custom initialization, activation functions, and architectural
    variations.

    Attributes:
        config (NetworkConfig): Network configuration

    Methods:
        build(): Construct and return configured network
        get_architecture_summary(): Get human-readable architecture description

    Example:
        >>> from prism.models.network_config import NetworkConfig
        >>> from prism.models.network_builder import NetworkBuilder
        >>>
        >>> # Create configuration
        >>> config = NetworkConfig(
        ...     input_size=1024,
        ...     output_size=512,
        ...     latent_channels=1024,
        ...     activation="relu"
        ... )
        >>>
        >>> # Build network
        >>> builder = NetworkBuilder(config)
        >>> network = builder.build()
        >>>
        >>> # View architecture
        >>> print(builder.get_architecture_summary())
    """

    def __init__(self, config: NetworkConfig):
        """
        Initialize NetworkBuilder with configuration.

        Args:
            config (NetworkConfig): Network configuration dataclass

        Raises:
            ValueError: If configuration validation fails
        """
        config.validate()  # Validate before building
        self.config = config

    def build(self) -> "ProgressiveDecoder":
        """
        Build ProgressiveDecoder from configuration.

        Constructs network with specified architecture, activation functions,
        and weight initialization. Applies custom initialization method
        after construction.

        Returns:
            ProgressiveDecoder: Configured network ready for training

        Example:
            >>> config = NetworkConfig(input_size=512, output_size=256)
            >>> builder = NetworkBuilder(config)
            >>> network = builder.build()
            >>> print(network)
        """
        # Import here to avoid circular dependency
        from prism.models.networks import ProgressiveDecoder

        # Build network with configuration
        network = ProgressiveDecoder(
            input_size=self.config.input_size,
            output_size=self.config.output_size,
            latent_channels=self.config.latent_channels,
            channel_progression=self.config.hidden_channels,
            use_bn=self.config.use_batch_norm,
            output_activation=self.config.output_activation,
        )

        # Apply custom initialization if specified
        if self.config.init_method != "kaiming":  # kaiming is PyTorch default
            self._apply_initialization(network)

        return network

    def _apply_initialization(self, network: nn.Module) -> None:
        """
        Apply custom weight initialization to network.

        Args:
            network (nn.Module): Network to initialize

        Notes:
            Uses initialization utilities from prism.models.layers
        """
        init_fn_map = {
            "kaiming": init_weights_kaiming,
            "xavier": init_weights_xavier,
            "orthogonal": init_weights_orthogonal,
        }

        init_fn = init_fn_map.get(self.config.init_method.lower())
        if init_fn:
            network.apply(init_fn)  # type: ignore[arg-type]

    def get_architecture_summary(self) -> str:
        """
        Get human-readable architecture description.

        Returns:
            str: Multi-line architecture summary including layer counts,
                 channel progression, and configuration details

        Example:
            >>> config = NetworkConfig(input_size=1024, output_size=512)
            >>> builder = NetworkBuilder(config)
            >>> print(builder.get_architecture_summary())
            ProgressiveDecoder Architecture:
              Input Size: 1024 x 1024
              Output Size: 512 x 512
              Latent Channels: 512
              Architecture Type: Decoder-only (generative)
              ...
        """
        import numpy as np

        # Calculate architecture details
        half_depth = int(np.log2(self.config.input_size)) - 2
        latent_size = 2 ** (half_depth + 2)

        # Calculate upsampling layers needed
        current_size = 4
        upsample_layers = 0
        while current_size < self.config.output_size:
            current_size *= 2
            upsample_layers += 1

        lines = [
            "ProgressiveDecoder Architecture:",
            f"  Input Size: {self.config.input_size} x {self.config.input_size}",
            f"  Output Size: {self.config.output_size} x {self.config.output_size}",
            f"  Latent Channels: {self.config.latent_channels}",
            f"  Actual Latent Size: {latent_size} (computed from input_size)",
            "  Architecture Type: Decoder-only (generative)",
            f"  Upsampling Layers: {upsample_layers} (4x4 → {current_size}x{current_size})",
            "",
            "Configuration:",
            f"  Activation: {self.config.activation}",
            f"  Output Activation: {self.config.output_activation}",
            f"  Batch Normalization: {self.config.use_batch_norm}",
            f"  Dropout: {self.config.use_dropout}",
        ]

        if self.config.use_dropout:
            lines.append(f"  Dropout Rate: {self.config.dropout_rate}")

        lines.append(f"  Initialization: {self.config.init_method}")

        return "\n".join(lines)

    def estimate_parameters(self) -> int:
        """
        Estimate total number of trainable parameters.

        Returns:
            int: Estimated parameter count

        Notes:
            This is an approximation based on configuration.
            Use model.parameters() for exact count after building.

        Example:
            >>> config = NetworkConfig(input_size=1024, output_size=512)
            >>> builder = NetworkBuilder(config)
            >>> param_count = builder.estimate_parameters()
            >>> print(f"Estimated parameters: {param_count:,}")
        """
        # Build the network to get exact count
        network = self.build()
        return sum(p.numel() for p in network.parameters() if p.requires_grad)
