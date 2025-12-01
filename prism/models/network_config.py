"""
Configuration dataclasses for network architectures.

This module provides configuration classes for customizing neural network
architectures in SPIDS, particularly for ProgressiveDecoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class NetworkConfig:
    """
    Configuration for ProgressiveDecoder architecture.

    Allows customization of network architecture including latent size,
    channel counts, activation functions, normalization, and initialization.

    Attributes:
        input_size (int): Full image grid size (e.g., 1024). Must be power of 2.
        output_size (int): Actual object size to reconstruct (e.g., 512).
        latent_channels (int): Number of channels in latent vector (default: 512).
        hidden_channels (List[int]): Channel progression through decoder layers.
                                     If None, uses default progressive halving.
        activation (str): Activation function type (default: "leakyrelu").
                         Options: "relu", "leakyrelu", "tanh", "sigmoid"
        use_batch_norm (bool): Whether to use batch normalization (default: True).
        use_dropout (bool): Whether to use dropout layers (default: False).
        dropout_rate (float): Dropout probability if use_dropout=True (default: 0.1).
        init_method (str): Weight initialization method (default: "kaiming").
                          Options: "kaiming", "xavier", "orthogonal"
        output_activation (str): Final layer activation (default: "sigmoid").

    Example:
        >>> # Default configuration
        >>> config = NetworkConfig(input_size=1024, output_size=512)
        >>>
        >>> # Custom configuration with larger latent space
        >>> config = NetworkConfig(
        ...     input_size=1024,
        ...     output_size=512,
        ...     latent_channels=1024,
        ...     activation="relu",
        ...     init_method="xavier"
        ... )
        >>>
        >>> # Configuration with dropout
        >>> config = NetworkConfig(
        ...     input_size=512,
        ...     output_size=256,
        ...     use_dropout=True,
        ...     dropout_rate=0.2
        ... )
    """

    # Required parameters
    input_size: int
    output_size: int

    # Architecture parameters
    latent_channels: int = 512
    hidden_channels: List[int] | None = None

    # Activation and normalization
    activation: str = "relu"
    use_batch_norm: bool = True
    use_dropout: bool = False
    dropout_rate: float = 0.1

    # Initialization
    init_method: str = "kaiming"  # kaiming, xavier, orthogonal

    # Output
    output_activation: str = "sigmoid"

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid.

        Example:
            >>> config = NetworkConfig(input_size=1024, output_size=512)
            >>> config.validate()  # OK
            >>>
            >>> bad_config = NetworkConfig(input_size=100, output_size=512)
            >>> bad_config.validate()  # Raises ValueError
        """
        # Check input_size is power of 2
        if not self._is_power_of_2(self.input_size):
            raise ValueError(f"input_size must be power of 2, got {self.input_size}")

        # Check output_size is power of 2
        if not self._is_power_of_2(self.output_size):
            raise ValueError(f"output_size must be power of 2, got {self.output_size}")

        # Check output_size <= input_size
        if self.output_size > self.input_size:
            raise ValueError(
                f"output_size ({self.output_size}) cannot exceed input_size ({self.input_size})"
            )

        # Check latent_channels is positive
        if self.latent_channels <= 0:
            raise ValueError(f"latent_channels must be positive, got {self.latent_channels}")

        # Check dropout_rate range
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")

        # Check activation type
        valid_activations = {"relu", "leakyrelu", "tanh", "sigmoid", "elu", "selu"}
        if self.activation.lower() not in valid_activations:
            raise ValueError(
                f"activation must be one of {valid_activations}, got {self.activation}"
            )

        # Check init_method
        valid_init_methods = {"kaiming", "xavier", "orthogonal"}
        if self.init_method.lower() not in valid_init_methods:
            raise ValueError(
                f"init_method must be one of {valid_init_methods}, got {self.init_method}"
            )

        # Check output_activation
        valid_output_activations = {"sigmoid", "tanh", "relu", "identity"}
        if self.output_activation.lower() not in valid_output_activations:
            raise ValueError(
                f"output_activation must be one of {valid_output_activations}, "
                f"got {self.output_activation}"
            )

    @staticmethod
    def _is_power_of_2(n: int) -> bool:
        """Check if n is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    def get_summary(self) -> str:
        """
        Get human-readable configuration summary.

        Returns:
            str: Multi-line summary of configuration.

        Example:
            >>> config = NetworkConfig(input_size=1024, output_size=512)
            >>> print(config.get_summary())
            Network Configuration:
              Input Size: 1024
              Output Size: 512
              Latent Channels: 512
              Activation: relu
              Batch Norm: True
              Dropout: False
              Init Method: kaiming
              Output Activation: sigmoid
        """
        lines = [
            "Network Configuration:",
            f"  Input Size: {self.input_size}",
            f"  Output Size: {self.output_size}",
            f"  Latent Channels: {self.latent_channels}",
            f"  Activation: {self.activation}",
            f"  Batch Norm: {self.use_batch_norm}",
            f"  Dropout: {self.use_dropout}",
        ]

        if self.use_dropout:
            lines.append(f"  Dropout Rate: {self.dropout_rate}")

        lines.extend(
            [
                f"  Init Method: {self.init_method}",
                f"  Output Activation: {self.output_activation}",
            ]
        )

        if self.hidden_channels is not None:
            lines.append(f"  Hidden Channels: {self.hidden_channels}")

        return "\n".join(lines)
