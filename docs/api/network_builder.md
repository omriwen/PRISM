# prism.models.network_builder

Builder pattern for constructing neural networks from configuration.

This module provides a builder class that constructs ProgressiveDecoder
instances from NetworkConfig dataclasses, enabling flexible and
configurable network architectures.

## Classes

### NetworkBuilder

```python
NetworkBuilder(config: prism.models.network_config.NetworkConfig)
```

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

#### Methods

##### `__init__`

Initialize NetworkBuilder with configuration.

Args:
    config (NetworkConfig): Network configuration dataclass

Raises:
    ValueError: If configuration validation fails

##### `build`

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

##### `estimate_parameters`

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

##### `get_architecture_summary`

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
