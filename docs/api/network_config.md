# prism.models.network_config

Configuration dataclasses for network architectures.

This module provides configuration classes for customizing neural network
architectures in PRISM, particularly for ProgressiveDecoder.

## Classes

### NetworkConfig

```python
NetworkConfig(input_size: int, output_size: int, latent_channels: int = 512, hidden_channels: Optional[List[int]] = None, activation: str = 'relu', use_batch_norm: bool = True, use_dropout: bool = False, dropout_rate: float = 0.1, init_method: str = 'kaiming', output_activation: str = 'sigmoid') -> None
```

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

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

##### `get_summary`

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

##### `validate`

Validate configuration parameters.

Raises:
    ValueError: If any parameter is invalid.

Example:
    >>> config = NetworkConfig(input_size=1024, output_size=512)
    >>> config.validate()  # OK
    >>>
    >>> bad_config = NetworkConfig(input_size=100, output_size=512)
    >>> bad_config.validate()  # Raises ValueError
