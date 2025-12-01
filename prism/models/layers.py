"""
Module: layers.py
Purpose: Custom neural network layers and components for SPIDS
Dependencies: torch, image_utils
Main Classes:
    - ConditionalBatchNorm: Conditional batch normalization (can be disabled)
    - EncoderUnit: Convolution + activation + batch norm for encoding
    - DecoderUnit: Transposed convolution + activation + batch norm for decoding
    - ComplexAct: Activation function for complex-valued tensors
    - ScaleSigmoid: Scaled sigmoid with adjustable range
    - CropPad: Automatic crop/pad layer to match target size
    - ToComplex: Converts real tensor to complex dtype

Usage:
    from prism.models.layers import ConditionalBatchNorm, ComplexAct, CropPad

    # Conditional batch normalization
    bn = ConditionalBatchNorm(num_features=64, enable=True)

    # Complex activation
    complex_act = ComplexAct(act_type='relu')

    # Automatic cropping/padding
    crop_pad = CropPad(target_size=256)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from prism.utils import crop_pad


def activation(act_type: str, inplace: bool = True) -> nn.Module:
    """
    Factory function for activation layers.

    Args:
        act_type: Activation type string ('relu', 'leaky', 'sigmoid', 'tanh', etc.)
        inplace: Whether to use in-place operations (default: True for memory efficiency)

    Returns:
        Activation module instance

    Raises:
        ValueError: If activation type is not supported

    Performance:
        In-place activations save memory and can improve performance by ~5-15%
    """
    if act_type == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_type == "leaky":
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif act_type == "sigmoid":
        return nn.Sigmoid()
    elif act_type == "hardsigmoid":
        return nn.Hardsigmoid(inplace=inplace)
    elif act_type == "scalesigmoid":
        return ScaleSigmoid()
    elif act_type == "tanh":
        return nn.Tanh()
    elif act_type == "none":
        return nn.Identity()
    else:
        raise ValueError(
            f"Activation type {act_type} is not supported. "
            f"Supported types are: 'relu', 'leaky', 'sigmoid', 'hard_sigmoid', 'tanh', 'none'"
        )


# ============================================================================
# Activation Registry Pattern
# ============================================================================

ACTIVATION_REGISTRY = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}


def get_activation(name: str, **kwargs: Any) -> nn.Module:
    """
    Get activation function by name from registry.

    Args:
        name: Activation name (lowercase with underscores)
        **kwargs: Activation-specific parameters (e.g., negative_slope for LeakyReLU)

    Returns:
        Activation module instance

    Raises:
        ValueError: If activation name is unknown

    Examples:
        >>> act = get_activation('relu')
        >>> act = get_activation('leaky_relu', negative_slope=0.2)
        >>> act = get_activation('elu', alpha=1.0)

    Notes:
        - Registry pattern allows easy extension with custom activations
        - Use lowercase with underscores for naming consistency
        - Available activations: relu, leaky_relu, elu, silu, gelu, tanh, sigmoid, softplus
    """
    if name not in ACTIVATION_REGISTRY:
        available = ", ".join(ACTIVATION_REGISTRY.keys())
        raise ValueError(f"Unknown activation: {name}. Available: {available}")

    return ACTIVATION_REGISTRY[name](**kwargs)  # type: ignore[no-any-return]


# ============================================================================
# Weight Initialization Utilities
# ============================================================================


def init_weights_kaiming(
    module: nn.Module, mode: str = "fan_in", nonlinearity: str = "relu"
) -> None:
    """
    Initialize weights using Kaiming (He) initialization.

    Kaiming initialization is designed for ReLU-based networks and helps maintain
    variance of activations across layers during forward pass.

    Args:
        module: Module to initialize (Conv2d or Linear layers)
        mode: 'fan_in' (default) or 'fan_out' - determines variance scaling
        nonlinearity: Activation function name (default: 'relu')

    Example:
        >>> layer = nn.Conv2d(3, 64, 3)
        >>> init_weights_kaiming(layer, nonlinearity='leaky_relu')
        >>> # Weights initialized appropriately for LeakyReLU activation

    Notes:
        - Biases initialized to zeros
        - Only applies to Conv2d and Linear layers
        - Preserves variance across layers with ReLU activations
        - Reference: He et al., "Delving Deep into Rectifiers" (2015)
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)  # type: ignore[arg-type]
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_xavier(module: nn.Module, gain: float = 1.0) -> None:
    """
    Initialize weights using Xavier (Glorot) initialization.

    Xavier initialization maintains variance of activations across layers
    for sigmoid and tanh activations.

    Args:
        module: Module to initialize (Conv2d or Linear layers)
        gain: Scaling factor (default: 1.0)
              - Use 1.0 for sigmoid/tanh
              - Use nn.init.calculate_gain('relu') for ReLU

    Example:
        >>> layer = nn.Linear(128, 64)
        >>> init_weights_xavier(layer)
        >>> # Or with custom gain for ReLU
        >>> import torch.nn.init as init
        >>> init_weights_xavier(layer, gain=init.calculate_gain('relu'))

    Notes:
        - Biases initialized to zeros
        - Only applies to Conv2d and Linear layers
        - Best for tanh/sigmoid activations
        - Reference: Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010)
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_orthogonal(module: nn.Module, gain: float = 1.0) -> None:
    """
    Initialize weights using orthogonal initialization.

    Orthogonal initialization creates weight matrices with orthonormal rows/columns,
    preserving norms during forward/backward passes. Particularly useful for RNNs.

    Args:
        module: Module to initialize (Conv2d or Linear layers)
        gain: Scaling factor (default: 1.0)

    Example:
        >>> layer = nn.LSTM(128, 64)
        >>> layer.apply(lambda m: init_weights_orthogonal(m, gain=1.0))
        >>> # Or for a single layer
        >>> linear = nn.Linear(256, 128)
        >>> init_weights_orthogonal(linear)

    Notes:
        - Biases initialized to zeros
        - Only applies to Conv2d and Linear layers
        - Helps with gradient flow in deep/recurrent networks
        - Weight matrix has orthonormal rows or columns
        - Reference: Saxe et al., "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" (2013)
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConditionalBatchNorm(nn.Module):
    """
    Conditional Batch Normalization.

    Wrapper around BatchNorm2d that can be disabled, returning Identity instead.
    Useful for experiments comparing with/without batch normalization.

    Args:
        num_features: Number of features for BatchNorm2d
        enable: Whether to enable batch normalization (default: True)
        **kwargs: Additional arguments passed to BatchNorm2d

    Example:
        >>> bn = ConditionalBatchNorm(num_features=64, enable=True)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = bn(x)  # Applies batch normalization
    """

    def __init__(self, num_features: int, enable: bool = True, **kwargs: Any) -> None:
        """
        Initialize conditional batch normalization layer.

        Parameters
        ----------
        num_features : int
            Number of channels/features in the input tensor.
        enable : bool, default=True
            If True, applies batch normalization. If False, uses identity (no-op).
        **kwargs : Any
            Additional keyword arguments passed to nn.BatchNorm2d.
        """
        super().__init__()
        self.main: nn.Module = nn.BatchNorm2d(num_features, **kwargs) if enable else nn.Identity()

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Apply conditional batch normalization.

        Parameters
        ----------
        tensor : Tensor
            Input tensor of shape (batch, num_features, height, width).

        Returns
        -------
        Tensor
            Normalized tensor (if enabled) or unchanged tensor (if disabled).
        """
        result: Tensor = self.main(tensor)  # type: ignore[assignment]
        return result


class EncoderUnit(nn.Module):
    """
    Standard encoder unit: Conv2d + Activation + BatchNorm.

    Downsampling block used in encoder architectures. Applies convolution,
    activation (LeakyReLU by default), and optional batch normalization.

    Args:
        chan_in: Number of input channels
        chan_out: Number of output channels
        use_bn: Whether to use batch normalization (default: True)
        kernel_size: Convolution kernel size (default: 4)
        stride: Convolution stride (default: 2)
        padding: Convolution padding (default: 1)
        is_first: Whether this is the first layer (disables BN) (default: False)
        is_last: Whether this is the last layer (uses Sigmoid) (default: False)

    Example:
        >>> encoder = EncoderUnit(chan_in=3, chan_out=64)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> output = encoder(x)  # Shape: (1, 64, 128, 128)
    """

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        use_bn: bool = True,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        is_first: bool = False,
        is_last: bool = False,
    ) -> None:
        """
        Initialize encoder unit with convolution, activation, and batch norm.

        Parameters
        ----------
        chan_in : int
            Number of input channels.
        chan_out : int
            Number of output channels.
        use_bn : bool, default=True
            Whether to use batch normalization (disabled for first layer).
        kernel_size : int, default=4
            Size of convolutional kernel.
        stride : int, default=2
            Convolution stride (downsampling factor).
        padding : int, default=1
            Padding applied to input.
        is_first : bool, default=False
            If True, disables batch normalization for this layer.
        is_last : bool, default=False
            If True, uses Sigmoid activation instead of LeakyReLU.
        """
        super().__init__()
        use_bn = use_bn and not is_first
        conv = nn.Conv2d(
            in_channels=chan_in,
            out_channels=chan_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn,
        )
        # Use in-place activations for memory efficiency
        act = nn.LeakyReLU(0.2, inplace=True) if not is_last else nn.Sigmoid()
        bn = ConditionalBatchNorm(num_features=chan_out, enable=use_bn)
        self.main = nn.Sequential(conv, act, bn)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Forward pass through encoder unit.

        Applies convolution (with downsampling), activation, and batch norm.

        Parameters
        ----------
        tensor : Tensor
            Input tensor of shape (batch, chan_in, height, width).

        Returns
        -------
        Tensor
            Downsampled tensor of shape (batch, chan_out, height//stride, width//stride).
        """
        result: Tensor = self.main(tensor)  # type: ignore[assignment]
        return result


class DecoderUnit(nn.Module):
    """
    Standard decoder unit: ConvTranspose2d + Activation + BatchNorm.

    Upsampling block used in decoder architectures. Applies transposed convolution,
    activation (ReLU by default), and optional batch normalization.

    Args:
        chan_in: Number of input channels
        chan_out: Number of output channels
        use_bn: Whether to use batch normalization (default: True)
        kernel_size: Convolution kernel size (default: 4)
        stride: Convolution stride (default: 2)
        padding: Convolution padding (default: 1)
        is_last: Whether this is the last layer (uses Sigmoid) (default: False)

    Example:
        >>> decoder = DecoderUnit(chan_in=64, chan_out=3, is_last=True)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = decoder(x)  # Shape: (1, 3, 256, 256)
    """

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        use_bn: bool = True,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        is_last: bool = False,
    ) -> None:
        """
        Initialize decoder unit with transposed convolution, activation, and batch norm.

        Parameters
        ----------
        chan_in : int
            Number of input channels.
        chan_out : int
            Number of output channels.
        use_bn : bool, default=True
            Whether to use batch normalization (disabled for last layer).
        kernel_size : int, default=4
            Size of convolutional kernel.
        stride : int, default=2
            Convolution stride (upsampling factor).
        padding : int, default=1
            Padding applied to input.
        is_last : bool, default=False
            If True, uses Sigmoid activation and disables batch normalization.
        """
        super().__init__()
        use_bn = use_bn and not is_last
        conv = nn.ConvTranspose2d(
            in_channels=chan_in,
            out_channels=chan_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn,
        )
        # Use in-place activations for memory efficiency (except for final layer sigmoid)
        act = nn.Sigmoid() if is_last else nn.ReLU(inplace=True)
        bn = ConditionalBatchNorm(num_features=chan_out, enable=use_bn)
        self.main = nn.Sequential(conv, act, bn)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Forward pass through decoder unit.

        Applies transposed convolution (upsampling), activation, and batch norm.

        Parameters
        ----------
        tensor : Tensor
            Input tensor of shape (batch, chan_in, height, width).

        Returns
        -------
        Tensor
            Upsampled tensor of shape (batch, chan_out, height*stride, width*stride).
        """
        result: Tensor = self.main(tensor)  # type: ignore[assignment]
        return result


class ToComplex(nn.Module):
    """
    Converts real tensor to complex dtype.

    Simple utility layer that converts a real-valued tensor to complex float (cfloat).
    Used in models that need to transition from real to complex representations.

    Example:
        >>> to_complex = ToComplex()
        >>> x = torch.randn(1, 1, 256, 256)
        >>> complex_x = to_complex(x)
        >>> assert complex_x.dtype == torch.cfloat
    """

    def __init__(self) -> None:
        """Initialize ToComplex layer."""
        super().__init__()

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Convert real tensor to complex dtype.

        Parameters
        ----------
        tensor : Tensor
            Real-valued input tensor.

        Returns
        -------
        Tensor
            Complex tensor with same values, dtype changed to torch.cfloat.
        """
        return tensor.to(torch.cfloat)


class ComplexAct(nn.Module):
    """
    Activation function for complex-valued tensors.

    Applies real-valued activation separately to real and imaginary parts.
    Internally converts complex tensor to real view, applies activation,
    then converts back to complex.

    Args:
        act_type: Type of activation ('relu', 'sigmoid', 'tanh', etc.)

    Example:
        >>> complex_act = ComplexAct(act_type='relu')
        >>> x = torch.randn(1, 1, 256, 256, dtype=torch.cfloat)
        >>> output = complex_act(x)
    """

    def __init__(self, act_type: str) -> None:
        """
        Initialize complex activation layer.

        Parameters
        ----------
        act_type : str
            Type of activation to apply ('relu', 'sigmoid', 'tanh', etc.).
        """
        super().__init__()
        self.act = activation(act_type)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Apply activation to complex tensor.

        Converts complex tensor to real view, applies activation to both
        real and imaginary parts, then converts back to complex.

        Parameters
        ----------
        tensor : Tensor
            Complex-valued input tensor.

        Returns
        -------
        Tensor
            Complex tensor with activation applied separately to real
            and imaginary components.
        """
        return torch.view_as_complex(self.act(torch.view_as_real(tensor)))


class ScaleSigmoid(nn.Module):
    """
    Scaled sigmoid activation with adjustable range.

    Standard sigmoid output is [0, 1]. This layer scales it to
    [-scale, 1+scale] for more flexible range.

    Args:
        scale: Amount to extend range beyond [0, 1] (default: 0.01)

    Example:
        >>> scaled_sigmoid = ScaleSigmoid(scale=0.1)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = scaled_sigmoid(x)  # Range: [-0.1, 1.1]
    """

    def __init__(self, scale: float = 0.01) -> None:
        """
        Initialize scaled sigmoid activation.

        Parameters
        ----------
        scale : float, default=0.01
            Amount to extend output range beyond [0, 1].
            Output range becomes [-scale, 1+scale].
        """
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Apply scaled sigmoid activation.

        Parameters
        ----------
        tensor : Tensor
            Input tensor of any shape.

        Returns
        -------
        Tensor
            Tensor with scaled sigmoid applied, range [-scale, 1+scale].
        """
        # Apply sigmoid, then scale in-place for efficiency
        output = self.sigmoid(tensor)
        scale_val = self.scale if isinstance(self.scale, Tensor) else torch.tensor(0.01)
        output.mul_(1 + 2 * scale_val).sub_(scale_val)
        result: Tensor = output
        return result


class CropPad(nn.Module):
    """
    Automatic crop/pad layer to match target size.

    Automatically crops or pads input tensor to match target spatial size.
    Handles both cases: when input is larger (crop) or smaller (pad) than target.

    Args:
        target_size: Desired output spatial size (H and W)

    Example:
        >>> crop_pad = CropPad(target_size=256)
        >>> x = torch.randn(1, 1, 512, 512)
        >>> output = crop_pad(x)  # Shape: (1, 1, 256, 256)
        >>> y = torch.randn(1, 1, 128, 128)
        >>> output2 = crop_pad(y)  # Shape: (1, 1, 256, 256)
    """

    def __init__(self, target_size: int) -> None:
        """
        Initialize CropPad layer.

        Parameters
        ----------
        target_size : int
            Desired output spatial size (applied to both height and width).
        """
        super().__init__()
        self.size = target_size

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Crop or pad tensor to match target size.

        If input is larger than target, crops from center.
        If input is smaller than target, pads with zeros symmetrically.

        Parameters
        ----------
        tensor : Tensor
            Input tensor of shape (batch, channels, height, width).

        Returns
        -------
        Tensor
            Tensor with spatial dimensions adjusted to (target_size, target_size).
        """
        return crop_pad(tensor, self.size)
