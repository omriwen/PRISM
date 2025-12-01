"""
Module: networks.py
Purpose: Neural network architectures for SPIDS reconstruction
Dependencies: torch, numpy, spids.models.layers

Main Classes:
    - ProgressiveDecoder: **PRIMARY MODEL** - Generative decoder-only network with progressive upsampling

Architecture Details:
    ProgressiveDecoder (Primary Model):
    - Decoder-only architecture (no encoder needed)
    - Learns from single learnable latent vector (configurable size)
    - Progressive upsampling: 1x1 → 4x4 → ... → output_size
    - Automatic cropping to handle obj_size != image_size
    - Flexible architecture: Manual or automatic layer configuration

Usage Pattern:
    from prism.models.networks import ProgressiveDecoder

    # Automatic configuration (legacy behavior)
    model = ProgressiveDecoder(input_size=1024, output_size=512)

    # Manual configuration
    model = ProgressiveDecoder(
        input_size=1024,
        output_size=512,
        latent_channels=2048,
        channel_progression=[1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    )

    # Configuration-driven (recommended)
    from prism.models.network_config import NetworkConfig
    config = NetworkConfig(input_size=1024, output_size=512, latent_channels=1024)
    model = ProgressiveDecoder.from_config(config)

    # Training (decoder-only, no input needed)
    output = model()
    loss = criterion(output, target)
    loss.backward()
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from torch import Tensor, nn

from prism.models.layers import (
    ConditionalBatchNorm,
    CropPad,
    activation,
)


if TYPE_CHECKING:
    from prism.models.network_config import NetworkConfig


class ProgressiveDecoder(nn.Module):
    """
    Progressive generative decoder with flexible architecture.

    A decoder-only generative network that progressively upsamples from a
    learnable latent vector (1x1) to the target output size. Unlike traditional
    autoencoders, this model doesn't require an encoder - the latent vector is
    directly optimized during training.

    Architecture:
        1. Learnable latent vector: [1, C_latent, 1, 1]
        2. Initial upsampling: 1x1 → 4x4 (ConvTranspose2d)
        3. Progressive doubling: 4x4 → 8x8 → 16x16 → ... (until >= output_size)
        4. Channel refinement: Reduce channels to 1 while maintaining size
        5. Crop/Pad: Adjust to exact output_size and input_size

    Key Features:
        - Fully automatic configuration: Just specify input/output size
        - Manual control: Override latent size, channels, layer depth
        - Mixed precision training: Enable with use_amp=True
        - Inference optimization: Call prepare_for_inference() after training
        - Configuration-driven: Use NetworkConfig for advanced setups

    Args:
        input_size (int): Full image grid size (e.g., 1024). Must be power of 2.
        output_size (int, optional): Actual object size to reconstruct (e.g., 512).
            If None, uses input_size.
        latent_channels (int, optional): Number of channels in latent vector.
            If None, automatically computed as 2^(log2(input_size)).
            Typical values: 256, 512, 1024, 2048.
        channel_progression (list[int], optional): Explicit channel count for each layer.
            If None, automatically generated as powers-of-2 halving.
            Example: [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        num_upsample_layers (int, optional): Number of upsampling (2x) layers.
            If None, automatically computed to reach output_size from 4x4.
        use_bn (bool): Whether to use batch normalization. Default: True.
        output_activation (str): Final activation function. Options: "sigmoid",
            "tanh", "relu", "none". Default: "sigmoid".
        use_amp (bool): Use Automatic Mixed Precision (FP16/FP32 mixed).
            Provides ~20-30% speedup on Ampere+ GPUs. Default: False.
        use_leaky (bool): Deprecated parameter kept for API compatibility.
        middle_activation (str): Deprecated parameter kept for API compatibility.
        complex_data (bool): Deprecated parameter kept for API compatibility.

    Examples:
        Basic usage (automatic configuration):
        >>> model = ProgressiveDecoder(input_size=1024, output_size=512)
        >>> output = model()  # No input needed!
        >>> print(output.shape)  # torch.Size([1, 1, 1024, 1024])

        Manual architecture control:
        >>> model = ProgressiveDecoder(
        ...     input_size=1024,
        ...     output_size=512,
        ...     latent_channels=2048,  # Larger latent space
        ...     channel_progression=[2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        ... )

        Configuration-driven (recommended for complex setups):
        >>> from prism.models.network_config import NetworkConfig
        >>> config = NetworkConfig(
        ...     input_size=1024,
        ...     output_size=512,
        ...     latent_channels=1024,
        ...     activation="relu",
        ...     init_method="xavier"
        ... )
        >>> model = ProgressiveDecoder.from_config(config)

        Training loop:
        >>> model = ProgressiveDecoder(input_size=1024, output_size=512, use_amp=True)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> for epoch in range(num_epochs):
        ...     optimizer.zero_grad()
        ...     reconstruction = model()  # Decoder-only: no input!
        ...     loss = criterion(reconstruction, measurement)
        ...     loss.backward()
        ...     optimizer.step()

        Inference optimization:
        >>> model.prepare_for_inference()  # Fuse Conv-BN, freeze params
        >>> with torch.no_grad():
        ...     final_output = model.generate_fp32()  # Force FP32 for accuracy

    Notes:
        - No encoder required: The latent vector is directly learned
        - Decoder-only design: forward() takes no arguments
        - Two-stage cropping: First to output_size, then pad to input_size
          (enables obj_size != image_size in SPIDS)
        - For training, use forward(). For inference, use generate_fp32().

    See Also:
        - NetworkConfig: Advanced configuration dataclass
        - NetworkBuilder: Build networks from configuration
        - spids.models.layers: Layer components (DecoderUnit, CropPad, etc.)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        latent_channels: int | None = None,
        channel_progression: list[int] | None = None,
        num_upsample_layers: int | None = None,
        use_bn: bool = True,
        output_activation: str = "sigmoid",
        use_amp: bool = False,
        # Deprecated parameters (ignored with warning)
        use_leaky: bool = True,
        middle_activation: str = "tanh",
        complex_data: bool = False,
    ):
        """Initialize ProgressiveDecoder with flexible architecture configuration."""
        super().__init__()

        # Emit warnings for deprecated parameters
        if use_leaky is not True or middle_activation != "tanh" or complex_data is not False:
            warnings.warn(
                "Parameters 'use_leaky', 'middle_activation', and 'complex_data' are "
                "deprecated and will be removed in v2.0",
                DeprecationWarning,
                stacklevel=2,
            )

        # Store configuration
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        self.use_amp = use_amp

        # Compute architecture parameters
        arch_params = self._compute_architecture_params(
            input_size=input_size,
            output_size=self.output_size,
            latent_channels=latent_channels,
            channel_progression=channel_progression,
            num_upsample_layers=num_upsample_layers,
        )

        self.latent_channels = arch_params["latent_channels"]
        self.channel_progression = arch_params["channel_progression"]
        self.num_upsample_layers = arch_params["num_upsample_layers"]

        # Create learnable latent vector
        self.input_vec = nn.Parameter(torch.randn(1, self.latent_channels, 1, 1))

        # Build decoder
        self.decoder = self._build_decoder(
            use_bn=use_bn,
            output_activation=output_activation,
        )

    def _compute_architecture_params(
        self,
        input_size: int,
        output_size: int,
        latent_channels: int | None,
        channel_progression: list[int] | None,
        num_upsample_layers: int | None,
    ) -> dict:
        """
        Compute architecture parameters (automatic or manual).

        Validates inputs and computes missing parameters automatically based
        on input_size and output_size.

        Args:
            input_size: Full image grid size
            output_size: Actual object size to reconstruct
            latent_channels: Optional manual latent channel count
            channel_progression: Optional manual channel progression
            num_upsample_layers: Optional manual upsampling layer count

        Returns:
            dict with keys: latent_channels, channel_progression, num_upsample_layers

        Raises:
            ValueError: If input_size is not power of 2, or output_size > input_size
        """
        # Validation: input_size must be power of 2
        if not (input_size & (input_size - 1)) == 0:
            raise ValueError(
                f"input_size must be power of 2, got {input_size}. "
                f"Valid values: 64, 128, 256, 512, 1024, 2048"
            )

        # Validation: output_size must be <= input_size
        if output_size > input_size:
            raise ValueError(f"output_size ({output_size}) cannot exceed input_size ({input_size})")

        # Validation: output_size should be power of 2 (warning only)
        if not (output_size & (output_size - 1)) == 0:
            warnings.warn(
                f"output_size={output_size} is not a power of 2. This may cause "
                f"suboptimal architecture. Consider using: 64, 128, 256, 512, 1024",
                UserWarning,
            )

        # Automatic latent channels (default)
        if latent_channels is None:
            latent_depth = int(np.log2(input_size)) - 2
            latent_channels = 2 ** (latent_depth + 2)

        # Automatic num_upsample_layers (default)
        if num_upsample_layers is None:
            current_size = 4  # After first layer: 1x1 → 4x4
            num_upsample_layers = 0
            while current_size < output_size:
                current_size *= 2
                num_upsample_layers += 1

        # Automatic channel progression (default)
        if channel_progression is None:
            # Start from latent_channels, halve each layer
            channel_progression = []
            current_ch = latent_channels

            # Initial layer: latent_channels → latent_channels // 2
            channel_progression.append(current_ch // 2)
            current_ch //= 2

            # Upsampling layers
            for _ in range(num_upsample_layers):
                current_ch //= 2
                channel_progression.append(current_ch)

            # Refinement layers: reduce to 1 channel
            while current_ch > 1:
                current_ch = max(1, current_ch // 2)
                channel_progression.append(current_ch)
        else:
            # Validation: channel_progression consistency
            if channel_progression[-1] != 1:
                raise ValueError(
                    f"channel_progression must end with 1 (final output channel), "
                    f"got {channel_progression[-1]}"
                )
            if len(channel_progression) < 2:
                raise ValueError(
                    f"channel_progression must have at least 2 elements, "
                    f"got {len(channel_progression)}"
                )

        return {
            "latent_channels": latent_channels,
            "channel_progression": channel_progression,
            "num_upsample_layers": num_upsample_layers,
        }

    def _build_decoder(
        self,
        use_bn: bool,
        output_activation: str,
    ) -> nn.Sequential:
        """
        Build decoder network from architecture parameters.

        Constructs the complete decoder pipeline including initial upsampling,
        progressive upsampling layers, refinement layers, and final cropping.

        Args:
            use_bn: Whether to use batch normalization
            output_activation: Final activation function name

        Returns:
            nn.Sequential decoder module
        """
        layers = []

        # Initial upsampling: 1x1 → 4x4
        layers.extend(
            self._build_initial_layer(
                in_channels=self.latent_channels,
                out_channels=self.channel_progression[0],
                use_bn=use_bn,
            )
        )

        # Progressive upsampling layers
        for i in range(self.num_upsample_layers):
            layers.extend(
                self._build_upsample_layer(
                    in_channels=self.channel_progression[i],
                    out_channels=self.channel_progression[i + 1],
                    use_bn=use_bn,
                    layer_idx=i,
                )
            )

        # Refinement layers (reduce channels, maintain size)
        refinement_start = self.num_upsample_layers + 1
        for i in range(refinement_start, len(self.channel_progression)):
            is_final = i == len(self.channel_progression) - 1
            layers.extend(
                self._build_refinement_layer(
                    in_channels=self.channel_progression[i - 1],
                    out_channels=self.channel_progression[i],
                    use_bn=use_bn and not is_final,
                    is_final=is_final,
                )
            )

        # Output activation
        layers.append(activation(output_activation))

        # Crop/Pad to exact sizes
        layers.append(CropPad(self.output_size))
        layers.append(CropPad(self.input_size))

        return nn.Sequential(*layers)

    def _build_initial_layer(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool,
    ) -> list[nn.Module]:
        """
        Build initial upsampling layer: 1x1 → 4x4.

        Shape transformation:
            Input:  [B, in_channels, 1, 1]
            Output: [B, out_channels, 4, 4]

        Args:
            in_channels: Number of input channels (latent_channels)
            out_channels: Number of output channels
            use_bn: Whether to use batch normalization

        Returns:
            List of layer modules
        """
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=not use_bn,
            )
        ]
        if use_bn:
            layers.append(ConditionalBatchNorm(num_features=out_channels, enable=True))
        layers.append(nn.ReLU())
        return layers

    def _build_upsample_layer(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool,
        layer_idx: int,
    ) -> list[nn.Module]:
        """
        Build upsampling layer (doubles spatial size).

        Shape transformation:
            Input:  [B, in_channels, H, W]
            Output: [B, out_channels, 2*H, 2*W]

        Example:
            [B, 256, 8, 8] → [B, 128, 16, 16]

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_bn: Whether to use batch normalization
            layer_idx: Layer index for debugging/logging

        Returns:
            List of layer modules
        """
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not use_bn,
            )
        ]
        if use_bn:
            layers.append(ConditionalBatchNorm(num_features=out_channels, enable=True))
        layers.append(nn.ReLU())
        return layers

    def _build_refinement_layer(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool,
        is_final: bool,
    ) -> list[nn.Module]:
        """
        Build refinement layer (maintains size, reduces channels).

        Shape transformation:
            Input:  [B, in_channels, H, W]
            Output: [B, out_channels, H-2, W-2] (if is_final)
                    [B, out_channels, H, W]     (otherwise)

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_bn: Whether to use batch normalization
            is_final: Whether this is the final refinement layer

        Returns:
            List of layer modules
        """
        kernel_size = 3
        padding = 0 if is_final else 1

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=(out_channels == 1) or (not use_bn),
            )
        ]

        if use_bn and out_channels > 1:
            layers.append(ConditionalBatchNorm(num_features=out_channels, enable=True))
            layers.append(nn.ReLU())

        return layers

    @classmethod
    def from_config(cls, config: "NetworkConfig") -> "ProgressiveDecoder":
        """
        Create ProgressiveDecoder from NetworkConfig.

        This is the recommended way to create networks when using advanced
        configurations. The NetworkBuilder validates the config and applies
        custom initialization.

        Args:
            config (NetworkConfig): Network configuration dataclass

        Returns:
            ProgressiveDecoder: Configured network instance

        Example:
            >>> from prism.models.network_config import NetworkConfig
            >>> config = NetworkConfig(
            ...     input_size=1024,
            ...     output_size=512,
            ...     latent_channels=1024,
            ...     activation="relu",
            ...     init_method="xavier"
            ... )
            >>> network = ProgressiveDecoder.from_config(config)
            >>> print(f"Parameters: {sum(p.numel() for p in network.parameters()):,}")
        """
        from prism.models.network_builder import NetworkBuilder

        builder = NetworkBuilder(config)
        return builder.build()

    def forward(self) -> Tensor:
        """
        Generate reconstruction from learnable latent vector.

        Uses Automatic Mixed Precision (AMP) if use_amp=True was specified
        during initialization. For inference with maximum accuracy, use
        generate_fp32() instead.

        Returns:
            Tensor: Generated image of shape (1, 1, input_size, input_size)

        Notes:
            - Training mode: Uses AMP if enabled (FP16/FP32 mixed precision)
            - Inference mode: Use generate_fp32() for best accuracy
        """
        if self.use_amp:
            with torch.cuda.amp.autocast():
                return self._forward_impl()
        else:
            return self._forward_impl()

    def _forward_impl(self) -> Tensor:
        """Internal forward pass implementation."""
        if hasattr(self, "_use_gradient_checkpointing") and self._use_gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(self.decoder, self.input_vec, use_reentrant=False)  # type: ignore[no-any-return]
        else:
            return self.decoder(self.input_vec)  # type: ignore[no-any-return]

    def generate_fp32(self) -> Tensor:
        """
        Generate reconstruction in FP32 for inference.

        Always uses FP32 precision regardless of use_amp setting,
        ensuring maximum accuracy for final reconstructions.

        Returns:
            Tensor: Generated image in FP32 precision

        Example:
            >>> # Training with AMP
            >>> model = ProgressiveDecoder(input_size=1024, use_amp=True)
            >>> train_output = model()  # Uses mixed precision
            >>>
            >>> # Inference with full precision
            >>> model.eval()
            >>> final_output = model.generate_fp32()  # Always FP32
        """
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return self._forward_impl()

    def prepare_for_inference(
        self,
        compile_mode: str | None = None,
        free_memory: bool = True,
    ) -> "ProgressiveDecoder":
        """
        Optimize network for inference.

        Applies multiple optimizations:
        1. Set to eval mode
        2. Conv-BN fusion (~10-20% speedup)
        3. Freeze parameters (disable gradients)
        4. Optional: torch.compile (~30% additional speedup)
        5. Optional: Free unused CUDA memory

        Args:
            compile_mode (str, optional): If specified, compile with torch.compile.
                Options: "default", "reduce-overhead", "max-autotune".
            free_memory (bool): Whether to call torch.cuda.empty_cache().
                Default: True.

        Returns:
            self: Returns self for method chaining

        Example:
            >>> model = ProgressiveDecoder(input_size=1024)
            >>> # ... training ...
            >>> model.prepare_for_inference(compile_mode="max-autotune")
            >>> with torch.no_grad():
            ...     output = model.generate_fp32()

        Notes:
            - Call this once after training completes
            - Model cannot be trained after this operation
            - Total expected speedup: ~30-50% vs standard eval mode
        """
        # Set to eval mode
        self.eval()

        # Fuse Conv-BN layers
        self._fuse_conv_bn_layers()

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Optional: Compile
        if compile_mode is not None:
            self.compile(mode=compile_mode)

        # Optional: Free memory
        if free_memory and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Freed unused CUDA memory")

        return self

    def compile(self, mode: str = "default") -> "ProgressiveDecoder":
        """
        Compile model using torch.compile for faster execution.

        Requires PyTorch 2.0+. Provides significant speedup for both
        training and inference through graph optimizations.

        Args:
            mode (str): Compilation mode. Options:
                - "default": Balanced optimization (~30% speedup)
                - "reduce-overhead": Minimize Python overhead
                - "max-autotune": Maximum optimization (slower compile time)

        Returns:
            self: Returns self for method chaining

        Example:
            >>> model = ProgressiveDecoder(input_size=1024)
            >>> model = model.compile(mode="max-autotune")
            >>> # Now forward() is ~30-40% faster

        Notes:
            - First call to forward() will trigger compilation (slow)
            - Subsequent calls use compiled version (fast)
            - Requires PyTorch 2.0+
            - Works with both CPU and CUDA
        """
        if not hasattr(torch, "compile"):
            warnings.warn(
                "torch.compile not available (requires PyTorch 2.0+). "
                "Update PyTorch for this optimization.",
                UserWarning,
            )
            return self

        try:
            self.decoder = torch.compile(self.decoder, mode=mode)  # type: ignore[assignment]
            logger.info(f"Model compiled with mode='{mode}'")
        except Exception as e:  # noqa: BLE001 - torch.compile may fail for various reasons
            warnings.warn(
                f"torch.compile failed: {e}. Continuing without compilation.",
                UserWarning,
            )

        return self

    def _fuse_conv_bn_layers(self) -> None:
        """
        Fuse Conv2d + BatchNorm2d layer pairs for faster inference.

        This optimization combines convolution and batch normalization into
        a single operation, reducing memory access and computation.

        Expected speedup: ~10-20% for inference

        Notes:
            - Only effective when use_bn=True
            - Must be called in eval mode
            - Modifies network in-place
            - Only fuses standard Conv2d + BatchNorm2d pairs
            - ConvTranspose2d and ConditionalBatchNorm are not fusable
        """
        # Collect Conv-BN pairs to fuse
        modules_to_fuse = []

        # Iterate through decoder layers
        i = 0
        while i < len(self.decoder):
            current_layer = self.decoder[i]

            # Only fuse standard Conv2d + BatchNorm2d (not ConvTranspose2d or ConditionalBatchNorm)
            # PyTorch fusion only supports Conv2d + BatchNorm2d pairs
            if isinstance(current_layer, nn.Conv2d) and not isinstance(
                current_layer, nn.ConvTranspose2d
            ):
                if i + 1 < len(self.decoder):
                    next_layer = self.decoder[i + 1]
                    if isinstance(next_layer, nn.BatchNorm2d) and not isinstance(
                        next_layer, ConditionalBatchNorm
                    ):
                        # Found fusable Conv2d-BN pair
                        modules_to_fuse.append([f"decoder.{i}", f"decoder.{i + 1}"])
                        i += 2  # Skip next layer
                        continue
            i += 1

        # Apply fusion if any pairs found
        if modules_to_fuse:
            try:
                from torch.ao.quantization import fuse_modules

                fuse_modules(self, modules_to_fuse, inplace=True)
                logger.info(f"Fused {len(modules_to_fuse)} Conv-BN pairs for inference")
            except (ImportError, AssertionError) as e:
                warnings.warn(
                    f"Conv-BN fusion skipped: {e}. "
                    "This is expected for models using ConvTranspose2d or ConditionalBatchNorm.",
                    UserWarning,
                )
        else:
            logger.debug(
                "No fusable Conv-BN pairs found (expected for ConvTranspose2d-based models)"
            )

    def enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing for memory-efficient training.

        Reduces memory usage by ~50% at the cost of ~20% slower training.
        Useful for training very large models or with large batch sizes.

        Example:
            >>> model = ProgressiveDecoder(input_size=2048, latent_channels=2048)
            >>> model.enable_gradient_checkpointing()
            >>> # Now can train with 2x larger batch size

        Notes:
            - Only affects training (requires_grad=True)
            - No effect during inference
            - Trades compute for memory
            - Modifies forward pass to use checkpointing
        """
        # Store flag to enable checkpointing in forward pass
        self._use_gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing (memory reduction: ~50%)")

    def benchmark(
        self,
        num_iterations: int = 100,
        warmup: int = 10,
        measure_memory: bool = True,
    ) -> dict:
        """
        Benchmark model performance.

        Args:
            num_iterations (int): Number of forward passes to average
            warmup (int): Number of warmup iterations (excluded from timing)
            measure_memory (bool): Whether to measure CUDA memory usage

        Returns:
            dict: Benchmark results with keys:
                - 'avg_time_ms': Average forward pass time in milliseconds
                - 'fps': Reconstructions per second
                - 'memory_mb': Peak CUDA memory (if measure_memory=True)
                - 'num_parameters': Total parameter count

        Example:
            >>> model = ProgressiveDecoder(input_size=1024)
            >>> results = model.benchmark(num_iterations=100)
            >>> print(f"Average time: {results['avg_time_ms']:.2f} ms")
            >>> print(f"FPS: {results['fps']:.2f}")
        """
        import time

        self.eval()
        device = next(self.parameters()).device

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self()

        # Synchronize if CUDA
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Measure memory (before)
        if measure_memory and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = self()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        # Measure memory (after)
        if measure_memory and device.type == "cuda":
            memory_peak = torch.cuda.max_memory_allocated()
            memory_mb = (memory_peak - memory_before) / 1024 / 1024

        # Compute statistics
        avg_time_ms = sum(times) / len(times)
        fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0

        results = {
            "avg_time_ms": avg_time_ms,
            "fps": fps,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }

        if measure_memory and device.type == "cuda":
            results["memory_mb"] = memory_mb

        return results


# GenCropSpidsNet has been removed. Use ProgressiveDecoder instead.
# If you have code using GenCropSpidsNet, please update:
#   Old: from prism.models.networks import GenCropSpidsNet
#   New: from prism.models.networks import ProgressiveDecoder
