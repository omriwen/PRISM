"""
Comprehensive unit tests for ProgressiveDecoder.

Tests cover:
- Basic initialization and architecture computation
- Forward pass and output shapes
- Manual configuration modes
- Parameter validation and error handling
- Inference optimization features
- Checkpoint compatibility
- Deprecation warnings
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from prism.models.network_config import NetworkConfig
from prism.models.networks import ProgressiveDecoder


@pytest.fixture
def device():
    """Get device for testing (CPU only to avoid CUDA dependencies)."""
    return torch.device("cpu")


class TestProgressiveDecoderInitialization:
    """Test basic initialization and parameter computation."""

    def test_basic_initialization(self, device):
        """Test basic model creation with default parameters."""
        model = ProgressiveDecoder(input_size=256, output_size=128)
        model = model.to(device)

        assert model.input_size == 256
        assert model.output_size == 128
        assert isinstance(model.decoder, nn.Sequential)
        assert isinstance(model.input_vec, nn.Parameter)

    def test_automatic_latent_channels(self, device):
        """Test automatic latent channel computation."""
        # For input_size=256: latent_depth = log2(256) - 2 = 8 - 2 = 6
        # latent_channels = 2^(6 + 2) = 2^8 = 256
        model = ProgressiveDecoder(input_size=256, output_size=128)
        assert model.latent_channels == 256

        # For input_size=1024: latent_depth = log2(1024) - 2 = 10 - 2 = 8
        # latent_channels = 2^(8 + 2) = 2^10 = 1024
        model = ProgressiveDecoder(input_size=1024, output_size=512)
        assert model.latent_channels == 1024

    def test_automatic_num_upsample_layers(self, device):
        """Test automatic upsampling layer count computation."""
        # From 4x4 to 128x128: 4 -> 8 -> 16 -> 32 -> 64 -> 128 (5 layers)
        model = ProgressiveDecoder(input_size=256, output_size=128)
        assert model.num_upsample_layers == 5

        # From 4x4 to 512x512: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 (7 layers)
        model = ProgressiveDecoder(input_size=1024, output_size=512)
        assert model.num_upsample_layers == 7

    def test_automatic_channel_progression(self, device):
        """Test automatic channel progression generation."""
        model = ProgressiveDecoder(input_size=256, output_size=128)

        # Check progression ends with 1
        assert model.channel_progression[-1] == 1

        # Check progression is decreasing (monotonic)
        for i in range(len(model.channel_progression) - 1):
            assert model.channel_progression[i] >= model.channel_progression[i + 1]

    def test_output_size_defaults_to_input_size(self, device):
        """Test that output_size defaults to input_size when not specified."""
        model = ProgressiveDecoder(input_size=256)
        assert model.output_size == 256

    @pytest.mark.parametrize(
        "input_size,output_size",
        [
            (64, 64),
            (128, 64),
            (256, 128),
            (512, 256),
            (1024, 512),
            (2048, 1024),
        ],
    )
    def test_various_size_combinations(self, input_size, output_size, device):
        """Test model creation with various valid size combinations."""
        model = ProgressiveDecoder(input_size=input_size, output_size=output_size)
        model = model.to(device)

        # Verify model can be created
        assert model.input_size == input_size
        assert model.output_size == output_size

        # Verify forward pass works
        with torch.no_grad():
            output = model()
            assert output.shape == (1, 1, input_size, input_size)


class TestProgressiveDecoderManualConfiguration:
    """Test manual configuration modes."""

    def test_manual_latent_channels(self, device):
        """Test manually specifying latent channels."""
        model = ProgressiveDecoder(
            input_size=256,
            output_size=128,
            latent_channels=512,  # Override automatic value
        )

        assert model.latent_channels == 512

    def test_manual_channel_progression(self, device):
        """Test manually specifying channel progression."""
        custom_progression = [256, 128, 64, 32, 16, 8, 4, 2, 1]
        model = ProgressiveDecoder(
            input_size=256, output_size=128, channel_progression=custom_progression
        )

        assert model.channel_progression == custom_progression

    def test_manual_num_upsample_layers(self, device):
        """Test manually specifying number of upsampling layers."""
        model = ProgressiveDecoder(input_size=256, output_size=128, num_upsample_layers=4)

        assert model.num_upsample_layers == 4

    def test_full_manual_configuration(self, device):
        """Test full manual control over all parameters."""
        model = ProgressiveDecoder(
            input_size=256,
            output_size=128,
            latent_channels=512,
            channel_progression=[256, 128, 64, 32, 16, 8, 4, 2, 1],
            num_upsample_layers=5,
            use_bn=False,
            output_activation="tanh",
        )

        assert model.latent_channels == 512
        assert model.channel_progression == [256, 128, 64, 32, 16, 8, 4, 2, 1]
        assert model.num_upsample_layers == 5


class TestProgressiveDecoderValidation:
    """Test parameter validation and error handling."""

    def test_invalid_input_size_not_power_of_2(self, device):
        """Test that non-power-of-2 input_size raises error."""
        with pytest.raises(ValueError, match="input_size must be power of 2"):
            ProgressiveDecoder(input_size=100, output_size=50)

    def test_invalid_output_size_exceeds_input_size(self, device):
        """Test that output_size > input_size raises error."""
        with pytest.raises(ValueError, match="output_size.*cannot exceed.*input_size"):
            ProgressiveDecoder(input_size=256, output_size=512)

    def test_warning_output_size_not_power_of_2(self, device):
        """Test warning when output_size is not power of 2."""
        with pytest.warns(UserWarning, match="output_size.*is not a power of 2"):
            ProgressiveDecoder(input_size=256, output_size=100)

    def test_invalid_channel_progression_not_ending_with_1(self, device):
        """Test that channel_progression must end with 1."""
        with pytest.raises(ValueError, match="channel_progression must end with 1"):
            ProgressiveDecoder(
                input_size=256,
                output_size=128,
                channel_progression=[128, 64, 32, 16],  # Doesn't end with 1
            )

    def test_invalid_channel_progression_too_short(self, device):
        """Test that channel_progression must have at least 2 elements."""
        with pytest.raises(ValueError, match="channel_progression must have at least 2 elements"):
            ProgressiveDecoder(
                input_size=256,
                output_size=128,
                channel_progression=[1],  # Too short
            )

    def test_deprecated_parameters_warning(self, device):
        """Test that deprecated parameters trigger deprecation warning."""
        with pytest.warns(DeprecationWarning, match="deprecated.*will be removed in v2.0"):
            ProgressiveDecoder(
                input_size=256,
                output_size=128,
                use_leaky=False,  # Deprecated parameter
            )


class TestProgressiveDecoderForward:
    """Test forward pass and output shapes."""

    def test_forward_pass_shape(self, device):
        """Test that forward pass produces correct output shape."""
        model = ProgressiveDecoder(input_size=256, output_size=128)
        model = model.to(device)

        with torch.no_grad():
            output = model()

        assert output.shape == (1, 1, 256, 256)
        assert output.dtype == torch.float32

    def test_forward_pass_no_input_needed(self, device):
        """Test that forward pass works without input (decoder-only)."""
        model = ProgressiveDecoder(input_size=256, output_size=128)
        model = model.to(device)

        with torch.no_grad():
            output = model()  # No input argument

        assert output is not None

    def test_forward_pass_deterministic_without_training(self, device):
        """Test that forward pass is deterministic in eval mode."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output1 = model()
            output2 = model()

        # Should be identical in eval mode with same latent vector
        assert torch.allclose(output1, output2)

    @pytest.mark.parametrize("input_size", [64, 128, 256, 512])
    def test_forward_pass_various_sizes(self, input_size, device):
        """Test forward pass with various input sizes."""
        model = ProgressiveDecoder(input_size=input_size, output_size=input_size // 2)
        model = model.to(device)

        with torch.no_grad():
            output = model()

        assert output.shape == (1, 1, input_size, input_size)


class TestProgressiveDecoderInferenceOptimization:
    """Test inference optimization features."""

    def test_prepare_for_inference(self, device):
        """Test prepare_for_inference sets eval mode and freezes parameters."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)

        # Before preparation
        assert model.training is True

        # Prepare for inference
        model.prepare_for_inference(free_memory=False)

        # After preparation
        assert model.training is False
        for param in model.parameters():
            assert param.requires_grad is False

    def test_generate_fp32(self, device):
        """Test generate_fp32 produces FP32 output."""
        model = ProgressiveDecoder(input_size=128, output_size=64, use_amp=True)
        model = model.to(device)

        with torch.no_grad():
            output = model.generate_fp32()

        assert output.dtype == torch.float32

    def test_compile_method_available(self, device):
        """Test compile method exists and can be called."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)

        # Should not raise error (will warn if torch.compile not available)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.compile(mode="default")

        assert result is model  # Should return self for chaining

    def test_enable_gradient_checkpointing(self, device):
        """Test gradient checkpointing can be enabled."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)

        model.enable_gradient_checkpointing()

        assert hasattr(model, "_use_gradient_checkpointing")
        assert model._use_gradient_checkpointing is True


class TestProgressiveDecoderBenchmarking:
    """Test benchmarking functionality."""

    def test_benchmark_runs(self, device):
        """Test that benchmark method runs without error."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)

        results = model.benchmark(num_iterations=5, warmup=2, measure_memory=False)

        assert "avg_time_ms" in results
        assert "fps" in results
        assert "num_parameters" in results
        assert results["avg_time_ms"] > 0
        assert results["fps"] > 0
        assert results["num_parameters"] > 0

    def test_benchmark_returns_correct_keys(self, device):
        """Test that benchmark returns expected result keys."""
        model = ProgressiveDecoder(input_size=64, output_size=32)
        model = model.to(device)

        results = model.benchmark(num_iterations=3, warmup=1, measure_memory=False)

        required_keys = ["avg_time_ms", "fps", "num_parameters"]
        for key in required_keys:
            assert key in results


class TestProgressiveDecoderFromConfig:
    """Test configuration-driven creation."""

    def test_from_config_basic(self, device):
        """Test creating model from NetworkConfig."""
        config = NetworkConfig(input_size=256, output_size=128, latent_channels=256)

        model = ProgressiveDecoder.from_config(config)
        model = model.to(device)

        assert model.input_size == 256
        assert model.output_size == 128

    def test_from_config_with_custom_parameters(self, device):
        """Test creating model from NetworkConfig with custom parameters."""
        config = NetworkConfig(
            input_size=512,
            output_size=256,
            latent_channels=512,
            activation="relu",
            use_batch_norm=True,
            init_method="xavier",
        )

        model = ProgressiveDecoder.from_config(config)
        model = model.to(device)

        assert model.input_size == 512
        assert model.output_size == 256
        assert model.latent_channels == 512

    def test_from_config_forward_pass(self, device):
        """Test that model from config can perform forward pass."""
        config = NetworkConfig(input_size=128, output_size=64)
        model = ProgressiveDecoder.from_config(config)
        model = model.to(device)

        with torch.no_grad():
            output = model()

        assert output.shape == (1, 1, 128, 128)


class TestProgressiveDecoderAMP:
    """Test Automatic Mixed Precision support."""

    def test_amp_mode_disabled_by_default(self, device):
        """Test that AMP is disabled by default."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        assert model.use_amp is False

    def test_amp_mode_can_be_enabled(self, device):
        """Test that AMP can be enabled."""
        model = ProgressiveDecoder(input_size=128, output_size=64, use_amp=True)
        assert model.use_amp is True

    def test_forward_with_amp(self, device):
        """Test forward pass with AMP enabled."""
        model = ProgressiveDecoder(input_size=128, output_size=64, use_amp=True)
        model = model.to(device)

        # Should not raise error (AMP handled internally)
        with torch.no_grad():
            output = model()

        assert output is not None


class TestProgressiveDecoderCheckpointCompatibility:
    """Test checkpoint loading and state dict compatibility."""

    def test_state_dict_save_and_load(self, device):
        """Test that state dict can be saved and loaded."""
        model1 = ProgressiveDecoder(input_size=128, output_size=64)
        model1 = model1.to(device)

        # Save state dict
        state_dict = model1.state_dict()

        # Create new model and load state dict
        model2 = ProgressiveDecoder(input_size=128, output_size=64)
        model2 = model2.to(device)
        model2.load_state_dict(state_dict)

        # Verify outputs are identical
        model1.eval()
        model2.eval()
        with torch.no_grad():
            output1 = model1()
            output2 = model2()

        assert torch.allclose(output1, output2)

    def test_checkpoint_format_compatibility(self, device):
        """Test that checkpoint format is compatible between old and new names."""
        # Create model with ProgressiveDecoder
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)

        # Save checkpoint in typical format
        checkpoint = {
            "model": model.state_dict(),
            "epoch": 10,
        }

        # Load into new model
        model2 = ProgressiveDecoder(input_size=128, output_size=64)
        model2 = model2.to(device)
        model2.load_state_dict(checkpoint["model"])

        # Should work without issues
        assert checkpoint["epoch"] == 10


class TestProgressiveDecoderOutputActivations:
    """Test different output activation functions."""

    @pytest.mark.parametrize("activation", ["sigmoid", "tanh", "relu", "none"])
    def test_output_activations(self, activation, device):
        """Test model with different output activations."""
        model = ProgressiveDecoder(input_size=128, output_size=64, output_activation=activation)
        model = model.to(device)

        with torch.no_grad():
            output = model()

        assert output.shape == (1, 1, 128, 128)

        # Check activation ranges
        if activation == "sigmoid":
            assert output.min() >= 0 and output.max() <= 1
        elif activation == "tanh":
            assert output.min() >= -1 and output.max() <= 1


class TestProgressiveDecoderBatchNorm:
    """Test batch normalization configurations."""

    def test_with_batch_norm(self, device):
        """Test model creation with batch normalization enabled."""
        model = ProgressiveDecoder(input_size=128, output_size=64, use_bn=True)
        model = model.to(device)

        # Check that ConditionalBatchNorm layers exist
        has_bn = False
        for module in model.modules():
            if "BatchNorm" in module.__class__.__name__:
                has_bn = True
                break

        assert has_bn is True

    def test_without_batch_norm(self, device):
        """Test model creation with batch normalization disabled."""
        model = ProgressiveDecoder(input_size=128, output_size=64, use_bn=False)
        model = model.to(device)

        # Model should still work
        with torch.no_grad():
            output = model()

        assert output.shape == (1, 1, 128, 128)


class TestProgressiveDecoderParameterCount:
    """Test parameter counting and model size."""

    def test_parameter_count_positive(self, device):
        """Test that model has positive parameter count."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_parameter_count_increases_with_latent_channels(self, device):
        """Test that parameter count increases with larger latent channels."""
        model_small = ProgressiveDecoder(input_size=128, output_size=64, latent_channels=64)
        model_large = ProgressiveDecoder(input_size=128, output_size=64, latent_channels=256)

        params_small = sum(p.numel() for p in model_small.parameters())
        params_large = sum(p.numel() for p in model_large.parameters())

        assert params_large > params_small


class TestProgressiveDecoderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_size(self, device):
        """Test model with minimum valid size (64x64)."""
        model = ProgressiveDecoder(input_size=64, output_size=32)
        model = model.to(device)

        with torch.no_grad():
            output = model()

        assert output.shape == (1, 1, 64, 64)

    def test_equal_input_output_size(self, device):
        """Test model with input_size == output_size."""
        model = ProgressiveDecoder(input_size=256, output_size=256)
        model = model.to(device)

        with torch.no_grad():
            output = model()

        assert output.shape == (1, 1, 256, 256)

    def test_training_mode_gradient_flow(self, device):
        """Test that gradients flow in training mode."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)
        model.train()

        output = model()
        loss = output.mean()
        loss.backward()

        # Check that gradients exist for input_vec
        assert model.input_vec.grad is not None
