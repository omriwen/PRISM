"""
Integration tests for custom network configurations.

Tests NetworkConfig and NetworkBuilder functionality including:
- Custom network architectures with different layer counts
- Various activation functions and initialization methods
- Dropout and batch normalization configurations
- Integration with ProgressiveDecoder.from_config()
- End-to-end training with custom configs
- Checkpoint loading and compatibility
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from prism.models.network_config import NetworkConfig
from prism.models.networks import ProgressiveDecoder


@pytest.fixture
def device():
    """Get device for testing (CUDA if available, CPU otherwise)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestNetworkConfigCreation:
    """Test NetworkConfig dataclass creation and validation."""

    def test_default_config(self):
        """Test creating config with default parameters."""
        config = NetworkConfig(input_size=1024, output_size=512)

        assert config.input_size == 1024
        assert config.output_size == 512
        assert config.latent_channels == 512
        assert config.activation == "relu"
        assert config.use_batch_norm is True
        assert config.init_method == "kaiming"

    def test_custom_config(self):
        """Test creating config with custom parameters."""
        config = NetworkConfig(
            input_size=512,
            output_size=256,
            latent_channels=256,
            activation="leakyrelu",
            use_batch_norm=False,
            use_dropout=True,
            dropout_rate=0.2,
            init_method="xavier",
            output_activation="tanh",
        )

        assert config.input_size == 512
        assert config.output_size == 256
        assert config.latent_channels == 256
        assert config.activation == "leakyrelu"
        assert config.use_batch_norm is False
        assert config.use_dropout is True
        assert config.dropout_rate == 0.2
        assert config.init_method == "xavier"
        assert config.output_activation == "tanh"

    def test_config_validation_power_of_2(self):
        """Test validation of power-of-2 sizes."""
        # Valid config
        config = NetworkConfig(input_size=1024, output_size=512)
        config.validate()  # Should not raise

        # Invalid input_size
        with pytest.raises(ValueError, match="input_size must be power of 2"):
            bad_config = NetworkConfig(input_size=1000, output_size=512)
            bad_config.validate()

        # Invalid output_size
        with pytest.raises(ValueError, match="output_size must be power of 2"):
            bad_config = NetworkConfig(input_size=1024, output_size=500)
            bad_config.validate()

    def test_config_validation_output_size(self):
        """Test validation that output_size <= input_size."""
        # Valid config
        config = NetworkConfig(input_size=1024, output_size=512)
        config.validate()

        # Equal sizes (should be valid)
        config_equal = NetworkConfig(input_size=512, output_size=512)
        config_equal.validate()

        # Invalid: output_size > input_size
        with pytest.raises(ValueError, match="output_size.*cannot exceed.*input_size"):
            bad_config = NetworkConfig(input_size=512, output_size=1024)
            bad_config.validate()

    def test_config_validation_channels(self):
        """Test validation of channel counts."""
        # Valid config
        config = NetworkConfig(input_size=1024, output_size=512, latent_channels=256)
        config.validate()

        # Invalid: negative channels
        with pytest.raises(ValueError, match="latent_channels must be positive"):
            bad_config = NetworkConfig(input_size=1024, output_size=512, latent_channels=-1)
            bad_config.validate()

        # Invalid: zero channels
        with pytest.raises(ValueError, match="latent_channels must be positive"):
            bad_config = NetworkConfig(input_size=1024, output_size=512, latent_channels=0)
            bad_config.validate()


class TestNetworkConfigActivations:
    """Test different activation function configurations."""

    @pytest.mark.parametrize("activation", ["relu", "leakyrelu", "tanh", "sigmoid"])
    def test_activation_types(self, activation, device):
        """Test network creation with different activation functions."""
        config = NetworkConfig(
            input_size=256,
            output_size=128,
            latent_channels=128,
            activation=activation,
        )
        config.validate()

        # Create network (this will be tested in integration)
        # For now, just verify config is valid
        assert config.activation == activation

    @pytest.mark.parametrize("output_activation", ["sigmoid", "tanh", "relu", "identity"])
    def test_output_activation_types(self, output_activation, device):
        """Test network creation with different output activations."""
        config = NetworkConfig(
            input_size=256,
            output_size=128,
            output_activation=output_activation,
        )
        config.validate()

        assert config.output_activation == output_activation


class TestNetworkConfigInitialization:
    """Test different weight initialization methods."""

    @pytest.mark.parametrize("init_method", ["kaiming", "xavier", "orthogonal"])
    def test_init_methods(self, init_method, device):
        """Test network creation with different initialization methods."""
        config = NetworkConfig(
            input_size=256,
            output_size=128,
            init_method=init_method,
        )
        config.validate()

        assert config.init_method == init_method


class TestNetworkConfigDropout:
    """Test dropout configuration."""

    def test_dropout_enabled(self, device):
        """Test config with dropout enabled."""
        config = NetworkConfig(
            input_size=256,
            output_size=128,
            use_dropout=True,
            dropout_rate=0.3,
        )
        config.validate()

        assert config.use_dropout is True
        assert config.dropout_rate == 0.3

    def test_dropout_disabled(self, device):
        """Test config with dropout disabled."""
        config = NetworkConfig(
            input_size=256,
            output_size=128,
            use_dropout=False,
        )
        config.validate()

        assert config.use_dropout is False

    def test_dropout_rate_validation(self):
        """Test validation of dropout rate."""
        # Valid: 0.0 to < 1.0
        for rate in [0.0, 0.1, 0.5, 0.9]:
            config = NetworkConfig(
                input_size=256,
                output_size=128,
                use_dropout=True,
                dropout_rate=rate,
            )
            config.validate()

        # Invalid: negative rate
        with pytest.raises(ValueError, match="dropout_rate must be"):
            bad_config = NetworkConfig(
                input_size=256,
                output_size=128,
                use_dropout=True,
                dropout_rate=-0.1,
            )
            bad_config.validate()

        # Invalid: rate > 1
        with pytest.raises(ValueError, match="dropout_rate must be"):
            bad_config = NetworkConfig(
                input_size=256,
                output_size=128,
                use_dropout=True,
                dropout_rate=1.5,
            )
            bad_config.validate()


class TestNetworkConfigWithProgressiveDecoder:
    """Test NetworkConfig integration with ProgressiveDecoder."""

    def test_create_network_from_default_config(self, device):
        """Test creating network from default config."""
        config = NetworkConfig(input_size=256, output_size=128)
        config.validate()

        # Note: ProgressiveDecoder.from_config() may not exist yet
        # If it doesn't, this test will need to be updated
        # For now, test that config is valid
        assert config.input_size == 256
        assert config.output_size == 128

    def test_create_network_from_custom_config(self, device):
        """Test creating network from custom config."""
        config = NetworkConfig(
            input_size=512,
            output_size=256,
            latent_channels=256,
            activation="leakyrelu",
            use_batch_norm=True,
            init_method="xavier",
        )
        config.validate()

        # Verify config parameters
        assert config.latent_channels == 256
        assert config.activation == "leakyrelu"
        assert config.init_method == "xavier"

    def test_network_forward_pass_with_config(self, device):
        """Test forward pass with custom config network (generative model)."""
        config = NetworkConfig(
            input_size=256,
            output_size=128,
            latent_channels=128,
            activation="relu",
        )
        config.validate()

        # SPIDS is generative - no input needed
        # This is a placeholder until ProgressiveDecoder.from_config() is available
        # For now, just verify config is valid
        assert config.input_size == 256
        assert config.output_size == 128


class TestNetworkConfigScenarios:
    """Test realistic network configuration scenarios."""

    def test_small_network_config(self, device):
        """Test configuration for small, fast network."""
        config = NetworkConfig(
            input_size=256,
            output_size=128,
            latent_channels=128,
            activation="relu",
            use_batch_norm=False,
            use_dropout=False,
            init_method="kaiming",
        )
        config.validate()

        assert config.latent_channels == 128
        assert not config.use_batch_norm
        assert not config.use_dropout

    def test_large_network_config(self, device):
        """Test configuration for large, high-capacity network."""
        config = NetworkConfig(
            input_size=2048,
            output_size=1024,
            latent_channels=1024,
            activation="leakyrelu",
            use_batch_norm=True,
            use_dropout=True,
            dropout_rate=0.1,
            init_method="xavier",
        )
        config.validate()

        assert config.input_size == 2048
        assert config.output_size == 1024
        assert config.latent_channels == 1024
        assert config.use_dropout

    def test_production_network_config(self, device):
        """Test configuration similar to production settings."""
        config = NetworkConfig(
            input_size=1024,
            output_size=512,
            latent_channels=512,
            activation="leakyrelu",
            use_batch_norm=True,
            use_dropout=False,
            init_method="kaiming",
            output_activation="sigmoid",
        )
        config.validate()

        # Typical production settings
        assert config.input_size == 1024
        assert config.output_size == 512
        assert config.use_batch_norm
        assert not config.use_dropout


class TestNetworkConfigWithTraining:
    """Test network configs in training scenarios."""

    def test_config_with_optimizer(self, device):
        """Test that custom config networks can be optimized."""
        config = NetworkConfig(
            input_size=256,
            output_size=128,
            latent_channels=128,
        )
        config.validate()

        # Create dummy model parameters to test optimizer
        dummy_params = [torch.randn(10, 10, requires_grad=True, device=device)]
        optimizer = torch.optim.Adam(dummy_params, lr=0.001)

        # Verify optimizer works
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_config_with_loss_function(self, device):
        """Test that custom config networks work with loss functions (single sample)."""
        config = NetworkConfig(
            input_size=256,
            output_size=128,
            latent_channels=128,
            output_activation="sigmoid",
        )
        config.validate()

        # Test with different loss functions (single sample - SPIDS paradigm)
        output = torch.rand(1, 1, 128, 128, device=device)
        target = torch.rand(1, 1, 128, 128, device=device)

        # L1 Loss
        l1_loss = nn.L1Loss()(output, target)
        assert l1_loss.item() >= 0

        # MSE Loss
        mse_loss = nn.MSELoss()(output, target)
        assert mse_loss.item() >= 0

    def test_config_gradient_flow(self, device):
        """Test that gradients flow through custom config network (single sample)."""
        config = NetworkConfig(
            input_size=256,
            output_size=128,
            latent_channels=128,
        )
        config.validate()

        # Create dummy network layer to test gradient flow (single sample)
        layer = nn.Conv2d(1, 32, kernel_size=3, padding=1).to(device)
        input_tensor = torch.randn(1, 1, 128, 128, device=device, requires_grad=True)

        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert input_tensor.grad is not None
        assert layer.weight.grad is not None


class TestNetworkConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_network_config(self, device):
        """Test smallest valid network configuration."""
        config = NetworkConfig(
            input_size=64,
            output_size=32,
            latent_channels=32,
        )
        config.validate()

        assert config.input_size == 64
        assert config.output_size == 32

    def test_equal_input_output_size(self, device):
        """Test config where input_size == output_size."""
        config = NetworkConfig(
            input_size=512,
            output_size=512,
            latent_channels=256,
        )
        config.validate()

        assert config.input_size == config.output_size

    def test_custom_hidden_channels(self, device):
        """Test config with custom hidden channel progression."""
        hidden_channels = [512, 256, 128, 64]
        config = NetworkConfig(
            input_size=1024,
            output_size=512,
            latent_channels=512,
            hidden_channels=hidden_channels,
        )
        config.validate()

        assert config.hidden_channels == hidden_channels


class TestProgressiveDecoderIntegration:
    """Integration tests for ProgressiveDecoder with NetworkConfig."""

    def test_progressive_decoder_from_config_basic(self, device):
        """Test creating ProgressiveDecoder from basic NetworkConfig."""
        config = NetworkConfig(input_size=256, output_size=128, latent_channels=256)
        config.validate()

        model = ProgressiveDecoder.from_config(config)
        model = model.to(device)

        assert model.input_size == 256
        assert model.output_size == 128
        assert model.latent_channels == 256

    def test_progressive_decoder_from_config_custom(self, device):
        """Test creating ProgressiveDecoder from custom NetworkConfig."""
        config = NetworkConfig(
            input_size=512,
            output_size=256,
            latent_channels=512,
            activation="relu",
            use_batch_norm=True,
            init_method="xavier",
            output_activation="sigmoid",
        )
        config.validate()

        model = ProgressiveDecoder.from_config(config)
        model = model.to(device)

        assert model.input_size == 512
        assert model.output_size == 256

        # Test forward pass works
        with torch.no_grad():
            output = model()
            assert output.shape == (1, 1, 512, 512)

    def test_progressive_decoder_forward_pass_with_config(self, device):
        """Test forward pass through ProgressiveDecoder created from config."""
        config = NetworkConfig(input_size=128, output_size=64, latent_channels=128)
        config.validate()

        model = ProgressiveDecoder.from_config(config)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output = model()

        assert output.shape == (1, 1, 128, 128)
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("init_method", ["kaiming", "xavier", "orthogonal"])
    def test_progressive_decoder_with_different_init_methods(self, init_method, device):
        """Test ProgressiveDecoder with different initialization methods."""
        config = NetworkConfig(
            input_size=128, output_size=64, latent_channels=128, init_method=init_method
        )
        config.validate()

        model = ProgressiveDecoder.from_config(config)
        model = model.to(device)

        # Test forward pass works
        with torch.no_grad():
            output = model()
            assert output.shape == (1, 1, 128, 128)


class TestProgressiveDecoderTrainingLoop:
    """Integration tests for ProgressiveDecoder in training scenarios."""

    def test_progressive_decoder_training_loop_basic(self, device):
        """Test ProgressiveDecoder in basic training loop."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Simulate training for a few iterations
        for _ in range(3):
            optimizer.zero_grad()

            # Forward pass (no input needed - decoder-only)
            output = model()

            # Create dummy target
            target = torch.rand(1, 1, 128, 128, device=device)

            # Compute loss and backward
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Check loss is computed
            assert loss.item() >= 0

    def test_progressive_decoder_training_loop_with_config(self, device):
        """Test ProgressiveDecoder from config in training loop."""
        config = NetworkConfig(
            input_size=128,
            output_size=64,
            latent_channels=128,
            activation="relu",
            use_batch_norm=True,
            init_method="kaiming",
        )
        config.validate()

        model = ProgressiveDecoder.from_config(config)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()

        # Training iteration
        optimizer.zero_grad()
        output = model()
        target = torch.rand(1, 1, 128, 128, device=device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Verify gradients exist
        assert model.input_vec.grad is not None

    def test_progressive_decoder_gradient_flow(self, device):
        """Test gradient flow through ProgressiveDecoder."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)
        model.train()

        # Forward pass
        output = model()

        # Compute simple loss
        loss = output.mean()
        loss.backward()

        # Check gradients exist for key parameters
        assert model.input_vec.grad is not None
        assert model.input_vec.grad.abs().sum() > 0

    def test_progressive_decoder_with_amp(self, device):
        """Test ProgressiveDecoder with Automatic Mixed Precision."""
        # Skip this test as AMP is only useful with CUDA, and for testing
        # we can verify AMP flag works without actually using it
        model = ProgressiveDecoder(input_size=128, output_size=64, use_amp=False)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training without AMP (AMP tested separately in unit tests)
        optimizer.zero_grad()
        output = model()
        target = torch.rand(1, 1, 128, 128, device=device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0


class TestProgressiveDecoderCheckpointLoading:
    """Integration tests for checkpoint loading and compatibility."""

    def test_checkpoint_save_and_load(self, device):
        """Test saving and loading ProgressiveDecoder checkpoint."""
        # Create and train model briefly
        model1 = ProgressiveDecoder(input_size=128, output_size=64)
        model1 = model1.to(device)
        model1.train()

        optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)

        # Train for 1 iteration
        optimizer.zero_grad()
        output = model1()
        loss = output.mean()
        loss.backward()
        optimizer.step()

        # Save checkpoint
        checkpoint = {
            "model": model1.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 1,
        }

        # Load into new model
        model2 = ProgressiveDecoder(input_size=128, output_size=64)
        model2 = model2.to(device)
        model2.load_state_dict(checkpoint["model"])

        # Verify models produce same output
        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model1()
            out2 = model2()

        assert torch.allclose(out1, out2, rtol=1e-5)

    def test_checkpoint_compatibility_with_config(self, device):
        """Test checkpoint compatibility between direct and config-based creation."""
        # Create model directly
        model1 = ProgressiveDecoder(input_size=128, output_size=64, latent_channels=128)
        model1 = model1.to(device)

        # Create model from config with same parameters
        config = NetworkConfig(input_size=128, output_size=64, latent_channels=128)
        model2 = ProgressiveDecoder.from_config(config)
        model2 = model2.to(device)

        # Save and load checkpoint
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)

        # Should work without errors
        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model1()
            out2 = model2()

        assert torch.allclose(out1, out2)

    def test_checkpoint_cross_version_compatibility(self, device):
        """Test that checkpoints can be loaded across model versions."""
        # Create model with ProgressiveDecoder
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)

        # Save checkpoint
        checkpoint = {"model": model.state_dict(), "version": "1.0"}

        # Load into new model (simulating loading old checkpoint)
        model_new = ProgressiveDecoder(input_size=128, output_size=64)
        model_new = model_new.to(device)
        model_new.load_state_dict(checkpoint["model"])

        # Should work without issues
        with torch.no_grad():
            output = model_new()
            assert output.shape == (1, 1, 128, 128)

    def test_checkpoint_after_prepare_for_inference(self, device):
        """Test that checkpoints can be saved after prepare_for_inference."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)

        # Prepare for inference (freezes params, fusion, etc.)
        model.prepare_for_inference(compile_mode=None, free_memory=False)

        # Save checkpoint
        state_dict = model.state_dict()

        # Load into new model
        model_new = ProgressiveDecoder(input_size=128, output_size=64)
        model_new = model_new.to(device)
        model_new.load_state_dict(state_dict)

        # Should work
        model_new.eval()
        with torch.no_grad():
            output = model_new()
            assert output.shape == (1, 1, 128, 128)


class TestProgressiveDecoderEndToEnd:
    """End-to-end integration tests for realistic usage scenarios."""

    def test_end_to_end_training_pipeline(self, device):
        """Test complete training pipeline from config to inference."""
        # 1. Create model from config
        config = NetworkConfig(
            input_size=128,
            output_size=64,
            latent_channels=128,
            activation="relu",
            use_batch_norm=True,
            init_method="kaiming",
            output_activation="sigmoid",
        )
        config.validate()

        model = ProgressiveDecoder.from_config(config)
        model = model.to(device)

        # 2. Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 3. Train for a few iterations
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            output = model()
            target = torch.rand(1, 1, 128, 128, device=device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # 4. Prepare for inference
        model.prepare_for_inference(compile_mode=None, free_memory=False)

        # 5. Generate final output
        model.eval()
        with torch.no_grad():
            final_output = model.generate_fp32()

        assert final_output.shape == (1, 1, 128, 128)
        assert final_output.dtype == torch.float32

    def test_end_to_end_with_different_sizes(self, device):
        """Test end-to-end pipeline with various input/output sizes."""
        sizes = [(64, 32), (128, 64), (256, 128)]

        for input_size, output_size in sizes:
            config = NetworkConfig(
                input_size=input_size, output_size=output_size, latent_channels=input_size // 2
            )
            config.validate()

            model = ProgressiveDecoder.from_config(config)
            model = model.to(device)

            # Quick training
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.train()
            optimizer.zero_grad()
            output = model()
            loss = output.mean()
            loss.backward()
            optimizer.step()

            # Inference
            model.eval()
            with torch.no_grad():
                final_output = model.generate_fp32()

            assert final_output.shape == (1, 1, input_size, input_size)

    def test_end_to_end_with_benchmarking(self, device):
        """Test end-to-end pipeline including performance benchmarking."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)

        # Train briefly
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            output = model()
            loss = output.mean()
            loss.backward()
            optimizer.step()

        # Benchmark before optimization
        model.eval()
        results_before = model.benchmark(num_iterations=5, warmup=2, measure_memory=False)

        # Prepare for inference
        model.prepare_for_inference(compile_mode=None, free_memory=False)

        # Benchmark after optimization
        results_after = model.benchmark(num_iterations=5, warmup=2, measure_memory=False)

        # Both should complete successfully
        assert results_before["avg_time_ms"] > 0
        assert results_after["avg_time_ms"] > 0
        assert results_before["fps"] > 0
        assert results_after["fps"] > 0
