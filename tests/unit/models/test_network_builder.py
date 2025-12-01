"""
Unit tests for NetworkBuilder.

Tests network construction from configuration, initialization methods,
and architecture summary generation.
"""

from __future__ import annotations

import pytest
import torch

from prism.models.network_builder import NetworkBuilder
from prism.models.network_config import NetworkConfig
from prism.models.networks import ProgressiveDecoder


class TestNetworkBuilderConstruction:
    """Test network builder construction."""

    def test_build_default_config(self):
        """Test building network with default configuration."""
        config = NetworkConfig(input_size=512, output_size=256)
        builder = NetworkBuilder(config)
        network = builder.build()

        assert isinstance(network, ProgressiveDecoder)
        assert network.input_size == 512
        assert network.output_size == 256

    def test_build_custom_config(self):
        """Test building network with custom configuration."""
        config = NetworkConfig(
            input_size=1024,
            output_size=512,
            latent_channels=1024,
            use_batch_norm=False,
            output_activation="tanh",
        )
        builder = NetworkBuilder(config)
        network = builder.build()

        assert isinstance(network, ProgressiveDecoder)
        assert network.input_size == 1024
        assert network.output_size == 512

    def test_build_validates_config(self):
        """Test that build validates configuration."""
        config = NetworkConfig(input_size=1000, output_size=512)  # Invalid: not power of 2
        with pytest.raises(ValueError, match="input_size must be power of 2"):
            NetworkBuilder(config)

    def test_forward_pass(self):
        """Test that built network can perform forward pass."""
        config = NetworkConfig(input_size=512, output_size=256)
        builder = NetworkBuilder(config)
        network = builder.build()

        # Test forward pass
        output = network()
        assert output.shape == (1, 1, 512, 512)  # Padded to input_size

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
    def test_build_on_device(self, device):
        """Test building network on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = NetworkConfig(input_size=256, output_size=128)
        builder = NetworkBuilder(config)
        network = builder.build().to(device)

        output = network()
        assert output.device.type == device


class TestNetworkBuilderInitialization:
    """Test weight initialization methods."""

    @pytest.mark.parametrize("init_method", ["kaiming", "xavier", "orthogonal"])
    def test_initialization_methods(self, init_method):
        """Test different initialization methods."""
        config = NetworkConfig(input_size=256, output_size=128, init_method=init_method)
        builder = NetworkBuilder(config)
        network = builder.build()

        # Check that weight parameters are initialized (not all zero)
        # Bias parameters can be zero by design
        weight_count = 0
        for name, param in network.named_parameters():
            if "weight" in name:
                weight_count += 1
                # At least some weights should be non-zero
                assert param.abs().max() > 0

        assert weight_count > 0

    def test_kaiming_default(self):
        """Test that kaiming is the default initialization."""
        config = NetworkConfig(input_size=256, output_size=128)
        builder = NetworkBuilder(config)
        network = builder.build()

        # Network should be constructed successfully
        assert isinstance(network, ProgressiveDecoder)


class TestNetworkBuilderSummary:
    """Test architecture summary generation."""

    def test_get_architecture_summary(self):
        """Test that architecture summary is generated."""
        config = NetworkConfig(input_size=1024, output_size=512)
        builder = NetworkBuilder(config)
        summary = builder.get_architecture_summary()

        assert "ProgressiveDecoder Architecture:" in summary
        assert "Input Size: 1024 x 1024" in summary
        assert "Output Size: 512 x 512" in summary
        assert "Architecture Type: Decoder-only" in summary
        assert "Activation: relu" in summary

    def test_summary_includes_upsampling_layers(self):
        """Test that summary includes upsampling layer count."""
        config = NetworkConfig(input_size=1024, output_size=512)
        builder = NetworkBuilder(config)
        summary = builder.get_architecture_summary()

        assert "Upsampling Layers:" in summary

    def test_estimate_parameters(self):
        """Test parameter count estimation."""
        config = NetworkConfig(input_size=512, output_size=256)
        builder = NetworkBuilder(config)
        param_count = builder.estimate_parameters()

        assert param_count > 0
        assert isinstance(param_count, int)

    def test_parameter_count_matches_network(self):
        """Test that estimated parameters match actual network."""
        config = NetworkConfig(input_size=512, output_size=256)
        builder = NetworkBuilder(config)

        estimated = builder.estimate_parameters()
        network = builder.build()
        actual = sum(p.numel() for p in network.parameters() if p.requires_grad)

        assert estimated == actual


class TestNetworkBuilderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_network(self):
        """Test building very small network."""
        config = NetworkConfig(input_size=4, output_size=4)
        builder = NetworkBuilder(config)
        network = builder.build()

        output = network()
        assert output.shape == (1, 1, 4, 4)

    def test_large_network(self):
        """Test building large network."""
        config = NetworkConfig(input_size=2048, output_size=1024)
        builder = NetworkBuilder(config)
        network = builder.build()

        assert isinstance(network, ProgressiveDecoder)

    def test_equal_input_output_size(self):
        """Test network with equal input and output size."""
        config = NetworkConfig(input_size=512, output_size=512)
        builder = NetworkBuilder(config)
        network = builder.build()

        output = network()
        assert output.shape == (1, 1, 512, 512)
