"""
Unit tests for NetworkConfig dataclass.

Tests configuration validation, parameter ranges, and edge cases.
"""

from __future__ import annotations

import pytest

from prism.models.network_config import NetworkConfig


class TestNetworkConfigValidation:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test that valid configuration passes validation."""
        config = NetworkConfig(input_size=1024, output_size=512)
        config.validate()  # Should not raise

    def test_input_size_power_of_2(self):
        """Test that input_size must be power of 2."""
        config = NetworkConfig(input_size=1000, output_size=512)
        with pytest.raises(ValueError, match="input_size must be power of 2"):
            config.validate()

    def test_output_size_power_of_2(self):
        """Test that output_size must be power of 2."""
        config = NetworkConfig(input_size=1024, output_size=500)
        with pytest.raises(ValueError, match="output_size must be power of 2"):
            config.validate()

    def test_output_size_exceeds_input_size(self):
        """Test that output_size cannot exceed input_size."""
        config = NetworkConfig(input_size=512, output_size=1024)
        with pytest.raises(ValueError, match="output_size.*cannot exceed.*input_size"):
            config.validate()

    def test_negative_latent_channels(self):
        """Test that latent_channels must be positive."""
        config = NetworkConfig(input_size=1024, output_size=512, latent_channels=-10)
        with pytest.raises(ValueError, match="latent_channels must be positive"):
            config.validate()

    def test_invalid_dropout_rate(self):
        """Test that dropout_rate must be in [0, 1)."""
        config = NetworkConfig(input_size=1024, output_size=512, dropout_rate=1.5)
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            config.validate()

    def test_invalid_activation(self):
        """Test that activation must be valid."""
        config = NetworkConfig(input_size=1024, output_size=512, activation="invalid")
        with pytest.raises(ValueError, match="activation must be one of"):
            config.validate()

    def test_invalid_init_method(self):
        """Test that init_method must be valid."""
        config = NetworkConfig(input_size=1024, output_size=512, init_method="invalid")
        with pytest.raises(ValueError, match="init_method must be one of"):
            config.validate()


class TestNetworkConfigDefaults:
    """Test default configuration values."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = NetworkConfig(input_size=1024, output_size=512)

        assert config.latent_channels == 512
        assert config.activation == "relu"
        assert config.use_batch_norm is True
        assert config.use_dropout is False
        assert config.dropout_rate == 0.1
        assert config.init_method == "kaiming"
        assert config.output_activation == "sigmoid"

    def test_custom_values(self):
        """Test that custom values override defaults."""
        config = NetworkConfig(
            input_size=1024,
            output_size=512,
            latent_channels=1024,
            activation="tanh",
            use_dropout=True,
            dropout_rate=0.2,
            init_method="xavier",
        )

        assert config.latent_channels == 1024
        assert config.activation == "tanh"
        assert config.use_dropout is True
        assert config.dropout_rate == 0.2
        assert config.init_method == "xavier"


class TestNetworkConfigSummary:
    """Test configuration summary generation."""

    def test_get_summary(self):
        """Test that summary is generated correctly."""
        config = NetworkConfig(input_size=1024, output_size=512)
        summary = config.get_summary()

        assert "Input Size: 1024" in summary
        assert "Output Size: 512" in summary
        assert "Latent Channels: 512" in summary
        assert "Activation: relu" in summary
        assert "Batch Norm: True" in summary
        assert "Dropout: False" in summary

    def test_get_summary_with_dropout(self):
        """Test that summary includes dropout rate when enabled."""
        config = NetworkConfig(input_size=1024, output_size=512, use_dropout=True, dropout_rate=0.3)
        summary = config.get_summary()

        assert "Dropout: True" in summary
        assert "Dropout Rate: 0.3" in summary


class TestNetworkConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_sizes(self):
        """Test minimum valid sizes (2^1 = 2)."""
        config = NetworkConfig(input_size=2, output_size=2)
        config.validate()  # Should not raise

    def test_large_sizes(self):
        """Test large valid sizes."""
        config = NetworkConfig(input_size=4096, output_size=2048)
        config.validate()  # Should not raise

    def test_equal_input_output_size(self):
        """Test that input_size == output_size is valid."""
        config = NetworkConfig(input_size=1024, output_size=1024)
        config.validate()  # Should not raise

    def test_dropout_rate_zero(self):
        """Test that dropout_rate=0 is valid."""
        config = NetworkConfig(input_size=1024, output_size=512, dropout_rate=0.0)
        config.validate()  # Should not raise

    def test_dropout_rate_near_one(self):
        """Test that dropout_rate < 1.0 is valid."""
        config = NetworkConfig(input_size=1024, output_size=512, dropout_rate=0.999)
        config.validate()  # Should not raise
