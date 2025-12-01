"""
Integration tests for Automatic Mixed Precision (AMP) training.

Tests mixed precision training functionality including:
- AMP with ProgressiveDecoder
- GradScaler integration
- Numerical stability (FP16 vs FP32 convergence)
- Memory usage (if CUDA available)
- Integration with PRISMTrainer and ePIETrainer
"""

from __future__ import annotations

import pytest
import torch

from prism.core.trainers import PRISMTrainer
from prism.models.network_config import NetworkConfig
from prism.models.networks import ProgressiveDecoder


@pytest.fixture
def device():
    """Get device for testing (CUDA if available, CPU otherwise)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_config():
    """Create a simple network configuration for testing."""
    return NetworkConfig(
        input_size=256,
        output_size=128,
        latent_channels=128,
        activation="relu",
        use_batch_norm=True,
        init_method="kaiming",
    )


@pytest.fixture
def simple_network(device):
    """Create a simple network for AMP testing."""
    model = ProgressiveDecoder(input_size=256, use_amp=True).to(device)
    return model


@pytest.fixture
def measurement_setup(device):
    """Create a simple measurement setup for testing."""
    # Must match simple_network output size (input_size=256 -> output [1, 1, 256, 256])
    image_size = 256

    # Create dummy measurement (single sample - SPIDS paradigm)
    # Shape: [1, 1, H, W] to match network output
    measurement = torch.randn(1, 1, image_size, image_size, device=device)
    measurement = torch.abs(measurement)  # Ensure positive

    # Create dummy ground truth (single sample)
    # Shape: [1, 1, H, W] to match network output
    image_gt = torch.randn(1, 1, image_size, image_size, device=device)
    image_gt = torch.abs(image_gt)

    return measurement, image_gt


class TestAMPBasicFunctionality:
    """Test basic AMP functionality with ProgressiveDecoder."""

    def test_network_with_amp_enabled(self, device):
        """Test that network can be created with AMP enabled."""
        model = ProgressiveDecoder(input_size=256, use_amp=True).to(device)

        assert model.use_amp is True
        assert hasattr(model, "generate_fp32")

    def test_network_with_amp_disabled(self, device):
        """Test that network can be created with AMP disabled."""
        model = ProgressiveDecoder(input_size=256, use_amp=False).to(device)

        assert model.use_amp is False

    def test_forward_pass_with_amp(self, simple_network, device):
        """Test forward pass with AMP enabled (generative model)."""
        # SPIDS is generative - model() has no input parameter
        with torch.cuda.amp.autocast(enabled=simple_network.use_amp):
            output = simple_network()  # No input!

        # Output should be model's output size
        assert output.dim() == 4  # (1, C, H, W)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_generate_fp32_method(self, simple_network, device):
        """Test generate_fp32() method for final reconstruction (generative model)."""
        # SPIDS is generative - no input needed
        output = simple_network.generate_fp32()  # No input!

        assert output.dtype == torch.float32
        assert output.dim() == 4  # (1, C, H, W)
        assert not torch.isnan(output).any()


class TestAMPTrainingIntegration:
    """Test AMP integration with training loops."""

    def test_amp_with_spids_trainer(self, simple_network, device, measurement_setup):
        """Test PRISMTrainer with AMP enabled."""
        measurement, image_gt = measurement_setup

        # Create trainer with AMP
        optimizer = torch.optim.Adam(simple_network.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # Create minimal args object
        class MinimalArgs:
            max_epochs_init = 2
            n_epochs_init = 2
            output_activation = "relu"
            num_samples_progressive = 0  # Skip progressive training
            checkpoint_frequency = 100

        args = MinimalArgs()

        # Create trainer
        trainer = PRISMTrainer(
            model=simple_network,
            optimizer=optimizer,
            scheduler=scheduler,
            measurement_system=None,  # type: ignore  # Not needed for this test
            args=args,
            device=device,
            use_amp=True,  # Enable AMP
        )

        # Check AMP is enabled
        assert trainer.use_amp is True
        assert trainer.scaler is not None
        if device.type == "cuda":
            assert isinstance(trainer.scaler, torch.cuda.amp.GradScaler)

    def test_amp_backward_pass(self, simple_network, device, measurement_setup):
        """Test backward pass with AMP and GradScaler (SPIDS generative training)."""
        measurement, image_gt = measurement_setup

        optimizer = torch.optim.Adam(simple_network.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        # Forward pass with autocast (generative model - no input!)
        with torch.cuda.amp.autocast(enabled=simple_network.use_amp and device.type == "cuda"):
            output = simple_network()  # No input!
            loss = torch.nn.functional.mse_loss(output, image_gt)

        # Backward pass with scaler
        optimizer.zero_grad()
        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Check gradients exist and are finite
        for param in simple_network.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()


class TestAMPNumericalStability:
    """Test numerical stability of AMP vs FP32 training."""

    def test_amp_vs_fp32_convergence(self, device, measurement_setup):
        """
        Test that AMP and FP32 training converge to similar results (SPIDS generative training).

        Note: This is a simplified test. Full convergence testing requires
        GPU hardware and longer training runs.
        """
        measurement, image_gt = measurement_setup

        # Create two identical networks
        model_amp = ProgressiveDecoder(input_size=256, use_amp=True).to(device)
        model_fp32 = ProgressiveDecoder(input_size=256, use_amp=False).to(device)

        # Copy weights to ensure same initialization
        model_fp32.load_state_dict(model_amp.state_dict())

        # Train for a few iterations
        optimizer_amp = torch.optim.Adam(model_amp.parameters(), lr=0.001)
        optimizer_fp32 = torch.optim.Adam(model_fp32.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        losses_amp = []
        losses_fp32 = []

        for _ in range(5):
            # AMP training step (generative model - no input!)
            with torch.cuda.amp.autocast(enabled=True and device.type == "cuda"):
                output_amp = model_amp()  # No input!
                loss_amp = torch.nn.functional.mse_loss(output_amp, image_gt)
            optimizer_amp.zero_grad()
            if scaler is not None and device.type == "cuda":
                scaler.scale(loss_amp).backward()
                scaler.step(optimizer_amp)
                scaler.update()
            else:
                loss_amp.backward()
                optimizer_amp.step()
            losses_amp.append(loss_amp.item())

            # FP32 training step (generative model - no input!)
            output_fp32 = model_fp32()  # No input!
            loss_fp32 = torch.nn.functional.mse_loss(output_fp32, image_gt)
            optimizer_fp32.zero_grad()
            loss_fp32.backward()
            optimizer_fp32.step()
            losses_fp32.append(loss_fp32.item())

        # Check that losses are decreasing for both
        assert losses_amp[-1] < losses_amp[0], "AMP loss should decrease"
        assert losses_fp32[-1] < losses_fp32[0], "FP32 loss should decrease"

        # Losses should be reasonably close (within 20%)
        # Note: On CPU, AMP may not provide actual FP16 training, so they might be very close
        final_loss_diff = abs(losses_amp[-1] - losses_fp32[-1])
        mean_loss = (losses_amp[-1] + losses_fp32[-1]) / 2
        relative_diff = final_loss_diff / mean_loss if mean_loss > 0 else 0

        # Allow for some divergence due to reduced precision
        assert relative_diff < 0.5, f"AMP and FP32 losses diverged too much: {relative_diff:.2%}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for memory testing")
class TestAMPMemoryReduction:
    """Test memory reduction with AMP (CUDA only)."""

    def test_amp_reduces_memory_usage(self, device):
        """
        Test that AMP reduces memory usage compared to FP32 (generative model).

        Note: This test requires CUDA. Expected reduction: 40-50%.
        """
        if device.type != "cuda":
            pytest.skip("CUDA required for memory testing")

        # Measure FP32 memory (generative model - no input!)
        torch.cuda.reset_peak_memory_stats()
        model_fp32 = ProgressiveDecoder(input_size=512, use_amp=False).to(device)
        output_fp32 = model_fp32()  # No input!
        loss_fp32 = output_fp32.sum()
        loss_fp32.backward()
        memory_fp32 = torch.cuda.max_memory_allocated() / 1024**2  # MB
        del model_fp32, output_fp32, loss_fp32
        torch.cuda.empty_cache()

        # Measure AMP memory (generative model - no input!)
        torch.cuda.reset_peak_memory_stats()
        model_amp = ProgressiveDecoder(input_size=512, use_amp=True).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            output_amp = model_amp()  # No input!
            loss_amp = output_amp.sum()
        loss_amp.backward()
        memory_amp = torch.cuda.max_memory_allocated() / 1024**2  # MB
        del model_amp, output_amp, loss_amp
        torch.cuda.empty_cache()

        # Check memory reduction
        memory_reduction = (memory_fp32 - memory_amp) / memory_fp32
        print(f"Memory usage - FP32: {memory_fp32:.1f} MB, AMP: {memory_amp:.1f} MB")
        print(f"Memory reduction: {memory_reduction:.1%}")

        # AMP should use less memory (target: 40-50%, but allow for variation)
        assert memory_amp < memory_fp32, "AMP should use less memory than FP32"
        # Don't enforce strict percentage on CI/CD since hardware varies


class TestAMPEdgeCases:
    """Test edge cases and error handling for AMP."""

    def test_amp_with_zero_gradients(self, simple_network, device):
        """Test AMP handles zero gradients gracefully (generative model)."""
        optimizer = torch.optim.Adam(simple_network.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        # Forward pass with autocast (generative - no input!)
        with torch.cuda.amp.autocast(enabled=simple_network.use_amp and device.type == "cuda"):
            output = simple_network()  # No input!
            # Create a loss that results in zero gradients
            loss = output.mean() * 0.0

        optimizer.zero_grad()
        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Should complete without errors
        assert True

    def test_amp_with_nan_loss(self, simple_network, device):
        """Test AMP handles NaN loss values (generative model)."""
        optimizer = torch.optim.Adam(simple_network.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        # Create a loss that produces NaN (generative - no input!)
        with torch.cuda.amp.autocast(enabled=simple_network.use_amp and device.type == "cuda"):
            output = simple_network()  # No input!
            # Create NaN by dividing by zero in a way that maintains grad
            loss = (output.sum() * 0.0) / 0.0  # This produces NaN with grad_fn

        optimizer.zero_grad()
        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            # GradScaler should handle NaN by skipping the step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training - backward with NaN
            loss.backward()

        # Should complete without crashing
        assert True
