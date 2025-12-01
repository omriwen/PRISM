"""
Integration tests for SSIM in training loop.

Tests cover:
    - Configuration validation with SSIM loss types
    - LossAggregator initialization with SSIM
    - Backward pass integration
    - Training loop compatibility
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from prism.config.base import PRISMConfig, TrainingConfig
from prism.models.losses import LossAggregator


class TestSSIMConfigurationIntegration:
    """Integration tests for SSIM configuration system."""

    def test_ssim_config_validation(self):
        """Test SSIM loss type validates correctly."""
        # Create minimal config with SSIM
        config = PRISMConfig()
        config.training.loss_type = "ssim"

        # Should not raise
        config.validate()

        assert config.training.loss_type == "ssim"

    def test_ms_ssim_config_validation(self):
        """Test MS-SSIM loss type validates correctly."""
        config = PRISMConfig()
        config.training.loss_type = "ms-ssim"

        # Should not raise
        config.validate()

        assert config.training.loss_type == "ms-ssim"

    def test_l1_config_validation(self):
        """Test L1 loss type still validates correctly (regression test)."""
        config = PRISMConfig()
        config.training.loss_type = "l1"

        # Should not raise
        config.validate()

        assert config.training.loss_type == "l1"

    def test_l2_config_validation(self):
        """Test L2 loss type still validates correctly (regression test)."""
        config = PRISMConfig()
        config.training.loss_type = "l2"

        # Should not raise
        config.validate()

        assert config.training.loss_type == "l2"

    def test_invalid_loss_type(self):
        """Test invalid loss type raises error."""
        config = PRISMConfig()
        config.training.loss_type = "invalid"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Invalid loss_type"):
            config.validate()

    def test_training_config_creation(self):
        """Test TrainingConfig creation with SSIM."""
        training_config = TrainingConfig(loss_type="ssim")
        assert training_config.loss_type == "ssim"

        training_config_ms = TrainingConfig(loss_type="ms-ssim")
        assert training_config_ms.loss_type == "ms-ssim"


class TestSSIMLossAggregatorIntegration:
    """Integration tests for LossAggregator with SSIM."""

    def test_ssim_loss_creation(self):
        """Test creating LossAggregator with SSIM loss type."""
        criterion = LossAggregator(loss_type="ssim")

        assert criterion.loss_type == "ssim"
        assert criterion.window_size == 11
        assert criterion.sigma == 1.5
        assert criterion.data_range == 1.0

    def test_ms_ssim_loss_creation(self):
        """Test creating LossAggregator with MS-SSIM loss type."""
        criterion = LossAggregator(loss_type="ms-ssim")

        assert criterion.loss_type == "ms-ssim"
        assert criterion.window_size == 11
        assert criterion.sigma == 1.5
        assert criterion.data_range == 1.0

    def test_ssim_forward_pass(self):
        """Test SSIM forward pass produces valid outputs."""
        criterion = LossAggregator(loss_type="ssim")

        inputs = torch.rand(1, 1, 128, 128)
        target = torch.rand(2, 1, 128, 128)

        loss_old, loss_new = criterion(inputs, target)

        # Check outputs are valid scalar tensors
        assert isinstance(loss_old, torch.Tensor)
        assert isinstance(loss_new, torch.Tensor)
        assert loss_old.dim() == 0
        assert loss_new.dim() == 0

        # Check loss values are in expected range
        assert 0 <= loss_old <= 0.6  # DSSIM range (allowing slight tolerance)
        assert 0 <= loss_new <= 0.6

    def test_ms_ssim_forward_pass(self):
        """Test MS-SSIM forward pass produces valid outputs."""
        criterion = LossAggregator(loss_type="ms-ssim")

        # Need larger images for MS-SSIM
        inputs = torch.rand(1, 1, 256, 256)
        target = torch.rand(2, 1, 256, 256)

        loss_old, loss_new = criterion(inputs, target)

        # Check outputs are valid scalar tensors
        assert isinstance(loss_old, torch.Tensor)
        assert isinstance(loss_new, torch.Tensor)
        assert loss_old.dim() == 0
        assert loss_new.dim() == 0

        # Check loss values are in expected range
        assert 0 <= loss_old <= 0.6
        assert 0 <= loss_new <= 0.6


class TestSSIMBackwardPassIntegration:
    """Integration tests for SSIM backward pass in training context."""

    def test_ssim_backward_pass(self):
        """Test SSIM loss works in backward pass."""
        criterion = LossAggregator(loss_type="ssim")

        inputs = torch.rand(1, 1, 128, 128, requires_grad=True)
        target = torch.rand(2, 1, 128, 128)

        loss_old, loss_new = criterion(inputs, target)
        loss = loss_old + loss_new

        # Backward pass
        loss.backward()

        # Check gradients exist and are valid
        assert inputs.grad is not None
        assert inputs.grad.shape == inputs.shape
        assert not torch.isnan(inputs.grad).any()
        assert not torch.isinf(inputs.grad).any()

    def test_ms_ssim_backward_pass(self):
        """Test MS-SSIM loss works in backward pass."""
        criterion = LossAggregator(loss_type="ms-ssim")

        inputs = torch.rand(1, 1, 256, 256, requires_grad=True)
        target = torch.rand(2, 1, 256, 256)

        loss_old, loss_new = criterion(inputs, target)
        loss = loss_old + loss_new

        # Backward pass
        loss.backward()

        # Check gradients exist and are valid
        assert inputs.grad is not None
        assert inputs.grad.shape == inputs.shape
        assert not torch.isnan(inputs.grad).any()
        assert not torch.isinf(inputs.grad).any()

    def test_ssim_gradient_accumulation(self):
        """Test SSIM works with gradient accumulation."""
        criterion = LossAggregator(loss_type="ssim")

        inputs = torch.rand(1, 1, 128, 128, requires_grad=True)
        target = torch.rand(2, 1, 128, 128)

        # First backward pass
        loss_old, loss_new = criterion(inputs, target)
        loss1 = loss_old + loss_new
        loss1.backward()

        # Store first gradient
        grad1 = inputs.grad.clone()

        # Second backward pass (without zeroing gradients)
        loss_old, loss_new = criterion(inputs, target)
        loss2 = loss_old + loss_new
        loss2.backward()

        # Gradients should have accumulated
        assert not torch.allclose(inputs.grad, grad1)
        # Gradient should be approximately 2x (though not exact due to different losses)
        assert inputs.grad.abs().mean() > grad1.abs().mean()

    def test_ssim_optimizer_step(self):
        """Test SSIM loss with optimizer step."""
        criterion = LossAggregator(loss_type="ssim")

        # Create a simple parameter tensor (simulating model parameters)
        params = torch.rand(1, 1, 128, 128, requires_grad=True)
        optimizer = torch.optim.SGD([params], lr=0.01)

        target = torch.rand(2, 1, 128, 128)

        # Training step
        optimizer.zero_grad()
        loss_old, loss_new = criterion(params, target)
        loss = loss_old + loss_new
        loss.backward()

        # Store parameter values before step
        params_before = params.clone().detach()

        optimizer.step()

        # Parameters should have changed
        assert not torch.allclose(params, params_before)


class TestSSIMTrainingLoopIntegration:
    """Integration tests simulating training loop usage."""

    def test_ssim_training_loop_simulation(self):
        """Test SSIM in a simulated training loop."""

        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.rand(1, 1, 128, 128))

            def forward(self):
                return self.param

        model = SimpleModel()
        criterion = LossAggregator(loss_type="ssim")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Simulated target
        target = torch.rand(2, 1, 128, 128)

        # Training loop (3 iterations)
        losses = []
        for epoch in range(3):
            optimizer.zero_grad()

            output = model()
            loss_old, loss_new = criterion(output, target)
            loss = loss_old + loss_new

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Check loss is valid
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            assert loss >= 0

        # Losses should be recorded
        assert len(losses) == 3

    def test_ms_ssim_training_loop_simulation(self):
        """Test MS-SSIM in a simulated training loop."""

        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.rand(1, 1, 256, 256))

            def forward(self):
                return self.param

        model = SimpleModel()
        criterion = LossAggregator(loss_type="ms-ssim")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Simulated target
        target = torch.rand(2, 1, 256, 256)

        # Training loop (3 iterations)
        losses = []
        for epoch in range(3):
            optimizer.zero_grad()

            output = model()
            loss_old, loss_new = criterion(output, target)
            loss = loss_old + loss_new

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Check loss is valid
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            assert loss >= 0

        # Losses should be recorded
        assert len(losses) == 3

    def test_loss_type_switching(self):
        """Test that different loss types can be used interchangeably."""
        inputs = torch.rand(1, 1, 256, 256, requires_grad=True)
        target = torch.rand(2, 1, 256, 256)

        # Test all loss types work
        for loss_type in ["l1", "l2", "ssim", "ms-ssim"]:
            criterion = LossAggregator(loss_type=loss_type)  # type: ignore[arg-type]

            if inputs.grad is not None:
                inputs.grad.zero_()

            if loss_type in ["l1", "l2"]:
                # L1/L2 need telescope parameter (use None for direct comparison)
                loss_old, loss_new = criterion(inputs, target, telescope=None)
            else:
                # SSIM/MS-SSIM don't need telescope
                loss_old, loss_new = criterion(inputs, target)

            loss = loss_old + loss_new
            loss.backward()

            # All should produce valid gradients
            assert inputs.grad is not None
            assert not torch.isnan(inputs.grad).any()

    def test_ssim_convergence_behavior(self):
        """Test that SSIM loss can drive convergence."""

        # Create model that learns to match a target
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.rand(1, 1, 128, 128))

            def forward(self):
                return self.param

        model = SimpleModel()
        criterion = LossAggregator(loss_type="ssim")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # Create target (both old and new are the same for simplicity)
        target_img = torch.rand(1, 1, 128, 128)
        target = torch.cat([target_img, target_img], dim=0)

        # Train for several iterations
        initial_loss = 0.0
        final_loss = 0.0

        for epoch in range(20):
            optimizer.zero_grad()

            output = model()
            loss_old, loss_new = criterion(output, target)
            loss = loss_old + loss_new

            if epoch == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Loss should decrease (model should learn to match target)
        assert initial_loss > 0, "Initial loss should be positive"
        assert final_loss < initial_loss, (
            f"Loss should decrease during training: {initial_loss} -> {final_loss}"
        )


class TestSSIMDeviceCompatibility:
    """Test SSIM works on different devices."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ssim_cuda_forward_backward(self):
        """Test SSIM on CUDA device."""
        inputs = torch.rand(1, 1, 128, 128, requires_grad=True, device="cuda")
        target = torch.rand(2, 1, 128, 128, device="cuda")

        criterion = LossAggregator(loss_type="ssim").cuda()

        loss_old, loss_new = criterion(inputs, target)
        loss = loss_old + loss_new

        # Check on CUDA
        assert loss.is_cuda

        loss.backward()

        # Check gradients on CUDA
        assert inputs.grad is not None
        assert inputs.grad.is_cuda
        assert not torch.isnan(inputs.grad).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ms_ssim_cuda_forward_backward(self):
        """Test MS-SSIM on CUDA device."""
        inputs = torch.rand(1, 1, 256, 256, requires_grad=True, device="cuda")
        target = torch.rand(2, 1, 256, 256, device="cuda")

        criterion = LossAggregator(loss_type="ms-ssim").cuda()

        loss_old, loss_new = criterion(inputs, target)
        loss = loss_old + loss_new

        # Check on CUDA
        assert loss.is_cuda

        loss.backward()

        # Check gradients on CUDA
        assert inputs.grad is not None
        assert inputs.grad.is_cuda
        assert not torch.isnan(inputs.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
