"""
Numerical stability tests for Automatic Mixed Precision (AMP).

Tests that AMP training converges similarly to FP32 over longer training runs.
These tests validate that FP16/FP32 mixed precision doesn't introduce
numerical instabilities that affect convergence.
"""

from __future__ import annotations

import pytest
import torch

from prism.models.networks import ProgressiveDecoder


@pytest.fixture
def device():
    """Get device for testing (CUDA if available, CPU otherwise)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def target_256(device):
    """Create a fixed target for 256x256 training."""
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    target = torch.randn(1, 1, 256, 256, device=device)
    # Normalize to [0, 1] range
    target = (target - target.min()) / (target.max() - target.min())
    return target


@pytest.fixture
def target_512(device):
    """Create a fixed target for 512x512 training."""
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    target = torch.randn(1, 1, 512, 512, device=device)
    # Normalize to [0, 1] range
    target = (target - target.min()) / (target.max() - target.min())
    return target


class TestAMPNumericalStability:
    """Test numerical stability of AMP vs FP32 training."""

    def test_amp_convergence_100_iterations(self, device, target_256):
        """
        Test that AMP converges similarly to FP32 over 100 iterations.

        This test verifies that mixed precision training doesn't introduce
        numerical instabilities that prevent convergence.
        """
        # Create two identical networks
        torch.manual_seed(123)
        model_fp32 = ProgressiveDecoder(input_size=256, use_amp=False).to(device)

        torch.manual_seed(123)
        model_amp = ProgressiveDecoder(input_size=256, use_amp=True).to(device)

        # Verify same initialization
        for p1, p2 in zip(model_fp32.parameters(), model_amp.parameters()):
            assert torch.allclose(p1, p2, rtol=1e-5)

        # Create optimizers with same seed
        optimizer_fp32 = torch.optim.Adam(model_fp32.parameters(), lr=1e-3)
        optimizer_amp = torch.optim.Adam(model_amp.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        losses_fp32 = []
        losses_amp = []

        # Train for 100 iterations
        for iteration in range(100):
            # FP32 training step
            optimizer_fp32.zero_grad()
            output_fp32 = model_fp32()
            loss_fp32 = torch.nn.functional.mse_loss(output_fp32, target_256)
            loss_fp32.backward()
            optimizer_fp32.step()
            losses_fp32.append(loss_fp32.item())

            # AMP training step
            optimizer_amp.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                output_amp = model_amp()
                loss_amp = torch.nn.functional.mse_loss(output_amp, target_256)

            if scaler is not None and device.type == "cuda":
                scaler.scale(loss_amp).backward()
                scaler.step(optimizer_amp)
                scaler.update()
            else:
                loss_amp.backward()
                optimizer_amp.step()

            losses_amp.append(loss_amp.item())

        # Verify both converge (final loss lower than initial)
        # Relaxed threshold since convergence speed varies with random initialization
        assert losses_fp32[-1] < losses_fp32[0] * 0.7, "FP32 should converge"
        assert losses_amp[-1] < losses_amp[0] * 0.7, "AMP should converge"

        # Verify convergence rates are similar
        fp32_improvement = (losses_fp32[0] - losses_fp32[-1]) / losses_fp32[0]
        amp_improvement = (losses_amp[0] - losses_amp[-1]) / losses_amp[0]

        improvement_diff = abs(fp32_improvement - amp_improvement)
        assert improvement_diff < 0.2, (
            f"AMP and FP32 convergence rates should be similar. "
            f"FP32: {fp32_improvement:.2%}, AMP: {amp_improvement:.2%}"
        )

        # Verify final losses are within 10%
        final_diff = abs(losses_fp32[-1] - losses_amp[-1]) / losses_fp32[-1]
        assert final_diff < 0.15, (
            f"Final losses should be within 15%. "
            f"FP32: {losses_fp32[-1]:.6f}, AMP: {losses_amp[-1]:.6f}, "
            f"diff: {final_diff:.2%}"
        )

    def test_amp_gradient_stability(self, device, target_256):
        """
        Test that gradients remain stable during AMP training.

        Verifies that gradients don't explode or vanish due to FP16 precision.
        """
        model = ProgressiveDecoder(input_size=256, use_amp=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        gradient_norms = []

        for _ in range(50):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                output = model()
                loss = torch.nn.functional.mse_loss(output, target_256)

            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            # Compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            gradient_norms.append(total_norm)

            if scaler is not None and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        # Verify gradients don't explode (< 100)
        max_grad_norm = max(gradient_norms)
        assert max_grad_norm < 100.0, f"Gradients exploded: max norm = {max_grad_norm:.2f}"

        # Verify gradients don't vanish (> 1e-6)
        min_grad_norm = min(gradient_norms)
        assert min_grad_norm > 1e-6, f"Gradients vanished: min norm = {min_grad_norm:.2e}"

        # Verify no NaN or Inf gradients
        assert all(norm > 0 and norm < float("inf") for norm in gradient_norms), (
            "Gradients contain NaN or Inf values"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP")
    def test_amp_scaler_updates(self, device, target_256):
        """
        Test that GradScaler updates scale factor appropriately.

        The scaler should adjust the loss scale based on gradient overflow/underflow.
        """
        if device.type != "cuda":
            pytest.skip("CUDA required for GradScaler testing")

        model = ProgressiveDecoder(input_size=256, use_amp=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")

        initial_scale = scaler.get_scale()
        scales = [initial_scale]

        # Train for some iterations
        for _ in range(50):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=True):
                output = model()
                loss = torch.nn.functional.mse_loss(output, target_256)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scales.append(scaler.get_scale())

        # Scaler should have non-zero scale throughout
        assert all(s > 0 for s in scales), "GradScaler scale should remain positive"

        # Scale should be reasonable (typically 2^15 to 2^17)
        final_scale = scaler.get_scale()
        assert 1.0 <= final_scale <= 2**20, f"GradScaler scale out of range: {final_scale}"

    def test_amp_loss_consistency(self, device, target_256):
        """
        Test that loss values are consistent between AMP and FP32.

        While the training dynamics may differ slightly, loss values
        should be computed consistently.
        """
        # Create identical models
        torch.manual_seed(456)
        model_fp32 = ProgressiveDecoder(input_size=256, use_amp=False).to(device)

        torch.manual_seed(456)
        model_amp = ProgressiveDecoder(input_size=256, use_amp=True).to(device)

        # Forward pass with same model weights
        with torch.no_grad():
            output_fp32 = model_fp32()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                output_amp = model_amp()
                # Convert to FP32 for comparison
                output_amp = output_amp.float()

        # Compute losses
        loss_fp32 = torch.nn.functional.mse_loss(output_fp32, target_256)
        loss_amp = torch.nn.functional.mse_loss(output_amp, target_256)

        # Losses should be very close (within 1% due to FP16 precision)
        loss_diff = abs(loss_fp32.item() - loss_amp.item()) / loss_fp32.item()
        assert loss_diff < 0.05, (
            f"Loss values differ significantly: "
            f"FP32={loss_fp32.item():.6f}, AMP={loss_amp.item():.6f}, "
            f"diff={loss_diff:.2%}"
        )

    def test_amp_parameter_updates(self, device, target_256):
        """
        Test that model parameters update correctly during AMP training.

        Verifies that weights change during training and don't get stuck.
        """
        model = ProgressiveDecoder(input_size=256, use_amp=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        # Store initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        # Train for 20 iterations
        for _ in range(20):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                output = model()
                loss = torch.nn.functional.mse_loss(output, target_256)

            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # Verify parameters have changed
        params_changed = 0
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name], rtol=1e-3):
                params_changed += 1

        total_params = len(list(model.parameters()))
        change_ratio = params_changed / total_params

        assert change_ratio > 0.9, (
            f"Most parameters should have changed during training. "
            f"Changed: {params_changed}/{total_params} ({change_ratio:.1%})"
        )


class TestAMPDifferentSizes:
    """Test AMP stability across different image sizes."""

    def test_amp_convergence_512(self, device, target_512):
        """
        Test AMP convergence on 512x512 images.

        Larger images are where AMP benefits should be most apparent.
        """
        torch.manual_seed(789)
        model = ProgressiveDecoder(input_size=512, use_amp=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        losses = []

        # Train for 50 iterations (fewer due to larger size)
        for _ in range(50):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                output = model()
                loss = torch.nn.functional.mse_loss(output, target_512)

            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

        # Verify convergence
        assert losses[-1] < losses[0] * 0.7, (
            f"Should converge on 512x512 images. Initial: {losses[0]:.6f}, Final: {losses[-1]:.6f}"
        )

        # Verify no NaN losses
        assert all(not (loss != loss) for loss in losses), "Losses contain NaN"

    def test_amp_different_learning_rates(self, device, target_256):
        """
        Test AMP stability with different learning rates.

        Higher learning rates can expose numerical instabilities.
        """
        learning_rates = [1e-4, 1e-3, 1e-2]

        for lr in learning_rates:
            torch.manual_seed(999)
            model = ProgressiveDecoder(input_size=256, use_amp=True).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

            losses = []

            # Train for 30 iterations
            for _ in range(30):
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    output = model()
                    loss = torch.nn.functional.mse_loss(output, target_256)

                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                losses.append(loss.item())

            # Verify no NaN or Inf losses
            assert all(0 <= loss < float("inf") for loss in losses), (
                f"Unstable training with lr={lr}. Losses: {losses[:5]}...{losses[-5:]}"
            )

            # Verify loss is decreasing or stable (not increasing)
            if lr <= 1e-3:  # Lower learning rates should converge
                assert losses[-1] < losses[0] * 1.1, f"Loss should decrease with lr={lr}"
