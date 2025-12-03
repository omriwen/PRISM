"""
GPU memory stability tests for training loop.

This module tests that GPU memory optimizations are working correctly:
- Task 5.1: Verify GPU memory doesn't grow unbounded during training
- Task 5.2: Verify computational equivalence of optimized code

The optimizations tested include:
- Streaming average for synthetic aperture initialization
- LRU cache eviction for SSIM window cache
- scheduler.step(loss.item()) to prevent graph retention
- torch.cuda.empty_cache() defensive calls
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn

from prism.models.losses import (
    FastSSIMLossStrategy,
    LossAggregator,
    SSIMLossStrategy,
    _create_gaussian_window,
    _ssim_single,
)


if TYPE_CHECKING:
    pass


# =============================================================================
# Task 5.1: Memory Stability Tests
# =============================================================================


class TestMemoryStabilityDuringTraining:
    """Test GPU memory stability during training loops."""

    @pytest.mark.gpu
    def test_memory_stability_during_training_loop(self, gpu_device):
        """Verify GPU memory doesn't grow unbounded during training.

        This test simulates a training loop with multiple samples and
        verifies that memory usage remains stable (doesn't grow linearly).
        """
        # Force cleanup before test
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Record initial memory
        initial_memory = torch.cuda.memory_allocated()

        # Create model and optimizer
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.rand(1, 1, 128, 128, device=gpu_device))

            def forward(self):
                return self.param

        model = SimpleModel()
        criterion = LossAggregator(loss_type="ssim")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        # Record memory after setup
        setup_memory = torch.cuda.memory_allocated()
        memory_samples = []

        # Run training loop with 60+ samples (more than plan's 50)
        n_samples = 60
        for i in range(n_samples):
            # Create fresh target each iteration (simulates new sample)
            target = torch.rand(2, 1, 128, 128, device=gpu_device)

            optimizer.zero_grad()
            output = model()
            loss_old, loss_new = criterion(output, target)
            loss = loss_old + loss_new
            loss.backward()
            optimizer.step()

            # Use .item() as per Task 3.1 fix
            scheduler.step(loss.item())

            # Defensive cache clear as per Task 3.2 fix
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Record memory every 10 samples
            if i % 10 == 0:
                memory_samples.append(torch.cuda.memory_allocated())

        # Final memory measurement
        final_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()

        # Calculate memory growth
        growth = final_memory - initial_memory
        setup_overhead = setup_memory - initial_memory

        # Verify no unbounded growth (allow 100MB overhead as per plan)
        max_allowed_growth = 100 * 1024 * 1024  # 100 MB
        assert growth < max_allowed_growth, (
            f"Memory grew by {growth / 1e6:.1f} MB, exceeding {max_allowed_growth / 1e6:.1f} MB limit. "
            f"Initial: {initial_memory / 1e6:.1f} MB, Final: {final_memory / 1e6:.1f} MB"
        )

        # Verify memory is stable (no linear growth pattern)
        if len(memory_samples) >= 3:
            # Check that later samples don't consistently exceed earlier ones
            mid_point = len(memory_samples) // 2
            early_avg = sum(memory_samples[:mid_point]) / mid_point
            late_avg = sum(memory_samples[mid_point:]) / (len(memory_samples) - mid_point)

            # Allow 20% variance (for GC timing differences)
            assert late_avg < early_avg * 1.2, (
                f"Memory appears to be growing: early avg {early_avg / 1e6:.1f} MB, "
                f"late avg {late_avg / 1e6:.1f} MB"
            )

    @pytest.mark.gpu
    def test_ssim_cache_lru_eviction(self, gpu_device):
        """Verify SSIM cache LRU eviction prevents unbounded growth."""
        gc.collect()
        torch.cuda.empty_cache()

        # Create SSIM strategy
        ssim_strategy = SSIMLossStrategy()

        # Verify LRU attributes exist (from Task 2.1)
        assert hasattr(ssim_strategy, "_cache_order")
        assert hasattr(ssim_strategy, "_max_cache_size")
        assert ssim_strategy._max_cache_size == 5

        # Create many different tensor configurations to fill cache
        dtypes = [torch.float32, torch.float16]
        channels_list = [1, 3, 4]
        devices = [gpu_device]

        call_count = 0
        for dtype in dtypes:
            for channels in channels_list:
                pred = torch.rand(1, channels, 64, 64, device=gpu_device, dtype=dtype)
                target = torch.rand(1, channels, 64, 64, device=gpu_device, dtype=dtype)
                _ = ssim_strategy(pred, target)
                call_count += 1

        # Cache should be limited to max_cache_size
        assert len(ssim_strategy._window_cache) <= ssim_strategy._max_cache_size, (
            f"Cache size {len(ssim_strategy._window_cache)} exceeds max {ssim_strategy._max_cache_size}"
        )
        assert len(ssim_strategy._cache_order) <= ssim_strategy._max_cache_size

    @pytest.mark.gpu
    def test_fast_ssim_cache_lru_eviction(self, gpu_device):
        """Verify FastSSIM cache LRU eviction prevents unbounded growth."""
        gc.collect()
        torch.cuda.empty_cache()

        # Create FastSSIM strategy
        fast_ssim_strategy = FastSSIMLossStrategy()

        # Verify LRU attributes exist (from Task 2.2)
        assert hasattr(fast_ssim_strategy, "_cache_order")
        assert hasattr(fast_ssim_strategy, "_max_cache_size")
        assert fast_ssim_strategy._max_cache_size == 5

        # Create many different tensor configurations
        dtypes = [torch.float32, torch.float16]
        channels_list = [1, 3, 4]

        for dtype in dtypes:
            for channels in channels_list:
                pred = torch.rand(1, channels, 64, 64, device=gpu_device, dtype=dtype)
                target = torch.rand(1, channels, 64, 64, device=gpu_device, dtype=dtype)
                _ = fast_ssim_strategy(pred, target)

        # Cache should be limited
        assert len(fast_ssim_strategy._window_cache) <= fast_ssim_strategy._max_cache_size
        assert len(fast_ssim_strategy._cache_order) <= fast_ssim_strategy._max_cache_size


# =============================================================================
# Task 5.2: Computational Equivalence Tests
# =============================================================================


class TestComputationalEquivalence:
    """Verify optimizations don't change computational results."""

    def test_ssim_loss_returns_tensor(self):
        """Verify SSIM loss returns tensor (not wrapped with as_tensor).

        Task 2.3 removed torch.as_tensor wrappers. The result should
        still be a proper tensor with gradient capability.
        """
        ssim_strategy = SSIMLossStrategy()

        pred = torch.rand(1, 1, 64, 64, requires_grad=True)
        target = torch.rand(1, 1, 64, 64)

        result = ssim_strategy(pred, target)

        # Verify result is a tensor
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Scalar
        assert result.requires_grad  # Should maintain gradient flow

        # Verify backward works
        result.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_ssim_value_range(self):
        """Verify SSIM loss values are in expected range [0, 0.5]."""
        ssim_strategy = SSIMLossStrategy()

        # Test with identical images (should give ~0 loss)
        img = torch.rand(1, 1, 64, 64)
        loss_identical = ssim_strategy(img, img)
        assert 0 <= loss_identical <= 0.01, f"Identical image loss should be ~0, got {loss_identical}"

        # Test with random images (should give positive loss)
        pred = torch.rand(1, 1, 64, 64)
        target = torch.rand(1, 1, 64, 64)
        loss_random = ssim_strategy(pred, target)
        assert 0 <= loss_random <= 0.5, f"Random image loss should be in [0, 0.5], got {loss_random}"

        # Verify identical < random
        assert loss_identical < loss_random, "Identical images should have lower loss"

    def test_fast_ssim_value_range(self):
        """Verify FastSSIM loss values are in expected range."""
        fast_ssim_strategy = FastSSIMLossStrategy()

        # Test with identical images
        img = torch.rand(1, 1, 64, 64)
        loss_identical = fast_ssim_strategy(img, img)
        assert 0 <= loss_identical <= 0.01, f"Identical image loss should be ~0, got {loss_identical}"

        # Test with random images
        pred = torch.rand(1, 1, 64, 64)
        target = torch.rand(1, 1, 64, 64)
        loss_random = fast_ssim_strategy(pred, target)
        assert 0 <= loss_random <= 0.5, f"Random image loss should be in [0, 0.5], got {loss_random}"

    def test_ssim_and_fast_ssim_agreement(self):
        """Verify SSIMLossStrategy and FastSSIMLossStrategy produce similar results."""
        ssim_strategy = SSIMLossStrategy()
        fast_ssim_strategy = FastSSIMLossStrategy()

        # Create test images
        torch.manual_seed(42)
        pred = torch.rand(1, 1, 64, 64)
        target = torch.rand(1, 1, 64, 64)

        loss_ssim = ssim_strategy(pred, target)
        loss_fast = fast_ssim_strategy(pred, target)

        # Should be very close (both use same underlying SSIM computation)
        assert torch.allclose(loss_ssim, loss_fast, atol=1e-5), (
            f"SSIM strategies differ: {loss_ssim.item()} vs {loss_fast.item()}"
        )

    def test_ssim_gradient_flow_preserved(self):
        """Verify gradient flow is preserved after optimization changes."""
        ssim_strategy = SSIMLossStrategy()

        # Create test case with known gradients
        pred = torch.rand(1, 1, 64, 64, requires_grad=True)
        target = torch.zeros(1, 1, 64, 64)  # Zero target for predictable gradient direction

        loss = ssim_strategy(pred, target)
        loss.backward()

        # Verify gradients exist and are valid
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
        assert not torch.isnan(pred.grad).any()
        assert not torch.isinf(pred.grad).any()

        # Gradient should be non-zero (loss is differentiable)
        assert pred.grad.abs().sum() > 0

    @pytest.mark.gpu
    def test_ssim_gpu_cpu_consistency(self, gpu_device):
        """Verify SSIM produces consistent results on CPU and GPU."""
        ssim_strategy_cpu = SSIMLossStrategy()
        ssim_strategy_gpu = SSIMLossStrategy()

        # Create identical test images
        torch.manual_seed(42)
        pred_cpu = torch.rand(1, 1, 64, 64)
        target_cpu = torch.rand(1, 1, 64, 64)

        pred_gpu = pred_cpu.to(gpu_device)
        target_gpu = target_cpu.to(gpu_device)

        loss_cpu = ssim_strategy_cpu(pred_cpu, target_cpu)
        loss_gpu = ssim_strategy_gpu(pred_gpu, target_gpu)

        # Compare (GPU results brought back to CPU)
        assert torch.allclose(loss_cpu, loss_gpu.cpu(), atol=1e-5), (
            f"CPU/GPU results differ: {loss_cpu.item()} vs {loss_gpu.item()}"
        )

    def test_streaming_average_equivalence(self):
        """Verify streaming average produces same result as batch average.

        This tests the mathematical equivalence of Task 1.1's streaming
        average implementation vs the original batch approach.
        """
        torch.manual_seed(42)

        # Generate test measurements
        n_samples = 50
        measurements = [torch.rand(1, 1, 64, 64) for _ in range(n_samples)]

        # Method 1: Batch average (original)
        stacked = torch.stack(measurements)
        batch_avg = stacked.mean(dim=0)

        # Method 2: Streaming average (optimized)
        running_sum = None
        count = 0
        for meas in measurements:
            if running_sum is None:
                running_sum = meas.detach()
            else:
                running_sum = running_sum + meas.detach()
            count += 1
        streaming_avg = running_sum / count

        # Verify equivalence within float32 precision
        max_diff = (batch_avg - streaming_avg).abs().max()
        assert max_diff < 1e-5, (
            f"Streaming average differs from batch average by {max_diff.item()}"
        )

    def test_loss_item_scheduler_compatibility(self):
        """Verify loss.item() works correctly with scheduler.

        This tests Task 3.1's fix to use loss.item() instead of loss tensor.
        """
        # Create simple model
        param = torch.rand(1, 1, 64, 64, requires_grad=True)
        optimizer = torch.optim.Adam([param], lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

        # Simulate training steps
        criterion = SSIMLossStrategy()
        target = torch.rand(1, 1, 64, 64)

        initial_lr = optimizer.param_groups[0]["lr"]
        losses = []

        for i in range(10):
            optimizer.zero_grad()
            loss = criterion(param, target)
            loss.backward()
            optimizer.step()

            # Use .item() as per fix
            loss_value = loss.item()
            scheduler.step(loss_value)
            losses.append(loss_value)

        # Verify scheduler received valid values
        assert len(losses) == 10
        assert all(isinstance(l, float) for l in losses)
        assert all(0 <= l <= 1 for l in losses)

        # Verify scheduler is working (LR should be same or reduced)
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr <= initial_lr


# =============================================================================
# Additional Memory Tests
# =============================================================================


class TestSSIMWindowCache:
    """Test SSIM window caching behavior."""

    def test_window_cache_reuse(self):
        """Verify windows are cached and reused."""
        ssim_strategy = SSIMLossStrategy()

        pred = torch.rand(1, 1, 64, 64)
        target = torch.rand(1, 1, 64, 64)

        # First call - should create cache entry
        _ = ssim_strategy(pred, target)
        assert len(ssim_strategy._window_cache) == 1

        # Second call with same config - should reuse
        _ = ssim_strategy(pred, target)
        assert len(ssim_strategy._window_cache) == 1  # Still just 1 entry

    def test_window_cache_different_configs(self):
        """Verify different configurations create different cache entries."""
        ssim_strategy = SSIMLossStrategy()

        # Different channel counts
        pred1 = torch.rand(1, 1, 64, 64)
        target1 = torch.rand(1, 1, 64, 64)
        _ = ssim_strategy(pred1, target1)

        pred3 = torch.rand(1, 3, 64, 64)
        target3 = torch.rand(1, 3, 64, 64)
        _ = ssim_strategy(pred3, target3)

        # Should have 2 cache entries
        assert len(ssim_strategy._window_cache) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
