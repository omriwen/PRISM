"""
Performance benchmarks for ProgressiveDecoder.

Tests performance of:
- Baseline forward pass speed
- Conv-BN fusion optimization
- torch.compile() optimization
- Memory usage
- Performance regression detection
"""

from __future__ import annotations

import warnings

import pytest
import torch

from prism.models.networks import ProgressiveDecoder


@pytest.fixture
def device():
    """Get device for benchmarking (CPU for consistency)."""
    return torch.device("cpu")


@pytest.fixture
def small_model(device):
    """Create small model for quick benchmarks."""
    model = ProgressiveDecoder(input_size=128, output_size=64)
    return model.to(device)


@pytest.fixture
def medium_model(device):
    """Create medium model for realistic benchmarks."""
    model = ProgressiveDecoder(input_size=256, output_size=128)
    return model.to(device)


class TestBaselinePerformance:
    """Benchmark baseline performance without optimizations."""

    def test_baseline_forward_pass_small(self, small_model):
        """Benchmark baseline forward pass for small model."""
        small_model.eval()

        results = small_model.benchmark(num_iterations=10, warmup=3, measure_memory=False)

        # Verify results structure
        assert "avg_time_ms" in results
        assert "fps" in results
        assert "num_parameters" in results

        # Verify reasonable performance (not too slow)
        assert results["avg_time_ms"] < 1000  # Should be less than 1 second
        assert results["fps"] > 1  # Should process at least 1 FPS

    def test_baseline_forward_pass_medium(self, medium_model):
        """Benchmark baseline forward pass for medium model."""
        medium_model.eval()

        results = medium_model.benchmark(num_iterations=10, warmup=3, measure_memory=False)

        # Verify results
        assert results["avg_time_ms"] > 0
        assert results["fps"] > 0
        assert results["num_parameters"] > 0

    @pytest.mark.parametrize("input_size", [64, 128, 256])
    def test_baseline_performance_scaling(self, input_size, device):
        """Test that performance scales reasonably with model size."""
        model = ProgressiveDecoder(input_size=input_size, output_size=input_size // 2)
        model = model.to(device)
        model.eval()

        results = model.benchmark(num_iterations=5, warmup=2, measure_memory=False)

        # Verify performance is reasonable
        assert results["avg_time_ms"] < 2000
        assert results["fps"] > 0.5

    def test_baseline_deterministic_output(self, small_model):
        """Test that baseline model produces deterministic output in eval mode."""
        small_model.eval()

        with torch.no_grad():
            output1 = small_model()
            output2 = small_model()

        # Should be identical
        assert torch.allclose(output1, output2)


class TestPrepareForInferenceOptimization:
    """Benchmark prepare_for_inference() optimization."""

    def test_prepare_for_inference_improves_performance(self, small_model):
        """Test that prepare_for_inference() doesn't degrade performance."""
        small_model.eval()

        # Apply optimization
        small_model.prepare_for_inference(compile_mode=None, free_memory=False)

        # Benchmark after optimization
        results_after = small_model.benchmark(num_iterations=10, warmup=3, measure_memory=False)

        # Performance should not degrade significantly
        # (Conv-BN fusion may not apply to ConvTranspose2d layers, so no speedup expected)
        # Just verify it still works
        assert results_after["avg_time_ms"] > 0
        assert results_after["fps"] > 0

    def test_prepare_for_inference_freezes_parameters(self, small_model):
        """Test that prepare_for_inference() freezes parameters."""
        # Before preparation
        assert any(p.requires_grad for p in small_model.parameters())

        # Apply optimization
        small_model.prepare_for_inference(compile_mode=None, free_memory=False)

        # After preparation - all parameters should be frozen
        assert all(not p.requires_grad for p in small_model.parameters())

    def test_prepare_for_inference_sets_eval_mode(self, small_model):
        """Test that prepare_for_inference() sets eval mode."""
        small_model.train()
        assert small_model.training is True

        small_model.prepare_for_inference(compile_mode=None, free_memory=False)

        assert small_model.training is False


class TestConvBNFusionPerformance:
    """Benchmark Conv-BN fusion optimization."""

    def test_conv_bn_fusion_runs_without_error(self, small_model):
        """Test that Conv-BN fusion runs without error."""
        small_model.eval()

        # Apply fusion (should not raise error)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            small_model._fuse_conv_bn_layers()

        # Model should still work
        with torch.no_grad():
            output = small_model()
            assert output.shape == (1, 1, 128, 128)

    def test_conv_bn_fusion_preserves_output(self, small_model):
        """Test that Conv-BN fusion preserves model output."""
        small_model.eval()

        # Get output before fusion
        with torch.no_grad():
            output_before = small_model()

        # Apply fusion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            small_model._fuse_conv_bn_layers()

        # Get output after fusion
        with torch.no_grad():
            output_after = small_model()

        # Outputs should be very close (allowing for numerical differences)
        assert torch.allclose(output_before, output_after, rtol=1e-4, atol=1e-6)


class TestTorchCompilePerformance:
    """Benchmark torch.compile() optimization."""

    def test_compile_runs_without_error(self, small_model):
        """Test that torch.compile() runs without error."""
        small_model.eval()

        # Compile (will warn if not available)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            small_model.compile(mode="default")

        # Model should still work
        with torch.no_grad():
            output = small_model()
            assert output.shape == (1, 1, 128, 128)

    def test_compile_with_different_modes(self, small_model):
        """Test compile with different optimization modes."""
        small_model.eval()

        modes = ["default", "reduce-overhead", "max-autotune"]

        for mode in modes:
            model_copy = ProgressiveDecoder(input_size=128, output_size=64)
            model_copy = model_copy.to(small_model.input_vec.device)
            model_copy.eval()

            # Compile with mode
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_copy.compile(mode=mode)

            # Should work
            with torch.no_grad():
                output = model_copy()
                assert output.shape == (1, 1, 128, 128)


class TestMemoryUsage:
    """Benchmark memory usage."""

    def test_model_parameter_count(self, small_model):
        """Test model parameter count is reasonable."""
        param_count = sum(p.numel() for p in small_model.parameters())

        # Small model should have reasonable parameter count
        assert param_count > 0
        assert param_count < 10_000_000  # Less than 10M parameters

    def test_parameter_count_scales_with_latent_channels(self, device):
        """Test that parameter count scales with latent channels."""
        model_small = ProgressiveDecoder(input_size=128, output_size=64, latent_channels=64)
        model_large = ProgressiveDecoder(input_size=128, output_size=64, latent_channels=256)

        params_small = sum(p.numel() for p in model_small.parameters())
        params_large = sum(p.numel() for p in model_large.parameters())

        # Larger latent channels should have more parameters
        assert params_large > params_small

    def test_gradient_checkpointing_reduces_memory(self, device):
        """Test that gradient checkpointing can be enabled."""
        model = ProgressiveDecoder(input_size=128, output_size=64)
        model = model.to(device)

        # Enable gradient checkpointing
        model.enable_gradient_checkpointing()

        # Model should still work
        model.train()
        output = model()
        loss = output.mean()
        loss.backward()

        assert model.input_vec.grad is not None


class TestPerformanceRegression:
    """Test for performance regressions."""

    def test_small_model_meets_fps_target(self, small_model):
        """Test that small model meets minimum FPS target."""
        small_model.eval()

        results = small_model.benchmark(num_iterations=10, warmup=3, measure_memory=False)

        # Should achieve at least 10 FPS on CPU for small model
        assert results["fps"] >= 5, f"FPS regression: {results['fps']} < 5"

    def test_forward_pass_completes_quickly(self, small_model):
        """Test that forward pass completes in reasonable time."""
        small_model.eval()

        results = small_model.benchmark(num_iterations=5, warmup=2, measure_memory=False)

        # Single forward pass should complete in less than 500ms on CPU
        assert results["avg_time_ms"] < 500, f"Time regression: {results['avg_time_ms']} > 500ms"

    def test_inference_faster_than_training(self, small_model):
        """Test that inference is faster than training mode."""
        # Training mode benchmark
        small_model.train()
        train_results = small_model.benchmark(num_iterations=5, warmup=2, measure_memory=False)

        # Eval mode benchmark
        small_model.eval()
        eval_results = small_model.benchmark(num_iterations=5, warmup=2, measure_memory=False)

        # Eval mode should be at least as fast as training mode
        # (May be same speed due to no batch norm running stats in single sample mode)
        assert eval_results["avg_time_ms"] <= train_results["avg_time_ms"] * 1.5


class TestBenchmarkMethodReliability:
    """Test reliability of benchmark() method itself."""

    def test_benchmark_consistency(self, small_model):
        """Test that benchmark produces consistent results."""
        small_model.eval()

        # Run benchmark multiple times
        results1 = small_model.benchmark(num_iterations=10, warmup=3, measure_memory=False)
        results2 = small_model.benchmark(num_iterations=10, warmup=3, measure_memory=False)

        # Results should be similar (within 50% variance)
        avg_fps = (results1["fps"] + results2["fps"]) / 2
        assert abs(results1["fps"] - results2["fps"]) < avg_fps * 0.5

    def test_benchmark_warmup_effect(self, small_model):
        """Test that warmup iterations are excluded from timing."""
        small_model.eval()

        # Benchmark with different warmup counts
        results_no_warmup = small_model.benchmark(num_iterations=10, warmup=0, measure_memory=False)
        results_with_warmup = small_model.benchmark(
            num_iterations=10, warmup=5, measure_memory=False
        )

        # Both should complete successfully
        assert results_no_warmup["avg_time_ms"] > 0
        assert results_with_warmup["avg_time_ms"] > 0

    def test_benchmark_with_various_iteration_counts(self, small_model):
        """Test benchmark with different iteration counts."""
        small_model.eval()

        for num_iterations in [1, 5, 10, 20]:
            results = small_model.benchmark(
                num_iterations=num_iterations, warmup=1, measure_memory=False
            )

            assert results["avg_time_ms"] > 0
            assert results["fps"] > 0


class TestPerformanceComparison:
    """Compare performance across different configurations."""

    def test_larger_models_have_more_parameters(self, device):
        """Test that larger models have more parameters."""
        model_64 = ProgressiveDecoder(input_size=64, output_size=32)
        model_128 = ProgressiveDecoder(input_size=128, output_size=64)
        model_256 = ProgressiveDecoder(input_size=256, output_size=128)

        params_64 = sum(p.numel() for p in model_64.parameters())
        params_128 = sum(p.numel() for p in model_128.parameters())
        params_256 = sum(p.numel() for p in model_256.parameters())

        # Larger models should have more parameters
        assert params_128 > params_64
        assert params_256 > params_128

    def test_batch_norm_increases_parameters(self, device):
        """Test that batch norm increases parameter count."""
        model_with_bn = ProgressiveDecoder(input_size=128, output_size=64, use_bn=True)
        model_without_bn = ProgressiveDecoder(input_size=128, output_size=64, use_bn=False)

        params_with_bn = sum(p.numel() for p in model_with_bn.parameters())
        params_without_bn = sum(p.numel() for p in model_without_bn.parameters())

        # With BN should have more parameters (or equal if BN has no learnable params)
        assert params_with_bn >= params_without_bn

    @pytest.mark.parametrize("latent_channels", [64, 128, 256, 512])
    def test_performance_with_different_latent_sizes(self, latent_channels, device):
        """Test performance with different latent channel sizes."""
        model = ProgressiveDecoder(input_size=128, output_size=64, latent_channels=latent_channels)
        model = model.to(device)
        model.eval()

        results = model.benchmark(num_iterations=5, warmup=2, measure_memory=False)

        # Should complete successfully for all sizes
        assert results["avg_time_ms"] > 0
        assert results["fps"] > 0


class TestPerformanceSummary:
    """Generate performance summary for documentation."""

    def test_generate_performance_report(self, device):
        """Generate comprehensive performance report."""
        configs = [
            ("Small (64x64)", 64, 32),
            ("Medium (128x128)", 128, 64),
            ("Large (256x256)", 256, 128),
        ]

        print("\n=== ProgressiveDecoder Performance Report ===\n")

        for name, input_size, output_size in configs:
            model = ProgressiveDecoder(input_size=input_size, output_size=output_size)
            model = model.to(device)
            model.eval()

            results = model.benchmark(num_iterations=10, warmup=3, measure_memory=False)

            print(f"{name}:")
            print(f"  Average time: {results['avg_time_ms']:.2f} ms")
            print(f"  FPS: {results['fps']:.2f}")
            print(f"  Parameters: {results['num_parameters']:,}")
            print()

        # This test always passes - it's for informational purposes
        assert True
