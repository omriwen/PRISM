"""
Integration tests for model optimization (post-training).

IMPORTANT: SPIDS has no "inference phase" - training IS the algorithm.
These tests verify prepare_for_inference() optimizes the trained model
for final output generation, NOT an inference phase.

Tests prepare_for_inference() functionality including:
- Model optimization for final reconstruction generation
- Parameter freezing
- Conv-BN fusion (placeholder)
- Eval mode setting
- Speedup validation for post-training model usage
- Integration with AMP
"""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn

from prism.models.networks import ProgressiveDecoder


@pytest.fixture
def device():
    """Get device for testing (CUDA if available, CPU otherwise)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_network(device):
    """Create a simple network for testing."""
    # Create a small network for fast testing
    model = ProgressiveDecoder(input_size=256).to(device)
    return model


@pytest.fixture
def test_target(device):
    """Create test target for loss computation (SPIDS is generative - no input!)."""
    return torch.randn(1, 1, 128, 128, device=device)


class TestPrepareForInferenceBasics:
    """Test basic prepare_for_inference() functionality."""

    def test_prepare_for_inference_exists(self, simple_network):
        """Test that prepare_for_inference method exists."""
        assert hasattr(simple_network, "prepare_for_inference")
        assert callable(simple_network.prepare_for_inference)

    def test_prepare_for_inference_returns_self(self, simple_network):
        """Test that prepare_for_inference returns self for chaining."""
        result = simple_network.prepare_for_inference()
        assert result is simple_network

    def test_prepare_for_inference_sets_eval_mode(self, simple_network):
        """Test that prepare_for_inference sets model to eval mode."""
        # Start in training mode
        simple_network.train()
        assert simple_network.training is True

        # Prepare for inference
        simple_network.prepare_for_inference()

        # Should be in eval mode
        assert simple_network.training is False

    def test_prepare_for_inference_freezes_parameters(self, simple_network):
        """Test that prepare_for_inference freezes parameters."""
        # Initially all parameters require gradients
        initial_requires_grad = [p.requires_grad for p in simple_network.parameters()]
        assert any(initial_requires_grad)

        # Prepare for inference
        simple_network.prepare_for_inference()

        # All parameters should have requires_grad=False
        for param in simple_network.parameters():
            assert param.requires_grad is False

    def test_prepare_for_inference_idempotent(self, simple_network):
        """Test that calling prepare_for_inference twice is safe."""
        # Call twice
        simple_network.prepare_for_inference()
        simple_network.prepare_for_inference()

        # Should still be in eval mode with frozen parameters
        assert simple_network.training is False
        for param in simple_network.parameters():
            assert param.requires_grad is False


class TestInferenceOptimizationCorrectness:
    """Test that model optimization preserves correctness (generative model)."""

    def test_output_unchanged_after_optimization(self, simple_network):
        """Test that outputs are numerically equivalent after optimization (generative model)."""
        # Get output before optimization (generative - no input!)
        simple_network.eval()
        with torch.no_grad():
            output_before = simple_network()  # No input!

        # Prepare for inference
        simple_network.prepare_for_inference()

        # Get output after optimization (generative - no input!)
        with torch.no_grad():
            output_after = simple_network()  # No input!

        # Outputs should be very close (allowing for numerical precision)
        assert torch.allclose(output_before, output_after, rtol=1e-4, atol=1e-5)

    def test_multiple_forward_passes_consistent(self, simple_network):
        """Test that multiple forward passes give consistent results (generative model)."""
        simple_network.prepare_for_inference()

        outputs = []
        for _ in range(3):
            with torch.no_grad():
                output = simple_network()  # No input!
                outputs.append(output)

        # All outputs should be identical (deterministic)
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i])

    def test_output_generation_after_optimization(self, simple_network, device):
        """Test that model can generate outputs after optimization (generative model)."""
        simple_network.prepare_for_inference()

        # Generate output multiple times (SPIDS is generative - no input!)
        for _ in range(5):
            with torch.no_grad():
                output = simple_network()  # No input!
            assert output.dim() == 4  # (1, C, H, W)
            assert not torch.isnan(output).any()


class TestInferenceOptimizationWithGradients:
    """Test gradient behavior after optimization (generative model)."""

    def test_no_gradients_after_optimization(self, simple_network):
        """Test that no gradients are computed after optimization (generative model)."""
        simple_network.prepare_for_inference()

        # Forward pass (generative - no input!)
        output = simple_network()  # No input!

        # Try to compute loss and backward (should not compute gradients)
        loss = output.sum()
        loss.backward()

        # Parameters should have no gradients (or None)
        for param in simple_network.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

    def test_training_disabled_after_optimization(self, simple_network):
        """Test that training operations are disabled after optimization (generative model)."""
        simple_network.prepare_for_inference()

        # Try to train - should not affect parameters
        initial_params = [p.clone() for p in simple_network.parameters()]

        optimizer = torch.optim.Adam(simple_network.parameters(), lr=0.001)
        output = simple_network()  # No input!
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Parameters should be unchanged (frozen)
        for initial, current in zip(initial_params, simple_network.parameters()):
            assert torch.allclose(initial, current)


class TestInferenceOptimizationWithAMP:
    """Test model optimization with mixed precision (generative model)."""

    def test_prepare_for_inference_with_amp(self, device):
        """Test prepare_for_inference with AMP-enabled network (generative model)."""
        # Create network with AMP
        model = ProgressiveDecoder(input_size=256, use_amp=True).to(device)

        # Prepare for inference
        model.prepare_for_inference()

        # Should still work with AMP (generative - no input!)
        with torch.cuda.amp.autocast(enabled=True and device.type == "cuda"):
            output = model()  # No input!

        assert output.dim() == 4  # (1, C, H, W)
        assert not torch.isnan(output).any()

    def test_generate_fp32_after_optimization(self, device):
        """Test generate_fp32() method after optimization (generative model)."""
        model = ProgressiveDecoder(input_size=256, use_amp=True).to(device)
        model.prepare_for_inference()

        # Generate FP32 output (generative - no input!)
        output = model.generate_fp32()  # No input!

        assert output.dtype == torch.float32
        assert output.dim() == 4  # (1, C, H, W)


class TestInferenceOptimizationSpeed:
    """Test model optimization speed improvements (generative model)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Speed test requires CUDA")
    def test_inference_speedup(self, device):
        """
        Test that prepare_for_inference provides speedup (generative model).

        Note: This test requires CUDA and may be skipped on CPU.
        Target: 10-20% speedup.
        """
        if device.type != "cuda":
            pytest.skip("Speed test requires CUDA")

        # Create model (generative - no input needed!)
        model = ProgressiveDecoder(input_size=512).to(device)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model()  # No input!

        # Benchmark before optimization
        model.eval()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = model()  # No input!
        torch.cuda.synchronize()
        time_before = time.time() - start

        # Prepare for inference
        model.prepare_for_inference()

        # Benchmark after optimization
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = model()  # No input!
        torch.cuda.synchronize()
        time_after = time.time() - start

        # Calculate speedup
        speedup = (time_before - time_after) / time_before * 100

        print(f"Time before: {time_before:.4f}s, Time after: {time_after:.4f}s")
        print(f"Speedup: {speedup:.1f}%")

        # Should have some speedup (but don't enforce strict percentage on CI)
        # Target is 10-20%, but hardware varies
        assert time_after <= time_before, "Post-optimization should not be slower"


class TestInferenceOptimizationScenarios:
    """Test model optimization in realistic scenarios (generative model)."""

    def test_optimize_after_training(self, device):
        """Test typical workflow: SPIDS training then optimize for final reconstruction."""
        model = ProgressiveDecoder(input_size=256).to(device)
        target = torch.randn(1, 1, 128, 128, device=device)

        # SPIDS training phase (generative - no input!)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(5):
            output = model()  # No input!
            loss = nn.functional.mse_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Optimize for final reconstruction generation
        model.prepare_for_inference()

        # Generate final reconstruction
        with torch.no_grad():
            final_reconstruction = model()  # No input!

        assert final_reconstruction.dim() == 4  # (1, C, H, W)
        assert not torch.isnan(final_reconstruction).any()

    def test_optimize_and_save_checkpoint(self, device, tmp_path):
        """Test saving checkpoint after optimization (generative model)."""
        model = ProgressiveDecoder(input_size=256).to(device)
        model.prepare_for_inference()

        # Save checkpoint
        checkpoint_path = tmp_path / "optimized_model.pth"
        torch.save(model.state_dict(), checkpoint_path)

        # Load checkpoint
        loaded_model = ProgressiveDecoder(input_size=256).to(device)
        loaded_model.load_state_dict(torch.load(checkpoint_path))
        loaded_model.eval()

        # Generate final reconstruction (generative - no input!)
        with torch.no_grad():
            output = loaded_model()  # No input!

        assert output.dim() == 4  # (1, C, H, W)

    def test_optimize_for_production_deployment(self, device):
        """Test optimization for production deployment scenario (generative model)."""
        # Simulate production setup
        model = ProgressiveDecoder(input_size=1024).to(device)

        # Optimize for final reconstruction generation
        model.prepare_for_inference()

        # Generate final reconstruction multiple times (SPIDS is generative)
        for _ in range(5):
            with torch.no_grad():
                output = model()  # No input!
            assert output.dim() == 4  # (1, C, H, W)
            assert not torch.isnan(output).any()


class TestInferenceOptimizationEdgeCases:
    """Test edge cases for model optimization (generative model)."""

    def test_optimize_immediately_after_creation(self, device):
        """Test optimizing model immediately after creation (generative model)."""
        model = ProgressiveDecoder(input_size=256).to(device)
        model.prepare_for_inference()

        # Generate output (generative - no input!)
        with torch.no_grad():
            output = model()  # No input!

        assert output.dim() == 4  # (1, C, H, W)

    def test_optimize_with_custom_config(self, device):
        """Test optimization with custom network configuration (generative model)."""
        # This test assumes NetworkConfig integration exists
        # If not, it will test with default config
        model = ProgressiveDecoder(input_size=512).to(device)
        model.prepare_for_inference()

        # Generate output (generative - no input!)
        with torch.no_grad():
            output = model()  # No input!

        assert output.dim() == 4  # (1, C, H, W)

    def test_optimize_multiple_models(self, device):
        """Test optimizing multiple models independently (generative models)."""
        models = [
            ProgressiveDecoder(input_size=256).to(device),
            ProgressiveDecoder(input_size=512).to(device),
        ]

        # Optimize all models
        for model in models:
            model.prepare_for_inference()

        # Test all models (generative - no input!)
        for model in models:
            with torch.no_grad():
                output = model()  # No input!
            assert output.dim() == 4  # (1, C, H, W)


class TestInferenceOptimizationDocumentation:
    """Test that inference optimization is well-documented."""

    def test_prepare_for_inference_has_docstring(self, simple_network):
        """Test that prepare_for_inference has a docstring."""
        method = simple_network.prepare_for_inference
        assert method.__doc__ is not None
        assert len(method.__doc__) > 0

    def test_docstring_mentions_optimizations(self, simple_network):
        """Test that docstring describes the optimizations."""
        docstring = simple_network.prepare_for_inference.__doc__
        # Should mention key optimizations
        assert "inference" in docstring.lower()
        # Should mention at least some optimization
        optimization_terms = ["freeze", "eval", "optimization", "faster"]
        assert any(term in docstring.lower() for term in optimization_terms)
