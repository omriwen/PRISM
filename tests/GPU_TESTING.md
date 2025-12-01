# GPU Testing Guide

This guide covers how to run and write GPU tests in the PRISM test suite.

## Table of Contents

- [Running GPU Tests](#running-gpu-tests)
- [Writing GPU Tests](#writing-gpu-tests)
- [GPU Test Fixtures Reference](#gpu-test-fixtures-reference)
- [Best Practices](#best-practices)

---

## Running GPU Tests

### Basic Commands

Run all GPU tests (skips automatically if CUDA unavailable):
```bash
uv run pytest -m gpu
```

Run GPU tests excluding slow tests:
```bash
uv run pytest -m "gpu and not slow"
```

Run specific GPU test file:
```bash
uv run pytest tests/unit/models/test_losses.py -m gpu -v
```

### Parallel Execution with pytest-xdist

GPU tests can be run in parallel, but limit workers to avoid GPU memory exhaustion:

```bash
# Run with 2 workers (recommended for GPU tests)
uv run pytest -m gpu -n 2 -v

# Run non-GPU tests with auto-detection
uv run pytest -m "not gpu" -n auto -v
```

### Skipping GPU Tests on CPU-Only Machines

GPU tests are automatically skipped on CPU-only machines using pytest markers. No special configuration needed.

To explicitly run only CPU tests:
```bash
uv run pytest -m "not gpu"
```

### Collecting GPU Tests Without Running

To see which tests will run without executing them:
```bash
uv run pytest -m gpu --collect-only
```

---

## Writing GPU Tests

### Using the `@pytest.mark.gpu` Marker

All GPU tests should use the `@pytest.mark.gpu` decorator:

```python
import pytest
import torch

@pytest.mark.gpu
def test_model_on_gpu(gpu_device):
    """Test that model works on GPU."""
    model = MyModel().to(gpu_device)
    output = model(torch.randn(1, 3, 224, 224, device=gpu_device))
    assert output.device.type == "cuda"
```

### Using the `device` Fixture (CPU + GPU Parametrization)

The `device` fixture runs tests on **both CPU and GPU** (if available):

```python
@pytest.mark.gpu  # Still needed for the GPU variant
def test_forward_pass_cross_platform(device):
    """Test forward pass on both CPU and GPU.

    This test will run twice:
    1. On CPU (always)
    2. On GPU (if CUDA available, skipped otherwise)
    """
    model = MyModel().to(device)
    x = torch.randn(1, 3, 224, 224, device=device)
    output = model(x)

    # Verify output is on correct device
    assert output.device == device
    assert output.shape == (1, 10)
```

**When to use**: Tests that should verify behavior is consistent across CPU and GPU.

### Using the `gpu_device` Fixture (GPU-Only Tests)

The `gpu_device` fixture is for **GPU-only** tests:

```python
@pytest.mark.gpu
def test_cuda_specific_feature(gpu_device):
    """Test CUDA-specific functionality.

    This test only runs on GPU and is skipped on CPU-only machines.
    """
    # Test CUDA-specific operations
    x = torch.randn(1000, 1000, device=gpu_device)
    torch.cuda.synchronize()

    # Check GPU memory usage
    assert torch.cuda.memory_allocated() > 0
```

**When to use**: Tests for GPU-specific features (memory management, multi-GPU, etc.).

### Example: Testing Loss Functions

```python
import pytest
import torch
from prism.models.losses import L1Loss

class TestL1Loss:
    """Test L1 loss on both CPU and GPU."""

    @pytest.mark.gpu
    def test_l1_loss_computation(self, device):
        """Test L1 loss computation on CPU and GPU."""
        loss_fn = L1Loss()

        pred = torch.randn(10, 3, 256, 256, device=device)
        target = torch.randn(10, 3, 256, 256, device=device)

        loss = loss_fn(pred, target)

        # Verify loss properties
        assert loss.device == device
        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # L1 loss is non-negative

    @pytest.mark.gpu
    def test_l1_loss_gradient(self, gpu_device):
        """Test L1 loss backpropagation on GPU."""
        loss_fn = L1Loss()

        pred = torch.randn(10, 3, 256, 256, device=gpu_device, requires_grad=True)
        target = torch.randn(10, 3, 256, 256, device=gpu_device)

        loss = loss_fn(pred, target)
        loss.backward()

        # Check gradients computed
        assert pred.grad is not None
        assert pred.grad.device.type == "cuda"
```

### Example: Testing Neural Network Models

```python
import pytest
import torch
from prism.models.networks import ProgressiveDecoder

class TestProgressiveDecoder:
    """Test ProgressiveDecoder on both CPU and GPU."""

    @pytest.mark.gpu
    def test_forward_pass(self, device):
        """Test forward pass on CPU and GPU."""
        model = ProgressiveDecoder(input_size=256, output_channels=1).to(device)
        output = model()

        assert output.shape == (1, 1, 256, 256)
        assert output.device == device

    @pytest.mark.gpu
    def test_training_step(self, device):
        """Test that model can be trained on CPU and GPU."""
        model = ProgressiveDecoder(input_size=256, output_channels=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Forward pass
        output = model()
        target = torch.randn_like(output, device=device)
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify gradients exist and are on correct device
        for param in model.parameters():
            assert param.grad is not None
            assert param.grad.device == device
```

---

## GPU Test Fixtures Reference

### `device` - Parametrized CPU/GPU Fixture

**Signature**: `device(request) -> torch.device`

**Description**: Returns `torch.device("cpu")` or `torch.device("cuda")` depending on parametrization.

**Behavior**:
- First run: Returns `torch.device("cpu")`
- Second run: Returns `torch.device("cuda")` (skipped if CUDA unavailable)

**Usage**:
```python
@pytest.mark.gpu
def test_with_device(device):
    tensor = torch.randn(10, device=device)
    assert tensor.device.type == device.type
```

**When to use**: Tests that should verify cross-platform consistency (CPU vs GPU).

---

### `gpu_device` - GPU-Only Fixture

**Signature**: `gpu_device() -> torch.device`

**Description**: Returns `torch.device("cuda")` or skips test if CUDA unavailable.

**Behavior**:
- Returns `torch.device("cuda")` if available
- Calls `pytest.skip("CUDA not available")` if GPU unavailable

**Usage**:
```python
@pytest.mark.gpu
def test_gpu_only(gpu_device):
    tensor = torch.randn(1000, 1000, device=gpu_device)
    assert tensor.device.type == "cuda"
```

**When to use**: Tests that only make sense on GPU (memory tests, multi-GPU, CUDA-specific ops).

---

### `cleanup_gpu_memory` - Auto-Cleanup Fixture

**Signature**: `cleanup_gpu_memory() -> None`

**Scope**: Function (runs after each test)

**Autouse**: Yes (applied automatically to all tests)

**Description**: Automatically calls `torch.cuda.empty_cache()` after each test to prevent GPU memory accumulation.

**Behavior**:
- Runs after every test function
- Clears PyTorch's GPU memory cache
- No manual invocation needed

**Usage**: Automatic, no explicit usage required.

```python
# No need to call manually - runs automatically
def test_something(gpu_device):
    large_tensor = torch.randn(10000, 10000, device=gpu_device)
    # Memory automatically cleaned up after test
```

---

### `configure_gpu_for_testing` - Session GPU Configuration

**Signature**: `configure_gpu_for_testing() -> None`

**Scope**: Session (runs once per test session)

**Autouse**: Yes (applied automatically)

**Description**: Configures GPU settings for deterministic and performant testing.

**Configuration**:
- Disables TF32 for deterministic results (`torch.backends.cuda.matmul.allow_tf32 = False`)
- Disables cuDNN TF32 (`torch.backends.cudnn.allow_tf32 = False`)
- Enables cuDNN benchmarking for performance (`torch.backends.cudnn.benchmark = True`)

**Usage**: Automatic, no explicit usage required.

**Why this matters**:
- **TF32 disabled**: Ensures numerical consistency between CPU and GPU (TF32 can cause small numerical differences)
- **cuDNN benchmark enabled**: Speeds up tests by selecting optimal convolution algorithms

---

## Best Practices

### 1. Memory Management

#### Use Cleanup Fixture (Automatic)

The `cleanup_gpu_memory` fixture automatically cleans up GPU memory after each test. No manual cleanup needed.

```python
@pytest.mark.gpu
def test_large_model(gpu_device):
    # Create large model
    model = VeryLargeModel().to(gpu_device)
    output = model(torch.randn(10, 3, 512, 512, device=gpu_device))

    # No manual cleanup needed - automatic via cleanup_gpu_memory fixture
```

#### Manual Cleanup (Advanced)

For tests that need explicit control over memory:

```python
@pytest.mark.gpu
def test_memory_intensive_operation(gpu_device):
    """Test with explicit memory management."""
    try:
        # Allocate large tensor
        x = torch.randn(5000, 5000, device=gpu_device)

        # Do computation
        result = torch.matmul(x, x.T)

        # Verify
        assert result.shape == (5000, 5000)
    finally:
        # Explicit cleanup before automatic fixture cleanup
        del x, result
        torch.cuda.empty_cache()
```

### 2. Synchronization

#### Always Synchronize for Timing

CUDA operations are asynchronous. Always synchronize before timing:

```python
import time
import pytest
import torch

@pytest.mark.gpu
def test_performance(gpu_device):
    """Test GPU performance with proper synchronization."""
    model = MyModel().to(gpu_device)
    x = torch.randn(10, 3, 224, 224, device=gpu_device)

    # Warm-up
    _ = model(x)
    torch.cuda.synchronize()  # Wait for warm-up to complete

    # Timed run
    start = time.time()
    output = model(x)
    torch.cuda.synchronize()  # Wait for computation to complete
    elapsed = time.time() - start

    assert elapsed < 0.1  # Should be fast
```

#### Synchronize for Deterministic Tests

Some operations need synchronization for deterministic results:

```python
@pytest.mark.gpu
def test_deterministic_operation(gpu_device):
    """Test with deterministic CUDA operations."""
    torch.manual_seed(42)

    x = torch.randn(100, 100, device=gpu_device)
    result = torch.matmul(x, x.T)

    torch.cuda.synchronize()  # Ensure computation complete

    # Now safe to check results
    assert result.shape == (100, 100)
```

### 3. Device Assertions

#### Always Verify Tensor Device

Explicitly check that tensors are on the expected device:

```python
@pytest.mark.gpu
def test_model_device_placement(device):
    """Verify model and tensors on correct device."""
    model = MyModel().to(device)
    x = torch.randn(10, 3, 224, 224, device=device)

    # Forward pass
    output = model(x)

    # Device assertions
    assert output.device == device
    assert output.device.type == device.type

    # Check all model parameters on correct device
    for param in model.parameters():
        assert param.device == device
```

### 4. Handling dtype Differences

#### CPU vs GPU dtype Consistency

Some operations may produce slightly different dtypes on CPU vs GPU:

```python
@pytest.mark.gpu
def test_consistent_dtype(device):
    """Test dtype consistency across devices."""
    # Explicitly set dtype
    x = torch.randn(10, 10, dtype=torch.float32, device=device)
    y = torch.randn(10, 10, dtype=torch.float32, device=device)

    result = torch.matmul(x, y)

    # Verify dtype preserved
    assert result.dtype == torch.float32
    assert result.device == device
```

#### Mixed Precision (AMP) Compatibility

For tests involving Automatic Mixed Precision:

```python
import pytest
import torch
from torch.cuda.amp import autocast

@pytest.mark.gpu
def test_amp_compatible_operation(gpu_device):
    """Test operation works with AMP."""
    model = MyModel().to(gpu_device)
    x = torch.randn(10, 3, 224, 224, device=gpu_device)

    # Test with AMP
    with autocast():
        output = model(x)

    # AMP may use float16 internally
    assert output.device.type == "cuda"
    assert output.dtype in [torch.float16, torch.float32]
```

### 5. Parametrization Best Practices

#### Use `device` Fixture for Cross-Platform Tests

When testing that behavior is identical on CPU and GPU:

```python
@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_model_batching(device, batch_size):
    """Test model with different batch sizes on CPU and GPU.

    This creates 6 test cases:
    - batch_size=1, device=cpu
    - batch_size=1, device=cuda
    - batch_size=8, device=cpu
    - batch_size=8, device=cuda
    - batch_size=32, device=cpu
    - batch_size=32, device=cuda
    """
    model = MyModel().to(device)
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    output = model(x)

    assert output.shape == (batch_size, 10)
    assert output.device == device
```

#### Use `gpu_device` for GPU-Specific Tests

When testing GPU-only features:

```python
@pytest.mark.gpu
@pytest.mark.parametrize("num_gpus", [1, 2, 4])
def test_multi_gpu_training(gpu_device, num_gpus):
    """Test multi-GPU training (GPU-only)."""
    if torch.cuda.device_count() < num_gpus:
        pytest.skip(f"Need {num_gpus} GPUs, only {torch.cuda.device_count()} available")

    # Multi-GPU test logic
    model = torch.nn.DataParallel(MyModel(), device_ids=list(range(num_gpus)))
    model = model.to(gpu_device)

    # Test training
    x = torch.randn(num_gpus * 4, 3, 224, 224, device=gpu_device)
    output = model(x)

    assert output.shape[0] == num_gpus * 4
```

### 6. Common Patterns Summary

```python
# Pattern 1: Cross-platform test (CPU + GPU)
@pytest.mark.gpu
def test_cross_platform(device):
    model = Model().to(device)
    result = model(torch.randn(10, device=device))
    assert result.device == device

# Pattern 2: GPU-only test
@pytest.mark.gpu
def test_gpu_only(gpu_device):
    large_model = HugeModel().to(gpu_device)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() > 0

# Pattern 3: Performance test with timing
@pytest.mark.gpu
def test_performance(gpu_device):
    model = Model().to(gpu_device)
    x = torch.randn(100, device=gpu_device)

    torch.cuda.synchronize()  # Before timing
    start = time.time()
    output = model(x)
    torch.cuda.synchronize()  # After timing
    elapsed = time.time() - start

    assert elapsed < threshold

# Pattern 4: Parametrized cross-platform test
@pytest.mark.gpu
@pytest.mark.parametrize("size", [64, 128, 256])
def test_sizes(device, size):
    x = torch.randn(size, size, device=device)
    result = torch.fft.fft2(x)
    assert result.shape == (size, size)
    assert result.device == device
```

---

## Additional Resources

- **PyTorch CUDA Documentation**: https://pytorch.org/docs/stable/cuda.html
- **pytest Documentation**: https://docs.pytest.org/
- **pytest-xdist for Parallel Testing**: https://pytest-xdist.readthedocs.io/

---

## Troubleshooting

### "CUDA out of memory" errors

Reduce batch size or grid size in tests, or limit pytest-xdist workers:
```bash
uv run pytest -m gpu -n 1 -v
```

### Tests pass on CPU but fail on GPU

Check for:
- Numerical precision differences (use `torch.allclose` with appropriate tolerances)
- Missing synchronization (`torch.cuda.synchronize()`)
- Device mismatches (tensor on CPU, model on GPU)

### GPU tests always skipped

Check CUDA availability:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

If False, ensure CUDA-enabled PyTorch is installed:
```bash
uv run pip list | grep torch
```
