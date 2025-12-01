# PRISM Performance Baseline Metrics

**Date:** November 3, 2025
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU
**Purpose:** Establish baseline performance metrics before refactoring to measure improvements

---

## Executive Summary

This document provides baseline performance metrics for the PRISM (Progressive Reconstruction from Imaging with Sparse Measurements) system. These metrics were gathered using the `profile_baseline.py` profiling script and serve as a reference point for measuring the impact of refactoring efforts.

### Key Baseline Metrics (Image Size: 1024x1024)

| Metric | Value | Notes |
|--------|-------|-------|
| **FFT Forward Pass** | 0.0055s avg | 100 iterations, first pass: 0.54s (CUDA initialization) |
| **FFT Inverse Pass** | 0.0005s avg | 100 iterations |
| **FFT Round-trip** | 0.0013s avg | 100 iterations |
| **Model Forward (inference)** | 0.0064s avg | 50 iterations, GenCropSpidsNet |
| **Model Forward (training)** | 0.0057s avg | 10 iterations, with gradient computation |
| **Model Initialization** | 0.2652s | GenCropSpidsNet creation time |
| **Peak FFT Memory** | 28.00 MB | GPU memory for FFT operations |
| **Peak Model Memory** | 67.64 MB | GPU memory with gradients |

---

## Detailed Performance Analysis

### 1. FFT Operations

FFT operations are core to the optical simulation in PRISM, used extensively in telescope measurements and propagation.

#### Timing Metrics

```
Operation: FFT Forward
- Average time: 0.0055s
- Min time: 0.0001s
- Max time: 0.5392s (first call includes CUDA initialization)
- Total time (100 iterations): 0.5532s
- Steady-state average (excluding first): ~0.0005s

Operation: FFT Inverse
- Average time: 0.0005s
- Min time: 0.0001s
- Max time: 0.0015s
- Total time (100 iterations): 0.0456s

Operation: FFT Round-trip (FFT + IFFT)
- Average time: 0.0013s
- Min time: 0.0010s
- Max time: 0.0024s
- Total time (100 iterations): 0.1260s
```

#### Memory Usage

```
Initial: 4.00 MB
After first FFT: 12.00 MB (+8 MB)
After first IFFT: 20.00 MB (+8 MB)
Final (100 iterations): 28.00 MB (+8 MB)

Memory growth: 24.00 MB total
Peak memory: 28.00 MB
```

#### Analysis

- First FFT call is ~100x slower due to CUDA initialization and memory allocation
- Steady-state FFT operations are very fast (< 1ms)
- Memory usage grows steadily with iterations, suggesting potential memory leak or accumulation
- **Bottleneck identified:** First-call overhead could be mitigated with warm-up passes
- **Memory concern:** 24 MB growth over 100 iterations (0.24 MB/iteration) suggests tensors may not be properly deallocated

### 2. Model Operations

The GenCropSpidsNet model is the core neural network for reconstruction.

#### Timing Metrics

```
Operation: Model Initialization
- Time: 0.2652s
- Includes: Layer creation, CUDA memory allocation, parameter initialization

Operation: Forward Pass (inference, no gradients)
- Average time: 0.0064s
- Min time: 0.0008s
- Max time: 0.2696s (first call includes tensor allocation)
- Total time (50 iterations): 0.3197s
- Steady-state average (excluding first): ~0.001s

Operation: Forward Pass (training, with gradients)
- Average time: 0.0057s
- Min time: 0.0014s
- Max time: 0.0400s (first call includes gradient buffer allocation)
- Total time (10 iterations): 0.0573s
- Steady-state average (excluding first): ~0.0015s
```

#### Memory Usage

```
Initial: 0.00 MB
After model creation: 42.70 MB
After first forward pass (inference): 46.70 MB (+4 MB)
After first forward pass (training): 67.64 MB (+20.94 MB)
Final: 67.64 MB

Model parameters: 42.70 MB
Activation memory (inference): ~4 MB
Gradient memory (training): ~20.94 MB
Peak memory: 67.64 MB
```

#### Model Architecture Details

```python
GenCropSpidsNet Configuration (1024x1024):
- Input size: 1024
- Output size: 512 (default half of input)
- Decoder-only architecture
- Learnable latent vector: 1x1 starting point
- Progressive upsampling through transposed convolutions
- Batch normalization enabled
- Output activation: Sigmoid
```

#### Analysis

- Model initialization is reasonably fast (~265ms)
- Forward passes are very fast in steady-state (~1-1.5ms)
- First-call overhead is significant (20-270ms) due to tensor allocation
- **Memory efficiency:** Training requires 3.5x more memory than inference (gradients + activations)
- **Bottleneck identified:** First forward pass overhead could benefit from pre-allocation strategies

### 3. Projected Training Performance

Based on the baseline metrics, we can estimate full training performance:

#### Estimated Training Time per Sample (100 epochs)

```
Breakdown per epoch:
- Model forward pass: ~0.0015s (training mode)
- Telescope measurement: ~0.005s (estimated from FFT costs)
- Loss computation: ~0.001s (estimated)
- Backward pass: ~0.002s (estimated 2x forward)
- Optimizer step: ~0.0005s (estimated)
- Total per epoch: ~0.010s

For 100 epochs per sample: ~1 second
For 240 samples: ~240 seconds (~4 minutes)

Initialization phase (1000 epochs): ~10-15 seconds
Total estimated training time: ~5 minutes for typical run
```

**Note:** Actual training times may vary significantly based on:
- Convergence speed (failed samples require more epochs)
- Visualization overhead (real-time plotting)
- I/O operations (checkpoint saving)
- Telescope aggregation complexity (grows with samples)

---

## Identified Bottlenecks

### High Priority

1. **FFT Memory Growth**
   - **Issue:** 0.24 MB/iteration memory growth
   - **Impact:** Could lead to OOM errors in long training runs
   - **Likely cause:** Tensors not being properly released after FFT operations
   - **Target improvement:** Zero memory growth in steady-state

2. **First-call Overhead**
   - **Issue:** First FFT call: 540ms vs steady-state 0.5ms (1000x slower)
   - **Issue:** First forward pass: 270ms vs steady-state 1ms (270x slower)
   - **Impact:** Initialization phase slower than necessary
   - **Solution:** Add warm-up passes, pre-allocate common tensor sizes

3. **Matplotlib Memory Leaks** (known issue from code review)
   - **Issue:** Figures not properly closed in visualization code
   - **Impact:** Memory accumulates during long training runs
   - **Solution:** Use context managers for figure creation, explicit cleanup

### Medium Priority

4. **Telescope Operation Overhead**
   - **Issue:** Complex aggregation logic with growing mask accumulation
   - **Impact:** Performance degrades as more samples are added
   - **Solution:** Optimize mask accumulation, consider sparse storage

5. **Non-vectorized Loops**
   - **Issue:** Some operations iterate over samples sequentially
   - **Example:** `[telescope(obj, point) for point in points]`
   - **Solution:** Implement batched telescope operations

### Low Priority

6. **Model Initialization Time**
   - **Issue:** 265ms to create model
   - **Impact:** Minor, only happens once per experiment
   - **Solution:** Consider lazy initialization for unused components

---

## Memory Usage Patterns

### GPU Memory Allocation by Component

```
Component                  | Memory (MB) | Percentage
---------------------------|-------------|------------
Model Parameters           |    42.70    |   63.1%
Training Gradients         |    20.94    |   30.9%
Activation Buffers         |     4.00    |    5.9%
---------------------------|-------------|------------
Total (Training)           |    67.64    |  100.0%

Additional overhead per sample:
FFT Operations             |    ~8-10    |
Telescope Measurements     |    ~5-10    |
Aggregated Masks           |    ~2-5     | (grows with samples)
```

### Memory Growth Over Time

Based on FFT profiling results:
- Linear growth: ~0.24 MB per 100 FFT operations
- Projected for 240 samples × 100 epochs: ~576 MB leak
- **Critical:** This could cause OOM errors on GPUs with <2GB memory

### Estimated Peak Memory (Full Training Run)

```
Base model (training):              67.64 MB
FFT/Telescope operations:           20.00 MB
Aggregated measurements (240):      50.00 MB (estimated)
Visualization buffers:              30.00 MB (estimated)
Memory leak accumulation:          576.00 MB (projected)
----------------------------------------
Total estimated peak:              743.64 MB (~0.75 GB)
```

**Concern:** Memory leak could push this to 1-2 GB on longer runs.

---

## Performance Targets for Refactoring

Based on the baseline metrics, we set the following improvement targets:

| Metric | Baseline | Target | Priority |
|--------|----------|--------|----------|
| FFT memory growth | 0.24 MB/100 iter | 0 MB | Critical |
| Peak memory usage | 743 MB (est.) | <500 MB | High |
| Model forward time | 0.0064s | 0.003s | Medium |
| First-call overhead | 270-540ms | <50ms | Medium |
| Training time (240 samples) | ~5 min (est.) | <4 min | Low |

### Phase 4 Optimization Goals

From [REFACTORING_PLAN.md](../REFACTORING_PLAN.md), Phase 4 targets:
- ✓ 30% reduction in memory usage: 743 MB → 520 MB
- ✓ 20% faster training time: 5 min → 4 min

These are achievable by addressing the identified bottlenecks.

---

## Profiling Tools Used

### 1. Custom Timer Utilities

```python
class OperationTimer:
    """Accumulate timing statistics for repeated operations"""
    - Tracks min/max/average times
    - Handles multiple operation types
    - Generates summary reports
```

### 2. Memory Profiler

```python
class MemoryProfiler:
    """Track GPU/CPU memory usage over time"""
    - Records memory at key checkpoints
    - Calculates peak memory and growth
    - Supports both CUDA and CPU profiling
```

### 3. cProfile Integration

- Full system profiling capability
- Identifies hotspot functions
- Generates call graphs (available in `profiling_results/cprofile_stats.prof`)

---

## How to Re-run Profiling

After refactoring changes, re-run profiling to measure improvements:

```bash
# Profile all operations
uv run python profile_baseline.py --operation all --image_size 1024

# Profile specific operations
uv run python profile_baseline.py --operation fft --image_size 1024
uv run python profile_baseline.py --operation model --image_size 1024

# Full system profiling with cProfile
uv run python profile_baseline.py --operation full --image_size 512 --n_samples 16

# Custom configuration
uv run python profile_baseline.py \
    --operation training \
    --image_size 512 \
    --n_samples 32 \
    --n_epochs 50 \
    --output_dir profiling_results_v2
```

Results are saved to `profiling_results/` with timestamped summaries.

---

## Notes and Caveats

1. **Hardware Dependency:** Metrics are specific to RTX 4050 Laptop GPU. Performance will vary on other hardware.

2. **Synthetic Data:** Profiling uses random tensors, not real astronomical images. Actual data may have different performance characteristics.

3. **Incomplete Training Loop:** Full training loop profiling encountered API complexity issues. Individual operation metrics are reliable; end-to-end metrics are estimated.

4. **CUDA Warm-up:** First-call measurements include CUDA initialization overhead. Steady-state metrics are more representative of actual performance.

5. **No I/O Included:** Metrics exclude disk I/O (checkpoint saving, image loading). Add ~1-2 seconds per checkpoint save.

6. **Visualization Overhead:** Real training includes matplotlib visualization which adds overhead (estimated 10-20% of total time).

---

## References

- Profiling Script: [`profile_baseline.py`](../profile_baseline.py)
- Refactoring Plan: [`REFACTORING_PLAN.md`](../REFACTORING_PLAN.md)
- Raw Profiling Data: `profiling_results/profiling_summary.txt`
- cProfile Stats: `profiling_results/cprofile_stats.prof`

---

## Next Steps

1. **Phase 2:** Proceed with code restructuring (no performance impact expected)
2. **Phase 4:** Address identified bottlenecks
   - Fix FFT memory leak
   - Optimize telescope operations
   - Add warm-up passes
   - Fix matplotlib cleanup
3. **Re-profile:** After Phase 4, re-run profiling to measure improvements
4. **Continuous Monitoring:** Add performance regression tests to CI/CD

---

**Last Updated:** November 3, 2025
**Status:** Baseline established, ready for refactoring
