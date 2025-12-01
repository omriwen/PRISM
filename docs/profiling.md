# PRISM Training Profiler Documentation

**Version**: 1.0
**Last Updated**: 2025-12-01

Low-overhead performance profiling for PyTorch training loops with post-hoc analysis, bottleneck detection, and interactive visualizations.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [Basic Usage](#basic-usage)
   - [Command-Line Integration](#command-line-integration)
   - [Programmatic Usage](#programmatic-usage)
   - [Context Manager for Custom Regions](#context-manager-for-custom-regions)
5. [CLI Commands](#cli-commands)
   - [analyze](#prism-profile-analyze)
   - [view](#prism-profile-view)
   - [compare](#prism-profile-compare)
   - [export](#prism-profile-export)
6. [Understanding Visualizations](#understanding-visualizations)
   - [Timing Breakdown Chart](#timing-breakdown-chart)
   - [Memory Timeline](#memory-timeline)
   - [Flame Graph](#flame-graph)
   - [Sunburst Chart](#sunburst-chart)
   - [Bottleneck Summary](#bottleneck-summary)
7. [Interpreting Bottlenecks](#interpreting-bottlenecks)
   - [CPU-GPU Sync Issues](#cpu-gpu-sync-issues)
   - [Memory Leaks](#memory-leaks)
   - [Hot Operations](#hot-operations)
8. [Integration Examples](#integration-examples)
9. [Configuration Reference](#configuration-reference)
10. [Performance Impact](#performance-impact)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The PRISM profiler is a **low-overhead performance analysis tool** designed specifically for PyTorch training loops. It collects timing, memory, and operation data during training with minimal impact (<2% overhead), then provides rich post-hoc analysis with interactive visualizations.

### Key Features

- **Minimal Overhead**: <2% training slowdown via non-blocking CUDA events and adaptive sampling
- **Automatic Bottleneck Detection**: Identifies CPU-GPU sync, memory leaks, and hot operations
- **Rich Visualizations**: Static reports (matplotlib), interactive dashboards (Plotly), flame graphs
- **Hierarchical Call Graphs**: Understand operation hierarchy with flame graphs and sunburst charts
- **Memory Tracking**: GPU memory usage with leak detection via linear regression
- **Multiple Export Formats**: Chrome trace, JSON, TensorBoard integration
- **Seamless Integration**: Works with existing PRISMTrainer via callbacks

### Design Philosophy

> **Profile with minimal overhead during training; defer all analysis and visualization to post-processing.**

The profiler follows a two-phase approach:

1. **Collection Phase** (during training): Record timing/memory events non-blocking
2. **Analysis Phase** (after training): Analyze data, detect bottlenecks, generate visualizations

This design ensures profiling doesn't interfere with training performance.

---

## Installation & Setup

### Prerequisites

The profiler is included with PRISM and has no additional dependencies beyond the standard PRISM installation:

- Python 3.11+
- PyTorch 2.0+
- matplotlib (for static visualizations)
- plotly (for interactive visualizations)
- dash (for web dashboard)

All dependencies are automatically installed with PRISM:

```bash
# Standard PRISM installation includes profiler
uv sync

# Activate environment
source .venv/bin/activate
```

### Verify Installation

```python
from prism.profiling import TrainingProfiler, ProfilerConfig

# Should import without errors
print("Profiler installed successfully!")
```

---

## Quick Start

**TL;DR**: Enable profiling with a single flag:

```bash
# Enable profiling during training
python main.py --profile --profile-output my_profile.pt

# Analyze the profile
prism profile analyze my_profile.pt

# Launch interactive viewer
prism profile view my_profile.pt
```

That's it! The profiler will:
1. Collect timing and memory data during training
2. Save profile to `my_profile.pt`
3. Allow post-hoc analysis via CLI or Python API

---

## Basic Usage

### Command-Line Integration

The easiest way to use the profiler is via the `--profile` flag in `main.py`:

```bash
# Enable profiling (saves to runs/{name}/profile.pt by default)
python main.py --profile

# Specify custom output path
python main.py --profile --profile-output experiments/run_001/profile.pt

# Profile with custom sampling rate (via environment or code modification)
python main.py --profile  # Uses default 10% sampling
```

### Programmatic Usage

For more control, use the profiler programmatically:

```python
from prism.profiling import TrainingProfiler, ProfilerConfig
from prism.core.trainers import PRISMTrainer

# 1. Configure profiler
config = ProfilerConfig(
    enabled=True,
    sample_rate=0.1,          # Profile 10% of epochs
    collect_memory=True,       # Track GPU memory
    collect_timing=True,       # Collect timing data
    collect_gpu_ops=True,      # Detailed operator stats
    adaptive_sampling=True,    # More sampling during early training
)

# 2. Create profiler instance
profiler = TrainingProfiler(config)

# 3. Pass callback to trainer
trainer = PRISMTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    measurement_system=measurement_system,
    args=args,
    device=device,
    callbacks=[profiler.callback],  # Add profiler callback
)

# 4. Run training (profiling happens automatically)
trainer.run_progressive_training(...)

# 5. Save profile data
profiler.save("my_profile.pt")
print("Profile saved! Analyze with: prism profile analyze my_profile.pt")
```

### Context Manager for Custom Regions

Profile specific code regions using the context manager:

```python
from prism.profiling import TrainingProfiler, ProfilerConfig

profiler = TrainingProfiler(ProfilerConfig(enabled=True))

# Profile custom regions
with profiler.profile_region("data_loading"):
    data = load_batch()

with profiler.profile_region("forward_pass"):
    output = model(data)

with profiler.profile_region("backward_pass"):
    loss.backward()

# Save and analyze
profiler.save("custom_regions.pt")
```

**Note**: The context manager requires the profiler to be actively sampling. It works best when integrated with the callback system or when manually controlling sampling.

---

## CLI Commands

The profiler provides a rich CLI for post-training analysis via `prism profile` subcommands.

### `prism profile analyze`

Analyze a profile and print a summary report.

**Usage**:
```bash
prism profile analyze <profile_path> [options]
```

**Options**:
- `--output`, `-o`: Save report to file instead of printing
- `--format`: Output format (`txt`, `html`, `json`); default: `txt`

**Examples**:

```bash
# Print analysis to console
prism profile analyze my_profile.pt

# Save report to file
prism profile analyze my_profile.pt --output report.txt

# Generate JSON report for programmatic parsing
prism profile analyze my_profile.pt --output report.json --format json
```

**Output Example**:
```
============================================================
PRISM Training Profile Report
============================================================

Summary:
  Total samples: 50
  Total epochs: 500
  Total time: 12543.2 ms
  Avg epoch time: 25.09 ms
  Peak GPU memory: 1024.5 MB

Top Operations:
  1. forward_pass: 6234.1 ms (245 calls)
  2. backward_pass: 4123.4 ms (245 calls)
  3. optimizer_step: 1234.5 ms (245 calls)
  4. loss_computation: 456.7 ms (245 calls)
  5. measurement_system: 494.5 ms (50 calls)

Bottlenecks Detected:
  [HIGH] hot_operation: 'forward_pass': 49.7% of total time
    Fix: Consider optimizing forward_pass
  [MEDIUM] hot_operation: 'backward_pass': 32.9% of total time
    Fix: Consider optimizing backward_pass

============================================================
```

---

### `prism profile view`

Launch an interactive web-based profile viewer with visualizations.

**Usage**:
```bash
prism profile view <profile_path> [options]
```

**Options**:
- `--port`: Port for dashboard server; default: `8051`

**Examples**:

```bash
# Launch viewer on default port 8051
prism profile view my_profile.pt

# Use custom port
prism profile view my_profile.pt --port 8080
```

The viewer opens a web dashboard at `http://localhost:8051` with tabs for:
- **Summary**: High-level statistics and metrics
- **Operations**: Interactive bar chart of operation times
- **Memory**: GPU memory timeline with leak detection
- **Call Graph**: Interactive sunburst chart for drill-down
- **Bottlenecks**: Detected performance issues with recommendations

Press `Ctrl+C` to stop the server.

---

### `prism profile compare`

Compare two profile files side-by-side.

**Usage**:
```bash
prism profile compare <profile1> <profile2> [options]
```

**Options**:
- `--output`, `-o`: Save comparison report to file

**Examples**:

```bash
# Compare two profiles
prism profile compare baseline.pt optimized.pt

# Save comparison
prism profile compare before.pt after.pt --output comparison.txt
```

**Output Example**:
```
Profile Comparison
============================================================
Metric                    Profile 1         Profile 2
------------------------------------------------------------
total_samples                    50                50
total_epochs                    500               500
total_time_ms              12543.20          8234.10
avg_epoch_time_ms             25.09             16.47
peak_gpu_memory_mb          1024.50           896.30
memory_leak_detected          False             False
```

Use this to validate optimizations or compare different configurations.

---

### `prism profile export`

Export profile to different formats for external tools.

**Usage**:
```bash
prism profile export <profile_path> --format <format> --output <output_path>
```

**Options**:
- `--format`: Export format (required)
  - `chrome-trace`: Chrome DevTools trace format (view at `chrome://tracing`)
  - `csv`: CSV format for spreadsheet analysis
  - `tensorboard`: TensorBoard-compatible format (future)
- `--output`, `-o`: Output file path (required)

**Examples**:

```bash
# Export to Chrome trace format
prism profile export my_profile.pt --format chrome-trace --output trace.json

# Open in Chrome:
# 1. Navigate to chrome://tracing
# 2. Click "Load" and select trace.json
# 3. Explore interactive timeline

# Export to CSV for analysis in Excel/Python
prism profile export my_profile.pt --format csv --output profile_data.csv
```

---

## Understanding Visualizations

The profiler generates multiple visualization types, each revealing different aspects of training performance.

### Timing Breakdown Chart

**Type**: Horizontal bar chart
**Shows**: Top operations by total time
**Use for**: Identifying which operations consume the most time

**Interpretation**:
- **Long bars**: Operations taking the most time (candidates for optimization)
- **Many calls**: High call count may indicate loop overhead
- **Short bars**: Minor operations (usually safe to ignore)

**Example**:
```
forward_pass     ████████████████████████ 6234.1 ms
backward_pass    ████████████████ 4123.4 ms
optimizer_step   ████ 1234.5 ms
loss_computation █ 456.7 ms
```

**Actionable Insights**:
- If `forward_pass` dominates: Profile model architecture, consider operator fusion
- If `backward_pass` dominates: Check gradient accumulation, mixed precision
- If `optimizer_step` dominates: Consider optimizer choice (Adam vs SGD)

---

### Memory Timeline

**Type**: Line chart with trend line
**Shows**: GPU memory usage over training
**Use for**: Detecting memory leaks, understanding memory patterns

**Interpretation**:
- **Flat line**: Stable memory usage (good!)
- **Upward slope**: Potential memory leak (red trend line)
- **Spikes**: Large allocations (check peak sample for cause)
- **Oscillations**: Normal allocation/deallocation patterns

**Example**:
```
GPU Memory (MB)
1200 |                              *  *
1000 |                    *    *
 800 |          *    *
 600 |    *
 400 | *
     +----------------------------------
       0    100   200   300   400   500
              Sample Index
```

**Actionable Insights**:
- **Memory leak detected**: Look for tensors retained in lists/closures
- **High peak memory**: Consider gradient checkpointing or batch size reduction
- **Oscillating pattern**: Normal if cleaning up between samples

**Common Causes of Memory Leaks**:
```python
# BAD: Accumulating tensors in lists
losses = []
for epoch in range(n_epochs):
    loss = criterion(output, target)
    losses.append(loss)  # Retains computation graph!

# GOOD: Detach or use .item()
losses = []
for epoch in range(n_epochs):
    loss = criterion(output, target)
    losses.append(loss.item())  # Only stores scalar
```

---

### Flame Graph

**Type**: Hierarchical stacked bars
**Shows**: Call hierarchy and time distribution
**Use for**: Understanding nested operation relationships

**How to Read**:
- **Width**: Proportional to time spent in that function
- **Height**: Call stack depth (deeper = more nested)
- **Color**: Different operations (color by depth or module)
- **Hover**: Shows exact timing and percentage

**Example Structure**:
```
┌──────────────────────────────────────────────────┐
│              root (100%)                         │ <- Total time
├─────────────────┬──────────────┬────────────────┤
│  forward (50%)  │ backward(32%)│  other (18%)   │ <- Top-level ops
├────┬────┬───────┼─────┬────────┤                │
│conv│relu│linear │ ... │        │                │ <- Sub-operations
└────┴────┴───────┴─────┴────────┴────────────────┘
```

**Actionable Insights**:
- **Wide bars at depth 0**: Hot top-level functions
- **Many narrow bars**: Fragmented execution (consider batching)
- **Deep stacks**: Complex call hierarchies (consider inlining)

---

### Sunburst Chart

**Type**: Interactive circular hierarchy
**Shows**: Same as flame graph but radial layout
**Use for**: Interactive drill-down into operation hierarchy

**How to Read**:
- **Center**: Root node (total time)
- **Rings**: Each ring is one level deeper in call hierarchy
- **Angle**: Proportional to time spent
- **Click**: Drill down into that operation

**Interaction**:
1. **Click** a segment to zoom in
2. **Hover** to see exact timing and percentage
3. **Click center** to zoom back out

**Example**:
```
        ┌─────────────┐
        │    root     │ <- Click to zoom out
    ┌───┼─────────────┼───┐
    │fwd│    bwd      │opt│ <- Click to drill down
┌───┼───┼──┬───┬──┬───┼───┼───┐
│...│...│..│...│..│...│...│...│ <- Detailed operations
```

**Best Practices**:
- Start at root to see overall distribution
- Click large segments to investigate hotspots
- Look for unexpected large segments (bugs, inefficiencies)

---

### Bottleneck Summary

**Type**: Table with severity and recommendations
**Shows**: Automatically detected performance issues
**Use for**: Quick prioritization of optimization efforts

**Severity Levels**:
- **HIGH** (red): >30% of time or critical issue → Fix immediately
- **MEDIUM** (yellow): 10-30% of time → Investigate when possible
- **LOW** (green): <10% of time → Low priority

**Example Table**:
```
┌──────────────────┬──────────┬─────────────────────────────────────┬─────────────────────────────┐
│ Type             │ Severity │ Description                         │ Recommendation              │
├──────────────────┼──────────┼─────────────────────────────────────┼─────────────────────────────┤
│ hot_operation    │ HIGH     │ 'forward_pass': 49.7% of total time│ Consider optimizing forward │
│ memory_leak      │ HIGH     │ Memory leak: 12.5 MB/sample         │ Check tensors in lists      │
│ cpu_gpu_sync     │ MEDIUM   │ CPU-GPU sync: 15.2% of GPU time     │ Avoid .item() in loop       │
└──────────────────┴──────────┴─────────────────────────────────────┴─────────────────────────────┘
```

**Priority Order**:
1. Fix HIGH severity issues first
2. Address MEDIUM issues if easy wins
3. LOW issues only if performance still inadequate

---

## Interpreting Bottlenecks

The profiler automatically detects four types of bottlenecks:

### CPU-GPU Sync Issues

**What it means**: CPU waits for GPU results (pipeline stall)

**Detection criteria**: Sync time > 10% of total GPU time

**Common causes**:
```python
# BAD: Forces CPU-GPU sync
for epoch in range(n_epochs):
    loss = criterion(output, target)
    loss_value = loss.item()  # SYNC! CPU waits for GPU
    print(f"Loss: {loss_value}")

# GOOD: Batch logging
losses = []
for epoch in range(n_epochs):
    loss = criterion(output, target)
    losses.append(loss.detach())  # No sync
# Log after loop
print(f"Losses: {[l.item() for l in losses]}")  # Single sync
```

**Other causes**:
- `.cpu()` calls in training loop
- `.numpy()` conversions during training
- Excessive logging/printing
- Metrics computed on CPU

**Fix**:
1. Defer CPU operations to end of epoch/sample
2. Use `.detach()` instead of `.item()` in loops
3. Keep tensors on GPU as long as possible
4. Use `torch.cuda.synchronize()` only when necessary

---

### Memory Leaks

**What it means**: Memory usage grows linearly over training

**Detection criteria**: Linear memory growth > 10 MB/sample

**Common causes**:

```python
# CAUSE 1: Accumulating tensors with computation graphs
self.losses = []  # Instance variable
def training_step(self):
    loss = self.criterion(output, target)
    self.losses.append(loss)  # Leak! Retains graph

# FIX: Detach or use .item()
self.losses.append(loss.detach())  # Or loss.item()

# CAUSE 2: Closure captures in callbacks
def create_callback(tensor):
    def callback():
        print(tensor.sum())  # Captures tensor!
    return callback

# FIX: Capture only what's needed
def create_callback(value):
    def callback():
        print(value)  # Captures scalar
    return callback

# CAUSE 3: Matplotlib figures not closed
for epoch in range(n_epochs):
    fig, ax = plt.subplots()
    ax.plot(data)
    # Leak! Figure not closed

# FIX: Always close figures
for epoch in range(n_epochs):
    fig, ax = plt.subplots()
    ax.plot(data)
    plt.close(fig)  # Release memory
```

**Debugging memory leaks**:
1. Check the profiler's memory timeline for slope
2. Use `torch.cuda.empty_cache()` periodically
3. Use `del` on large tensors when done
4. Profile with `torch.profiler` for detailed allocation traces

---

### Hot Operations

**What it means**: Single operation consumes >10% of total time

**Detection criteria**: Operation time > `bottleneck_threshold_pct` (default 10%)

**Interpretation**:
- **>50%**: Extreme hotspot, likely the main algorithm (expected)
- **30-50%**: Significant hotspot, investigate if unexpected
- **10-30%**: Minor hotspot, optimize if easy

**Optimization strategies**:

```python
# Hot operation: forward_pass (49.7% of time)

# 1. Profile the model itself
with torch.profiler.profile() as prof:
    output = model(input)
print(prof.key_averages().table(sort_by="cuda_time_total"))

# 2. Check for inefficient operations
# - Small kernels (use operator fusion)
# - Tensor copies (reduce data movement)
# - CPU operations (move to GPU)

# 3. Use mixed precision (if not already)
with torch.cuda.amp.autocast():
    output = model(input)  # 20-30% faster

# 4. Consider architectural changes
# - Reduce layer count
# - Use more efficient operations (GroupNorm vs LayerNorm)
# - Fuse operations (Conv+ReLU -> Conv with inplace ReLU)
```

**PRISM-specific hotspots**:
- `measurement_system`: Expected in PRISM (Fraunhofer diffraction)
- `forward_pass`: Model complexity (consider simplifying)
- `loss_computation`: SSIM can be slow (use mixed precision)

---

### Memory-Bound Operations

**What it means**: Operation limited by memory bandwidth, not compute

**Detection criteria**: Low compute intensity (FLOPS/byte ratio)

**Characteristics**:
- High memory usage
- Low GPU utilization
- Bandwidth-heavy operations (copies, transposes)

**Common causes in PRISM**:
- Large FFTs (measurement system)
- Image transformations (crops, resizes)
- Large tensor reshapes

**Fixes**:
```python
# 1. Reduce memory movement
# BAD: Multiple copies
temp1 = tensor.cpu()
temp2 = temp1.numpy()
result = torch.from_numpy(temp2).cuda()

# GOOD: Minimize copies
result = tensor.clone()  # Stay on same device

# 2. Use in-place operations
# BAD: New allocation
tensor = tensor * 2

# GOOD: In-place
tensor *= 2  # Or tensor.mul_(2)

# 3. Fuse operations
# BAD: Separate passes
x = x + bias
x = torch.relu(x)

# GOOD: Fused (if possible)
x = torch.relu(x + bias)
```

---

## Integration Examples

### Example 1: Basic Integration with PRISMTrainer

```python
from prism.profiling import TrainingProfiler, ProfilerConfig
from prism.core.trainers import PRISMTrainer

# Configure profiler
config = ProfilerConfig(
    enabled=True,
    sample_rate=0.1,  # Profile 10% of epochs
)

profiler = TrainingProfiler(config)

# Create trainer with profiler callback
trainer = PRISMTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    measurement_system=measurement_system,
    args=args,
    device=device,
    callbacks=[profiler.callback],
)

# Train (profiling happens automatically)
trainer.run_progressive_training(
    target_image=target,
    sample_centers=sample_centers,
)

# Save profile
profiler.save("runs/my_experiment/profile.pt")
print("Profile saved! Analyze with: prism profile analyze runs/my_experiment/profile.pt")
```

---

### Example 2: Custom Profiling Regions

```python
from prism.profiling import TrainingProfiler, ProfilerConfig

profiler = TrainingProfiler(ProfilerConfig(enabled=True))

# Profile specific training phases
for sample_idx, center in enumerate(sample_centers):
    with profiler.profile_region("data_preparation"):
        measurement = measurement_system.measure(image, reconstruction, [center])

    for epoch in range(n_epochs):
        with profiler.profile_region("forward"):
            output = model()

        with profiler.profile_region("loss"):
            loss = criterion(output, measurement)

        with profiler.profile_region("backward"):
            optimizer.zero_grad()
            loss.backward()

        with profiler.profile_region("optimizer"):
            optimizer.step()

profiler.save("detailed_profile.pt")
```

---

### Example 3: Profiling with Multiple Callbacks

```python
from prism.profiling import TrainingProfiler, ProfilerConfig
from prism.core.convergence import ConvergenceMonitor

# Create multiple callbacks
profiler = TrainingProfiler(ProfilerConfig(enabled=True))
convergence = ConvergenceMonitor(patience=10)

# Pass all callbacks to trainer
trainer = PRISMTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    measurement_system=measurement_system,
    args=args,
    device=device,
    callbacks=[
        profiler.callback,      # Profiling
        convergence.callback,   # Convergence monitoring
    ],
)

trainer.run_progressive_training(...)

# Both callbacks collect data independently
profiler.save("profile.pt")
print(f"Converged: {convergence.has_converged}")
```

---

### Example 4: Post-Training Analysis in Python

```python
from prism.profiling import ProfileAnalyzer
from prism.profiling.visualization import ProfilePlotter
from prism.profiling.call_graph import CallGraphBuilder

# Load profile
analyzer = ProfileAnalyzer("my_profile.pt")

# 1. Get summary statistics
summary = analyzer.get_summary()
print(f"Total time: {summary['total_time_ms']:.1f} ms")
print(f"Peak memory: {summary['peak_gpu_memory_mb']:.1f} MB")

# 2. Get top operations
top_ops = analyzer.get_top_operations(n=10)
for i, op in enumerate(top_ops, 1):
    print(f"{i}. {op['name']}: {op['total_ms']:.1f} ms")

# 3. Detect bottlenecks
bottlenecks = analyzer.identify_bottlenecks()
for b in bottlenecks:
    print(f"[{b.severity}] {b.type.value}: {b.description}")
    print(f"  Fix: {b.recommendation}")

# 4. Generate report
report = analyzer.get_efficiency_report()
print(report)

# 5. Create visualizations
plotter = ProfilePlotter(analyzer)
plotter.plot_timing_breakdown().savefig("timing.png")
plotter.plot_memory_timeline().savefig("memory.png")
plotter.create_report("full_report.png")

# 6. Build call graph
builder = CallGraphBuilder()
call_graph = builder.build_from_regions(analyzer.data.region_times)
print(f"Root time: {call_graph.total_time_ms:.1f} ms")
```

---

### Example 5: Comparing Before/After Optimization

```python
from prism.profiling import ProfileAnalyzer

# Profile before optimization
analyzer_before = ProfileAnalyzer("before_optimization.pt")
before_summary = analyzer_before.get_summary()

# Profile after optimization
analyzer_after = ProfileAnalyzer("after_optimization.pt")
after_summary = analyzer_after.get_summary()

# Compare metrics
print("Optimization Impact:")
print(f"  Time: {before_summary['total_time_ms']:.1f} ms -> {after_summary['total_time_ms']:.1f} ms")
speedup = before_summary['total_time_ms'] / after_summary['total_time_ms']
print(f"  Speedup: {speedup:.2f}x")

memory_reduction = before_summary['peak_gpu_memory_mb'] - after_summary['peak_gpu_memory_mb']
print(f"  Memory saved: {memory_reduction:.1f} MB")

# Compare bottlenecks
before_bottlenecks = analyzer_before.identify_bottlenecks()
after_bottlenecks = analyzer_after.identify_bottlenecks()
print(f"  Bottlenecks fixed: {len(before_bottlenecks) - len(after_bottlenecks)}")
```

---

## Configuration Reference

The `ProfilerConfig` dataclass controls all profiling behavior:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `True` | Master switch for profiling |
| `collect_timing` | bool | `True` | Collect operation timing data |
| `collect_memory` | bool | `True` | Track GPU/CPU memory usage |
| `collect_gpu_ops` | bool | `True` | Detailed GPU operator statistics |
| `sample_rate` | float | `0.1` | Fraction of epochs to profile (0.1 = 10%) |
| `adaptive_sampling` | bool | `True` | Increase rate during early training |
| `use_cuda_events` | bool | `True` | Use CUDA events for GPU timing |
| `cuda_sync_per_epoch` | bool | `False` | Sync every epoch (debug only) |
| `warmup_samples` | int | `2` | Warmup samples for torch.profiler |
| `active_samples` | int | `5` | Active profiling samples |
| `record_shapes` | bool | `True` | Record tensor shapes |
| `with_flops` | bool | `True` | Estimate FLOPs |
| `with_modules` | bool | `True` | Record module hierarchy |
| `output_dir` | Path \| None | `None` | Output directory (defaults to experiment dir) |
| `storage_format` | str | `"binary"` | Storage format (`"binary"` or `"json"`) |
| `export_chrome_trace` | bool | `True` | Export Chrome trace format |
| `bottleneck_threshold_pct` | float | `10.0` | Bottleneck detection threshold (%) |
| `memory_leak_threshold_mb` | float | `10.0` | Memory leak threshold (MB/sample) |

### Configuration Examples

**Minimal overhead** (for production training):
```python
config = ProfilerConfig(
    enabled=True,
    sample_rate=0.05,          # Only 5% of epochs
    collect_gpu_ops=False,     # Skip detailed operator stats
    with_flops=False,          # Skip FLOP estimation
)
```

**Debugging** (maximum detail):
```python
config = ProfilerConfig(
    enabled=True,
    sample_rate=0.5,           # Profile 50% of epochs
    cuda_sync_per_epoch=True,  # Sync more frequently
    record_shapes=True,        # Record tensor shapes
    with_flops=True,           # Estimate FLOPs
    bottleneck_threshold_pct=5.0,  # Stricter detection
)
```

**Memory leak detection**:
```python
config = ProfilerConfig(
    enabled=True,
    collect_memory=True,       # Essential for leak detection
    sample_rate=0.2,           # More samples for regression
    memory_leak_threshold_mb=5.0,  # Sensitive threshold
)
```

---

## Performance Impact

The profiler is designed for minimal overhead:

| Configuration | Overhead | Use Case |
|--------------|----------|----------|
| Default (10% sampling) | **<2%** | General profiling |
| Minimal (5% sampling) | **<1%** | Production monitoring |
| Debug (50% sampling) | **3-5%** | Debugging bottlenecks |
| Full profiling (100%) | **8-12%** | Single-run analysis |

**Overhead breakdown**:
- CUDA events (non-blocking): ~0.5%
- Memory tracking: ~0.5%
- torch.profiler (5 samples): ~1%
- Total: <2% with defaults

**Best practices for minimal overhead**:
1. Use default `sample_rate=0.1` for training runs
2. Enable `adaptive_sampling` (more data early, less later)
3. Use `cuda_sync_per_epoch=False` (batch syncs)
4. Disable `collect_gpu_ops` if not needed

**When to use higher overhead settings**:
- Debugging specific performance issues
- One-off profiling runs
- Comparing before/after optimizations
- Investigating rare events (needs more samples)

---

## Troubleshooting

### "No CUDA device available"

**Symptom**: Profiler reports "No CUDA device" or timing data is empty.

**Cause**: Running on CPU-only machine.

**Fix**:
- Profiler still works on CPU (timing only, no CUDA events)
- Disable CUDA-specific features:
  ```python
  config = ProfilerConfig(
      enabled=True,
      use_cuda_events=False,  # Disable CUDA timing
      collect_gpu_ops=False,  # No GPU operators
  )
  ```

---

### "Profile file is empty or corrupted"

**Symptom**: `prism profile analyze` fails to load profile.

**Cause**: Training interrupted before `profiler.save()` was called.

**Fix**:
1. Ensure training completes successfully
2. Call `profiler.save()` even if training fails:
   ```python
   try:
       trainer.run_progressive_training(...)
   finally:
       profiler.save("profile_partial.pt")  # Save even on error
   ```

---

### "Memory leak detected but code looks clean"

**Symptom**: Profiler reports memory leak but code appears correct.

**Possible causes**:
1. **Matplotlib figures not closed**: Always use `plt.close(fig)`
2. **Cached gradients**: Call `optimizer.zero_grad()` properly
3. **Python references**: Use `del` on large objects
4. **CUDA cache growth**: Normal if `torch.cuda.empty_cache()` not called

**Debugging**:
```python
import gc
import torch

# Force garbage collection and CUDA cache clear
gc.collect()
torch.cuda.empty_cache()

# Check for retained tensors
for obj in gc.get_objects():
    if torch.is_tensor(obj):
        print(f"Tensor: {obj.shape}, device: {obj.device}")
```

---

### "Profiler slows down training significantly"

**Symptom**: Training is much slower with profiling enabled.

**Cause**: Too high `sample_rate` or `cuda_sync_per_epoch=True`.

**Fix**:
```python
config = ProfilerConfig(
    enabled=True,
    sample_rate=0.05,          # Reduce from 0.1 to 0.05
    cuda_sync_per_epoch=False,  # Must be False for low overhead
    collect_gpu_ops=False,     # Disable if not needed
)
```

---

### "Bottleneck detection reports too many issues"

**Symptom**: Bottleneck summary shows many MEDIUM/LOW issues.

**Cause**: Threshold too low for your workload.

**Fix**:
```python
config = ProfilerConfig(
    enabled=True,
    bottleneck_threshold_pct=20.0,  # Increase from 10.0
)
```

Or filter in analysis:
```python
bottlenecks = analyzer.identify_bottlenecks()
critical = [b for b in bottlenecks if b.severity == "high"]
print(f"Critical bottlenecks: {len(critical)}")
```

---

### "Chrome trace file won't load"

**Symptom**: Chrome trace export fails to load in `chrome://tracing`.

**Cause**: File too large or malformed.

**Fix**:
1. Reduce profile size:
   ```python
   config = ProfilerConfig(
       enabled=True,
       sample_rate=0.05,  # Smaller profile
   )
   ```
2. Check file size (Chrome has ~100MB limit)
3. Try JSON export instead:
   ```bash
   prism profile export profile.pt --format json --output profile.json
   ```

---

### "Interactive viewer doesn't start"

**Symptom**: `prism profile view` fails or hangs.

**Cause**: Port already in use or Dash not installed.

**Fix**:
1. Try different port:
   ```bash
   prism profile view profile.pt --port 8080
   ```
2. Check Dash installation:
   ```bash
   uv add dash plotly dash-bootstrap-components
   ```
3. Check for port conflicts:
   ```bash
   # Linux/macOS
   lsof -i :8051

   # Windows
   netstat -ano | findstr :8051
   ```

---

## Summary

The PRISM profiler provides a powerful, low-overhead tool for understanding and optimizing training performance:

1. **Enable profiling**: Add `--profile` flag or use programmatic API
2. **Analyze results**: Use `prism profile analyze` for text reports
3. **Explore interactively**: Use `prism profile view` for web dashboard
4. **Compare runs**: Use `prism profile compare` to validate optimizations
5. **Export data**: Use `prism profile export` for external tools

**Key takeaways**:
- <2% overhead with default settings
- Automatic bottleneck detection saves investigation time
- Rich visualizations (flame graphs, sunburst) reveal operation hierarchy
- Seamless integration with existing PRISMTrainer workflow

**Next steps**:
- Profile a baseline training run
- Identify top bottlenecks
- Apply optimizations (see [PERFORMANCE_OPTIMIZATION_GUIDE.md](PERFORMANCE_OPTIMIZATION_GUIDE.md))
- Re-profile and compare results

For questions or issues, see the [main PRISM documentation](README.md) or open an issue on GitHub.
