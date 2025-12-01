"""Configuration for PRISM training profiler.

The profiler uses minimal-overhead collection during training with deferred analysis.
Key design: record events non-blocking, synchronize once per epoch, analyze offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ProfilerConfig:
    """Configuration for PRISM training profiler.

    This configuration controls profiling behavior during training. The profiler
    is designed for minimal overhead (<2% impact) via:
    - Non-blocking CUDA events (batched synchronization)
    - Adaptive sampling (10% of epochs by default)
    - Post-hoc analysis and visualization

    Parameters
    ----------
    enabled : bool, default=True
        Master switch for profiling. When False, all profiling is disabled
        with zero overhead.

    collect_timing : bool, default=True
        Collect timing information for operations and regions.

    collect_memory : bool, default=True
        Track GPU and CPU memory usage over training.

    collect_gpu_ops : bool, default=True
        Collect detailed GPU operator statistics via torch.profiler.

    sample_rate : float, default=0.1
        Fraction of epochs to profile (0.1 = 10%). Higher rates provide
        more data but increase overhead.

    adaptive_sampling : bool, default=True
        Increase sample rate during early training (first 10 samples get 3x rate).
        Helps capture initialization and early convergence behavior.

    use_cuda_events : bool, default=True
        Use CUDA events for GPU timing. Provides accurate GPU timing with
        minimal overhead via batched synchronization.

    cuda_sync_per_epoch : bool, default=False
        Synchronize CUDA events every epoch. When False (recommended), batches
        syncs to end of sample for lower overhead. Only enable for debugging
        timing issues.

    warmup_samples : int, default=2
        Number of warmup samples for torch.profiler before active profiling.

    active_samples : int, default=5
        Number of samples to actively profile with torch.profiler.

    record_shapes : bool, default=True
        Record tensor shapes in torch.profiler output.

    with_flops : bool, default=True
        Estimate FLOPs for operations (requires torch.profiler).

    with_modules : bool, default=True
        Record module hierarchy in torch.profiler output.

    output_dir : Path | None, default=None
        Directory for profile output files. If None, defaults to experiment
        directory (runs/{name}/).

    storage_format : Literal["binary", "json"], default="binary"
        Storage format for profile data:
        - "binary": PyTorch .pt format (fast, compact)
        - "json": Human-readable JSON (slower, larger)

    export_chrome_trace : bool, default=True
        Export Chrome trace format for chrome://tracing visualization.

    bottleneck_threshold_pct : float, default=10.0
        Percentage threshold for bottleneck detection. Operations consuming
        more than this percentage of total time are flagged as bottlenecks.

    memory_leak_threshold_mb : float, default=10.0
        Memory leak detection threshold in MB/sample. A linear growth rate
        exceeding this value triggers leak detection.

    Examples
    --------
    Basic usage with defaults:

    >>> from prism.profiling import ProfilerConfig, TrainingProfiler
    >>> config = ProfilerConfig(enabled=True)
    >>> profiler = TrainingProfiler(config)

    Custom configuration for debugging:

    >>> config = ProfilerConfig(
    ...     enabled=True,
    ...     sample_rate=0.5,  # Profile 50% of epochs
    ...     cuda_sync_per_epoch=True,  # More frequent syncs for debugging
    ...     bottleneck_threshold_pct=5.0,  # Stricter bottleneck detection
    ... )

    Minimal overhead configuration:

    >>> config = ProfilerConfig(
    ...     enabled=True,
    ...     sample_rate=0.05,  # Profile only 5% of epochs
    ...     collect_gpu_ops=False,  # Skip detailed operator stats
    ...     with_flops=False,  # Skip FLOP estimation
    ... )

    See Also
    --------
    prism.profiling.collector.TrainingProfiler : Main profiler orchestrator
    prism.profiling.analyzer.ProfileAnalyzer : Post-training analysis
    """

    # Master switch
    enabled: bool = True

    # Collection options
    collect_timing: bool = True
    collect_memory: bool = True
    collect_gpu_ops: bool = True

    # Sampling (for minimal overhead)
    sample_rate: float = 0.1  # 10% of epochs
    adaptive_sampling: bool = True

    # GPU-specific
    use_cuda_events: bool = True
    cuda_sync_per_epoch: bool = False  # Batch syncs to end of sample

    # torch.profiler options
    warmup_samples: int = 2
    active_samples: int = 5
    record_shapes: bool = True
    with_flops: bool = True
    with_modules: bool = True

    # Storage
    output_dir: Path | None = None
    storage_format: Literal["binary", "json"] = "binary"
    export_chrome_trace: bool = True

    # Analysis thresholds
    bottleneck_threshold_pct: float = 10.0
    memory_leak_threshold_mb: float = 10.0
