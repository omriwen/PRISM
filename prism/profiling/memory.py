"""
Memory tracking and leak detection for PyTorch training.

This module provides GPU and CPU memory monitoring capabilities with automatic
memory leak detection through linear regression analysis. It captures snapshots
at sample boundaries and analyzes trends to identify persistent memory growth.

Classes:
    MemorySnapshot: Point-in-time memory state capture
    MemoryProfile: Complete memory profile with analysis results
    MemoryTracker: Main tracker with leak detection

Usage:
    from prism.profiling.memory import MemoryTracker

    tracker = MemoryTracker(leak_threshold_mb=10.0)

    for epoch in range(num_epochs):
        for sample_idx, batch in enumerate(dataloader):
            tracker.set_context(sample_idx, epoch)

            # Training step
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Capture memory state
            tracker.snapshot()

    # Analyze results
    profile = tracker.get_profile()
    if profile.leak_detected:
        print(f"Memory leak detected: {profile.leak_rate_mb_per_sample:.2f} MB/sample")
        print(f"Peak memory: {profile.peak_memory_mb:.2f} MB at sample {profile.peak_sample_idx}")

Notes:
    - Snapshots are taken at user-defined points (typically sample boundaries)
    - Leak detection requires at least 10 snapshots for statistical significance
    - GPU memory tracking requires CUDA availability
    - CPU memory tracking (RSS) is currently a placeholder for future implementation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class MemorySnapshot:
    """
    Point-in-time memory snapshot.

    Captures GPU memory statistics at a specific training iteration, including
    allocated memory, reserved memory, and peak usage. Links each snapshot to
    its training context (sample index and epoch).

    Attributes:
        timestamp: Unix timestamp when snapshot was captured
        sample_idx: Training sample index at snapshot time
        epoch: Training epoch at snapshot time
        gpu_allocated_mb: Currently allocated GPU memory in megabytes
        gpu_reserved_mb: Reserved GPU memory (memory pool) in megabytes
        gpu_peak_mb: Peak allocated GPU memory in megabytes
        cpu_rss_mb: CPU resident set size in megabytes (currently unused)

    Notes:
        - Memory values use torch.cuda.memory_stats() on CUDA devices
        - Peak memory is cumulative since last reset
        - Reserved memory may exceed allocated (memory pool caching)
    """

    timestamp: float
    sample_idx: int
    epoch: int
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_peak_mb: float
    cpu_rss_mb: float = 0.0


@dataclass
class MemoryProfile:
    """
    Complete memory profile with analysis results.

    Aggregates all memory snapshots and provides summary statistics including
    peak memory usage and memory leak detection results.

    Attributes:
        snapshots: List of all captured memory snapshots
        peak_memory_mb: Maximum GPU memory allocation across all snapshots
        peak_sample_idx: Sample index where peak memory occurred
        leak_detected: Whether a memory leak was detected
        leak_rate_mb_per_sample: Linear growth rate in MB per sample (if leak detected)

    Notes:
        - Empty profile (no snapshots) returns all-zero values
        - Leak detection uses linear regression over allocated memory
        - Peak memory uses gpu_peak_mb field (cumulative peak)
    """

    snapshots: list[MemorySnapshot] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    peak_sample_idx: int = 0
    leak_detected: bool = False
    leak_rate_mb_per_sample: float = 0.0


class MemoryTracker:
    """
    Track GPU/CPU memory with automatic leak detection.

    Monitors memory usage throughout training and detects persistent memory leaks
    using linear regression on allocated memory over time. Designed for minimal
    overhead by using existing CUDA memory stats.

    Attributes:
        leak_threshold_mb: Minimum growth rate (MB/sample) to flag as leak

    Example:
        >>> tracker = MemoryTracker(leak_threshold_mb=5.0)
        >>> for epoch in range(10):
        ...     for sample_idx in range(100):
        ...         tracker.set_context(sample_idx, epoch)
        ...         # ... training code ...
        ...         tracker.snapshot()
        >>> profile = tracker.get_profile()
        >>> profile.leak_detected
        False

    Notes:
        - Requires at least 10 snapshots for leak detection
        - Leak detection uses simple linear regression (np.polyfit)
        - Does not interfere with CUDA memory allocation
        - CPU memory tracking is reserved for future implementation
    """

    def __init__(self, leak_threshold_mb: float = 10.0):
        """
        Initialize memory tracker.

        Args:
            leak_threshold_mb: Minimum memory growth rate (MB/sample) to
                classify as a leak. Default is 10.0 MB/sample, suitable
                for detecting significant leaks over 100+ samples.
        """
        self.leak_threshold_mb = leak_threshold_mb
        self._snapshots: list[MemorySnapshot] = []
        self._current_sample = 0
        self._current_epoch = 0

    def set_context(self, sample_idx: int, epoch: int) -> None:
        """
        Set current sample and epoch context.

        Updates internal state to associate subsequent snapshots with the
        correct training iteration. Should be called before each snapshot.

        Args:
            sample_idx: Current training sample index (cumulative across epochs)
            epoch: Current training epoch number

        Example:
            >>> tracker = MemoryTracker()
            >>> tracker.set_context(sample_idx=42, epoch=2)
            >>> snapshot = tracker.snapshot()
            >>> snapshot.sample_idx
            42
            >>> snapshot.epoch
            2
        """
        self._current_sample = sample_idx
        self._current_epoch = epoch

    def snapshot(self) -> MemorySnapshot | None:
        """
        Capture current memory state.

        Takes a snapshot of GPU memory statistics using torch.cuda.memory_*()
        functions. Returns None if CUDA is not available.

        Returns:
            Memory snapshot with current GPU stats, or None if CUDA unavailable

        Example:
            >>> tracker = MemoryTracker()
            >>> tracker.set_context(0, 0)
            >>> snapshot = tracker.snapshot()
            >>> snapshot is not None  # If CUDA available
            True
            >>> snapshot.gpu_allocated_mb >= 0
            True

        Notes:
            - Does not call torch.cuda.synchronize() for minimal overhead
            - Uses direct memory functions for reliability and speed
            - Returns zero values if CUDA is available but not yet initialized
        """
        if not torch.cuda.is_available():
            return None

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            sample_idx=self._current_sample,
            epoch=self._current_epoch,
            gpu_allocated_mb=torch.cuda.memory_allocated() / 1e6,
            gpu_reserved_mb=torch.cuda.memory_reserved() / 1e6,
            gpu_peak_mb=torch.cuda.max_memory_allocated() / 1e6,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def detect_leak(self) -> tuple[bool, float]:
        """
        Detect memory leak via linear regression.

        Analyzes allocated memory trend across snapshots using simple linear
        regression. A positive slope exceeding the threshold indicates a leak.

        Returns:
            Tuple of (leak_detected, growth_rate_mb_per_sample)
            - leak_detected: True if slope > leak_threshold_mb
            - growth_rate_mb_per_sample: Linear regression slope in MB/sample

        Example:
            >>> tracker = MemoryTracker(leak_threshold_mb=5.0)
            >>> # ... collect snapshots with growing memory ...
            >>> leak_detected, rate = tracker.detect_leak()
            >>> if leak_detected:
            ...     print(f"Leak: {rate:.2f} MB/sample")

        Notes:
            - Requires at least 10 snapshots; returns (False, 0.0) otherwise
            - Uses numpy.polyfit(x, y, 1) for linear regression
            - X-axis: sample indices, Y-axis: allocated memory in MB
            - Does not account for expected memory growth (e.g., gradient accumulation)
        """
        if len(self._snapshots) < 10:
            return False, 0.0

        x = np.array([s.sample_idx for s in self._snapshots])
        y = np.array([s.gpu_allocated_mb for s in self._snapshots])

        # Simple linear regression
        slope, _ = np.polyfit(x, y, 1)
        return slope > self.leak_threshold_mb, slope

    def get_profile(self) -> MemoryProfile:
        """
        Get complete memory profile with analysis.

        Constructs a MemoryProfile summarizing all captured snapshots, including
        peak memory usage and leak detection results.

        Returns:
            MemoryProfile with snapshots, peak stats, and leak analysis

        Example:
            >>> tracker = MemoryTracker()
            >>> # ... training loop with snapshots ...
            >>> profile = tracker.get_profile()
            >>> print(f"Peak: {profile.peak_memory_mb:.1f} MB")
            >>> print(f"Leak: {profile.leak_detected}")

        Notes:
            - Returns empty MemoryProfile if no snapshots captured
            - Peak memory uses max of gpu_peak_mb across snapshots
            - Leak detection only reliable with 10+ snapshots
        """
        if not self._snapshots:
            return MemoryProfile()

        peak_snapshot = max(self._snapshots, key=lambda s: s.gpu_peak_mb)
        leak_detected, leak_rate = self.detect_leak()

        return MemoryProfile(
            snapshots=self._snapshots.copy(),
            peak_memory_mb=peak_snapshot.gpu_peak_mb,
            peak_sample_idx=peak_snapshot.sample_idx,
            leak_detected=leak_detected,
            leak_rate_mb_per_sample=leak_rate,
        )

    def reset(self) -> None:
        """
        Reset tracker state.

        Clears all snapshots and resets context counters. Useful for starting
        a fresh profiling session without creating a new tracker instance.

        Example:
            >>> tracker = MemoryTracker()
            >>> # ... first training run ...
            >>> profile1 = tracker.get_profile()
            >>> tracker.reset()
            >>> # ... second training run ...
            >>> profile2 = tracker.get_profile()
            >>> len(profile2.snapshots)  # Only second run
            100
        """
        self._snapshots.clear()
        self._current_sample = 0
        self._current_epoch = 0
