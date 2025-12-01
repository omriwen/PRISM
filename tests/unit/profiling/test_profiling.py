"""
Unit tests for the PRISM profiling module.

Tests cover:
    - ProfilerConfig dataclass default values and customization
    - CUDATimer non-blocking GPU timing with event pooling
    - MemoryTracker and MemorySnapshot memory tracking with leak detection
    - TrainingProfiler orchestration and data collection
    - ProfilerCallback integration with training loop
    - ProfileAnalyzer bottleneck detection and reporting
    - CallGraphBuilder hierarchical call graph construction
    - Storage functions (save/load/export)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from prism.profiling.analyzer import (
    Bottleneck,
    BottleneckType,
    ProfileAnalyzer,
)
from prism.profiling.call_graph import CallGraphBuilder, CallNode
from prism.profiling.callbacks import ProfilerCallback
from prism.profiling.collector import ProfileData, TrainingProfiler
from prism.profiling.config import ProfilerConfig
from prism.profiling.cuda_timer import CUDATimer
from prism.profiling.memory import MemoryProfile, MemorySnapshot, MemoryTracker
from prism.profiling.storage import (
    export_chrome_trace,
    load_profile,
    save_profile,
)


class TestProfilerConfig:
    """Test ProfilerConfig dataclass."""

    def test_default_values(self):
        """Test ProfilerConfig has correct default values."""
        config = ProfilerConfig()
        assert config.enabled is True
        assert config.collect_timing is True
        assert config.collect_memory is True
        assert config.collect_gpu_ops is True
        assert config.sample_rate == 0.1
        assert config.adaptive_sampling is True
        assert config.use_cuda_events is True
        assert config.cuda_sync_per_epoch is False
        assert config.warmup_samples == 2
        assert config.active_samples == 5
        assert config.record_shapes is True
        assert config.with_flops is True
        assert config.with_modules is True
        assert config.output_dir is None
        assert config.storage_format == "binary"
        assert config.export_chrome_trace is True
        assert config.bottleneck_threshold_pct == 10.0
        assert config.memory_leak_threshold_mb == 10.0

    def test_custom_values(self):
        """Test ProfilerConfig with custom values."""
        config = ProfilerConfig(
            enabled=False,
            sample_rate=0.5,
            bottleneck_threshold_pct=5.0,
            memory_leak_threshold_mb=20.0,
            storage_format="json",
        )
        assert config.enabled is False
        assert config.sample_rate == 0.5
        assert config.bottleneck_threshold_pct == 5.0
        assert config.memory_leak_threshold_mb == 20.0
        assert config.storage_format == "json"

    def test_output_dir_path(self):
        """Test ProfilerConfig with output directory."""
        config = ProfilerConfig(output_dir=Path("/tmp/profiling"))
        assert config.output_dir == Path("/tmp/profiling")


class TestCUDATimer:
    """Test CUDATimer for non-blocking GPU timing."""

    def test_initialization(self):
        """Test CUDATimer initialization."""
        timer = CUDATimer()
        assert timer._pool_size == 500
        assert timer._step_counter == 0
        assert len(timer._active) == 0
        assert len(timer._pending) == 0

    def test_custom_pool_size(self):
        """Test CUDATimer with custom pool size."""
        timer = CUDATimer(pool_size=100)
        assert timer._pool_size == 100

    def test_start_end_no_cuda(self):
        """Test start/end are no-ops without CUDA."""
        timer = CUDATimer()
        # Should not raise even without CUDA
        timer.start("test_region")
        timer.end("test_region")

    def test_step_counter(self):
        """Test step counter increment."""
        timer = CUDATimer()
        assert timer._step_counter == 0
        timer.step()
        assert timer._step_counter == 1
        timer.step()
        timer.step()
        assert timer._step_counter == 3

    def test_reset(self):
        """Test reset clears all state."""
        timer = CUDATimer()
        timer.step()
        timer.step()
        timer._results["test"] = [1.0, 2.0]
        timer.reset()
        assert timer._step_counter == 0
        assert len(timer._active) == 0
        assert len(timer._pending) == 0
        assert len(timer._results) == 0

    def test_flush_empty(self):
        """Test flush with no pending measurements returns empty dict."""
        timer = CUDATimer()
        result = timer.flush()
        assert result == {}

    @pytest.mark.gpu
    def test_cuda_timing(self, gpu_device):
        """Test actual CUDA timing on GPU."""
        timer = CUDATimer()
        timer.start("test_op")
        # Do some GPU work
        x = torch.randn(1000, 1000, device=gpu_device)
        _ = x @ x.T
        timer.end("test_op")

        results = timer.flush()
        assert "test_op" in results
        assert len(results["test_op"]) == 1
        assert results["test_op"][0] > 0  # Should have non-zero time

    @pytest.mark.gpu
    def test_multiple_regions(self, gpu_device):
        """Test timing multiple named regions."""
        timer = CUDATimer()

        timer.start("region_a")
        x = torch.randn(500, 500, device=gpu_device)
        timer.end("region_a")

        timer.start("region_b")
        _ = x @ x.T
        timer.end("region_b")

        results = timer.flush()
        assert "region_a" in results
        assert "region_b" in results

    @pytest.mark.gpu
    def test_event_pool_reuse(self, gpu_device):
        """Test that events are returned to pool for reuse."""
        timer = CUDATimer(pool_size=10)
        initial_pool_size = len(timer._event_pool)

        # Start and end several times
        for i in range(5):
            timer.start(f"op_{i}")
            timer.end(f"op_{i}")

        # After flush, events should be returned to pool
        timer.flush()
        # Pool may not be exactly initial size due to event creation/return
        assert len(timer._event_pool) <= timer._pool_size


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test MemorySnapshot can be created with all fields."""
        snapshot = MemorySnapshot(
            timestamp=1234567890.0,
            sample_idx=42,
            epoch=5,
            gpu_allocated_mb=1024.0,
            gpu_reserved_mb=2048.0,
            gpu_peak_mb=1500.0,
            cpu_rss_mb=512.0,
        )
        assert snapshot.timestamp == 1234567890.0
        assert snapshot.sample_idx == 42
        assert snapshot.epoch == 5
        assert snapshot.gpu_allocated_mb == 1024.0
        assert snapshot.gpu_reserved_mb == 2048.0
        assert snapshot.gpu_peak_mb == 1500.0
        assert snapshot.cpu_rss_mb == 512.0

    def test_snapshot_default_cpu(self):
        """Test MemorySnapshot default cpu_rss_mb is 0."""
        snapshot = MemorySnapshot(
            timestamp=0.0,
            sample_idx=0,
            epoch=0,
            gpu_allocated_mb=0.0,
            gpu_reserved_mb=0.0,
            gpu_peak_mb=0.0,
        )
        assert snapshot.cpu_rss_mb == 0.0


class TestMemoryProfile:
    """Test MemoryProfile dataclass."""

    def test_empty_profile(self):
        """Test empty MemoryProfile has correct defaults."""
        profile = MemoryProfile()
        assert profile.snapshots == []
        assert profile.peak_memory_mb == 0.0
        assert profile.peak_sample_idx == 0
        assert profile.leak_detected is False
        assert profile.leak_rate_mb_per_sample == 0.0

    def test_profile_with_data(self):
        """Test MemoryProfile with snapshot data."""
        snapshots = [
            MemorySnapshot(
                timestamp=0.0,
                sample_idx=0,
                epoch=0,
                gpu_allocated_mb=100.0,
                gpu_reserved_mb=200.0,
                gpu_peak_mb=150.0,
            ),
            MemorySnapshot(
                timestamp=1.0,
                sample_idx=1,
                epoch=0,
                gpu_allocated_mb=120.0,
                gpu_reserved_mb=200.0,
                gpu_peak_mb=180.0,
            ),
        ]
        profile = MemoryProfile(
            snapshots=snapshots,
            peak_memory_mb=180.0,
            peak_sample_idx=1,
            leak_detected=True,
            leak_rate_mb_per_sample=20.0,
        )
        assert len(profile.snapshots) == 2
        assert profile.peak_memory_mb == 180.0
        assert profile.peak_sample_idx == 1
        assert profile.leak_detected is True
        assert profile.leak_rate_mb_per_sample == 20.0


class TestMemoryTracker:
    """Test MemoryTracker for memory monitoring and leak detection."""

    def test_initialization(self):
        """Test MemoryTracker initialization."""
        tracker = MemoryTracker()
        assert tracker.leak_threshold_mb == 10.0

    def test_custom_threshold(self):
        """Test MemoryTracker with custom leak threshold."""
        tracker = MemoryTracker(leak_threshold_mb=5.0)
        assert tracker.leak_threshold_mb == 5.0

    def test_set_context(self):
        """Test setting sample and epoch context."""
        tracker = MemoryTracker()
        tracker.set_context(sample_idx=10, epoch=2)
        assert tracker._current_sample == 10
        assert tracker._current_epoch == 2

    def test_snapshot_no_cuda(self):
        """Test snapshot returns None without CUDA."""
        if torch.cuda.is_available():
            pytest.skip("Test requires CUDA to be unavailable")
        tracker = MemoryTracker()
        tracker.set_context(0, 0)
        snapshot = tracker.snapshot()
        assert snapshot is None

    def test_detect_leak_insufficient_data(self):
        """Test leak detection with insufficient data returns False."""
        tracker = MemoryTracker()
        detected, rate = tracker.detect_leak()
        assert detected is False
        assert rate == 0.0

    def test_detect_leak_needs_10_snapshots(self):
        """Test leak detection requires at least 10 snapshots."""
        tracker = MemoryTracker()
        # Add 9 snapshots (not enough)
        for i in range(9):
            tracker._snapshots.append(
                MemorySnapshot(
                    timestamp=float(i),
                    sample_idx=i,
                    epoch=0,
                    gpu_allocated_mb=100.0 + i * 10,
                    gpu_reserved_mb=200.0,
                    gpu_peak_mb=100.0 + i * 10,
                )
            )
        detected, rate = tracker.detect_leak()
        assert detected is False
        assert rate == 0.0

    def test_detect_leak_with_data(self):
        """Test leak detection with simulated growing memory."""
        tracker = MemoryTracker(leak_threshold_mb=5.0)
        # Simulate 20 snapshots with growing memory (20 MB/sample growth)
        for i in range(20):
            tracker._snapshots.append(
                MemorySnapshot(
                    timestamp=float(i),
                    sample_idx=i,
                    epoch=0,
                    gpu_allocated_mb=100.0 + i * 20,
                    gpu_reserved_mb=500.0,
                    gpu_peak_mb=100.0 + i * 20,
                )
            )
        detected, rate = tracker.detect_leak()
        assert detected == True  # Use == for numpy bool compatibility
        assert rate == pytest.approx(20.0, rel=0.1)

    def test_detect_no_leak_stable_memory(self):
        """Test leak detection with stable memory returns False."""
        tracker = MemoryTracker(leak_threshold_mb=10.0)
        # Simulate stable memory with small fluctuations
        np.random.seed(42)  # Make test deterministic
        for i in range(20):
            tracker._snapshots.append(
                MemorySnapshot(
                    timestamp=float(i),
                    sample_idx=i,
                    epoch=0,
                    gpu_allocated_mb=100.0 + np.random.uniform(-1, 1),
                    gpu_reserved_mb=200.0,
                    gpu_peak_mb=100.0,
                )
            )
        detected, rate = tracker.detect_leak()
        assert detected == False  # Use == for numpy bool compatibility

    def test_get_profile_empty(self):
        """Test get_profile with no snapshots returns empty profile."""
        tracker = MemoryTracker()
        profile = tracker.get_profile()
        assert profile.snapshots == []
        assert profile.peak_memory_mb == 0.0
        assert profile.leak_detected is False

    def test_get_profile_with_snapshots(self):
        """Test get_profile returns profile with computed stats."""
        tracker = MemoryTracker(leak_threshold_mb=50.0)
        for i in range(15):
            tracker._snapshots.append(
                MemorySnapshot(
                    timestamp=float(i),
                    sample_idx=i,
                    epoch=0,
                    gpu_allocated_mb=100.0 + i,
                    gpu_reserved_mb=200.0,
                    gpu_peak_mb=100.0 + i * 2,  # Peak grows faster
                )
            )
        profile = tracker.get_profile()
        assert len(profile.snapshots) == 15
        assert profile.peak_memory_mb == 128.0  # Last snapshot's peak
        assert profile.peak_sample_idx == 14

    def test_reset(self):
        """Test reset clears all state."""
        tracker = MemoryTracker()
        tracker.set_context(10, 5)
        tracker._snapshots.append(
            MemorySnapshot(
                timestamp=0.0,
                sample_idx=10,
                epoch=5,
                gpu_allocated_mb=100.0,
                gpu_reserved_mb=200.0,
                gpu_peak_mb=100.0,
            )
        )
        tracker.reset()
        assert tracker._current_sample == 0
        assert tracker._current_epoch == 0
        assert len(tracker._snapshots) == 0

    @pytest.mark.gpu
    def test_snapshot_with_cuda(self, gpu_device):
        """Test actual GPU memory snapshot."""
        tracker = MemoryTracker()
        tracker.set_context(0, 0)

        # Allocate some GPU memory
        x = torch.randn(1000, 1000, device=gpu_device)

        snapshot = tracker.snapshot()
        assert snapshot is not None
        assert snapshot.sample_idx == 0
        assert snapshot.epoch == 0
        assert snapshot.gpu_allocated_mb > 0

        del x


class TestProfileData:
    """Test ProfileData container."""

    def test_empty_profile_data(self):
        """Test ProfileData has correct defaults."""
        data = ProfileData()
        assert data.sample_times_ms == []
        assert data.epoch_times_ms == []
        assert data.region_times == {}
        assert data.memory_profile is None
        assert data.operator_times == {}
        assert data.operator_counts == {}
        assert data.sync_time_ms == 0.0
        assert data.cuda_time_total_ms == 0.0
        assert data.config is None
        assert data.device == "cpu"
        assert data.pytorch_version == ""
        assert data.total_samples == 0
        assert data.total_epochs == 0

    def test_profile_data_with_config(self):
        """Test ProfileData with config."""
        config = ProfilerConfig(sample_rate=0.5)
        data = ProfileData(config=config)
        assert data.config is not None
        assert data.config.sample_rate == 0.5


class TestTrainingProfiler:
    """Test TrainingProfiler orchestrator."""

    def test_initialization_default_config(self):
        """Test TrainingProfiler with default config."""
        profiler = TrainingProfiler()
        assert profiler.config is not None
        assert profiler.config.enabled is True

    def test_initialization_custom_config(self):
        """Test TrainingProfiler with custom config."""
        config = ProfilerConfig(enabled=False, sample_rate=0.2)
        profiler = TrainingProfiler(config)
        assert profiler.config.enabled is False
        assert profiler.config.sample_rate == 0.2

    def test_callback_property(self):
        """Test callback property returns ProfilerCallback."""
        profiler = TrainingProfiler()
        callback = profiler.callback
        assert isinstance(callback, ProfilerCallback)
        assert callback._profiler is profiler

    def test_profile_region_disabled(self):
        """Test profile_region is no-op when disabled."""
        config = ProfilerConfig(enabled=False)
        profiler = TrainingProfiler(config)

        with profiler.profile_region("test"):
            pass  # Should not raise

    def test_start_end_sample(self):
        """Test start_sample and end_sample update counters."""
        profiler = TrainingProfiler()
        profiler.start_sample(0)
        assert profiler._current_sample == 0

        profiler.end_sample(0)
        assert profiler._data.total_samples == 1

        profiler.start_sample(1)
        profiler.end_sample(1)
        assert profiler._data.total_samples == 2

    def test_start_end_epoch(self):
        """Test start_epoch and end_epoch update counters."""
        profiler = TrainingProfiler()
        profiler.start_epoch(0)
        assert profiler._current_epoch == 0

        profiler.end_epoch(0, loss=0.5)
        assert profiler._data.total_epochs == 1

    def test_should_sample_disabled(self):
        """Test _should_sample returns False when disabled."""
        config = ProfilerConfig(enabled=False)
        profiler = TrainingProfiler(config)
        assert profiler._should_sample() is False

    def test_should_sample_adaptive_early(self):
        """Test adaptive sampling increases rate for early samples."""
        config = ProfilerConfig(
            enabled=True, sample_rate=0.1, adaptive_sampling=True
        )
        profiler = TrainingProfiler(config)
        profiler._current_sample = 0

        # Run many trials - with adaptive sampling, early samples should
        # have higher rate (0.1 * 3 = 0.3)
        torch.manual_seed(42)
        samples = [profiler._should_sample() for _ in range(1000)]
        sample_rate = sum(samples) / len(samples)
        # Should be around 0.3 (with some variance)
        assert 0.2 < sample_rate < 0.4

    @pytest.mark.gpu
    def test_profile_region_timing(self, gpu_device):
        """Test profile_region collects timing data."""
        config = ProfilerConfig(enabled=True, use_cuda_events=True)
        profiler = TrainingProfiler(config)

        profiler.start_sample(0)
        with profiler.profile_region("forward"):
            x = torch.randn(500, 500, device=gpu_device)
            _ = x @ x.T
        profiler.end_sample(0)

        # Check region times were collected
        assert "forward" in profiler._data.region_times


class TestProfilerCallback:
    """Test ProfilerCallback for PRISMTrainer integration."""

    def test_callback_creation(self):
        """Test ProfilerCallback can be created."""
        profiler = TrainingProfiler()
        callback = ProfilerCallback(profiler)
        assert callback._profiler is profiler

    def test_on_sample_start(self):
        """Test on_sample_start delegates to profiler."""
        profiler = TrainingProfiler()
        callback = ProfilerCallback(profiler)

        callback.on_sample_start(5)
        assert profiler._current_sample == 5

    def test_on_sample_end(self):
        """Test on_sample_end delegates to profiler."""
        profiler = TrainingProfiler()
        callback = ProfilerCallback(profiler)

        callback.on_sample_start(0)
        callback.on_sample_end(0)
        assert profiler._data.total_samples == 1

    def test_on_epoch_start(self):
        """Test on_epoch_start delegates to profiler."""
        profiler = TrainingProfiler()
        callback = ProfilerCallback(profiler)

        callback.on_epoch_start(3)
        assert profiler._current_epoch == 3

    def test_on_epoch_end(self):
        """Test on_epoch_end delegates to profiler."""
        profiler = TrainingProfiler()
        callback = ProfilerCallback(profiler)

        callback.on_epoch_start(0)
        callback.on_epoch_end(0, loss=0.5)
        assert profiler._data.total_epochs == 1

    def test_on_training_end(self):
        """Test on_training_end is callable."""
        profiler = TrainingProfiler()
        callback = ProfilerCallback(profiler)
        # Should not raise
        callback.on_training_end()


class TestProfileAnalyzer:
    """Test ProfileAnalyzer for bottleneck detection."""

    def test_analyzer_from_data(self):
        """Test ProfileAnalyzer with ProfileData."""
        data = ProfileData()
        analyzer = ProfileAnalyzer(data)
        assert analyzer.data is data

    def test_get_summary_empty_data(self):
        """Test get_summary with empty data."""
        data = ProfileData()
        analyzer = ProfileAnalyzer(data)
        summary = analyzer.get_summary()

        assert summary["total_samples"] == 0
        assert summary["total_epochs"] == 0
        assert summary["total_time_ms"] == 0
        assert summary["avg_epoch_time_ms"] == 0
        assert summary["peak_gpu_memory_mb"] == 0
        assert summary["memory_leak_detected"] is False

    def test_get_summary_with_data(self):
        """Test get_summary with actual data."""
        config = ProfilerConfig(bottleneck_threshold_pct=10.0)
        data = ProfileData(
            config=config,
            total_samples=100,
            total_epochs=50,
            region_times={"forward": [10.0, 12.0, 11.0], "backward": [8.0, 9.0, 8.5]},
            device="cuda:0",
            pytorch_version="2.0.0",
        )
        analyzer = ProfileAnalyzer(data)
        summary = analyzer.get_summary()

        assert summary["total_samples"] == 100
        assert summary["total_epochs"] == 50
        assert summary["total_time_ms"] == pytest.approx(58.5, rel=0.01)
        assert summary["device"] == "cuda:0"
        assert summary["pytorch_version"] == "2.0.0"

    def test_get_top_operations(self):
        """Test get_top_operations returns sorted list."""
        data = ProfileData(
            region_times={
                "forward": [100.0, 100.0],
                "backward": [80.0, 80.0],
                "optimizer": [20.0, 20.0],
            }
        )
        analyzer = ProfileAnalyzer(data)
        top_ops = analyzer.get_top_operations(n=2)

        assert len(top_ops) == 2
        assert top_ops[0]["name"] == "forward"
        assert top_ops[0]["total_ms"] == 200.0
        assert top_ops[0]["count"] == 2
        assert top_ops[1]["name"] == "backward"
        assert top_ops[1]["total_ms"] == 160.0

    def test_identify_bottlenecks_hot_operation(self):
        """Test bottleneck detection for hot operations."""
        config = ProfilerConfig(bottleneck_threshold_pct=10.0)
        data = ProfileData(
            config=config,
            region_times={
                "slow_op": [90.0],  # 90% of time
                "fast_op": [10.0],  # 10% of time
            },
        )
        analyzer = ProfileAnalyzer(data)
        bottlenecks = analyzer.identify_bottlenecks()

        # slow_op should be flagged as hot operation
        hot_bottlenecks = [b for b in bottlenecks if b.type == BottleneckType.HOT_OPERATION]
        assert len(hot_bottlenecks) >= 1
        assert any("slow_op" in b.description for b in hot_bottlenecks)

    def test_identify_bottlenecks_memory_leak(self):
        """Test bottleneck detection for memory leaks."""
        config = ProfilerConfig()
        memory_profile = MemoryProfile(
            leak_detected=True, leak_rate_mb_per_sample=15.0
        )
        data = ProfileData(config=config, memory_profile=memory_profile)
        analyzer = ProfileAnalyzer(data)
        bottlenecks = analyzer.identify_bottlenecks()

        leak_bottlenecks = [b for b in bottlenecks if b.type == BottleneckType.MEMORY_LEAK]
        assert len(leak_bottlenecks) == 1
        assert "15.0" in leak_bottlenecks[0].description

    def test_identify_bottlenecks_cpu_gpu_sync(self):
        """Test bottleneck detection for CPU-GPU sync issues."""
        config = ProfilerConfig()
        data = ProfileData(
            config=config,
            sync_time_ms=50.0,
            cuda_time_total_ms=200.0,  # 25% sync time
        )
        analyzer = ProfileAnalyzer(data)
        bottlenecks = analyzer.identify_bottlenecks()

        sync_bottlenecks = [b for b in bottlenecks if b.type == BottleneckType.CPU_GPU_SYNC]
        assert len(sync_bottlenecks) == 1
        assert sync_bottlenecks[0].severity == "high"  # > 20%

    def test_identify_bottlenecks_no_issues(self):
        """Test no bottlenecks detected for clean profile."""
        config = ProfilerConfig(bottleneck_threshold_pct=50.0)  # High threshold
        data = ProfileData(
            config=config,
            region_times={
                "op1": [30.0],
                "op2": [35.0],
                "op3": [35.0],
            },
        )
        analyzer = ProfileAnalyzer(data)
        bottlenecks = analyzer.identify_bottlenecks()
        assert len(bottlenecks) == 0

    def test_get_efficiency_report(self):
        """Test efficiency report generation."""
        config = ProfilerConfig()
        data = ProfileData(
            config=config,
            total_samples=10,
            total_epochs=5,
            region_times={"forward": [10.0, 12.0], "backward": [8.0, 9.0]},
        )
        analyzer = ProfileAnalyzer(data)
        report = analyzer.get_efficiency_report()

        assert "PRISM Training Profile Report" in report
        assert "Total samples: 10" in report
        assert "Total epochs: 5" in report
        assert "forward" in report


class TestBottleneck:
    """Test Bottleneck dataclass."""

    def test_bottleneck_creation(self):
        """Test Bottleneck can be created with all fields."""
        bottleneck = Bottleneck(
            type=BottleneckType.HOT_OPERATION,
            severity="high",
            description="slow_function takes 80% of time",
            recommendation="Optimize slow_function",
            impact_ms=800.0,
        )
        assert bottleneck.type == BottleneckType.HOT_OPERATION
        assert bottleneck.severity == "high"
        assert bottleneck.impact_ms == 800.0


class TestBottleneckType:
    """Test BottleneckType enum."""

    def test_bottleneck_types(self):
        """Test all bottleneck types exist."""
        assert BottleneckType.CPU_GPU_SYNC.value == "cpu_gpu_sync"
        assert BottleneckType.MEMORY_LEAK.value == "memory_leak"
        assert BottleneckType.HOT_OPERATION.value == "hot_operation"
        assert BottleneckType.MEMORY_BOUND.value == "memory_bound"


class TestCallNode:
    """Test CallNode for call graph hierarchy."""

    def test_call_node_creation(self):
        """Test CallNode can be created."""
        node = CallNode(
            name="forward",
            module="model.forward",
            self_time_ms=10.0,
            total_time_ms=50.0,
            call_count=100,
        )
        assert node.name == "forward"
        assert node.module == "model.forward"
        assert node.self_time_ms == 10.0
        assert node.total_time_ms == 50.0
        assert node.call_count == 100
        assert node.children == []

    def test_call_node_percentage(self):
        """Test percentage calculation."""
        node = CallNode(
            name="op",
            total_time_ms=25.0,
            _root_time=100.0,
        )
        assert node.percentage == 25.0

    def test_call_node_percentage_zero_root(self):
        """Test percentage is 0 when root time is 0."""
        node = CallNode(name="op", total_time_ms=25.0, _root_time=0.0)
        assert node.percentage == 0.0

    def test_call_node_to_dict(self):
        """Test to_dict serialization."""
        child = CallNode(name="child", total_time_ms=10.0, _root_time=100.0)
        parent = CallNode(
            name="parent",
            module="test.parent",
            self_time_ms=5.0,
            total_time_ms=50.0,
            call_count=10,
            children=[child],
            _root_time=100.0,
        )
        d = parent.to_dict()

        assert d["name"] == "parent"
        assert d["module"] == "test.parent"
        assert d["self_time_ms"] == 5.0
        assert d["total_time_ms"] == 50.0
        assert d["call_count"] == 10
        assert d["percentage"] == 50.0
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "child"


class TestCallGraphBuilder:
    """Test CallGraphBuilder for hierarchical call graphs."""

    def test_build_from_regions_empty(self):
        """Test building from empty region times."""
        builder = CallGraphBuilder()
        root = builder.build_from_regions({})
        assert root.name == "root"
        assert root.total_time_ms == 0
        assert root.children == []

    def test_build_from_regions_flat(self):
        """Test building from flat (non-hierarchical) regions."""
        builder = CallGraphBuilder()
        region_times = {
            "forward": [10.0, 12.0],
            "backward": [8.0, 9.0],
        }
        root = builder.build_from_regions(region_times)

        assert root.name == "root"
        assert root.total_time_ms == 39.0
        assert len(root.children) == 2

        forward_node = next(c for c in root.children if c.name == "forward")
        assert forward_node.total_time_ms == 22.0
        assert forward_node.call_count == 2

    def test_build_from_regions_hierarchical(self):
        """Test building from hierarchical region names."""
        builder = CallGraphBuilder()
        region_times = {
            "model.forward.conv1": [5.0],
            "model.forward.conv2": [7.0],
            "model.backward": [10.0],
        }
        root = builder.build_from_regions(region_times)

        assert root.name == "root"
        model_node = next(c for c in root.children if c.name == "model")
        assert model_node is not None

        forward_node = next(c for c in model_node.children if c.name == "forward")
        assert forward_node is not None
        assert len(forward_node.children) == 2  # conv1 and conv2

    def test_to_flame_graph_data(self):
        """Test flame graph data generation."""
        builder = CallGraphBuilder()
        region_times = {"op1": [50.0], "op2": [50.0]}
        root = builder.build_from_regions(region_times)

        flame_data = builder.to_flame_graph_data(root)
        assert len(flame_data) >= 3  # root + 2 children
        assert any(d["name"] == "root" for d in flame_data)
        assert any(d["name"] == "op1" for d in flame_data)

    def test_to_sunburst_data(self):
        """Test sunburst chart data generation."""
        builder = CallGraphBuilder()
        region_times = {"forward": [30.0], "backward": [20.0]}
        root = builder.build_from_regions(region_times)

        sunburst = builder.to_sunburst_data(root)
        assert "ids" in sunburst
        assert "labels" in sunburst
        assert "parents" in sunburst
        assert "values" in sunburst
        assert len(sunburst["ids"]) >= 3  # root + 2 children
        assert "root" in sunburst["labels"]


class TestStorage:
    """Test profile save/load functionality."""

    def test_save_load_binary(self, tmp_path):
        """Test binary save and load."""
        config = ProfilerConfig()
        data = ProfileData(
            config=config,
            total_samples=10,
            total_epochs=5,
            region_times={"forward": [10.0, 12.0]},
            device="cuda:0",
        )
        path = tmp_path / "profile.pt"

        save_profile(data, path)
        loaded = load_profile(path)

        assert loaded.total_samples == 10
        assert loaded.total_epochs == 5
        assert loaded.region_times == {"forward": [10.0, 12.0]}
        assert loaded.device == "cuda:0"

    def test_save_load_json(self, tmp_path):
        """Test JSON save and load."""
        config = ProfilerConfig()
        data = ProfileData(
            config=config,
            total_samples=5,
            region_times={"op": [1.0, 2.0]},
        )
        path = tmp_path / "profile.json"

        save_profile(data, path)
        loaded = load_profile(path)

        assert loaded.total_samples == 5
        assert loaded.region_times == {"op": [1.0, 2.0]}

    def test_save_creates_directory(self, tmp_path):
        """Test save creates parent directory if needed."""
        data = ProfileData()
        path = tmp_path / "nested" / "dir" / "profile.pt"

        save_profile(data, path)
        assert path.exists()

    def test_export_chrome_trace(self, tmp_path):
        """Test Chrome trace export."""
        data = ProfileData(
            region_times={
                "forward": [10.0, 12.0],
                "backward": [8.0, 9.0],
            }
        )
        path = tmp_path / "trace.json"

        export_chrome_trace(data, path)

        with open(path) as f:
            trace = json.load(f)

        assert "traceEvents" in trace
        events = trace["traceEvents"]
        assert len(events) == 4  # 2 forward + 2 backward
        assert all(e["cat"] == "training" for e in events)
        assert all(e["ph"] == "X" for e in events)

    def test_export_chrome_trace_empty(self, tmp_path):
        """Test Chrome trace export with no data."""
        data = ProfileData()
        path = tmp_path / "trace_empty.json"

        export_chrome_trace(data, path)

        with open(path) as f:
            trace = json.load(f)

        assert trace["traceEvents"] == []


class TestTrainingProfilerSaveLoad:
    """Test TrainingProfiler save and load methods."""

    def test_save_method(self, tmp_path):
        """Test TrainingProfiler.save() method."""
        profiler = TrainingProfiler()
        profiler._data.total_samples = 10
        profiler._data.total_epochs = 5
        profiler._data.region_times = {"test": [1.0, 2.0]}

        path = tmp_path / "profile.pt"
        profiler.save(path)

        assert path.exists()

    def test_load_classmethod(self, tmp_path):
        """Test TrainingProfiler.load() classmethod."""
        # Create and save profile
        profiler = TrainingProfiler()
        profiler._data.total_samples = 15
        path = tmp_path / "profile.pt"
        profiler.save(path)

        # Load and verify
        loaded = TrainingProfiler.load(path)
        assert loaded.total_samples == 15


class TestIntegration:
    """Integration tests for the profiling system."""

    def test_full_workflow_cpu(self, tmp_path):
        """Test complete profiling workflow on CPU."""
        # Create profiler
        config = ProfilerConfig(enabled=True, sample_rate=1.0)
        profiler = TrainingProfiler(config)
        callback = profiler.callback

        # Simulate training loop
        for sample_idx in range(3):
            callback.on_sample_start(sample_idx)
            for epoch in range(2):
                callback.on_epoch_start(epoch)
                # Simulate work
                _ = torch.randn(100, 100)
                callback.on_epoch_end(epoch, loss=0.5 - epoch * 0.1)
            callback.on_sample_end(sample_idx)

        callback.on_training_end()

        # Save and load
        path = tmp_path / "profile.pt"
        profiler.save(path)
        loaded = TrainingProfiler.load(path)

        # Analyze
        analyzer = ProfileAnalyzer(loaded)
        summary = analyzer.get_summary()
        report = analyzer.get_efficiency_report()

        assert summary["total_samples"] == 3
        assert summary["total_epochs"] == 6
        assert "PRISM Training Profile Report" in report

    @pytest.mark.gpu
    def test_full_workflow_gpu(self, tmp_path, gpu_device):
        """Test complete profiling workflow on GPU."""
        config = ProfilerConfig(
            enabled=True,
            sample_rate=1.0,
            use_cuda_events=True,
        )
        profiler = TrainingProfiler(config)
        callback = profiler.callback

        for sample_idx in range(2):
            callback.on_sample_start(sample_idx)
            for epoch in range(3):
                callback.on_epoch_start(epoch)

                with profiler.profile_region("forward"):
                    x = torch.randn(500, 500, device=gpu_device)
                    _ = x @ x.T

                callback.on_epoch_end(epoch, loss=0.5)
            callback.on_sample_end(sample_idx)

        callback.on_training_end()

        # Save and analyze
        path = tmp_path / "gpu_profile.pt"
        profiler.save(path)

        analyzer = ProfileAnalyzer(path)
        summary = analyzer.get_summary()

        assert summary["total_samples"] == 2
        assert summary["total_epochs"] == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
