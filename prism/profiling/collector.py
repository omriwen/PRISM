# prism/profiling/collector.py
from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import torch

from prism.profiling.config import ProfilerConfig
from prism.profiling.cuda_timer import CUDATimer
from prism.profiling.memory import MemoryProfile, MemoryTracker


if TYPE_CHECKING:
    from prism.profiling.callbacks import ProfilerCallback


@dataclass
class ProfileData:
    """Container for all profiling data."""

    # Timing data
    sample_times_ms: list[float] = field(default_factory=list)
    epoch_times_ms: list[float] = field(default_factory=list)
    region_times: dict[str, list[float]] = field(default_factory=dict)

    # Memory data
    memory_profile: MemoryProfile | None = None

    # torch.profiler data
    operator_times: dict[str, float] = field(default_factory=dict)
    operator_counts: dict[str, int] = field(default_factory=dict)
    sync_time_ms: float = 0.0
    cuda_time_total_ms: float = 0.0

    # Metadata
    config: ProfilerConfig | None = None
    device: str = "cpu"
    pytorch_version: str = ""
    total_samples: int = 0
    total_epochs: int = 0


class TrainingProfiler:
    """Main profiler orchestrating all data collection."""

    def __init__(self, config: ProfilerConfig | None = None):
        self.config = config or ProfilerConfig()
        self._cuda_timer = CUDATimer() if self.config.use_cuda_events else None
        self._memory_tracker = MemoryTracker(self.config.memory_leak_threshold_mb)
        self._torch_profiler: torch.profiler.profile | None = None

        self._data = ProfileData(config=self.config)
        self._current_sample = 0
        self._current_epoch = 0
        self._sample_start_time: float | torch.cuda.Event = 0.0

    @property
    def callback(self) -> ProfilerCallback:
        """Get callback for PRISMTrainer integration."""
        from prism.profiling.callbacks import ProfilerCallback
        return ProfilerCallback(self)

    def _should_sample(self) -> bool:
        """Determine if this iteration should be sampled."""
        if not self.config.enabled:
            return False
        if self.config.adaptive_sampling:
            # Adaptive: sample more during early training
            rate = self.config.sample_rate
            if self._current_sample < 10:
                rate = min(1.0, rate * 3)
            return random.random() < rate
        return random.random() < self.config.sample_rate

    @contextmanager
    def profile_region(self, name: str) -> Generator[None, None, None]:
        """Profile a named region (context manager)."""
        if not self.config.enabled or self._cuda_timer is None:
            yield
            return

        self._cuda_timer.start(name)
        try:
            yield
        finally:
            self._cuda_timer.end(name)

    def start_sample(self, sample_idx: int) -> None:
        """Called at start of each sample."""
        self._current_sample = sample_idx
        self._sample_start_time = torch.cuda.Event(enable_timing=True)
        if torch.cuda.is_available():
            self._sample_start_time.record()

    def end_sample(self, sample_idx: int) -> None:
        """Called at end of each sample."""
        self._data.total_samples = sample_idx + 1

        # Flush CUDA timings with single sync
        if self._cuda_timer:
            timings = self._cuda_timer.flush()
            for name, times in timings.items():
                if name not in self._data.region_times:
                    self._data.region_times[name] = []
                self._data.region_times[name].extend(times)

    def start_epoch(self, epoch: int) -> None:
        """Called at start of each epoch."""
        self._current_epoch = epoch
        self._memory_tracker.set_context(self._current_sample, epoch)

    def end_epoch(self, epoch: int, loss: float) -> None:
        """Called at end of each epoch."""
        self._data.total_epochs += 1

        if self._should_sample():
            self._memory_tracker.snapshot()

    def save(self, path: Path | str) -> None:
        """Save profile data to file."""
        from prism.profiling.storage import save_profile
        self._data.memory_profile = self._memory_tracker.get_profile()
        self._data.device = str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
        self._data.pytorch_version = torch.__version__
        save_profile(self._data, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> ProfileData:
        """Load profile data from file."""
        from prism.profiling.storage import load_profile
        return load_profile(Path(path))
