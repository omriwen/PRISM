# PRISM Training Profiler
#
# Low-overhead profiling for PyTorch training loops with post-hoc analysis.
#
# Usage:
#     from prism.profiling import TrainingProfiler, ProfilerConfig
#
#     config = ProfilerConfig(enabled=True)
#     profiler = TrainingProfiler(config)
#
#     # Use as context manager or callback
#     trainer = PRISMTrainer(..., callbacks=[profiler.callback])

from __future__ import annotations

from prism.profiling.analyzer import Bottleneck, BottleneckType, ProfileAnalyzer
from prism.profiling.call_graph import CallGraphBuilder, CallNode
from prism.profiling.callbacks import ProfilerCallback
from prism.profiling.collector import ProfileData, TrainingProfiler
from prism.profiling.config import ProfilerConfig
from prism.profiling.memory import MemoryProfile, MemorySnapshot, MemoryTracker
from prism.profiling.torch_integration import (
    extract_operator_stats,
    torch_profiler_context,
)


__all__ = [
    "Bottleneck",
    "BottleneckType",
    "CallGraphBuilder",
    "CallNode",
    "ProfileAnalyzer",
    "ProfilerCallback",
    "ProfilerConfig",
    "ProfileData",
    "TrainingProfiler",
    "MemoryProfile",
    "MemorySnapshot",
    "MemoryTracker",
    "torch_profiler_context",
    "extract_operator_stats",
]
