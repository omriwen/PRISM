# prism/profiling/torch_integration.py
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable

import torch
from torch import profiler


if TYPE_CHECKING:
    from prism.profiling.config import ProfilerConfig


@contextmanager
def torch_profiler_context(
    config: ProfilerConfig,
    on_trace_ready: Callable[[profiler.profile], None] | None = None,
):
    """Context manager for torch.profiler integration."""
    if not config.collect_gpu_ops:
        yield None
        return

    activities = [profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(profiler.ProfilerActivity.CUDA)

    schedule = profiler.schedule(
        wait=0,
        warmup=config.warmup_samples,
        active=config.active_samples,
        repeat=1,
    )

    with profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=config.record_shapes,
        profile_memory=True,
        with_stack=False,
        with_flops=config.with_flops,
        with_modules=config.with_modules,
    ) as prof:
        yield prof


def extract_operator_stats(prof: profiler.profile) -> dict:
    """Extract operator statistics from torch.profiler."""
    key_averages = prof.key_averages()

    operators = {}
    for event in key_averages:
        if event.key and event.cuda_time_total > 0:
            operators[event.key] = {
                "cuda_time_ms": event.cuda_time_total / 1000,
                "cpu_time_ms": event.cpu_time_total / 1000,
                "count": event.count,
                "flops": event.flops if hasattr(event, "flops") else 0,
            }

    return operators
