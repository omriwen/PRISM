# prism/profiling/analyzer.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from prism.profiling.collector import ProfileData
from prism.profiling.storage import load_profile


class BottleneckType(Enum):
    CPU_GPU_SYNC = "cpu_gpu_sync"
    MEMORY_LEAK = "memory_leak"
    HOT_OPERATION = "hot_operation"
    MEMORY_BOUND = "memory_bound"


@dataclass
class Bottleneck:
    """Detected performance bottleneck."""

    type: BottleneckType
    severity: Literal["high", "medium", "low"]
    description: str
    recommendation: str
    impact_ms: float = 0.0


class ProfileAnalyzer:
    """Analyze profile data for bottlenecks and insights."""

    def __init__(self, data: ProfileData | Path | str):
        if isinstance(data, (Path, str)):
            data = load_profile(Path(data))
        self.data = data
        self._bottleneck_threshold = data.config.bottleneck_threshold_pct if data.config else 10.0

    def get_summary(self) -> dict:
        """Get summary statistics."""
        region_totals = {
            name: sum(times) for name, times in self.data.region_times.items()
        }
        total_time = sum(region_totals.values())

        return {
            "total_samples": self.data.total_samples,
            "total_epochs": self.data.total_epochs,
            "total_time_ms": total_time,
            "avg_epoch_time_ms": total_time / max(1, self.data.total_epochs),
            "peak_gpu_memory_mb": (
                self.data.memory_profile.peak_memory_mb
                if self.data.memory_profile else 0
            ),
            "memory_leak_detected": (
                self.data.memory_profile.leak_detected
                if self.data.memory_profile else False
            ),
            "device": self.data.device,
            "pytorch_version": self.data.pytorch_version,
        }

    def get_top_operations(self, n: int = 10) -> list[dict]:
        """Get top N operations by time."""
        region_totals = [
            {"name": name, "total_ms": sum(times), "count": len(times)}
            for name, times in self.data.region_times.items()
        ]
        region_totals.sort(key=lambda x: x["total_ms"], reverse=True)
        return region_totals[:n]

    def identify_bottlenecks(self) -> list[Bottleneck]:
        """Detect performance bottlenecks."""
        bottlenecks = []

        # 1. Memory leak detection
        if self.data.memory_profile and self.data.memory_profile.leak_detected:
            bottlenecks.append(Bottleneck(
                type=BottleneckType.MEMORY_LEAK,
                severity="high",
                description=(
                    f"Memory leak: {self.data.memory_profile.leak_rate_mb_per_sample:.1f} "
                    "MB/sample"
                ),
                recommendation="Check for tensors retained in lists/closures. Use del explicitly.",
            ))

        # 2. Hot operations (>threshold% of time)
        total_time = sum(sum(t) for t in self.data.region_times.values())
        if total_time > 0:
            for name, times in self.data.region_times.items():
                region_time = sum(times)
                ratio = region_time / total_time * 100
                if ratio > self._bottleneck_threshold:
                    bottlenecks.append(Bottleneck(
                        type=BottleneckType.HOT_OPERATION,
                        severity="high" if ratio > 30 else "medium",
                        description=f"'{name}': {ratio:.1f}% of total time",
                        recommendation=f"Consider optimizing {name}",
                        impact_ms=region_time,
                    ))

        # 3. CPU-GPU sync (if detected)
        if self.data.sync_time_ms > 0 and self.data.cuda_time_total_ms > 0:
            sync_ratio = self.data.sync_time_ms / self.data.cuda_time_total_ms
            if sync_ratio > 0.1:
                bottlenecks.append(Bottleneck(
                    type=BottleneckType.CPU_GPU_SYNC,
                    severity="high" if sync_ratio > 0.2 else "medium",
                    description=f"CPU-GPU sync: {sync_ratio*100:.1f}% of GPU time",
                    recommendation="Avoid .item(), .cpu(), .numpy() in training loop",
                    impact_ms=self.data.sync_time_ms,
                ))

        return bottlenecks

    def get_efficiency_report(self) -> str:
        """Generate human-readable efficiency report."""
        summary = self.get_summary()
        bottlenecks = self.identify_bottlenecks()
        top_ops = self.get_top_operations(5)

        lines = [
            "=" * 60,
            "PRISM Training Profile Report",
            "=" * 60,
            "",
            "Summary:",
            f"  Total samples: {summary['total_samples']}",
            f"  Total epochs: {summary['total_epochs']}",
            f"  Total time: {summary['total_time_ms']:.1f} ms",
            f"  Avg epoch time: {summary['avg_epoch_time_ms']:.2f} ms",
            f"  Peak GPU memory: {summary['peak_gpu_memory_mb']:.1f} MB",
            "",
            "Top Operations:",
        ]

        for i, op in enumerate(top_ops, 1):
            lines.append(f"  {i}. {op['name']}: {op['total_ms']:.1f} ms ({op['count']} calls)")

        if bottlenecks:
            lines.extend(["", "Bottlenecks Detected:"])
            for b in bottlenecks:
                lines.append(f"  [{b.severity.upper()}] {b.type.value}: {b.description}")
                lines.append(f"    Fix: {b.recommendation}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
