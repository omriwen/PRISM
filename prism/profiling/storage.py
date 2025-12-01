# prism/profiling/storage.py
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from prism.profiling.collector import ProfileData


def save_profile(data: ProfileData, path: Path) -> None:
    """Save profile data to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        _save_json(data, path)
    else:
        _save_binary(data, path)


def load_profile(path: Path) -> ProfileData:
    """Load profile data from file."""
    path = Path(path)

    if path.suffix == ".json":
        return _load_json(path)
    return _load_binary(path)


def _save_binary(data: ProfileData, path: Path) -> None:
    """Save as PyTorch binary format."""
    torch.save(data, path)


def _load_binary(path: Path) -> ProfileData:
    """Load from PyTorch binary format."""
    return torch.load(path, weights_only=False)


def _save_json(data: ProfileData, path: Path) -> None:
    """Save as JSON format."""
    # Convert dataclass to dict for JSON serialization
    import dataclasses
    data_dict = dataclasses.asdict(data)
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=2, default=str)


def _load_json(path: Path) -> ProfileData:
    """Load from JSON format."""
    from prism.profiling.collector import ProfileData
    with open(path) as f:
        data_dict = json.load(f)
    return ProfileData(**data_dict)


def export_chrome_trace(data: ProfileData, path: Path) -> None:
    """Export to Chrome trace format for chrome://tracing."""
    events = []

    # Convert region times to trace events
    ts = 0
    for name, times in data.region_times.items():
        for duration_ms in times:
            events.append({
                "name": name,
                "cat": "training",
                "ph": "X",  # Complete event
                "ts": ts * 1000,  # microseconds
                "dur": duration_ms * 1000,
                "pid": 1,
                "tid": 1,
            })
            ts += duration_ms

    trace = {"traceEvents": events}
    with open(path, "w") as f:
        json.dump(trace, f)
