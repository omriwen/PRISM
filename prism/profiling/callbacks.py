# prism/profiling/callbacks.py
"""Callback classes for PRISMTrainer integration with profiling."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from prism.profiling.collector import TrainingProfiler


class ProfilerCallback:
    """Callback for PRISMTrainer integration.

    This callback hooks into the training loop to collect profiling data
    at sample and epoch boundaries. It delegates all profiling operations
    to the TrainingProfiler instance.

    Parameters
    ----------
    profiler : TrainingProfiler
        The profiler instance to collect data with.

    Examples
    --------
    >>> from prism.profiling import TrainingProfiler, ProfilerConfig
    >>> config = ProfilerConfig(enabled=True)
    >>> profiler = TrainingProfiler(config)
    >>> callback = profiler.callback  # or ProfilerCallback(profiler)
    >>> # Pass to PRISMTrainer via callbacks=[callback]
    """

    def __init__(self, profiler: TrainingProfiler) -> None:
        self._profiler = profiler

    def on_sample_start(self, sample_idx: int) -> None:
        """Called at start of each sample.

        Parameters
        ----------
        sample_idx : int
            Index of the current sample being processed.
        """
        self._profiler.start_sample(sample_idx)

    def on_sample_end(
        self, sample_idx: int, stats: dict[str, Any] | None = None
    ) -> None:
        """Called at end of each sample.

        Parameters
        ----------
        sample_idx : int
            Index of the sample that just completed.
        stats : dict[str, Any] | None, optional
            Optional statistics from the sample (e.g., loss values).
        """
        self._profiler.end_sample(sample_idx)

    def on_epoch_start(self, epoch: int) -> None:
        """Called at start of each epoch.

        Parameters
        ----------
        epoch : int
            Index of the current epoch starting.
        """
        self._profiler.start_epoch(epoch)

    def on_epoch_end(self, epoch: int, loss: float) -> None:
        """Called at end of each epoch.

        Parameters
        ----------
        epoch : int
            Index of the epoch that just completed.
        loss : float
            Loss value at the end of the epoch.
        """
        self._profiler.end_epoch(epoch, loss)

    def on_training_end(self) -> None:
        """Called when training completes.

        This method can be used for final cleanup or summary operations.
        Currently a no-op but provides a hook for future extensions.
        """
        pass
