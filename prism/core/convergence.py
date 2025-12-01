"""
Adaptive convergence monitoring for per-sample training optimization.

This module provides tools to track and analyze convergence behavior during
SPIDS progressive reconstruction, enabling early exit for fast convergers
and escalated optimization for struggling samples.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Deque


if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class ConvergenceTier(Enum):
    """Optimization tier based on convergence behavior.

    Tiers determine training strategy:
    - FAST: Sample converged quickly, exit training
    - NORMAL: Standard training parameters
    - AGGRESSIVE: Struggling sample, increase optimization effort
    - RESCUE: Failed sample, try alternative strategy
    """

    FAST = "fast"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    RESCUE = "rescue"


@dataclass
class TierConfig:
    """Training configuration for a specific convergence tier.

    Parameters
    ----------
    lr_multiplier : float
        Multiplier applied to base learning rate
    extra_epochs : int
        Additional epochs beyond base allocation
    scheduler : str
        Scheduler type: 'plateau' or 'cosine_warm_restarts'
    """

    lr_multiplier: float = 1.0
    extra_epochs: int = 0
    scheduler: str = "plateau"


# Default tier configurations
DEFAULT_TIER_CONFIGS: dict[ConvergenceTier, TierConfig] = {
    ConvergenceTier.FAST: TierConfig(lr_multiplier=1.0, extra_epochs=0, scheduler="plateau"),
    ConvergenceTier.NORMAL: TierConfig(lr_multiplier=1.0, extra_epochs=0, scheduler="plateau"),
    ConvergenceTier.AGGRESSIVE: TierConfig(
        lr_multiplier=2.0, extra_epochs=500, scheduler="cosine_warm_restarts"
    ),
    ConvergenceTier.RESCUE: TierConfig(
        lr_multiplier=0.1, extra_epochs=1000, scheduler="cosine_warm_restarts"
    ),
}


@dataclass
class ConvergenceMonitor:
    """Track and analyze per-sample convergence behavior.

    Monitors loss progression to detect:
    - Convergence: loss below threshold with stability
    - Plateaus: loss stopped improving significantly
    - Tier transitions: when to escalate optimization strategy

    Parameters
    ----------
    loss_threshold : float
        Target loss value for convergence (default: 1e-3)
    patience : int
        Epochs without improvement before considering plateau (default: 10)
    plateau_window : int
        Window size for plateau detection (default: 50)
    plateau_threshold : float
        Relative improvement threshold for plateau (<1% = plateau) (default: 0.01)
    escalation_epochs : int
        Epochs before considering escalation to aggressive tier (default: 200)

    Examples
    --------
    >>> monitor = ConvergenceMonitor(loss_threshold=1e-3)
    >>> for epoch in range(max_epochs):
    ...     loss = train_step()
    ...     monitor.update(loss)
    ...     if monitor.should_stop():
    ...         break
    >>> print(f"Converged: {monitor.is_converged()}, Tier: {monitor.get_current_tier()}")
    """

    loss_threshold: float = 1e-3
    patience: int = 10
    plateau_window: int = 50
    plateau_threshold: float = 0.01
    escalation_epochs: int = 200

    # Internal state (initialized in __post_init__)
    _loss_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000), repr=False)
    _best_loss: float = field(default=float("inf"), repr=False)
    _epochs_since_improvement: int = field(default=0, repr=False)
    _current_tier: ConvergenceTier = field(default=ConvergenceTier.NORMAL, repr=False)
    _plateau_count: int = field(default=0, repr=False)
    _escalation_triggered: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize mutable state."""
        self._loss_history = deque(maxlen=1000)
        self._best_loss = float("inf")
        self._epochs_since_improvement = 0
        self._current_tier = ConvergenceTier.NORMAL
        self._plateau_count = 0
        self._escalation_triggered = False

    def update(self, loss: float) -> None:
        """Record new loss value and update convergence state.

        Parameters
        ----------
        loss : float
            Current epoch loss value
        """
        self._loss_history.append(loss)

        # Check for improvement
        if loss < self._best_loss * (1 - self.plateau_threshold):
            self._best_loss = loss
            self._epochs_since_improvement = 0
        else:
            self._epochs_since_improvement += 1

        # Update tier based on convergence state
        self._update_tier()

    def _update_tier(self) -> None:
        """Update optimization tier based on current state."""
        epochs = len(self._loss_history)

        # Fast convergence
        if self.is_converged():
            self._current_tier = ConvergenceTier.FAST
            return

        # Check for plateau
        if self.is_plateau():
            self._plateau_count += 1

            # Escalate if plateau persists
            if not self._escalation_triggered and epochs >= self.escalation_epochs:
                self._current_tier = ConvergenceTier.AGGRESSIVE
                self._escalation_triggered = True
            elif self._escalation_triggered and self._plateau_count > 2:
                self._current_tier = ConvergenceTier.RESCUE

    def loss_velocity(self) -> float:
        """Calculate rate of loss decrease (negative = improving).

        Uses linear regression over recent window to estimate velocity.

        Returns
        -------
        float
            Rate of change (negative values indicate improvement)
        """
        if len(self._loss_history) < 2:
            return 0.0

        window = min(self.plateau_window, len(self._loss_history))
        recent = list(self._loss_history)[-window:]

        # Simple finite difference approximation
        if len(recent) < 2:
            return 0.0

        # Calculate average velocity over window
        total_change = recent[-1] - recent[0]
        velocity = total_change / (len(recent) - 1)

        return velocity

    def is_plateau(self) -> bool:
        """Detect if loss has stopped improving.

        A plateau is detected when:
        1. Enough epochs have passed (patience)
        2. Recent improvement is below threshold

        Returns
        -------
        bool
            True if training is in a plateau state
        """
        if len(self._loss_history) < self.plateau_window:
            return False

        if self._epochs_since_improvement < self.patience:
            return False

        # Check relative improvement over window
        window = list(self._loss_history)[-self.plateau_window :]
        window_start = window[0]
        window_end = window[-1]

        if window_start == 0:
            return True

        relative_improvement = (window_start - window_end) / abs(window_start)

        return relative_improvement < self.plateau_threshold

    def is_converged(self) -> bool:
        """Check if loss is below threshold with stability.

        Convergence requires:
        1. Current loss below threshold
        2. Stable for at least `patience` epochs

        Returns
        -------
        bool
            True if sample has converged
        """
        if not self._loss_history:
            return False

        current_loss = self._loss_history[-1]

        # Check threshold
        if current_loss >= self.loss_threshold:
            return False

        # Check stability (at least patience epochs below threshold)
        if len(self._loss_history) < self.patience:
            below_threshold = sum(1 for loss in self._loss_history if loss < self.loss_threshold)
        else:
            recent = list(self._loss_history)[-self.patience :]
            below_threshold = sum(1 for loss in recent if loss < self.loss_threshold)

        return below_threshold >= min(self.patience, len(self._loss_history))

    def should_stop(self) -> bool:
        """Determine if training should stop (converged or hopeless).

        Stop conditions:
        1. Converged: loss below threshold with stability
        2. Hopeless: in RESCUE tier with plateau

        Returns
        -------
        bool
            True if training should stop
        """
        if self.is_converged():
            return True

        # Stop if in rescue tier and still plateauing
        if self._current_tier == ConvergenceTier.RESCUE and self.is_plateau():
            return True

        return False

    def should_escalate(self) -> bool:
        """Determine if should switch to more aggressive strategy.

        Returns
        -------
        bool
            True if escalation is recommended
        """
        if self._current_tier == ConvergenceTier.AGGRESSIVE:
            return False

        if self._current_tier == ConvergenceTier.RESCUE:
            return False

        return self.is_plateau() and len(self._loss_history) >= self.escalation_epochs

    def get_current_tier(self) -> ConvergenceTier:
        """Get current optimization tier based on convergence state.

        Returns
        -------
        ConvergenceTier
            Current tier (FAST, NORMAL, AGGRESSIVE, or RESCUE)
        """
        return self._current_tier

    def set_tier(self, tier: ConvergenceTier) -> None:
        """Manually set the optimization tier.

        Parameters
        ----------
        tier : ConvergenceTier
            Tier to set
        """
        self._current_tier = tier
        if tier in (ConvergenceTier.AGGRESSIVE, ConvergenceTier.RESCUE):
            self._escalation_triggered = True

    def reset(self) -> None:
        """Reset monitor state for new sample."""
        self._loss_history.clear()
        self._best_loss = float("inf")
        self._epochs_since_improvement = 0
        self._current_tier = ConvergenceTier.NORMAL
        self._plateau_count = 0
        self._escalation_triggered = False

    def get_statistics(self) -> dict:
        """Return convergence statistics for logging.

        Returns
        -------
        dict
            Dictionary containing:
            - epochs: Number of epochs tracked
            - best_loss: Best loss achieved
            - current_loss: Most recent loss
            - tier: Current optimization tier
            - is_converged: Whether converged
            - is_plateau: Whether in plateau
            - loss_velocity: Rate of loss change
            - plateau_count: Number of plateaus detected
        """
        return {
            "epochs": len(self._loss_history),
            "best_loss": self._best_loss,
            "current_loss": self._loss_history[-1] if self._loss_history else float("inf"),
            "tier": self._current_tier.value,
            "is_converged": self.is_converged(),
            "is_plateau": self.is_plateau(),
            "loss_velocity": self.loss_velocity(),
            "plateau_count": self._plateau_count,
            "epochs_since_improvement": self._epochs_since_improvement,
        }

    @property
    def epochs_used(self) -> int:
        """Number of epochs tracked.

        Returns
        -------
        int
            Total epochs recorded
        """
        return len(self._loss_history)


def get_tier_config(
    tier: ConvergenceTier,
    aggressive_lr_multiplier: float = 2.0,
    retry_lr_multiplier: float = 0.1,
) -> TierConfig:
    """Get training configuration for a given tier.

    Parameters
    ----------
    tier : ConvergenceTier
        Optimization tier
    aggressive_lr_multiplier : float
        LR multiplier for AGGRESSIVE tier (default: 2.0)
    retry_lr_multiplier : float
        LR multiplier for RESCUE tier (default: 0.1)

    Returns
    -------
    TierConfig
        Configuration for the specified tier
    """
    configs = {
        ConvergenceTier.FAST: TierConfig(lr_multiplier=1.0, extra_epochs=0, scheduler="plateau"),
        ConvergenceTier.NORMAL: TierConfig(lr_multiplier=1.0, extra_epochs=0, scheduler="plateau"),
        ConvergenceTier.AGGRESSIVE: TierConfig(
            lr_multiplier=aggressive_lr_multiplier,
            extra_epochs=500,
            scheduler="cosine_warm_restarts",
        ),
        ConvergenceTier.RESCUE: TierConfig(
            lr_multiplier=retry_lr_multiplier,
            extra_epochs=1000,
            scheduler="cosine_warm_restarts",
        ),
    }
    return configs[tier]


# ============================================================================
# Convergence Statistics
# ============================================================================


@dataclass
class ConvergenceStats:
    """Statistics for a single sample's convergence behavior.

    Tracks detailed convergence metrics for analysis and profiling.

    Parameters
    ----------
    sample_idx : int
        Index of the sample in the training sequence
    epochs_to_convergence : int
        Number of epochs used for this sample
    final_tier : ConvergenceTier
        Final optimization tier when training stopped
    retry_count : int
        Number of retry attempts (0 if converged on first try)
    final_loss : float
        Final loss value when training stopped
    total_time_seconds : float
        Wall-clock time for this sample
    loss_velocity_final : float
        Rate of loss change at end of training
    plateau_count : int
        Number of plateaus detected during training
    converged : bool
        Whether the sample successfully converged

    Examples
    --------
    >>> stats = ConvergenceStats(
    ...     sample_idx=0,
    ...     epochs_to_convergence=150,
    ...     final_tier=ConvergenceTier.FAST,
    ...     retry_count=0,
    ...     final_loss=0.0005,
    ...     total_time_seconds=2.3,
    ...     loss_velocity_final=-0.00001,
    ...     plateau_count=0,
    ...     converged=True,
    ... )
    """

    sample_idx: int
    epochs_to_convergence: int
    final_tier: ConvergenceTier
    retry_count: int
    final_loss: float
    total_time_seconds: float
    loss_velocity_final: float
    plateau_count: int
    converged: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary with all fields, tier as string
        """
        return {
            "sample_idx": self.sample_idx,
            "epochs_to_convergence": self.epochs_to_convergence,
            "final_tier": self.final_tier.value,
            "retry_count": self.retry_count,
            "final_loss": self.final_loss,
            "total_time_seconds": self.total_time_seconds,
            "loss_velocity_final": self.loss_velocity_final,
            "plateau_count": self.plateau_count,
            "converged": self.converged,
        }


class ConvergenceProfiler:
    """Collect and analyze convergence statistics across all samples.

    Provides summary statistics, tier distribution, and efficiency reports
    for analyzing adaptive convergence behavior.

    Examples
    --------
    >>> profiler = ConvergenceProfiler()
    >>> for sample_idx in range(n_samples):
    ...     # ... train sample ...
    ...     stats = ConvergenceStats(
    ...         sample_idx=sample_idx,
    ...         epochs_to_convergence=monitor.epochs_used,
    ...         final_tier=monitor.get_current_tier(),
    ...         # ... other fields ...
    ...     )
    ...     profiler.add_sample(stats)
    >>> print(profiler.get_efficiency_report())
    """

    def __init__(self) -> None:
        """Initialize empty profiler."""
        self._samples: list[ConvergenceStats] = []
        self._start_time: float = time.time()

    def add_sample(self, stats: ConvergenceStats) -> None:
        """Add sample convergence statistics.

        Parameters
        ----------
        stats : ConvergenceStats
            Statistics for a single sample
        """
        self._samples.append(stats)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics across all samples.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - total_samples: Number of samples processed
            - converged_count: Number that successfully converged
            - failed_count: Number that failed to converge
            - convergence_rate: Percentage that converged
            - avg_epochs: Average epochs per sample
            - median_epochs: Median epochs per sample
            - min_epochs: Minimum epochs
            - max_epochs: Maximum epochs
            - avg_time_per_sample: Average wall time per sample
            - total_time: Total wall time
            - tier_distribution: Dict of tier counts
        """
        if not self._samples:
            return {
                "total_samples": 0,
                "converged_count": 0,
                "failed_count": 0,
                "convergence_rate": 0.0,
                "avg_epochs": 0.0,
                "median_epochs": 0.0,
                "min_epochs": 0,
                "max_epochs": 0,
                "avg_time_per_sample": 0.0,
                "total_time": 0.0,
                "tier_distribution": {},
            }

        epochs = [s.epochs_to_convergence for s in self._samples]
        converged = [s for s in self._samples if s.converged]
        failed = [s for s in self._samples if not s.converged]

        # Sort for median calculation
        sorted_epochs = sorted(epochs)
        n = len(sorted_epochs)
        if n % 2 == 0:
            median = (sorted_epochs[n // 2 - 1] + sorted_epochs[n // 2]) / 2
        else:
            median = sorted_epochs[n // 2]

        total_time = sum(s.total_time_seconds for s in self._samples)

        return {
            "total_samples": len(self._samples),
            "converged_count": len(converged),
            "failed_count": len(failed),
            "convergence_rate": len(converged) / len(self._samples) * 100,
            "avg_epochs": sum(epochs) / len(epochs),
            "median_epochs": median,
            "min_epochs": min(epochs),
            "max_epochs": max(epochs),
            "avg_time_per_sample": total_time / len(self._samples),
            "total_time": total_time,
            "tier_distribution": self.get_tier_distribution(),
        }

    def get_tier_distribution(self) -> dict[str, int]:
        """Get distribution of final tiers.

        Returns
        -------
        dict[str, int]
            Dictionary mapping tier name to count
        """
        distribution: dict[str, int] = {tier.value: 0 for tier in ConvergenceTier}
        for sample in self._samples:
            distribution[sample.final_tier.value] += 1
        return distribution

    def get_efficiency_report(self, max_epochs: int | None = None) -> str:
        """Generate human-readable efficiency report.

        Parameters
        ----------
        max_epochs : int, optional
            Maximum epochs per sample for efficiency calculation.
            If None, uses the max observed epochs.

        Returns
        -------
        str
            Formatted report string
        """
        if not self._samples:
            return "No samples processed yet."

        summary = self.get_summary()
        tier_dist = summary["tier_distribution"]

        # Calculate efficiency (epochs used vs max possible)
        if max_epochs is None:
            max_epochs = summary["max_epochs"]
        total_epochs_used = sum(s.epochs_to_convergence for s in self._samples)
        total_epochs_possible = max_epochs * len(self._samples)
        efficiency = (
            (1 - total_epochs_used / total_epochs_possible) * 100
            if total_epochs_possible > 0
            else 0
        )

        lines = [
            "=" * 60,
            "CONVERGENCE PROFILER REPORT",
            "=" * 60,
            f"Total Samples:        {summary['total_samples']}",
            f"Converged:            {summary['converged_count']} ({summary['convergence_rate']:.1f}%)",
            f"Failed:               {summary['failed_count']}",
            "",
            "EPOCHS STATISTICS:",
            f"  Average:            {summary['avg_epochs']:.1f}",
            f"  Median:             {summary['median_epochs']:.1f}",
            f"  Min:                {summary['min_epochs']}",
            f"  Max:                {summary['max_epochs']}",
            f"  Efficiency:         {efficiency:.1f}% epochs saved",
            "",
            "TIER DISTRIBUTION:",
        ]

        for tier_name, count in tier_dist.items():
            pct = count / len(self._samples) * 100 if self._samples else 0
            lines.append(f"  {tier_name:12s}: {count:4d} ({pct:5.1f}%)")

        lines.extend(
            [
                "",
                "TIMING:",
                f"  Total Time:         {summary['total_time']:.1f}s",
                f"  Avg per Sample:     {summary['avg_time_per_sample']:.2f}s",
                "=" * 60,
            ]
        )

        return "\n".join(lines)

    def save_to_tensorboard(self, writer: "SummaryWriter", global_step: int = 0) -> None:
        """Save convergence statistics to TensorBoard.

        Parameters
        ----------
        writer : SummaryWriter
            TensorBoard summary writer
        global_step : int
            Global step for logging (default: 0)
        """
        if not self._samples:
            return

        summary = self.get_summary()

        # Log scalar summaries
        writer.add_scalar("convergence/convergence_rate", summary["convergence_rate"], global_step)
        writer.add_scalar("convergence/avg_epochs", summary["avg_epochs"], global_step)
        writer.add_scalar("convergence/median_epochs", summary["median_epochs"], global_step)
        writer.add_scalar("convergence/total_time", summary["total_time"], global_step)
        writer.add_scalar("convergence/failed_count", summary["failed_count"], global_step)

        # Log tier distribution
        tier_dist = summary["tier_distribution"]
        for tier_name, count in tier_dist.items():
            writer.add_scalar(f"convergence/tier_{tier_name}", count, global_step)

        # Log histogram of epochs to convergence
        try:
            import torch

            epochs_tensor = torch.tensor([s.epochs_to_convergence for s in self._samples])
            writer.add_histogram("convergence/epochs_histogram", epochs_tensor, global_step)
        except ImportError:
            pass  # Skip histogram if torch not available

        # Log histogram of final losses
        try:
            import torch

            losses_tensor = torch.tensor([s.final_loss for s in self._samples])
            writer.add_histogram("convergence/final_loss_histogram", losses_tensor, global_step)
        except ImportError:
            pass

    def get_all_stats(self) -> list[dict[str, Any]]:
        """Get all sample statistics as list of dicts.

        Returns
        -------
        list[dict[str, Any]]
            List of dictionaries, one per sample
        """
        return [s.to_dict() for s in self._samples]

    def clear(self) -> None:
        """Clear all collected statistics."""
        self._samples.clear()
        self._start_time = time.time()
