"""Enhanced progress tracking utilities backed by Rich."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
from loguru import logger
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table


# Define metric pairs that should be combined as "current/max"
# Key: (current_metric, max_metric) -> display_name
PROGRESS_PAIRS: dict[tuple[str, str], str] = {
    ("epoch", "max_epochs"): "epoch",
    ("sample", "n_samples"): "sample",
    ("init_epoch", "max_init_epochs"): "init_epoch",
    ("init_cycle", "max_init_cycles"): "init_cycle",
}

# Metrics that should be formatted as time (HH:MM:SS or MM:SS)
TIME_METRICS: set[str] = {"elapsed", "wall_time", "sample_time"}

# Metrics to hide from CLI display (still logged to tensorboard/wandb)
HIDDEN_METRICS: set[str] = {
    # Evaluation metrics (logged only)
    "rmse",
    "psnr",
    # Convergence monitor internals
    "epochs_since_best",
    "best_loss",
    "loss_velocity",
    # Redundant time metrics
    "wall_time",
    "sample_time",
    # Internal state
    "base_lr",
    "scheduler",
    "tier",
    "phase",
    # Max values (combined with current values)
    "max_epochs",
    "max_init_epochs",
    "max_init_cycles",
    "n_samples",
}

# Metric display groups and ordering
METRIC_GROUPS: dict[str, list[str]] = {
    "Progress": ["sample", "epoch", "init_cycle", "init_epoch", "lr"],
    "Loss": ["loss", "loss_old", "loss_new", "init_loss", "ssim"],
    "Timing": ["eta", "elapsed"],
    "System": ["gpu_mem_mb"],
}

# Human-readable display names
METRIC_DISPLAY_NAMES: dict[str, str] = {
    "loss_old": "Loss (old)",
    "loss_new": "Loss (new)",
    "init_loss": "Init Loss",
    "ssim": "SSIM",
    "gpu_mem_mb": "GPU Mem",
    "lr": "LR",
    "elapsed": "Total",
}


def render_sparkline(values: list[float], width: int = 20, use_unicode: bool = True) -> str:
    """Render sparkline using unicode block characters or ASCII fallback.

    Args:
        values: List of numeric values to visualize.
        width: Maximum width of the sparkline in characters.
        use_unicode: If True, use unicode block chars; else use ASCII.

    Returns:
        String representation of the sparkline.
    """
    if not values or len(values) == 0:
        return ""

    # Limit to width values (take the most recent)
    if len(values) > width:
        values = values[-width:]

    # Normalize to 0-1 range
    min_val = min(values)
    max_val = max(values)
    if max_val - min_val < 1e-10:
        # All values are the same
        if use_unicode:
            return "▄" * len(values)
        else:
            return "-" * len(values)

    normalized = [(v - min_val) / (max_val - min_val) for v in values]

    if use_unicode:
        # Unicode block characters: ▁▂▃▄▅▆▇█
        chars = "▁▂▃▄▅▆▇█"
        return "".join(chars[min(int(n * 7), 7)] for n in normalized)
    else:
        # ASCII fallback
        chars = "_.-'^*#"
        return "".join(chars[min(int(n * 6), 6)] for n in normalized)


class MetricTrend:
    """Analyzes metric trends to determine improvement/plateau/divergence."""

    def __init__(self, window_size: int = 50, plateau_threshold: float = 0.01) -> None:
        """Initialize trend analyzer.

        Args:
            window_size: Number of recent values to consider for trend analysis.
            plateau_threshold: Relative change threshold for plateau detection.
        """
        self.window_size = window_size
        self.plateau_threshold = plateau_threshold

    def analyze(self, values: list[float], metric_name: str = "metric") -> str:
        """Analyze trend and return status indicator.

        Args:
            values: Historical metric values.
            metric_name: Name of the metric (for direction detection).

        Returns:
            Colored emoji status indicator (🟢/🟡/🔴) or empty string.
        """
        if len(values) < 5:
            return ""  # Not enough data

        recent = values[-min(len(values), self.window_size) :]

        # Calculate trend direction
        if len(recent) >= 2:
            # Compare first half to second half
            mid = len(recent) // 2
            first_half_mean = np.mean(recent[:mid])
            second_half_mean = np.mean(recent[mid:])

            # Avoid division by zero
            if abs(first_half_mean) < 1e-10:
                relative_change = 0.0
            else:
                relative_change = float((second_half_mean - first_half_mean) / abs(first_half_mean))

            # Determine if metric should decrease (like loss) or increase (like SSIM/PSNR)
            metric_lower = metric_name.lower()
            should_decrease = "loss" in metric_lower or "error" in metric_lower

            if abs(relative_change) < self.plateau_threshold:
                return "🟡"  # Plateaued
            elif should_decrease:
                # For metrics that should decrease
                if relative_change < 0:
                    return "🟢"  # Improving (decreasing)
                else:
                    return "🔴"  # Diverging (increasing)
            else:
                # For metrics that should increase (SSIM, PSNR, accuracy, etc.)
                if relative_change > 0:
                    return "🟢"  # Improving (increasing)
                else:
                    return "🔴"  # Diverging (decreasing)

        return ""


class TrainingProgress:
    """Rich progress display and metric dashboard for training loops."""

    def __init__(
        self,
        console: Console | None = None,
        refresh_per_second: float = 10.0,
        enable_sparklines: bool = False,
        enable_emojis: bool = False,
        use_unicode: bool = True,
        sparkline_width: int = 15,
        metric_history_size: int = 100,
    ) -> None:
        """Initialise the training dashboard.

        Args:
            console: Optional console instance to render the dashboard to.
            refresh_per_second: Refresh frequency for the live dashboard.
            enable_sparklines: Whether to display sparkline charts for metrics.
            enable_emojis: Whether to display emoji status indicators.
            use_unicode: Whether to use unicode characters (False for ASCII fallback).
            sparkline_width: Width of sparkline charts in characters.
            metric_history_size: Number of historical values to retain per metric.
        """
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
        )
        self.metrics: dict[str, str] = {}
        self.metric_history: dict[str, list[float]] = defaultdict(list)
        self.previous_metrics: dict[str, float] = {}
        self.refresh_per_second = refresh_per_second
        self.live: Live | None = None
        self.trend_analyzer = MetricTrend()

        # Configuration options
        self.enable_sparklines = enable_sparklines
        self.enable_emojis = enable_emojis
        self.use_unicode = use_unicode
        self.sparkline_width = sparkline_width
        self.metric_history_size = metric_history_size
        self._suppressed_handler_id: int | None = None

    def __enter__(self) -> "TrainingProgress":
        """Enter the live dashboard context with logger suppression."""
        # Suppress console logs to prevent ghosting
        # Remove all handlers and re-add without stderr
        self._suppressed_handler_id = None
        try:
            # Get current handlers and remove stderr one
            # Loguru doesn't expose handlers directly, so we remove all and re-add file handlers
            logger.remove()
            # Note: This will be reconfigured on exit
            self._suppressed_handler_id = -1  # Flag that we suppressed
        except Exception:
            pass  # Continue even if suppression fails

        self.live = Live(
            self._render(),
            refresh_per_second=self.refresh_per_second,
            console=self.console,
        )
        self.live.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Tear down the live dashboard and restore logger."""
        if self.live is not None:
            self.live.__exit__(exc_type, exc_val, exc_tb)
            self.live = None

        # Restore logger configuration
        if self._suppressed_handler_id is not None:
            from prism.utils.logging_config import setup_logging

            setup_logging(level="INFO", show_time=True, show_level=True)

    def add_task(self, description: str, total: float) -> TaskID:
        """Register a new progress task and refresh the dashboard.

        Args:
            description: Task description displayed alongside the bar.
            total: Total number of steps expected for the task.

        Returns:
            The identifier of the newly created task.
        """
        task_id = self.progress.add_task(description, total=total)
        self.refresh()
        return task_id

    def advance(
        self,
        task_id: TaskID,
        *,
        advance: float = 1.0,
        metrics: dict[str, float | str] | None = None,
        eta_seconds: float | None = None,
        description: str | None = None,
    ) -> None:
        """Advance a task, update metrics, and refresh the dashboard.

        Args:
            task_id: Identifier of the task to advance.
            advance: Increment to apply to task completion.
            metrics: Optional metrics to surface in the dashboard table.
            eta_seconds: Optional ETA in seconds to display.
            description: Optional updated task description.
        """
        if description is not None:
            self.progress.update(task_id, description=description)
        self.progress.advance(task_id, advance)
        if metrics:
            self.update_metrics(metrics)
        if eta_seconds is not None:
            self.metrics["eta"] = self._format_eta(eta_seconds)
        self.refresh()

    def update_metrics(self, metrics: dict[str, float | str]) -> None:
        """Update the live metrics table with new values.

        Args:
            metrics: Mapping of metric names to values.
        """
        for key, value in metrics.items():
            # Store numeric values in history for trend analysis
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_value = float(value)
                self.metric_history[key].append(numeric_value)

                # Trim history to size limit
                if len(self.metric_history[key]) > self.metric_history_size:
                    self.metric_history[key] = self.metric_history[key][-self.metric_history_size :]

                # Store previous value for change calculation
                if key in self.previous_metrics:
                    pass  # Already have a previous value
                self.previous_metrics[key] = numeric_value
            elif hasattr(value, "item"):
                # Handle tensors
                try:
                    numeric_value = float(value.item())
                    self.metric_history[key].append(numeric_value)
                    if len(self.metric_history[key]) > self.metric_history_size:
                        self.metric_history[key] = self.metric_history[key][
                            -self.metric_history_size :
                        ]
                    self.previous_metrics[key] = numeric_value
                except (RuntimeError, ValueError, TypeError):
                    pass

            self.metrics[key] = self._format_metric_value(value)
        self.refresh()

    def complete(self, task_id: TaskID) -> None:
        """Mark a task as complete."""
        task = self.progress.tasks[task_id]
        self.progress.update(task_id, completed=task.total)
        self.refresh()

    def set_total(self, task_id: TaskID, total: float) -> None:
        """Update the total unit count for a task."""
        self.progress.update(task_id, total=total)
        self.refresh()

    def refresh(self) -> None:
        """Trigger a dashboard re-render."""
        if self.live is not None:
            self.live.update(self._render())

    def _render(self) -> Panel:
        """Construct the dashboard renderable."""
        return Panel(
            Group(self.progress, self._build_metrics_table()),
            title="Training Dashboard",
            border_style="cyan",
        )

    def _build_combined_values(self) -> dict[str, str]:
        """Build combined 'current/max' values for progress pairs."""
        combined: dict[str, str] = {}

        for (current_key, max_key), display_name in PROGRESS_PAIRS.items():
            if current_key in self.metrics and max_key in self.metrics:
                current_val = self.metrics[current_key]
                max_val = self.metrics[max_key]
                combined[current_key] = f"{current_val}/{max_val}"

        # Format time metrics
        for key in TIME_METRICS:
            if key in self.metric_history and self.metric_history[key]:
                if key not in HIDDEN_METRICS:
                    combined[key] = self._format_time(self.metric_history[key][-1])

        return combined

    def _build_metrics_table(self) -> Table:
        """Create a simplified, grouped metrics table."""
        table = Table(title="Training Metrics", expand=True, show_header=False)
        table.add_column("Metric", justify="left", style="dim")
        table.add_column("Value", justify="right")

        if not self.metrics:
            table.add_row("status", "initializing...")
            return table

        # Build combined values for progress pairs
        combined_values = self._build_combined_values()

        # Render metrics by group
        displayed_metrics: set[str] = set()

        for group_name, metric_keys in METRIC_GROUPS.items():
            group_rows = []
            for key in metric_keys:
                if key in displayed_metrics:
                    continue
                if key in HIDDEN_METRICS:
                    continue
                if key not in self.metrics and key not in combined_values:
                    continue

                value = combined_values.get(key, self.metrics.get(key, ""))
                if value is None or value == "None" or value == "":
                    continue

                display_name = METRIC_DISPLAY_NAMES.get(key, key)
                displayed_metrics.add(key)
                group_rows.append((f"  {display_name}", str(value)))

            if group_rows:
                # Add group header
                table.add_row(f"[bold cyan]{group_name}[/]", "")
                for name, val in group_rows:
                    table.add_row(name, val)

        return table

    def _format_metric_value(self, value: float | str | Any) -> str:
        """Convert a metric value to a human-readable string."""
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (RuntimeError, ValueError, TypeError):
                value = str(value)
        if value is None:
            return "—"
        if isinstance(value, (int, np.integer)):
            return f"{int(value)}"
        if isinstance(value, (float, np.floating)):
            abs_val = abs(float(value))
            if abs_val >= 1000 or (0 < abs_val < 1e-3):
                return f"{value:.2e}"
            return f"{value:.4f}"
        return str(value)

    def _format_eta(self, eta_seconds: float) -> str:
        """Format ETA seconds into a friendly string."""
        if eta_seconds <= 0:
            return "00:00"
        minutes, seconds = divmod(int(eta_seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS.

        Args:
            seconds: Time in seconds to format.

        Returns:
            Formatted time string.
        """
        if seconds < 0:
            return "00:00"
        if seconds < 3600:
            return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ETACalculator:
    """Calculate estimated time to completion for iterative loops with EMA smoothing."""

    def __init__(self, total_steps: int, ema_alpha: float = 0.1) -> None:
        """Initialise the ETA calculator.

        Args:
            total_steps: Total number of steps expected for the process.
            ema_alpha: Smoothing factor for exponential moving average (0-1).
                      Lower values = smoother but slower to adapt.
        """
        self.total_steps = max(total_steps, 1)
        self.start_time = time.time()
        self.last_update = self.start_time
        self.step_times: list[float] = []
        self.ema_alpha = ema_alpha
        self.ema_step_time: float | None = None

    def update(self, current_step: int) -> float:
        """Update ETA calculation based on the current step with EMA smoothing.

        Args:
            current_step: Step index (1-based) that has just completed.

        Returns:
            Estimated remaining time in seconds.
        """
        now = time.time()
        elapsed_step = now - self.last_update
        self.last_update = now

        if elapsed_step > 0:
            self.step_times.append(elapsed_step)

            # Update exponential moving average
            if self.ema_step_time is None:
                self.ema_step_time = elapsed_step
            else:
                self.ema_step_time = (
                    self.ema_alpha * elapsed_step + (1 - self.ema_alpha) * self.ema_step_time
                )

        # Use EMA if available, otherwise fall back to recent average
        if self.ema_step_time is not None:
            avg_time = self.ema_step_time
        else:
            recent_times = self.step_times[-100:] if self.step_times else []
            if recent_times:
                avg_time = float(np.mean(recent_times))
            else:
                elapsed_total = now - self.start_time
                avg_time = elapsed_total / max(current_step, 1)

        remaining_steps = max(self.total_steps - current_step, 0)
        return remaining_steps * avg_time

    def get_confidence_interval(
        self, current_step: int, confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate confidence interval for ETA estimate.

        Args:
            current_step: Current step index.
            confidence: Confidence level (default 0.95 for 95% CI).

        Returns:
            Tuple of (lower_bound, upper_bound) in seconds.
        """
        if len(self.step_times) < 5:
            # Not enough data for meaningful confidence interval
            eta = self.update(current_step)
            return (eta * 0.5, eta * 1.5)

        recent_times = self.step_times[-100:]
        mean_time = np.mean(recent_times)
        std_time = np.std(recent_times)

        # Use normal approximation for confidence interval
        # z-score for 95% confidence ≈ 1.96
        z_score = 1.96 if confidence == 0.95 else 2.576  # 99% CI

        margin = z_score * std_time / np.sqrt(len(recent_times))

        remaining_steps = max(self.total_steps - current_step, 0)
        lower = remaining_steps * max(mean_time - margin, 0)
        upper = remaining_steps * (mean_time + margin)

        return (lower, upper)
