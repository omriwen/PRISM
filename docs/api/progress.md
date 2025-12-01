# prism.utils.progress

Enhanced progress tracking utilities backed by Rich.

## Classes

### ETACalculator

```python
ETACalculator(total_steps: 'int', ema_alpha: 'float' = 0.1) -> 'None'
```

Calculate estimated time to completion for iterative loops with EMA smoothing.

#### Methods

##### `__init__`

Initialise the ETA calculator.

Args:
    total_steps: Total number of steps expected for the process.
    ema_alpha: Smoothing factor for exponential moving average (0-1).
              Lower values = smoother but slower to adapt.

##### `get_confidence_interval`

Calculate confidence interval for ETA estimate.

Args:
    current_step: Current step index.
    confidence: Confidence level (default 0.95 for 95% CI).

Returns:
    Tuple of (lower_bound, upper_bound) in seconds.

##### `update`

Update ETA calculation based on the current step with EMA smoothing.

Args:
    current_step: Step index (1-based) that has just completed.

Returns:
    Estimated remaining time in seconds.

### MetricTrend

```python
MetricTrend(window_size: 'int' = 50, plateau_threshold: 'float' = 0.01) -> 'None'
```

Analyzes metric trends to determine improvement/plateau/divergence.

#### Methods

##### `__init__`

Initialize trend analyzer.

Args:
    window_size: Number of recent values to consider for trend analysis.
    plateau_threshold: Relative change threshold for plateau detection.

##### `analyze`

Analyze trend and return status indicator.

Args:
    values: Historical metric values.
    metric_name: Name of the metric (for direction detection).

Returns:
    Colored emoji status indicator (🟢/🟡/🔴) or empty string.

### TrainingProgress

```python
TrainingProgress(console: 'Console | None' = None, refresh_per_second: 'float' = 10.0, enable_sparklines: 'bool' = True, enable_emojis: 'bool' = True, use_unicode: 'bool' = True, sparkline_width: 'int' = 15, metric_history_size: 'int' = 100) -> 'None'
```

Rich progress display and metric dashboard for training loops.

#### Methods

##### `__init__`

Initialise the training dashboard.

Args:
    console: Optional console instance to render the dashboard to.
    refresh_per_second: Refresh frequency for the live dashboard.
    enable_sparklines: Whether to display sparkline charts for metrics.
    enable_emojis: Whether to display emoji status indicators.
    use_unicode: Whether to use unicode characters (False for ASCII fallback).
    sparkline_width: Width of sparkline charts in characters.
    metric_history_size: Number of historical values to retain per metric.

##### `add_task`

Register a new progress task and refresh the dashboard.

Args:
    description: Task description displayed alongside the bar.
    total: Total number of steps expected for the task.

Returns:
    The identifier of the newly created task.

##### `advance`

Advance a task, update metrics, and refresh the dashboard.

Args:
    task_id: Identifier of the task to advance.
    advance: Increment to apply to task completion.
    metrics: Optional metrics to surface in the dashboard table.
    eta_seconds: Optional ETA in seconds to display.
    description: Optional updated task description.

##### `complete`

Mark a task as complete.

##### `refresh`

Trigger a dashboard re-render.

##### `set_total`

Update the total unit count for a task.

##### `update_metrics`

Update the live metrics table with new values.

Args:
    metrics: Mapping of metric names to values.

## Functions

### render_sparkline

```python
render_sparkline(values: 'list[float]', width: 'int' = 20, use_unicode: 'bool' = True) -> 'str'
```

Render sparkline using unicode block characters or ASCII fallback.

Args:
    values: List of numeric values to visualize.
    width: Maximum width of the sparkline in characters.
    use_unicode: If True, use unicode block chars; else use ASCII.

Returns:
    String representation of the sparkline.
