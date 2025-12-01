# prism.core.convergence

Adaptive convergence monitoring for per-sample training optimization.

This module provides tools to track and analyze convergence behavior during
PRISM progressive reconstruction, enabling early exit for fast convergers
and escalated optimization for struggling samples.

## Classes

### ConvergenceMonitor

```python
ConvergenceMonitor(loss_threshold: float = 0.001, patience: int = 10, plateau_window: int = 50, plateau_threshold: float = 0.01, escalation_epochs: int = 200, _loss_history: Deque[float] = <factory>, _best_loss: float = inf, _epochs_since_improvement: int = 0, _current_tier: prism.core.convergence.ConvergenceTier = <ConvergenceTier.NORMAL: 'normal'>, _plateau_count: int = 0, _escalation_triggered: bool = False) -> None
```

Track and analyze per-sample convergence behavior.

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

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

##### `get_current_tier`

Get current optimization tier based on convergence state.

Returns
-------
ConvergenceTier
    Current tier (FAST, NORMAL, AGGRESSIVE, or RESCUE)

##### `get_statistics`

Return convergence statistics for logging.

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

##### `is_converged`

Check if loss is below threshold with stability.

Convergence requires:
1. Current loss below threshold
2. Stable for at least `patience` epochs

Returns
-------
bool
    True if sample has converged

##### `is_plateau`

Detect if loss has stopped improving.

A plateau is detected when:
1. Enough epochs have passed (patience)
2. Recent improvement is below threshold

Returns
-------
bool
    True if training is in a plateau state

##### `loss_velocity`

Calculate rate of loss decrease (negative = improving).

Uses linear regression over recent window to estimate velocity.

Returns
-------
float
    Rate of change (negative values indicate improvement)

##### `reset`

Reset monitor state for new sample.

##### `set_tier`

Manually set the optimization tier.

Parameters
----------
tier : ConvergenceTier
    Tier to set

##### `should_escalate`

Determine if should switch to more aggressive strategy.

Returns
-------
bool
    True if escalation is recommended

##### `should_stop`

Determine if training should stop (converged or hopeless).

Stop conditions:
1. Converged: loss below threshold with stability
2. Hopeless: in RESCUE tier with plateau

Returns
-------
bool
    True if training should stop

##### `update`

Record new loss value and update convergence state.

Parameters
----------
loss : float
    Current epoch loss value

### ConvergenceProfiler

```python
ConvergenceProfiler() -> None
```

Collect and analyze convergence statistics across all samples.

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

#### Methods

##### `__init__`

Initialize empty profiler.

##### `add_sample`

Add sample convergence statistics.

Parameters
----------
stats : ConvergenceStats
    Statistics for a single sample

##### `clear`

Clear all collected statistics.

##### `get_all_stats`

Get all sample statistics as list of dicts.

Returns
-------
list[dict[str, Any]]
    List of dictionaries, one per sample

##### `get_efficiency_report`

Generate human-readable efficiency report.

Parameters
----------
max_epochs : int, optional
    Maximum epochs per sample for efficiency calculation.
    If None, uses the max observed epochs.

Returns
-------
str
    Formatted report string

##### `get_summary`

Get summary statistics across all samples.

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

##### `get_tier_distribution`

Get distribution of final tiers.

Returns
-------
dict[str, int]
    Dictionary mapping tier name to count

##### `save_to_tensorboard`

Save convergence statistics to TensorBoard.

Parameters
----------
writer : SummaryWriter
    TensorBoard summary writer
global_step : int
    Global step for logging (default: 0)

### ConvergenceStats

```python
ConvergenceStats(sample_idx: int, epochs_to_convergence: int, final_tier: prism.core.convergence.ConvergenceTier, retry_count: int, final_loss: float, total_time_seconds: float, loss_velocity_final: float, plateau_count: int, converged: bool = True) -> None
```

Statistics for a single sample's convergence behavior.

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

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

##### `to_dict`

Convert to dictionary for serialization.

Returns
-------
dict[str, Any]
    Dictionary with all fields, tier as string

### ConvergenceTier

```python
ConvergenceTier(*args, **kwds)
```

Optimization tier based on convergence behavior.

Tiers determine training strategy:
- FAST: Sample converged quickly, exit training
- NORMAL: Standard training parameters
- AGGRESSIVE: Struggling sample, increase optimization effort
- RESCUE: Failed sample, try alternative strategy

### TierConfig

```python
TierConfig(lr_multiplier: float = 1.0, extra_epochs: int = 0, scheduler: str = 'plateau') -> None
```

Training configuration for a specific convergence tier.

Parameters
----------
lr_multiplier : float
    Multiplier applied to base learning rate
extra_epochs : int
    Additional epochs beyond base allocation
scheduler : str
    Scheduler type: 'plateau' or 'cosine_warm_restarts'

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

## Functions

### get_tier_config

```python
get_tier_config(tier: prism.core.convergence.ConvergenceTier, aggressive_lr_multiplier: float = 2.0, retry_lr_multiplier: float = 0.1) -> prism.core.convergence.TierConfig
```

Get training configuration for a given tier.

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
