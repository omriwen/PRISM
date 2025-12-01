# prism.reporting.statistics

Experiment Statistics
====================

Calculate and summarize experiment metrics for PRISM experiments.

## Classes

### ExperimentStatistics

```python
ExperimentStatistics(metrics_history: Union[List[Dict], ForwardRef('pd.DataFrame')])
```

Calculate experiment statistics and summaries.

Parameters
----------
metrics_history : list of dict or pandas.DataFrame
    History of metrics over training iterations

Attributes
----------
df : pandas.DataFrame or None
    DataFrame of metrics (if pandas available)
data : list of dict
    Raw metrics data

Methods
-------
generate_summary()
    Generate comprehensive statistical summary
find_convergence(metric='loss', threshold=1e-3)
    Find convergence epoch
calculate_improvement_rate(metric='ssim')
    Calculate improvement rate over time

Examples
--------
>>> metrics = [
...     {'epoch': 0, 'loss': 0.5, 'ssim': 0.7},
...     {'epoch': 1, 'loss': 0.3, 'ssim': 0.8},
...     {'epoch': 2, 'loss': 0.1, 'ssim': 0.9}
... ]
>>> stats = ExperimentStatistics(metrics)
>>> summary = stats.generate_summary()

#### Methods

##### `__init__`

Initialize statistics calculator.

Parameters
----------
metrics_history : list of dict or pandas.DataFrame
    History of metrics

##### `calculate_descriptive_stats`

Calculate descriptive statistics for a metric.

Parameters
----------
metric : str
    Metric name

Returns
-------
dict
    Descriptive statistics (mean, std, min, max, etc.)

##### `calculate_improvement_rate`

Calculate improvement rate for a metric.

Parameters
----------
metric : str, default='ssim'
    Metric to analyze
method : str, default='linear'
    Method for calculating rate:
    - 'linear': Linear regression slope
    - 'exponential': Exponential fit rate
    - 'average': Average per-epoch change

Returns
-------
float or None
    Improvement rate, or None if insufficient data

Notes
-----
For SSIM, a positive rate indicates improvement.
For loss, a negative rate indicates improvement.

##### `compare_periods`

Compare statistics between early and late training periods.

Parameters
----------
metric : str
    Metric to compare
split_point : int, optional
    Epoch to split at (defaults to midpoint)

Returns
-------
dict
    Comparison statistics for early vs late periods

##### `find_convergence`

Find convergence epoch based on metric stability.

Parameters
----------
metric : str, default='loss'
    Metric to check for convergence
threshold : float, default=1e-3
    Maximum change allowed for convergence
window : int, default=10
    Number of epochs to check for stability

Returns
-------
int or None
    Epoch where convergence occurred, or None if not converged

Notes
-----
Convergence is defined as the point where the metric's change
remains below the threshold for `window` consecutive epochs.

##### `generate_summary`

Generate comprehensive statistical summary.

Returns
-------
dict
    Complete statistical summary containing:
    - final_metrics: Final values of all metrics
    - best_metrics: Best values achieved
    - convergence_epoch: Epoch where convergence occurred
    - improvement_rate: Rate of improvement
    - statistics: Descriptive statistics for all metrics

Examples
--------
>>> stats = ExperimentStatistics(metrics_history)
>>> summary = stats.generate_summary()
>>> print(f"Final SSIM: {summary['final_metrics']['ssim']}")
>>> print(f"Converged at epoch: {summary['convergence_epoch']}")

##### `get_trends`

Get trend analysis for a metric.

Parameters
----------
metric : str
    Metric to analyze
smooth_window : int, default=10
    Window size for smoothing

Returns
-------
dict
    Trend data containing:
    - values: Raw metric values
    - smoothed: Smoothed values
    - trend: Linear trend line

##### `to_dict`

Convert all statistics to dictionary.

Returns
-------
dict
    All statistics as nested dictionary
