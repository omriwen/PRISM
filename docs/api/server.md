# prism.web.server

Dashboard server for managing experiment data loading and parsing.

## Classes

### DashboardServer

```python
DashboardServer(runs_dir: pathlib.Path = PosixPath('runs'))
```

Manages dashboard server and data loading.

#### Methods

##### `__init__`

Initialize dashboard server.

Args:
    runs_dir: Path to directory containing experiment runs

##### `clear_cache`

Clear the experiment data cache.

##### `load_experiment_data`

Load experiment metrics and checkpoints.

Args:
    exp_id: Experiment ID (directory name)
    use_cache: Whether to use cached data if available

Returns:
    ExperimentData object or None if loading fails

##### `parse_tensorboard_logs`

Parse TensorBoard event files (if available).

Args:
    exp_path: Path to experiment directory

Returns:
    Dictionary of metric histories from TensorBoard

##### `refresh_experiment`

Force refresh experiment data from disk.

Args:
    exp_id: Experiment ID to refresh

Returns:
    Updated ExperimentData or None if loading fails

##### `scan_experiments`

Scan runs directory for experiments.

Returns:
    List of experiment metadata dictionaries

### ExperimentData

```python
ExperimentData(exp_id: str, path: pathlib.Path, config: Dict[str, Any], metrics: Dict[str, List[float]], final_metrics: Dict[str, float], timestamp: Optional[datetime.datetime] = None, reconstruction: Optional[numpy.ndarray] = None)
```

Container for experiment data.

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.
