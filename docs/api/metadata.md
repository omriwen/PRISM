# prism.output.metadata

Experiment metadata tracking

## Classes

### ExperimentMetadata

```python
ExperimentMetadata(experiment_name: str, start_time: Optional[datetime.datetime] = None, end_time: Optional[datetime.datetime] = None, total_samples: int = 0, failed_samples: List[int] = <factory>, best_metrics: Dict[str, float] = <factory>, final_metrics: Dict[str, float] = <factory>, hardware_info: prism.output.metadata.HardwareInfo = <factory>, config_snapshot: Dict[str, Any] = <factory>, git_commit: Optional[str] = None, notes: str = '') -> None
```

Track comprehensive experiment metadata.

Captures experiment runtime information, hardware details, metrics,
and failure tracking for complete experiment documentation.

Attributes
----------
experiment_name : str
    Name of the experiment
start_time : datetime or None
    Experiment start timestamp
end_time : datetime or None
    Experiment end timestamp
total_samples : int
    Total number of samples processed
failed_samples : list
    List of failed sample indices
best_metrics : dict
    Best metric values achieved
final_metrics : dict
    Final metric values
hardware_info : HardwareInfo
    Hardware configuration details
config_snapshot : dict
    Snapshot of experiment configuration
git_commit : str or None
    Git commit hash if available
notes : str
    Additional notes or comments

Examples
--------
>>> metadata = ExperimentMetadata(experiment_name="test_001")
>>> metadata.start()
>>> metadata.update_sample_count(100)
>>> metadata.record_best_metrics({'ssim': 0.95, 'loss': 0.01})
>>> metadata.end()
>>> metadata.save('metadata.json')

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

##### `add_failed_sample`

Record a failed sample.

Parameters
----------
sample_idx : int
    Index of failed sample

##### `add_note`

Append note to metadata.

Parameters
----------
note : str
    Note to add

##### `end`

Mark experiment end time.

##### `record_best_metrics`

Update best metrics if new values are better.

Parameters
----------
metrics : dict
    Dictionary of metric names to values

##### `record_final_metrics`

Record final metric values.

Parameters
----------
metrics : dict
    Dictionary of metric names to final values

##### `save`

Save metadata to JSON file.

Parameters
----------
filepath : str
    Path to output JSON file

##### `set_config`

Store configuration snapshot.

Parameters
----------
config : dict
    Configuration dictionary

##### `set_git_commit`

Record git commit hash.

Parameters
----------
commit_hash : str
    Git commit hash

##### `start`

Mark experiment start time.

##### `summary`

Generate human-readable summary.

Returns
-------
str
    Formatted summary string

##### `to_dict`

Convert metadata to dictionary for serialization.

Returns
-------
dict
    Dictionary representation with serializable types

##### `update_sample_count`

Update total sample count.

Parameters
----------
count : int
    New total sample count

### HardwareInfo

```python
HardwareInfo(cpu_model: str, cpu_cores: int, cpu_threads: int, ram_total_gb: float, gpu_available: bool, gpu_name: Optional[str] = None, gpu_count: int = 0, gpu_memory_gb: Optional[float] = None, cuda_version: Optional[str] = None, pytorch_version: str = '') -> None
```

Hardware information for experiment tracking.

Attributes
----------
cpu_model : str
    CPU model name
cpu_cores : int
    Number of CPU cores
cpu_threads : int
    Number of CPU threads
ram_total_gb : float
    Total RAM in GB
gpu_available : bool
    Whether GPU is available
gpu_name : str or None
    GPU model name
gpu_count : int
    Number of GPUs
gpu_memory_gb : float or None
    GPU memory in GB (per device)
cuda_version : str or None
    CUDA version
pytorch_version : str
    PyTorch version

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

## Functions

### get_git_commit

```python
get_git_commit() -> Optional[str]
```

Get current git commit hash if available.

Returns
-------
str or None
    Git commit hash, or None if not in a git repository

### get_hardware_info

```python
get_hardware_info() -> prism.output.metadata.HardwareInfo
```

Collect current hardware information.

Returns
-------
HardwareInfo
    Hardware configuration details

### get_system_info

```python
get_system_info() -> Dict[str, str]
```

Get general system information.

Returns
-------
dict
    System information including OS, hostname, Python version
