# prism.output.manager

Experiment output management

## Classes

### OutputManager

```python
OutputManager(base_dir: Union[pathlib.Path, str], experiment_name: str, versioning: bool = True, max_checkpoints: int = 5)
```

Manages experiment outputs with organized directory structure.

Creates and manages a standardized output directory structure for experiments,
including checkpoints, visualizations, logs, metrics, configs, and reports.

Parameters
----------
base_dir : Path or str
    Base directory for all outputs (typically 'runs/')
experiment_name : str
    Name of the experiment (creates subdirectory)
versioning : bool, optional
    Enable checkpoint versioning (default: True)
max_checkpoints : int, optional
    Maximum number of checkpoints to keep (default: 5, -1 for unlimited)

Attributes
----------
base_dir : Path
    Root directory for this experiment
versioning : bool
    Whether versioning is enabled
max_checkpoints : int
    Maximum checkpoints to retain

Examples
--------
>>> manager = OutputManager('runs/', 'experiment_001')
>>> manager.save_config({'lr': 0.001, 'epochs': 100})
>>> manager.save_checkpoint({'model': model.state_dict()}, epoch=10)

#### Methods

##### `__init__`

Initialize output manager and create directory structure.

##### `archive_experiment`

Archive entire experiment directory as compressed archive.

Parameters
----------
archive_dir : Path, optional
    Directory to store archive (default: parent of base_dir)

Returns
-------
Path
    Path to created archive file

##### `cleanup`

Remove all experiment outputs.

Parameters
----------
keep_best : bool, optional
    If True, preserve best.pt checkpoint (default: True)

##### `generate_all_reports`

Generate all available reports for experiment.

Parameters
----------
experiment_data : dict, optional
    Experiment data. If not provided, loads from saved data.
include_pdf : bool, optional
    Whether to generate PDF report (requires pdflatex), default False

Returns
-------
dict
    Dictionary mapping report type to file path

Examples
--------
>>> manager = OutputManager('runs/', 'experiment_001')
>>> reports = manager.generate_all_reports()
>>> print(f"HTML report: {reports['html']}")

##### `generate_html_report`

Generate HTML report for experiment.

Parameters
----------
experiment_data : dict, optional
    Experiment data. If not provided, loads from saved metrics and config.
template_path : Path, optional
    Custom HTML template path

Returns
-------
Path
    Path to generated HTML report

Raises
------
ImportError
    If reporting dependencies are not installed

Examples
--------
>>> manager = OutputManager('runs/', 'experiment_001')
>>> report_path = manager.generate_html_report()

##### `generate_pdf_report`

Generate PDF report for experiment.

Parameters
----------
experiment_data : dict, optional
    Experiment data. If not provided, loads from saved metrics and config.
template_path : Path, optional
    Custom LaTeX template path

Returns
-------
Path
    Path to generated PDF report

Raises
------
ImportError
    If reporting dependencies are not installed
RuntimeError
    If pdflatex is not available

Examples
--------
>>> manager = OutputManager('runs/', 'experiment_001')
>>> report_path = manager.generate_pdf_report()

##### `generate_statistics_summary`

Generate statistical summary of experiment metrics.

Parameters
----------
metrics_history : list of dict, optional
    Metrics history. If not provided, loads from saved metrics.

Returns
-------
dict
    Statistical summary

Raises
------
ImportError
    If reporting dependencies are not installed

Examples
--------
>>> manager = OutputManager('runs/', 'experiment_001')
>>> summary = manager.generate_statistics_summary()
>>> print(f"Final SSIM: {summary['final_metrics']['final_ssim']}")

##### `get_checkpoint_by_epoch`

Get checkpoint path by epoch number.

Parameters
----------
epoch : int
    Epoch number

Returns
-------
Path or None
    Path to checkpoint if it exists, None otherwise

##### `get_latest_checkpoint`

Get path to most recent checkpoint.

Returns
-------
Path or None
    Path to latest checkpoint, or None if no checkpoints exist

##### `list_checkpoints`

List all available checkpoints.

Returns
-------
list of Path
    List of checkpoint paths sorted by modification time (newest first)

##### `load_checkpoint`

Load checkpoint from file.

Parameters
----------
filename : str, optional
    Checkpoint filename (default: 'best.pt')

Returns
-------
dict
    Checkpoint state dictionary

##### `load_config`

Load configuration from YAML file.

Parameters
----------
filename : str, optional
    Config filename to load (default: 'experiment.yaml')

Returns
-------
dict
    Configuration dictionary

##### `load_metrics`

Load metrics from JSON file.

Parameters
----------
filename : str, optional
    Metrics filename (default: 'training_metrics.json')

Returns
-------
dict
    Dictionary of metric names to lists of values

##### `save_checkpoint`

Save training checkpoint with optional versioning.

Parameters
----------
state : dict
    Checkpoint state dictionary (model, optimizer, metrics, etc.)
epoch : int, optional
    Current epoch number (used in filename if provided)
is_best : bool, optional
    Whether this is the best checkpoint (creates copy as 'best.pt')
filename : str, optional
    Custom filename (overrides epoch-based naming)

Returns
-------
Path
    Path to saved checkpoint

Notes
-----
If max_checkpoints is set, automatically removes old checkpoints
to maintain the limit.

##### `save_config`

Save experiment configuration to YAML file.

Parameters
----------
config : dict or object
    Configuration dictionary or dataclass instance
filename : str, optional
    Output filename (default: 'experiment.yaml')

Returns
-------
Path
    Path to saved config file

Examples
--------
>>> config = {'lr': 0.001, 'batch_size': 32}
>>> path = manager.save_config(config)

##### `save_figure`

Save matplotlib figure to visualizations directory.

Parameters
----------
fig : matplotlib.figure.Figure
    Figure to save
filename : str
    Output filename (should include extension like .png, .pdf)
dpi : int, optional
    Resolution in dots per inch (default: 300)
**kwargs
    Additional arguments passed to fig.savefig()

Returns
-------
Path
    Path to saved figure

##### `save_metrics`

Save training metrics to JSON file.

Parameters
----------
metrics : dict
    Dictionary of metric names to lists of values
filename : str, optional
    Output filename (default: 'training_metrics.json')

Returns
-------
Path
    Path to saved metrics file

##### `save_report`

Save text report to reports directory.

Parameters
----------
content : str
    Report content (HTML, Markdown, or plain text)
filename : str, optional
    Output filename (default: 'report.html')

Returns
-------
Path
    Path to saved report

##### `setup_directories`

Create output directory structure.

Creates the following subdirectories:
- checkpoints: Model checkpoints
- visualizations: Plots and figures
- logs: Training logs and metrics
- metrics: Metric data files
- configs: Configuration files
- reports: Generated reports
