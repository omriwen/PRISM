# prism.web.launcher

Dashboard launcher for managing dashboard subprocess during training.

## Classes

### DashboardLauncher

```python
DashboardLauncher(runs_dir: pathlib.Path = PosixPath('runs'), port: int = 8050)
```

Launch and manage dashboard process alongside training.

This class provides a clean way to start the dashboard in a separate process
and ensures proper cleanup when training completes or fails.

Parameters
----------
runs_dir : Path
    Directory containing experiment runs
port : int
    Port number for dashboard server (default: 8050)

Examples
--------
>>> launcher = DashboardLauncher(port=8050)
>>> launcher.start()
>>> # ... run training ...
>>> launcher.stop()

Or use as context manager:
>>> with DashboardLauncher(port=8050) as launcher:
...     # ... run training ...
...     pass  # Dashboard stops automatically

#### Methods

##### `__init__`

Initialize dashboard launcher.

Parameters
----------
runs_dir : Path
    Directory containing experiment runs
port : int
    Port number for dashboard server

##### `is_port_available`

Check if the specified port is available.

Returns
-------
bool
    True if port is available, False otherwise

##### `is_running`

Check if dashboard is currently running.

Returns
-------
bool
    True if dashboard process is running, False otherwise

##### `start`

Launch dashboard in background process.

Returns
-------
bool
    True if dashboard started successfully, False otherwise

##### `stop`

Stop dashboard process.

Gracefully terminates the dashboard subprocess if it's running.

## Functions

### launch_dashboard_if_requested

```python
launch_dashboard_if_requested(args, runs_dir: Optional[pathlib.Path] = None) -> Optional[prism.web.launcher.DashboardLauncher]
```

Helper function to launch dashboard based on command-line arguments.

Parameters
----------
args : argparse.Namespace
    Parsed command-line arguments
runs_dir : Optional[Path]
    Directory containing experiment runs (defaults to args.log_dir)

Returns
-------
Optional[DashboardLauncher]
    Launcher instance if dashboard was started, None otherwise

Examples
--------
>>> launcher = launch_dashboard_if_requested(args)
>>> if launcher:
...     try:
...         run_training()
...     finally:
...         launcher.stop()
