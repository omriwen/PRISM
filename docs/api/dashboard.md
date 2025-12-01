# prism.web.dashboard

Main PRISM Dashboard application.

## Classes

## Functions

### create_app

```python
create_app(runs_dir: pathlib.Path = PosixPath('runs'), port: int = 8050) -> dash.dash.Dash
```

Create and configure the Dash application.

Args:
    runs_dir: Path to directory containing experiment runs
    port: Port number for the server

Returns:
    Configured Dash application

### run_dashboard

```python
run_dashboard(runs_dir: pathlib.Path = PosixPath('runs'), port: int = 8050, debug: bool = False, quiet: bool = True)
```

Run the dashboard server.

Args:
    runs_dir: Path to directory containing experiment runs
    port: Port number for the server
    debug: Whether to run in debug mode
    quiet: Whether to suppress HTTP request logging (default True)
