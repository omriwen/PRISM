# prism.web.layouts.comparison

Comparison layout components for PRISM dashboard.

## Classes

## Functions

### create_comparison_metrics_table

```python
create_comparison_metrics_table(experiments: List[Any]) -> Any
```

Create enhanced metrics comparison table with highlighting.

Args:
    experiments: List of ExperimentData objects

Returns:
    Dash DataTable with sortable, highlighted metrics

### create_config_diff_viewer

```python
create_config_diff_viewer(experiments: List[Any]) -> dash.html.Div.Div
```

Create configuration difference viewer with color coding.

Args:
    experiments: List of ExperimentData objects

Returns:
    HTML div with configuration differences

### create_side_by_side_comparison

```python
create_side_by_side_comparison(experiments: List[Any], view_mode: str = 'reconstruction', sync_axes: bool = True) -> plotly.graph_objs._figure.Figure
```

Create side-by-side reconstruction comparison with up to 4 experiments.

Args:
    experiments: List of ExperimentData objects (max 4)
    view_mode: One of 'reconstruction', 'difference', 'side_by_side'
    sync_axes: Whether to synchronize zoom/pan across subplots

Returns:
    Plotly figure with synchronized comparison

### create_training_curve_overlay

```python
create_training_curve_overlay(experiments: List[Any], smoothing_window: int = 50, opacity: float = 0.8) -> plotly.graph_objs._figure.Figure
```

Create training curve overlay with multiple experiments.

Args:
    experiments: List of ExperimentData objects
    smoothing_window: Window size for smoothing (if > 1)
    opacity: Opacity for traces (0-1)

Returns:
    Plotly figure with overlaid training curves
