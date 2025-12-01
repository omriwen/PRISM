# prism.web.layouts.live

Live monitoring layout components for PRISM dashboard.

## Classes

## Functions

### create_kspace_coverage_plot

```python
create_kspace_coverage_plot(exp_data: Any) -> plotly.graph_objs._figure.Figure
```

Create k-space coverage visualization.

Args:
    exp_data: ExperimentData object

Returns:
    Plotly figure with k-space coverage

### create_live_progress_panel

```python
create_live_progress_panel(exp_data: Any) -> dash_bootstrap_components._components.Card.Card
```

Create live progress panel with current training status.

Args:
    exp_data: ExperimentData object

Returns:
    Dash Card component with training progress

### create_live_reconstruction_preview

```python
create_live_reconstruction_preview(exp_data: Any) -> plotly.graph_objs._figure.Figure
```

Create live reconstruction preview with ground truth and difference.

Args:
    exp_data: ExperimentData object

Returns:
    Plotly figure with reconstruction comparison

### create_live_training_plot

```python
create_live_training_plot(exp_data: Any, smoothing_window: int = 50) -> plotly.graph_objs._figure.Figure
```

Create live training plot with smoothed curves.

Args:
    exp_data: ExperimentData object
    smoothing_window: Window size for smoothing (odd number)

Returns:
    Plotly figure with smoothed training curves
