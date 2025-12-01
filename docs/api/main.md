# prism.web.layouts.main

Main layout components for PRISM dashboard.

## Classes

## Functions

### create_config_table

```python
create_config_table(experiment: Any) -> Any
```

Create configuration table for an experiment.

Args:
    experiment: ExperimentData object

Returns:
    Dash DataTable component

### create_kspace_visualization

```python
create_kspace_visualization(experiment: Any) -> plotly.graph_objs._figure.Figure
```

Create k-space coverage visualization.

Args:
    experiment: ExperimentData object

Returns:
    Plotly figure with k-space visualization

### create_metrics_table

```python
create_metrics_table(experiments: List[Any]) -> Any
```

Create metrics comparison table.

Args:
    experiments: List of ExperimentData objects

Returns:
    Dash DataTable component

### create_reconstruction_comparison

```python
create_reconstruction_comparison(experiments: List[Any]) -> plotly.graph_objs._figure.Figure
```

Create side-by-side reconstruction comparison.

Args:
    experiments: List of ExperimentData objects

Returns:
    Plotly figure with reconstructions

### create_training_curves

```python
create_training_curves(experiments: List[Any]) -> plotly.graph_objs._figure.Figure
```

Create training curves plot with multiple experiments.

Args:
    experiments: List of ExperimentData objects

Returns:
    Plotly figure with training curves
