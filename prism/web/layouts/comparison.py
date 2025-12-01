"""Comparison layout components for SPIDS dashboard."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import plotly.graph_objects as go
from dash import dash_table, html
from plotly.subplots import make_subplots


def create_side_by_side_comparison(
    experiments: List[Any], view_mode: str = "reconstruction", sync_axes: bool = True
) -> go.Figure:
    """Create side-by-side reconstruction comparison with up to 4 experiments.

    Args:
        experiments: List of ExperimentData objects (max 4)
        view_mode: One of 'reconstruction', 'difference', 'side_by_side'
        sync_axes: Whether to synchronize zoom/pan across subplots

    Returns:
        Plotly figure with synchronized comparison
    """
    if not experiments:
        fig = go.Figure()
        fig.add_annotation(
            text="Select up to 4 experiments to compare",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=400)
        return fig

    # Limit to 4 experiments
    experiments = experiments[:4]
    n_exps = len(experiments)

    # Determine grid layout
    if n_exps == 1:
        rows, cols = 1, 1
    elif n_exps == 2:
        rows, cols = 1, 2
    elif n_exps <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 2  # Max 4

    # Create titles with key metrics
    subplot_titles = []
    for exp in experiments:
        fm = exp.final_metrics
        title = f"{exp.exp_id}<br>"
        title += f"SSIM: {fm.get('ssim', 0):.4f} | PSNR: {fm.get('psnr', 0):.2f} dB"
        subplot_titles.append(title)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.15,
    )

    # Track min/max values for consistent colorscale
    all_values = []

    for idx, exp in enumerate(experiments):
        row = idx // cols + 1
        col = idx % cols + 1

        if exp.reconstruction is None:
            # No reconstruction available
            fig.add_annotation(
                text="No reconstruction available",
                xref=f"x{idx + 1 if idx > 0 else ''}",
                yref=f"y{idx + 1 if idx > 0 else ''}",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=12),
                row=row,
                col=col,
            )
            continue

        # Handle different array shapes
        rec = exp.reconstruction
        if rec.ndim == 3:
            rec = rec[0]  # Take first channel if multi-channel

        # Collect values for consistent scaling
        all_values.extend(rec.flatten())

        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=rec,
                colorscale="gray",
                showscale=(idx == 0),  # Only show colorbar for first plot
                hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.4f}<extra></extra>",
                colorbar=dict(title="Intensity", len=0.5, y=0.5) if idx == 0 else None,
            ),
            row=row,
            col=col,
        )

    # Update layout
    height = 400 if n_exps <= 2 else 700
    fig.update_layout(
        height=height,
        showlegend=False,
        title=dict(
            text=f"Reconstruction Comparison ({view_mode.replace('_', ' ').title()})", x=0.5
        ),
    )

    # Configure axes
    for i in range(1, n_exps + 1):
        fig.update_xaxes(showticklabels=False, row=(i - 1) // cols + 1, col=(i - 1) % cols + 1)
        fig.update_yaxes(
            showticklabels=False,
            scaleanchor=f"x{i if i > 1 else ''}",
            scaleratio=1,
            row=(i - 1) // cols + 1,
            col=(i - 1) % cols + 1,
        )

    # Sync axes if requested
    if sync_axes and n_exps > 1:
        # All plots share the same axes range
        fig.update_xaxes(matches="x")
        fig.update_yaxes(matches="y")

    return fig


def create_comparison_metrics_table(experiments: List[Any]) -> Any:  # dash_table.DataTable
    """Create enhanced metrics comparison table with highlighting.

    Args:
        experiments: List of ExperimentData objects

    Returns:
        Dash DataTable with sortable, highlighted metrics
    """
    if not experiments:
        return dash_table.DataTable(  # type: ignore[attr-defined]
            data=[],
            columns=[
                {"name": "Experiment", "id": "experiment"},
                {"name": "Final Loss", "id": "loss"},
                {"name": "SSIM", "id": "ssim"},
                {"name": "PSNR (dB)", "id": "psnr"},
                {"name": "RMSE", "id": "rmse"},
                {"name": "Epochs", "id": "epochs"},
            ],
        )

    # Prepare data and track best/worst values
    data = []
    loss_values = []
    ssim_values = []
    psnr_values = []
    rmse_values = []

    for exp in experiments:
        fm = exp.final_metrics
        loss_val = fm.get("loss", float("inf"))
        ssim_val = fm.get("ssim", 0.0)
        psnr_val = fm.get("psnr", 0.0)
        rmse_val = fm.get("rmse", float("inf"))

        loss_values.append(loss_val)
        ssim_values.append(ssim_val)
        psnr_values.append(psnr_val)
        rmse_values.append(rmse_val)

        data.append(
            {
                "experiment": exp.exp_id,
                "loss": loss_val,
                "ssim": ssim_val,
                "psnr": psnr_val,
                "rmse": rmse_val,
                "epochs": int(fm.get("epochs", 0)),
                "timestamp": (
                    exp.timestamp.strftime("%Y-%m-%d %H:%M:%S") if exp.timestamp else "N/A"
                ),
            }
        )

    # Find best values (lowest loss/rmse, highest ssim/psnr)
    best_loss = min(loss_values) if loss_values else float("inf")
    best_ssim = max(ssim_values) if ssim_values else 0.0
    best_psnr = max(psnr_values) if psnr_values else 0.0
    best_rmse = min(rmse_values) if rmse_values else float("inf")

    # Style cells with conditional formatting
    style_data_conditional = [
        # Highlight best values with green background
        {
            "if": {"filter_query": f"{{loss}} = {best_loss}", "column_id": "loss"},
            "backgroundColor": "#d4edda",
            "fontWeight": "bold",
        },
        {
            "if": {"filter_query": f"{{ssim}} = {best_ssim}", "column_id": "ssim"},
            "backgroundColor": "#d4edda",
            "fontWeight": "bold",
        },
        {
            "if": {"filter_query": f"{{psnr}} = {best_psnr}", "column_id": "psnr"},
            "backgroundColor": "#d4edda",
            "fontWeight": "bold",
        },
        {
            "if": {"filter_query": f"{{rmse}} = {best_rmse}", "column_id": "rmse"},
            "backgroundColor": "#d4edda",
            "fontWeight": "bold",
        },
        # Alternate row colors
        {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
    ]

    # Create table
    table = dash_table.DataTable(  # type: ignore[attr-defined]
        id="comparison-metrics-table",
        data=data,
        columns=[
            {"name": "Experiment", "id": "experiment"},
            {"name": "Final Loss", "id": "loss", "type": "numeric", "format": {"specifier": ".6f"}},
            {"name": "SSIM", "id": "ssim", "type": "numeric", "format": {"specifier": ".4f"}},
            {
                "name": "PSNR (dB)",
                "id": "psnr",
                "type": "numeric",
                "format": {"specifier": ".2f"},
            },
            {"name": "RMSE", "id": "rmse", "type": "numeric", "format": {"specifier": ".6f"}},
            {"name": "Epochs", "id": "epochs", "type": "numeric"},
            {"name": "Timestamp", "id": "timestamp"},
        ],
        style_cell={"textAlign": "left", "padding": "10px", "fontFamily": "Arial, sans-serif"},
        style_header={
            "backgroundColor": "#1f77b4",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "center",
        },
        style_data_conditional=style_data_conditional,
        sort_action="native",
        filter_action="native",
        export_format="csv",
        export_headers="display",
        page_size=20,
    )

    return table


def create_config_diff_viewer(experiments: List[Any]) -> html.Div:
    """Create configuration difference viewer with color coding.

    Args:
        experiments: List of ExperimentData objects

    Returns:
        HTML div with configuration differences
    """
    if not experiments:
        return html.Div("No experiments selected", className="text-muted")

    if len(experiments) == 1:
        return html.Div(
            "Select 2 or more experiments to view configuration differences", className="text-muted"
        )

    # Get all unique config keys
    all_keys = set()
    for exp in experiments:
        all_keys.update(exp.config.keys())

    # Find differences
    differences = []
    same_params = []

    for key in sorted(all_keys):
        values = []
        exp_names = []

        for exp in experiments:
            value = exp.config.get(key, None)
            # Format value
            if isinstance(value, float):
                value_str = f"{value:.6g}"
            elif isinstance(value, (list, tuple)):
                value_str = str(value)[:50]  # Truncate
            else:
                value_str = str(value)

            values.append(value_str)
            exp_names.append(exp.exp_id)

        # Check if all values are the same
        if len(set(values)) == 1:
            same_params.append({"param": key, "value": values[0]})
        else:
            differences.append({"param": key, "experiments": exp_names, "values": values})

    # Create display
    if not differences:
        return html.Div(
            [
                html.H5("All configurations are identical!", className="text-success"),
                html.P(f"Total parameters: {len(all_keys)}"),
            ]
        )

    # Build difference table
    diff_rows = []
    for diff in differences:
        # Create row with parameter name
        param_cell = html.Td(html.Strong(diff["param"]), style={"width": "200px"})

        # Create cells for each experiment's value
        value_cells = []
        for exp_name, value in zip(diff["experiments"], diff["values"]):
            value_cells.append(
                html.Td(
                    [
                        html.Div(exp_name, className="small text-muted"),
                        html.Div(value, style={"fontFamily": "monospace"}),
                    ],
                    style={"backgroundColor": "#fff3cd", "padding": "8px"},
                )
            )

        diff_rows.append(html.Tr([param_cell] + value_cells))

    diff_table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [html.Th("Parameter", style={"width": "200px"})]
                    + [html.Th(exp.exp_id) for exp in experiments]
                )
            ),
            html.Tbody(diff_rows),
        ],
        className="table table-sm table-bordered",
        style={"fontSize": "12px"},
    )

    # Build summary
    summary = html.Div(
        [
            html.H5("Configuration Differences", className="mb-3"),
            html.P(
                [
                    html.Strong(f"{len(differences)}"),
                    " parameters differ | ",
                    html.Strong(f"{len(same_params)}"),
                    " parameters are the same",
                ],
                className="text-muted",
            ),
            html.Hr(),
            diff_table,
        ]
    )

    return summary


def create_training_curve_overlay(
    experiments: List[Any], smoothing_window: int = 50, opacity: float = 0.8
) -> go.Figure:
    """Create training curve overlay with multiple experiments.

    Args:
        experiments: List of ExperimentData objects
        smoothing_window: Window size for smoothing (if > 1)
        opacity: Opacity for traces (0-1)

    Returns:
        Plotly figure with overlaid training curves
    """
    if not experiments:
        fig = go.Figure()
        fig.add_annotation(
            text="Select experiments to overlay training curves",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=600)
        return fig

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Loss (log scale)", "SSIM", "PSNR (dB)"),
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.33, 0.33, 0.34],
    )

    # Color palette
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    for idx, exp in enumerate(experiments):
        color = colors[idx % len(colors)]
        metrics = exp.metrics

        if not metrics or "epoch" not in metrics:
            continue

        epochs = metrics.get("epoch", [])

        # Apply smoothing if requested
        def smooth(y: list[float], window: int) -> Any:
            if window <= 1 or len(y) < window:
                return y
            kernel = np.ones(window) / window
            return np.convolve(y, kernel, mode="valid")

        # Loss curve
        losses = metrics.get("loss", [])
        if losses:
            smoothed_loss = smooth(losses, smoothing_window)
            smoothed_epochs = epochs[: len(smoothed_loss)]

            fig.add_trace(
                go.Scatter(
                    x=smoothed_epochs,
                    y=smoothed_loss,
                    name=exp.exp_id,
                    mode="lines",
                    line=dict(color=color, width=2),
                    opacity=opacity,
                    showlegend=True,
                    legendgroup=exp.exp_id,
                    hovertemplate=f"{exp.exp_id}<br>Epoch: %{{x}}<br>Loss: %{{y:.6f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # SSIM curve
        ssims = metrics.get("ssim", [])
        if ssims:
            smoothed_ssim = smooth(ssims, smoothing_window)
            smoothed_epochs = epochs[: len(smoothed_ssim)]

            fig.add_trace(
                go.Scatter(
                    x=smoothed_epochs,
                    y=smoothed_ssim,
                    name=exp.exp_id,
                    mode="lines",
                    line=dict(color=color, width=2),
                    opacity=opacity,
                    showlegend=False,
                    legendgroup=exp.exp_id,
                    hovertemplate=f"{exp.exp_id}<br>Epoch: %{{x}}<br>SSIM: %{{y:.4f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # PSNR curve
        psnrs = metrics.get("psnr", [])
        if psnrs:
            smoothed_psnr = smooth(psnrs, smoothing_window)
            smoothed_epochs = epochs[: len(smoothed_psnr)]

            fig.add_trace(
                go.Scatter(
                    x=smoothed_epochs,
                    y=smoothed_psnr,
                    name=exp.exp_id,
                    mode="lines",
                    line=dict(color=color, width=2),
                    opacity=opacity,
                    showlegend=False,
                    legendgroup=exp.exp_id,
                    hovertemplate=f"{exp.exp_id}<br>Epoch: %{{x}}<br>PSNR: %{{y:.2f}} dB<extra></extra>",
                ),
                row=3,
                col=1,
            )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode="x unified",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=50, r=150, t=50, b=50),
        title=dict(text="Training Curves Overlay", x=0.5),
    )

    # Update axes
    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    fig.update_yaxes(title_text="Loss", type="log", row=1, col=1)
    fig.update_yaxes(title_text="SSIM", row=2, col=1)
    fig.update_yaxes(title_text="PSNR (dB)", row=3, col=1)

    return fig
