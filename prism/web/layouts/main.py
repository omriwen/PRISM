"""Main layout components for SPIDS dashboard."""

from __future__ import annotations

from typing import Any, List

import plotly.graph_objects as go
from dash import dash_table
from plotly.subplots import make_subplots


def create_training_curves(experiments: List[Any]) -> go.Figure:
    """Create training curves plot with multiple experiments.

    Args:
        experiments: List of ExperimentData objects

    Returns:
        Plotly figure with training curves
    """
    if not experiments:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Select experiments to display training curves",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=600)
        return fig

    # Create subplots for different metrics
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Loss", "SSIM", "PSNR (dB)"),
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.33, 0.33, 0.34],
    )

    # Define colors for different experiments
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, exp in enumerate(experiments):
        color = colors[idx % len(colors)]
        metrics = exp.metrics

        if not metrics or "epoch" not in metrics:
            continue

        epochs = metrics.get("epoch", [])

        # Loss curve
        losses = metrics.get("loss", [])
        if losses:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=losses,
                    name=exp.exp_id,
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=True,
                    legendgroup=exp.exp_id,
                    hovertemplate="Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # SSIM curve
        ssims = metrics.get("ssim", [])
        if ssims:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=ssims,
                    name=exp.exp_id,
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                    legendgroup=exp.exp_id,
                    hovertemplate="Epoch: %{x}<br>SSIM: %{y:.4f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # PSNR curve
        psnrs = metrics.get("psnr", [])
        if psnrs:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=psnrs,
                    name=exp.exp_id,
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                    legendgroup=exp.exp_id,
                    hovertemplate="Epoch: %{x}<br>PSNR: %{y:.2f} dB<extra></extra>",
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
    )

    # Update axes
    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    fig.update_yaxes(title_text="Loss", type="log", row=1, col=1)
    fig.update_yaxes(title_text="SSIM", row=2, col=1)
    fig.update_yaxes(title_text="PSNR (dB)", row=3, col=1)

    return fig


def create_reconstruction_comparison(
    experiments: List[Any], zoom_to_roi: bool = False
) -> go.Figure:
    """Create side-by-side reconstruction comparison.

    Args:
        experiments: List of ExperimentData objects
        zoom_to_roi: If True, zoom to show only the ROI (object part)

    Returns:
        Plotly figure with reconstructions
    """
    if not experiments:
        fig = go.Figure()
        fig.add_annotation(
            text="Select experiments to display reconstructions",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=400)
        return fig

    n_exps = len(experiments)
    cols = min(4, n_exps)  # Max 4 columns
    rows = (n_exps + cols - 1) // cols  # Ceiling division

    # Build subplot titles with ROI suffix if needed
    titles = []
    for exp in experiments:
        title = exp.exp_id
        if zoom_to_roi:
            obj_size = exp.config.get("obj_size")
            if obj_size:
                title += f" (ROI: {obj_size}×{obj_size})"
        titles.append(title)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    for idx, exp in enumerate(experiments):
        row = idx // cols + 1
        col = idx % cols + 1

        if exp.reconstruction is not None:
            # Handle different array shapes
            rec = exp.reconstruction
            if rec.ndim == 3:
                rec = rec[0]  # Take first channel if multi-channel

            # Apply ROI cropping if requested
            if zoom_to_roi:
                obj_size = exp.config.get("obj_size")
                if obj_size is not None and obj_size > 0:
                    # Calculate ROI bounds (center crop)
                    h, w = rec.shape
                    center_y, center_x = h // 2, w // 2
                    half_size = obj_size // 2

                    # Ensure bounds are within image
                    y_start = max(0, center_y - half_size)
                    y_end = min(h, center_y + half_size)
                    x_start = max(0, center_x - half_size)
                    x_end = min(w, center_x + half_size)

                    # Crop to ROI
                    rec = rec[y_start:y_end, x_start:x_end]

            # Get image dimensions for axis configuration
            img_height, img_width = rec.shape

            fig.add_trace(
                go.Heatmap(
                    z=rec,
                    colorscale="gray",
                    showscale=(col == cols),  # Only show colorbar for last column
                    hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            # Set explicit axis ranges for this subplot to fill the plot area
            fig.update_xaxes(
                range=[0, img_width],
                constrain="domain",
                row=row,
                col=col,
            )
            fig.update_yaxes(
                range=[img_height, 0],  # Flip y-axis so origin is top-left
                constrain="domain",
                row=row,
                col=col,
            )
        else:
            # No reconstruction available
            fig.add_annotation(
                text="No reconstruction available",
                xref=f"x{idx + 1}",
                yref=f"y{idx + 1}",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=12),
            )

    # Update layout
    fig.update_layout(
        height=300 * rows,
        showlegend=False,
        margin=dict(l=20, r=60, t=40, b=20),  # Tight margins
    )

    # Remove axis labels for cleaner look and maintain aspect ratio
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, scaleanchor="x", scaleratio=1)

    return fig


def create_kspace_visualization(experiment: Any) -> go.Figure:
    """Create k-space coverage visualization.

    Args:
        experiment: ExperimentData object

    Returns:
        Plotly figure with k-space visualization
    """
    # Try to load sample points from the experiment
    sample_points_path = experiment.path / "sample_points.pt"

    if not sample_points_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Sample points data not available for this experiment",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=400)
        return fig

    try:
        import numpy as np
        import torch

        data = torch.load(sample_points_path, map_location="cpu")
        centers = data["centers"]
        # diameter = data.get("diameter", 10)  # Not used currently

        # Convert to numpy
        if isinstance(centers, torch.Tensor):
            centers = centers.cpu().numpy()

        # Handle different center formats:
        # - Point samples: (N, 2) - just x, y coordinates
        # - Line samples: (N, 2, 2) - start and end points of each line
        if centers.ndim == 3 and centers.shape[1:] == (2, 2):
            # Line samples: compute midpoints for visualization
            centers = centers.mean(axis=1)  # Average start and end -> (N, 2)

        # Determine which samples have been processed
        # Load checkpoint to get last_center_idx
        checkpoint_path = experiment.path / "checkpoint.pt"
        last_processed_idx = -1  # Default: no samples processed
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if "last_center_idx" in checkpoint:
                    last_processed_idx = int(checkpoint["last_center_idx"].item())
            except Exception:  # noqa: BLE001 - Checkpoint loading failure is non-fatal
                pass  # If checkpoint loading fails, assume no samples processed

        # Split centers into processed and unprocessed
        n_samples = len(centers)
        processed_mask = np.arange(n_samples) <= last_processed_idx
        processed_centers = centers[processed_mask]
        unprocessed_centers = centers[~processed_mask]

        # Create figure with subplots
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Sample Positions", "Coverage Density"),
            horizontal_spacing=0.15,
        )

        # Plot 1: Sample positions scatter with color coding
        # Plot processed samples first (green)
        if len(processed_centers) > 0:
            fig.add_trace(
                go.Scatter(
                    x=processed_centers[:, 0],
                    y=processed_centers[:, 1],
                    mode="markers",
                    marker=dict(size=5, color="green", opacity=0.7),
                    name="Processed",
                    hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>Status: Processed<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Plot unprocessed samples (red)
        if len(unprocessed_centers) > 0:
            fig.add_trace(
                go.Scatter(
                    x=unprocessed_centers[:, 0],
                    y=unprocessed_centers[:, 1],
                    mode="markers",
                    marker=dict(size=5, color="red", opacity=0.5),
                    name="Unprocessed",
                    hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>Status: Unprocessed<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Plot 2: Coverage heatmap (2D histogram)
        hist, xedges, yedges = np.histogram2d(centers[:, 0], centers[:, 1], bins=50, density=True)

        fig.add_trace(
            go.Heatmap(
                z=hist.T,
                x=xedges,
                y=yedges,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Density", x=1.15),
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>density: %{z:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Update layout with legend
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Update axes
        fig.update_xaxes(title_text="kx", scaleanchor="y", scaleratio=1, row=1, col=1)
        fig.update_yaxes(title_text="ky", row=1, col=1)
        fig.update_xaxes(title_text="kx", scaleanchor="y2", scaleratio=1, row=1, col=2)
        fig.update_yaxes(title_text="ky", row=1, col=2)

        return fig

    except Exception as e:  # noqa: BLE001 - Dashboard visualization must handle all errors
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading k-space data: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=400)
        return fig


def create_metrics_table(experiments: List[Any]) -> Any:  # dash_table.DataTable
    """Create metrics comparison table.

    Args:
        experiments: List of ExperimentData objects

    Returns:
        Dash DataTable component
    """
    if not experiments:
        return dash_table.DataTable(  # type: ignore[attr-defined]
            data=[],
            columns=[
                {"name": "Experiment", "id": "experiment"},
                {"name": "Final Loss", "id": "loss"},
                {"name": "SSIM", "id": "ssim"},
                {"name": "PSNR (dB)", "id": "psnr"},
                {"name": "Epochs", "id": "epochs"},
            ],
        )

    # Prepare data
    data = []
    for exp in experiments:
        fm = exp.final_metrics
        data.append(
            {
                "experiment": exp.exp_id,
                "loss": f"{fm.get('loss', 0):.6f}" if fm.get("loss") is not None else "N/A",
                "ssim": f"{fm.get('ssim', 0):.4f}" if fm.get("ssim") is not None else "N/A",
                "psnr": f"{fm.get('psnr', 0):.2f}" if fm.get("psnr") is not None else "N/A",
                "rmse": f"{fm.get('rmse', 0):.6f}" if fm.get("rmse") is not None else "N/A",
                "epochs": fm.get("epochs", 0),
            }
        )

    # Create table
    table = dash_table.DataTable(  # type: ignore[attr-defined]
        data=data,
        columns=[
            {"name": "Experiment", "id": "experiment"},
            {"name": "Final Loss", "id": "loss"},
            {"name": "SSIM", "id": "ssim"},
            {"name": "PSNR (dB)", "id": "psnr"},
            {"name": "RMSE", "id": "rmse"},
            {"name": "Epochs", "id": "epochs"},
        ],
        style_cell={"textAlign": "left", "padding": "10px", "fontFamily": "Arial, sans-serif"},
        style_header={"backgroundColor": "#1f77b4", "color": "white", "fontWeight": "bold"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"}],
        sort_action="native",
        filter_action="native",
        page_size=20,
    )

    return table


def create_config_table(experiment: Any) -> Any:  # dash_table.DataTable
    """Create configuration table for an experiment.

    Args:
        experiment: ExperimentData object

    Returns:
        Dash DataTable component
    """
    if not experiment or not experiment.config:
        return dash_table.DataTable(  # type: ignore[attr-defined]
            data=[],
            columns=[{"name": "Parameter", "id": "parameter"}, {"name": "Value", "id": "value"}],
        )

    # Prepare data
    data = []
    for key, value in sorted(experiment.config.items()):
        # Format value
        if isinstance(value, float):
            value_str = f"{value:.6g}"
        elif isinstance(value, (list, tuple)):
            value_str = str(value)[:100]  # Truncate long lists
        else:
            value_str = str(value)

        data.append({"parameter": key, "value": value_str})

    # Create table
    table = dash_table.DataTable(  # type: ignore[attr-defined]
        data=data,
        columns=[{"name": "Parameter", "id": "parameter"}, {"name": "Value", "id": "value"}],
        style_cell={
            "textAlign": "left",
            "padding": "8px",
            "fontFamily": "monospace",
            "fontSize": "12px",
        },
        style_header={"backgroundColor": "#1f77b4", "color": "white", "fontWeight": "bold"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"}],
        filter_action="native",
        page_size=50,
    )

    return table
