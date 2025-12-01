"""Live monitoring layout components for SPIDS dashboard."""

from __future__ import annotations

from typing import Any, List

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import html
from plotly.subplots import make_subplots


def create_live_progress_panel(exp_data: Any) -> dbc.Card:
    """Create live progress panel with current training status.

    Args:
        exp_data: ExperimentData object

    Returns:
        Dash Card component with training progress
    """
    if not exp_data or not exp_data.metrics:
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Live Training Progress", className="card-title"),
                    html.P("No training data available", className="text-muted"),
                ]
            )
        )

    metrics = exp_data.metrics
    final_metrics = exp_data.final_metrics

    # Get current epoch and total epochs
    current_epoch = int(final_metrics.get("epochs", 0))
    max_epochs = exp_data.config.get("max_epochs", current_epoch)

    # Calculate progress percentage
    progress_pct = (current_epoch / max_epochs * 100) if max_epochs > 0 else 0

    # Get current metrics
    current_loss = final_metrics.get("loss", 0.0)
    current_ssim = final_metrics.get("ssim", 0.0)
    current_psnr = final_metrics.get("psnr", 0.0)
    current_lr = exp_data.config.get("lr", 0.0)

    # Calculate changes (comparing last 2 values)
    loss_change = _calculate_percent_change(metrics.get("loss", []))
    ssim_change = _calculate_percent_change(metrics.get("ssim", []))
    psnr_change = _calculate_percent_change(metrics.get("psnr", []))

    # Get status indicators
    loss_status = _get_status_indicator("loss", loss_change)
    ssim_status = _get_status_indicator("ssim", ssim_change)
    psnr_status = _get_status_indicator("psnr", psnr_change)

    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5("Live Training Progress", className="mb-0"),
                className="bg-primary text-white",
            ),
            dbc.CardBody(
                [
                    # Progress bar
                    html.Div(
                        [
                            html.Label(
                                f"Epoch {current_epoch}/{max_epochs}", className="fw-bold mb-2"
                            ),
                            dbc.Progress(
                                value=progress_pct,
                                label=f"{progress_pct:.1f}%",
                                className="mb-3",
                                style={"height": "25px"},
                            ),
                        ]
                    ),
                    html.Hr(),
                    # Metrics grid
                    dbc.Row(
                        [
                            # Loss
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.H6("Loss", className="text-muted mb-1"),
                                            html.H4(
                                                f"{current_loss:.6f}",
                                                className="mb-1",
                                                style={"fontFamily": "monospace"},
                                            ),
                                            html.Small(
                                                [
                                                    _get_change_arrow(loss_change),
                                                    f" {abs(loss_change):.1f}% ",
                                                    loss_status,
                                                ],
                                                className="text-muted",
                                            ),
                                        ],
                                        className="border rounded p-3 mb-3",
                                    )
                                ],
                                md=4,
                            ),
                            # SSIM
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.H6("SSIM", className="text-muted mb-1"),
                                            html.H4(
                                                f"{current_ssim:.4f}",
                                                className="mb-1",
                                                style={"fontFamily": "monospace"},
                                            ),
                                            html.Small(
                                                [
                                                    _get_change_arrow(ssim_change),
                                                    f" {abs(ssim_change):.1f}% ",
                                                    ssim_status,
                                                ],
                                                className="text-muted",
                                            ),
                                        ],
                                        className="border rounded p-3 mb-3",
                                    )
                                ],
                                md=4,
                            ),
                            # PSNR
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.H6("PSNR", className="text-muted mb-1"),
                                            html.H4(
                                                f"{current_psnr:.2f} dB",
                                                className="mb-1",
                                                style={"fontFamily": "monospace"},
                                            ),
                                            html.Small(
                                                [
                                                    _get_change_arrow(psnr_change),
                                                    f" {abs(psnr_change):.1f}% ",
                                                    psnr_status,
                                                ],
                                                className="text-muted",
                                            ),
                                        ],
                                        className="border rounded p-3 mb-3",
                                    )
                                ],
                                md=4,
                            ),
                        ]
                    ),
                    html.Hr(),
                    # Additional info
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Small(
                                        [
                                            html.I(className="fas fa-layer-group me-2"),
                                            f"Learning Rate: {current_lr:.2e}",
                                        ],
                                        className="text-muted",
                                    )
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Small(
                                        [
                                            html.I(className="fas fa-chart-line me-2"),
                                            f"Samples: {exp_data.config.get('n_samples', 'N/A')}",
                                        ],
                                        className="text-muted",
                                    )
                                ],
                                md=6,
                            ),
                        ]
                    ),
                ]
            ),
        ],
        className="mb-3",
    )


def create_live_training_plot(exp_data: Any, smoothing_window: int = 50) -> go.Figure:
    """Create live training plot with smoothed curves.

    Args:
        exp_data: ExperimentData object
        smoothing_window: Window size for smoothing (odd number)

    Returns:
        Plotly figure with smoothed training curves
    """
    if not exp_data or not exp_data.metrics:
        fig = go.Figure()
        fig.add_annotation(
            text="No training data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=500)
        return fig

    metrics = exp_data.metrics
    epochs = metrics.get("epoch", [])
    losses = metrics.get("loss", [])
    ssims = metrics.get("ssim", [])
    psnrs = metrics.get("psnr", [])

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Loss (Smoothed)", "SSIM (Smoothed)", "PSNR (Smoothed)"),
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.33, 0.33, 0.34],
    )

    # Smooth the curves
    if len(epochs) > smoothing_window:
        losses_smooth = _moving_average(losses, smoothing_window)
        ssims_smooth = _moving_average(ssims, smoothing_window)
        psnrs_smooth = _moving_average(psnrs, smoothing_window)
    else:
        losses_smooth = losses
        ssims_smooth = ssims
        psnrs_smooth = psnrs

    # Add loss traces (raw and smoothed)
    if losses:
        # Raw data (faint)
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=losses,
                name="Loss (raw)",
                mode="lines",
                line=dict(color="rgba(99, 110, 250, 0.2)", width=1),
                showlegend=True,
                hovertemplate="Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        # Smoothed data
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=losses_smooth,
                name="Loss (smooth)",
                mode="lines",
                line=dict(color="rgb(99, 110, 250)", width=2),
                showlegend=True,
                hovertemplate="Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Add SSIM traces
    if ssims:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=ssims,
                name="SSIM (raw)",
                mode="lines",
                line=dict(color="rgba(239, 85, 59, 0.2)", width=1),
                showlegend=True,
                hovertemplate="Epoch: %{x}<br>SSIM: %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=ssims_smooth,
                name="SSIM (smooth)",
                mode="lines",
                line=dict(color="rgb(239, 85, 59)", width=2),
                showlegend=True,
                hovertemplate="Epoch: %{x}<br>SSIM: %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Add PSNR traces
    if psnrs:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=psnrs,
                name="PSNR (raw)",
                mode="lines",
                line=dict(color="rgba(0, 204, 150, 0.2)", width=1),
                showlegend=True,
                hovertemplate="Epoch: %{x}<br>PSNR: %{y:.2f} dB<extra></extra>",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=psnrs_smooth,
                name="PSNR (smooth)",
                mode="lines",
                line=dict(color="rgb(0, 204, 150)", width=2),
                showlegend=True,
                hovertemplate="Epoch: %{x}<br>PSNR: %{y:.2f} dB<extra></extra>",
            ),
            row=3,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=700,
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


def create_live_reconstruction_preview(exp_data: Any, zoom_to_roi: bool = False) -> go.Figure:
    """Create live reconstruction preview with optional ROI zoom.

    Args:
        exp_data: ExperimentData object
        zoom_to_roi: If True, zoom to show only the ROI (object part)

    Returns:
        Plotly figure with reconstruction comparison
    """
    if not exp_data or exp_data.reconstruction is None:
        fig = go.Figure()
        fig.add_annotation(
            text="No reconstruction data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=400)
        return fig

    rec = exp_data.reconstruction
    if rec.ndim == 3:
        rec = rec[0]  # Take first channel if multi-channel

    # Apply ROI cropping if requested
    title_suffix = ""
    if zoom_to_roi:
        obj_size = exp_data.config.get("obj_size")
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
            title_suffix = f" (ROI: {obj_size}×{obj_size})"
        else:
            title_suffix = " (ROI size not available)"

    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=[f"Current Reconstruction - {exp_data.exp_id}{title_suffix}"],
    )

    # Get image dimensions for proper axis configuration
    img_height, img_width = rec.shape

    fig.add_trace(
        go.Heatmap(
            z=rec,
            colorscale="gray",
            showscale=True,
            colorbar=dict(title="Intensity"),
            hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Update layout with proper sizing
    # Set explicit axis ranges to match data dimensions exactly
    # Use constrain='domain' to keep axes within figure bounds
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=20, r=80, t=40, b=20),  # Tight margins, space for colorbar
    )
    fig.update_xaxes(
        showticklabels=False,
        range=[0, img_width],
        constrain="domain",
    )
    fig.update_yaxes(
        showticklabels=False,
        range=[img_height, 0],  # Flip y-axis so origin is top-left
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )

    return fig


def create_kspace_coverage_plot(exp_data: Any) -> go.Figure:
    """Create k-space coverage visualization.

    Args:
        exp_data: ExperimentData object

    Returns:
        Plotly figure with k-space coverage
    """
    # Try to load sample points from the experiment
    sample_points_path = exp_data.path / "sample_points.pt"

    if not sample_points_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Sample points data not available",
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
        import torch

        data = torch.load(sample_points_path, map_location="cpu", weights_only=False)
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
        checkpoint_path = exp_data.path / "checkpoint.pt"
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

        # Plot unprocessed samples (red/orange)
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
        # Create 2D histogram of sample positions
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


# Helper functions


def _calculate_percent_change(values: List[float]) -> float:
    """Calculate percentage change between last two values.

    Args:
        values: List of metric values

    Returns:
        Percentage change (negative for decrease, positive for increase)
    """
    if len(values) < 2:
        return 0.0

    last = values[-1]
    prev = values[-2]

    if prev == 0:
        return 0.0

    return ((last - prev) / abs(prev)) * 100


def _get_status_indicator(metric_name: str, percent_change: float) -> html.Span:
    """Get colored status indicator based on metric trend.

    Args:
        metric_name: Name of metric ('loss', 'ssim', 'psnr')
        percent_change: Percentage change in metric

    Returns:
        HTML span element with status indicator
    """
    threshold = 0.1  # 0.1% threshold for "plateaued"

    # For loss, decreasing is good
    if metric_name == "loss":
        if percent_change < -threshold:
            return html.Span("🟢", title="Improving")
        elif abs(percent_change) <= threshold:
            return html.Span("🟡", title="Plateaued")
        else:
            return html.Span("🔴", title="Diverging")
    # For SSIM and PSNR, increasing is good
    else:
        if percent_change > threshold:
            return html.Span("🟢", title="Improving")
        elif abs(percent_change) <= threshold:
            return html.Span("🟡", title="Plateaued")
        else:
            return html.Span("🔴", title="Degrading")


def _get_change_arrow(percent_change: float) -> str:
    """Get arrow symbol for metric change.

    Args:
        percent_change: Percentage change

    Returns:
        Arrow symbol string
    """
    if percent_change > 0.1:
        return "↑"
    elif percent_change < -0.1:
        return "↓"
    else:
        return "→"


def _moving_average(values: List[float], window: int) -> List[float]:
    """Calculate moving average with given window size.

    Args:
        values: List of values to smooth
        window: Window size for moving average

    Returns:
        Smoothed values (same length as input)
    """
    if len(values) < window:
        return values

    # Use numpy for efficient computation
    values_array = np.array(values)
    weights = np.ones(window) / window
    smoothed = np.convolve(values_array, weights, mode="same")

    # Convert to list with explicit type
    result: List[float] = smoothed.tolist()
    return result
