"""Real-time monitoring callbacks for SPIDS dashboard."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from loguru import logger

from ..layouts.live import (
    create_kspace_coverage_plot,
    create_live_progress_panel,
    create_live_reconstruction_preview,
    create_live_training_plot,
)


def register_realtime_callbacks(app, server):
    """Register real-time monitoring callbacks.

    Args:
        app: Dash application instance
        server: DashboardServer instance
    """

    @app.callback(
        Output("live-tab-content", "children"),
        [
            Input("live-subtabs", "value"),
            Input("experiment-selector", "value"),
            Input("interval-component", "n_intervals"),
            Input("smoothing-slider", "value"),
        ],
    )
    def update_live_tab_content(subtab, selected_exps, n, smoothing_window):
        """Update live monitoring tab content.

        Args:
            subtab: Currently selected subtab
            selected_exps: Selected experiment ID(s)
            n: Number of intervals elapsed
            smoothing_window: Smoothing window size for plots

        Returns:
            Dash component(s) for live tab content
        """
        # Live monitoring only works with single experiment
        if not selected_exps:
            return dbc.Alert(
                "Please select an experiment from the dropdown to view live monitoring",
                color="info",
                className="m-4",
            )

        # Ensure single experiment
        if isinstance(selected_exps, list):
            if len(selected_exps) > 1:
                return dbc.Alert(
                    "Live monitoring is only available for single experiment selection. "
                    "Please select only one experiment.",
                    color="warning",
                    className="m-4",
                )
            exp_id = selected_exps[0]
        else:
            exp_id = selected_exps

        try:
            # Force refresh to get latest data
            exp_data = server.refresh_experiment(exp_id)

            if not exp_data:
                return dbc.Alert(
                    f"Failed to load experiment data for '{exp_id}'. Please check the logs.",
                    color="danger",
                    className="m-4",
                )

            # Render content based on subtab
            if subtab == "live-overview":
                return _create_overview_layout(exp_data, smoothing_window)

            elif subtab == "live-metrics":
                return _create_metrics_layout(exp_data, smoothing_window)

            elif subtab == "live-reconstruction":
                return _create_reconstruction_layout(exp_data)

            elif subtab == "live-kspace":
                return _create_kspace_layout(exp_data)

            else:
                return dbc.Alert("Unknown subtab", color="warning", className="m-4")

        except Exception as e:  # noqa: BLE001 - Dash callback must handle all errors
            logger.error(f"Error updating live tab content: {e}")
            return dbc.Alert(f"Error loading live data: {str(e)}", color="danger", className="m-4")

    @app.callback(
        Output("live-reconstruction-graph", "figure"),
        [
            Input("roi-zoom-switch", "value"),
            Input("experiment-selector", "value"),
            Input("interval-component", "n_intervals"),
        ],
    )
    def update_reconstruction_zoom(zoom_to_roi, selected_exps, n):
        """Update reconstruction graph based on ROI zoom state.

        Args:
            zoom_to_roi: Whether to zoom to ROI
            selected_exps: Selected experiment ID(s)
            n: Number of intervals elapsed

        Returns:
            Updated Plotly figure
        """
        # Get experiment data
        if not selected_exps:
            fig = go.Figure()
            fig.add_annotation(
                text="Please select an experiment",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16),
            )
            return fig

        # Ensure single experiment
        if isinstance(selected_exps, list):
            if len(selected_exps) > 1:
                fig = go.Figure()
                fig.add_annotation(
                    text="Please select only one experiment",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
                return fig
            exp_id = selected_exps[0]
        else:
            exp_id = selected_exps

        try:
            # Get experiment data
            exp_data = server.refresh_experiment(exp_id)
            if not exp_data:
                fig = go.Figure()
                fig.add_annotation(
                    text="Failed to load experiment data",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
                return fig

            # Create figure with zoom state
            return create_live_reconstruction_preview(exp_data, zoom_to_roi=zoom_to_roi)

        except Exception as e:  # noqa: BLE001 - Dash callback must handle all errors
            logger.error(f"Error updating reconstruction zoom: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14),
            )
            return fig


def _create_overview_layout(exp_data, smoothing_window):
    """Create overview layout with all key information.

    Args:
        exp_data: ExperimentData object
        smoothing_window: Smoothing window size

    Returns:
        Dash component with overview layout
    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            create_live_progress_panel(exp_data),
                        ],
                        md=12,
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Training Curves", className="mb-0"),
                                        className="bg-primary text-white",
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                figure=create_live_training_plot(
                                                    exp_data, smoothing_window
                                                ),
                                                config={
                                                    "displayModeBar": True,
                                                    "displaylogo": False,
                                                },
                                            )
                                        ]
                                    ),
                                ]
                            )
                        ],
                        md=12,
                    )
                ]
            ),
        ],
        fluid=True,
    )


def _create_metrics_layout(exp_data, smoothing_window):
    """Create detailed metrics layout.

    Args:
        exp_data: ExperimentData object
        smoothing_window: Smoothing window size

    Returns:
        Dash component with metrics layout
    """
    return dbc.Container(
        [
            html.H4("Detailed Training Metrics", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                figure=create_live_training_plot(exp_data, smoothing_window),
                                config={"displayModeBar": True, "displaylogo": False},
                            )
                        ],
                        md=12,
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Smoothing Control", className="mb-3"),
                            html.P(
                                f"Current smoothing window: {smoothing_window} epochs",
                                className="text-muted",
                            ),
                            html.Small(
                                "Adjust the smoothing slider in the sidebar to change the window size.",
                                className="text-muted",
                            ),
                        ],
                        md=12,
                    )
                ]
            ),
        ],
        fluid=True,
    )


def _create_reconstruction_layout(exp_data):
    """Create reconstruction preview layout.

    Args:
        exp_data: ExperimentData object

    Returns:
        Dash component with reconstruction layout
    """
    # Get obj_size from config to show in the info card
    obj_size = exp_data.config.get("obj_size")
    obj_size_info = f"{obj_size}×{obj_size} pixels" if obj_size else "Not specified"

    return dbc.Container(
        [
            html.H4("Live Reconstruction Preview", className="mb-3"),
            # ROI Zoom Control
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Switch(
                                                        id="roi-zoom-switch",
                                                        label="Zoom to ROI (Object Part)",
                                                        value=False,
                                                        className="mb-2",
                                                    ),
                                                    html.Small(
                                                        "Toggle to zoom the view to only the object region of interest",
                                                        className="text-muted",
                                                    ),
                                                ]
                                            )
                                        ]
                                    )
                                ],
                                className="mb-3",
                            )
                        ],
                        md=12,
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="live-reconstruction-graph",
                                figure=create_live_reconstruction_preview(
                                    exp_data, zoom_to_roi=False
                                ),
                                config={"displayModeBar": True, "displaylogo": False},
                            )
                        ],
                        md=12,
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H6("Reconstruction Info", className="mb-3"),
                                            html.P(
                                                [
                                                    html.Strong("Experiment: "),
                                                    exp_data.exp_id,
                                                ]
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Current SSIM: "),
                                                    f"{exp_data.final_metrics.get('ssim', 0):.4f}",
                                                ]
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Current PSNR: "),
                                                    f"{exp_data.final_metrics.get('psnr', 0):.2f} dB",
                                                ]
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Object ROI Size: "),
                                                    obj_size_info,
                                                ]
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        md=4,
                    )
                ]
            ),
        ],
        fluid=True,
    )


def _create_kspace_layout(exp_data):
    """Create k-space coverage layout.

    Args:
        exp_data: ExperimentData object

    Returns:
        Dash component with k-space layout
    """
    return dbc.Container(
        [
            html.H4("K-Space Coverage", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                figure=create_kspace_coverage_plot(exp_data),
                                config={"displayModeBar": True, "displaylogo": False},
                            )
                        ],
                        md=12,
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H6("Sampling Info", className="mb-3"),
                                            html.P(
                                                [
                                                    html.Strong("Total Samples: "),
                                                    str(exp_data.config.get("n_samples", "N/A")),
                                                ]
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Pattern: "),
                                                    str(exp_data.config.get("pattern_fn", "N/A")),
                                                ]
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Sample Diameter: "),
                                                    f"{exp_data.config.get('sample_diameter', 'N/A')} px",
                                                ]
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        md=4,
                    )
                ]
            ),
        ],
        fluid=True,
    )
