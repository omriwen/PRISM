"""Comparison-related callbacks for SPIDS dashboard."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from loguru import logger

from ..layouts.comparison import (
    create_comparison_metrics_table,
    create_config_diff_viewer,
    create_side_by_side_comparison,
    create_training_curve_overlay,
)


def register_comparison_callbacks(app, server):
    """Register comparison-specific callbacks.

    Args:
        app: Dash application instance
        server: DashboardServer instance
    """

    @app.callback(Output("comparison-controls-container", "children"), Input("tabs", "value"))
    def toggle_comparison_controls(tab):
        """Show comparison controls only when in comparison tab.

        Args:
            tab: Currently selected tab

        Returns:
            Comparison controls or empty div
        """
        if tab == "tab-comparison":
            return create_comparison_controls()
        return html.Div()

    @app.callback(
        Output("comparison-content", "children"),
        [
            Input("comparison-view-selector", "value"),
            Input("experiment-selector", "value"),
            Input("smoothing-slider", "value"),
            Input("opacity-slider", "value"),
            Input("sync-axes-toggle", "value"),
        ],
    )
    def update_comparison_view(view_type, selected_exps, smoothing, opacity, sync_axes):
        """Update comparison view based on selections.

        Args:
            view_type: Type of comparison view
            selected_exps: List of selected experiment IDs
            smoothing: Smoothing window size
            opacity: Trace opacity (0-1)
            sync_axes: Whether to sync axes

        Returns:
            Dash component(s) for comparison view
        """
        if not selected_exps:
            return dbc.Alert(
                "Please select 2 or more experiments to compare",
                color="info",
                className="m-4",
            )

        # Ensure selected_exps is a list
        if isinstance(selected_exps, str):
            selected_exps = [selected_exps]

        if len(selected_exps) < 2:
            return dbc.Alert(
                "Please select at least 2 experiments for comparison",
                color="warning",
                className="m-4",
            )

        try:
            # Load experiment data
            experiments = []
            for exp_id in selected_exps:
                exp_data = server.load_experiment_data(exp_id)
                if exp_data:
                    experiments.append(exp_data)

            if len(experiments) < 2:
                return dbc.Alert(
                    "Could not load enough experiment data. Please check the logs.",
                    color="warning",
                    className="m-4",
                )

            # Limit to 4 experiments for side-by-side
            if len(experiments) > 4 and view_type == "side_by_side":
                logger.warning(
                    f"Limiting comparison to first 4 experiments (selected {len(experiments)})"
                )
                experiments = experiments[:4]

            # Determine opacity value (convert from percentage if needed)
            opacity_val = opacity / 100.0 if opacity > 1 else opacity

            # Render based on view type
            if view_type == "reconstructions":
                sync_bool = True if sync_axes else False
                fig = create_side_by_side_comparison(
                    experiments, view_mode="reconstruction", sync_axes=sync_bool
                )
                return html.Div(
                    [
                        html.H4("Side-by-Side Reconstruction Comparison", className="mb-3"),
                        dbc.Alert(
                            [
                                html.I(className="fas fa-info-circle me-2"),
                                "Hover over images to see pixel values. Use mouse to zoom/pan.",
                            ],
                            color="info",
                            className="mb-3",
                        ),
                        dcc.Graph(
                            figure=fig,
                            config={
                                "displayModeBar": True,
                                "displaylogo": False,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "reconstruction_comparison",
                                    "height": 800,
                                    "width": 1200,
                                    "scale": 2,
                                },
                            },
                        ),
                    ]
                )

            elif view_type == "metrics":
                table = create_comparison_metrics_table(experiments)
                return html.Div(
                    [
                        html.H4("Metrics Comparison Table", className="mb-3"),
                        dbc.Alert(
                            [
                                html.I(className="fas fa-star me-2"),
                                "Best values are highlighted in green. Click column headers to sort. Use export button to save as CSV.",
                            ],
                            color="info",
                            className="mb-3",
                        ),
                        table,
                    ]
                )

            elif view_type == "config":
                diff_viewer = create_config_diff_viewer(experiments)
                return html.Div(
                    [
                        html.H4("Configuration Differences", className="mb-3"),
                        dbc.Alert(
                            [
                                html.I(className="fas fa-wrench me-2"),
                                "Only parameters that differ between experiments are shown.",
                            ],
                            color="info",
                            className="mb-3",
                        ),
                        diff_viewer,
                    ]
                )

            elif view_type == "training_curves":
                fig = create_training_curve_overlay(
                    experiments, smoothing_window=smoothing, opacity=opacity_val
                )
                return html.Div(
                    [
                        html.H4("Training Curves Overlay", className="mb-3"),
                        dbc.Alert(
                            [
                                html.I(className="fas fa-chart-line me-2"),
                                f"Smoothing: {smoothing} epochs | Opacity: {int(opacity_val * 100)}%",
                            ],
                            color="info",
                            className="mb-3",
                        ),
                        dcc.Graph(
                            figure=fig,
                            config={
                                "displayModeBar": True,
                                "displaylogo": False,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "training_curves_comparison",
                                    "height": 800,
                                    "width": 1200,
                                    "scale": 2,
                                },
                            },
                        ),
                    ]
                )

            elif view_type == "all":
                # Comprehensive view with all comparisons
                sync_bool = True if sync_axes else False
                recon_fig = create_side_by_side_comparison(
                    experiments, view_mode="reconstruction", sync_axes=sync_bool
                )
                metrics_table = create_comparison_metrics_table(experiments)
                config_diff = create_config_diff_viewer(experiments)
                curves_fig = create_training_curve_overlay(
                    experiments, smoothing_window=smoothing, opacity=opacity_val
                )

                return html.Div(
                    [
                        # Reconstructions
                        html.H4("Reconstructions", className="mb-3 mt-3"),
                        dcc.Graph(
                            figure=recon_fig,
                            config={"displayModeBar": True, "displaylogo": False},
                        ),
                        html.Hr(),
                        # Metrics table
                        html.H4("Metrics Comparison", className="mb-3 mt-3"),
                        metrics_table,
                        html.Hr(),
                        # Training curves
                        html.H4("Training Curves", className="mb-3 mt-3"),
                        dcc.Graph(
                            figure=curves_fig,
                            config={"displayModeBar": True, "displaylogo": False},
                        ),
                        html.Hr(),
                        # Config diff
                        html.H4("Configuration Differences", className="mb-3 mt-3"),
                        config_diff,
                    ]
                )

            else:
                return dbc.Alert(
                    f"Unknown view type: {view_type}", color="warning", className="m-4"
                )

        except Exception as e:  # noqa: BLE001 - Dash callback must handle all errors
            logger.error(f"Error in comparison view: {e}")
            return dbc.Alert(f"Error: {str(e)}", color="danger", className="m-4")


def create_comparison_controls() -> html.Div:
    """Create control panel for comparison view.

    Returns:
        HTML div with comparison controls
    """
    return html.Div(
        [
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Comparison Controls", className="mb-0")),
                    dbc.CardBody(
                        [
                            # View selector
                            html.Label("View Type:", className="fw-bold"),
                            dcc.Dropdown(
                                id="comparison-view-selector",
                                options=[  # type: ignore[arg-type]
                                    {
                                        "label": "📊 All Views (Comprehensive)",
                                        "value": "all",
                                    },
                                    {
                                        "label": "🖼️ Reconstructions Side-by-Side",
                                        "value": "reconstructions",
                                    },
                                    {
                                        "label": "📈 Training Curves Overlay",
                                        "value": "training_curves",
                                    },
                                    {
                                        "label": "📋 Metrics Table",
                                        "value": "metrics",
                                    },
                                    {
                                        "label": "⚙️ Configuration Diff",
                                        "value": "config",
                                    },
                                ],
                                value="all",
                                clearable=False,
                                className="mb-3",
                            ),
                            html.Hr(),
                            # Opacity slider
                            html.Label("Curve Opacity:", className="fw-bold"),
                            dcc.Slider(
                                id="opacity-slider",
                                min=20,
                                max=100,
                                step=10,
                                value=80,
                                marks={i: f"{i}%" for i in range(20, 101, 20)},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Div(
                                className="mt-2 mb-3 text-muted small",
                                children=[
                                    html.I(className="fas fa-adjust me-2"),
                                    "Adjust transparency when comparing many experiments",
                                ],
                            ),
                            html.Hr(),
                            # Sync axes toggle
                            html.Label("Synchronize Axes:", className="fw-bold"),
                            dbc.Checklist(
                                id="sync-axes-toggle",
                                options=[
                                    {
                                        "label": " Sync zoom/pan across all plots",
                                        "value": True,
                                    }
                                ],
                                value=[True],
                                switch=True,
                                className="mb-3",
                            ),
                            html.Div(
                                className="mt-2 text-muted small",
                                children=[
                                    html.I(className="fas fa-link me-2"),
                                    "When enabled, zooming one plot affects all others",
                                ],
                            ),
                        ]
                    ),
                ]
            )
        ]
    )
