"""Training-related callbacks for SPIDS dashboard."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
from loguru import logger

from ..layouts.main import (
    create_config_table,
    create_kspace_visualization,
    create_metrics_table,
    create_reconstruction_comparison,
    create_training_curves,
)


def register_callbacks(app, server):
    """Register all dashboard callbacks.

    Args:
        app: Dash application instance
        server: DashboardServer instance
    """

    @app.callback(
        Output("experiment-selector", "options"), Input("interval-component", "n_intervals")
    )
    def update_experiment_list(n):
        """Update list of available experiments.

        Args:
            n: Number of intervals elapsed

        Returns:
            List of experiment options for dropdown
        """
        try:
            experiments = server.scan_experiments()
            options = [
                {
                    "label": f"{exp['id']} ({exp['last_modified_str']}, {exp['size_mb']:.1f} MB)",
                    "value": exp["id"],
                }
                for exp in experiments
            ]
            return options
        except Exception as e:  # noqa: BLE001 - Dash callback must handle all errors
            logger.error(f"Error updating experiment list: {e}")
            return []

    @app.callback(
        [
            Output("tab-content", "children"),
            Output("live-subtabs-container", "style"),
        ],
        [
            Input("tabs", "value"),
            Input("experiment-selector", "value"),
            Input("interval-component", "n_intervals"),
        ],
        [
            State("recon-roi-toggle-store", "data"),
        ],
    )
    def update_tab_content(tab, selected_exps, n, roi_toggle_state):
        """Update tab content based on selection.

        Args:
            tab: Currently selected tab
            selected_exps: List of selected experiment IDs
            n: Number of intervals elapsed
            roi_toggle_state: Persisted state of ROI toggle

        Returns:
            Tuple of (content, live_subtabs_style)
        """
        # Style to show/hide live subtabs container
        show_live = {"display": "block"}
        hide_live = {"display": "none"}

        if not selected_exps:
            return (
                dbc.Alert(
                    "Please select one or more experiments from the dropdown above",
                    color="info",
                    className="m-4",
                ),
                hide_live,
            )

        # Ensure selected_exps is a list
        if isinstance(selected_exps, str):
            selected_exps = [selected_exps]

        try:
            # Load experiment data
            experiments = []
            for exp_id in selected_exps:
                exp_data = server.load_experiment_data(exp_id)
                if exp_data:
                    experiments.append(exp_data)

            if not experiments:
                return (
                    dbc.Alert(
                        "Failed to load experiment data. Please check the logs.",
                        color="warning",
                        className="m-4",
                    ),
                    hide_live,
                )

            # Render content based on tab
            if tab == "tab-live":
                # Live monitoring - show subtabs container, hide main content
                return (html.Div(), show_live)

            elif tab == "tab-training":
                fig = create_training_curves(experiments)
                return (
                    dcc.Graph(figure=fig, config={"displayModeBar": True, "displaylogo": False}),
                    hide_live,
                )

            elif tab == "tab-recon":
                # Return container with ROI toggle and graph
                # Use persisted toggle state from Store
                zoom_to_roi = roi_toggle_state if roi_toggle_state is not None else False
                return (
                    dbc.Container(
                        [
                            # ROI Zoom Toggle
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Switch(
                                            id="recon-roi-toggle",
                                            label="Zoom to ROI (Object Region)",
                                            value=zoom_to_roi,
                                            className="mb-0",
                                        ),
                                        html.Small(
                                            "Toggle to zoom the view to only the object region",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                className="mb-3",
                            ),
                            # Reconstruction graph
                            dcc.Graph(
                                id="recon-comparison-graph",
                                figure=create_reconstruction_comparison(
                                    experiments, zoom_to_roi=zoom_to_roi
                                ),
                                config={"displayModeBar": True, "displaylogo": False},
                            ),
                        ],
                        fluid=True,
                    ),
                    hide_live,
                )

            elif tab == "tab-kspace":
                if len(experiments) == 1:
                    fig = create_kspace_visualization(experiments[0])
                    return (
                        dcc.Graph(
                            figure=fig, config={"displayModeBar": True, "displaylogo": False}
                        ),
                        hide_live,
                    )
                else:
                    return (
                        dbc.Alert(
                            "K-space visualization is only available for single experiment selection",
                            color="info",
                            className="m-4",
                        ),
                        hide_live,
                    )

            elif tab == "tab-config":
                if len(experiments) == 1:
                    return (
                        dbc.Container(
                            [
                                html.H4(
                                    f"Configuration: {experiments[0].exp_id}", className="mt-3 mb-3"
                                ),
                                create_config_table(experiments[0]),
                            ],
                            fluid=True,
                        ),
                        hide_live,
                    )
                else:
                    return (
                        dbc.Alert(
                            "Configuration view is only available for single experiment selection",
                            color="info",
                            className="m-4",
                        ),
                        hide_live,
                    )

            elif tab == "tab-metrics":
                return (
                    dbc.Container(
                        [
                            html.H4("Metrics Comparison", className="mt-3 mb-3"),
                            create_metrics_table(experiments),
                        ],
                        fluid=True,
                    ),
                    hide_live,
                )

            elif tab == "tab-comparison":
                # Comparison tab - placeholder for comparison content
                return (html.Div(id="comparison-content"), hide_live)

        except Exception as e:  # noqa: BLE001 - Dash callback must handle all errors
            logger.error(f"Error updating tab content: {e}")
            return (
                dbc.Alert(f"Error loading data: {str(e)}", color="danger", className="m-4"),
                hide_live,
            )

        return (dbc.Alert("Unknown tab", color="warning", className="m-4"), hide_live)

    @app.callback(Output("interval-component", "interval"), Input("refresh-rate", "value"))
    def update_refresh_rate(refresh_seconds):
        """Update refresh interval based on slider.

        Args:
            refresh_seconds: Refresh rate in seconds

        Returns:
            Interval in milliseconds
        """
        return refresh_seconds * 1000

    @app.callback(Output("experiment-info", "children"), Input("experiment-selector", "value"))
    def update_experiment_info(selected_exps):
        """Update experiment information panel.

        Args:
            selected_exps: List of selected experiment IDs

        Returns:
            Dash component with experiment info
        """
        if not selected_exps:
            return html.Div("No experiments selected", className="text-muted")

        if isinstance(selected_exps, str):
            selected_exps = [selected_exps]

        return html.Div(
            [
                html.P(f"Selected: {len(selected_exps)} experiment(s)", className="mb-0"),
                html.Ul([html.Li(exp_id, className="small") for exp_id in selected_exps]),
            ]
        )

    @app.callback(
        Output("recon-roi-toggle-store", "data"),
        Input("recon-roi-toggle", "value"),
        prevent_initial_call=True,
    )
    def save_roi_toggle_state(zoom_to_roi):
        """Save ROI toggle state to Store for persistence across re-renders.

        Args:
            zoom_to_roi: Current toggle value

        Returns:
            Toggle value to store
        """
        return zoom_to_roi

    @app.callback(
        Output("recon-comparison-graph", "figure"),
        [
            Input("recon-roi-toggle", "value"),
            Input("experiment-selector", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_reconstruction_roi_zoom(zoom_to_roi, selected_exps):
        """Update reconstruction comparison graph based on ROI zoom toggle.

        Args:
            zoom_to_roi: Whether to zoom to ROI
            selected_exps: List of selected experiment IDs

        Returns:
            Updated Plotly figure
        """
        import plotly.graph_objects as go

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

        # Ensure selected_exps is a list
        if isinstance(selected_exps, str):
            selected_exps = [selected_exps]

        try:
            # Load experiment data
            experiments = []
            for exp_id in selected_exps:
                exp_data = server.load_experiment_data(exp_id)
                if exp_data:
                    experiments.append(exp_data)

            if not experiments:
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

            return create_reconstruction_comparison(experiments, zoom_to_roi=zoom_to_roi)

        except Exception as e:  # noqa: BLE001 - Dash callback must handle all errors
            logger.error(f"Error updating reconstruction ROI zoom: {e}")
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
