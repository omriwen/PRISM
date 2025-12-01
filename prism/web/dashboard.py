"""Main PRISM Dashboard application."""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from loguru import logger

from .callbacks import (
    register_callbacks,
    register_comparison_callbacks,
    register_realtime_callbacks,
)
from .server import DashboardServer


def create_app(runs_dir: Path = Path("runs"), port: int = 8050) -> dash.Dash:
    """Create and configure the Dash application.

    Args:
        runs_dir: Path to directory containing experiment runs
        port: Port number for the server

    Returns:
        Configured Dash application
    """
    # Initialize Dash app with Bootstrap theme
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,
        title="PRISM Dashboard",
    )

    # Initialize server
    server = DashboardServer(runs_dir=runs_dir)

    # Define layout
    app.layout = dbc.Container(
        [
            # Navigation bar
            dbc.NavbarSimple(
                brand="PRISM Dashboard",
                brand_href="/",
                color="primary",
                dark=True,
                fluid=True,
                children=[dbc.NavItem(html.I(className="fas fa-chart-line fa-lg"))],
            ),
            html.Br(),
            # Main content area
            dbc.Row(
                [
                    # Sidebar
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Experiments", className="mb-0")),
                                    dbc.CardBody(
                                        [
                                            html.Label("Select experiments:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id="experiment-selector",
                                                options=[],
                                                multi=True,
                                                placeholder="Select experiments...",
                                                className="mb-3",
                                            ),
                                            html.Div(id="experiment-info", className="mb-3"),
                                            html.Hr(),
                                            html.Label("Refresh Rate:", className="fw-bold"),
                                            dcc.Slider(
                                                id="refresh-rate",
                                                min=1,
                                                max=10,
                                                step=1,
                                                value=2,
                                                marks={i: f"{i}s" for i in range(1, 11)},
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                            html.Div(
                                                className="mt-3 text-muted small",
                                                children=[
                                                    html.I(className="fas fa-info-circle me-2"),
                                                    "Dashboard auto-refreshes to show latest data",
                                                ],
                                            ),
                                            html.Hr(),
                                            html.Label("Smoothing Window:", className="fw-bold"),
                                            dcc.Slider(
                                                id="smoothing-slider",
                                                min=10,
                                                max=200,
                                                step=10,
                                                value=50,
                                                marks={i: str(i) for i in range(0, 201, 50)},
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                            html.Div(
                                                className="mt-3 text-muted small",
                                                children=[
                                                    html.I(className="fas fa-chart-line me-2"),
                                                    "Adjust smoothing for live training curves",
                                                ],
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            html.Br(),
                            # Comparison controls (shown only when in comparison tab)
                            html.Div(id="comparison-controls-container"),
                        ],
                        width=3,
                    ),
                    # Main content
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dcc.Tabs(
                                                id="tabs",
                                                value="tab-live",
                                                children=[
                                                    dcc.Tab(
                                                        label="🔴 Live Monitor",
                                                        value="tab-live",
                                                        className="custom-tab",
                                                        selected_className="custom-tab--selected",
                                                    ),
                                                    dcc.Tab(
                                                        label="Training Metrics",
                                                        value="tab-training",
                                                        className="custom-tab",
                                                        selected_className="custom-tab--selected",
                                                    ),
                                                    dcc.Tab(
                                                        label="Reconstructions",
                                                        value="tab-recon",
                                                        className="custom-tab",
                                                        selected_className="custom-tab--selected",
                                                    ),
                                                    dcc.Tab(
                                                        label="K-Space",
                                                        value="tab-kspace",
                                                        className="custom-tab",
                                                        selected_className="custom-tab--selected",
                                                    ),
                                                    dcc.Tab(
                                                        label="Configuration",
                                                        value="tab-config",
                                                        className="custom-tab",
                                                        selected_className="custom-tab--selected",
                                                    ),
                                                    dcc.Tab(
                                                        label="Metrics Table",
                                                        value="tab-metrics",
                                                        className="custom-tab",
                                                        selected_className="custom-tab--selected",
                                                    ),
                                                    dcc.Tab(
                                                        label="🔀 Comparison",
                                                        value="tab-comparison",
                                                        className="custom-tab",
                                                        selected_className="custom-tab--selected",
                                                    ),
                                                ],
                                            ),
                                            html.Div(id="tab-content", className="mt-3"),
                                            # Live subtabs (hidden when not on live tab)
                                            html.Div(
                                                id="live-subtabs-container",
                                                children=[
                                                    dcc.Tabs(
                                                        id="live-subtabs",
                                                        value="live-overview",
                                                        children=[
                                                            dcc.Tab(
                                                                label="Overview",
                                                                value="live-overview",
                                                                className="custom-tab",
                                                                selected_className="custom-tab--selected",
                                                            ),
                                                            dcc.Tab(
                                                                label="Detailed Metrics",
                                                                value="live-metrics",
                                                                className="custom-tab",
                                                                selected_className="custom-tab--selected",
                                                            ),
                                                            dcc.Tab(
                                                                label="Reconstruction",
                                                                value="live-reconstruction",
                                                                className="custom-tab",
                                                                selected_className="custom-tab--selected",
                                                            ),
                                                            dcc.Tab(
                                                                label="K-Space",
                                                                value="live-kspace",
                                                                className="custom-tab",
                                                                selected_className="custom-tab--selected",
                                                            ),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        id="live-tab-content", className="mt-3"
                                                    ),
                                                ],
                                                style={"display": "none"},  # Hidden by default
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        width=9,
                    ),
                ]
            ),
            # Interval component for auto-refresh
            dcc.Interval(
                id="interval-component",
                interval=2 * 1000,  # Update every 2 seconds (default)
                n_intervals=0,
            ),
            # Store for persisting ROI toggle state across re-renders
            dcc.Store(id="recon-roi-toggle-store", data=False),
            # Footer
            html.Hr(className="mt-5"),
            html.Footer(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.P(
                                        [
                                            "PRISM Dashboard | ",
                                            html.A(
                                                "Documentation",
                                                href="#",
                                                className="text-decoration-none",
                                            ),
                                            " | ",
                                            html.A(
                                                "GitHub",
                                                href="https://github.com/yourusername/SPIDS",
                                                target="_blank",
                                                className="text-decoration-none",
                                            ),
                                        ],
                                        className="text-muted text-center",
                                    )
                                ]
                            )
                        ]
                    )
                ],
                className="mb-3",
            ),
        ],
        fluid=True,
    )

    # Register callbacks
    register_callbacks(app, server)
    register_realtime_callbacks(app, server)
    register_comparison_callbacks(app, server)

    logger.info(f"Dashboard initialized with runs directory: {runs_dir}")

    return app


def run_dashboard(
    runs_dir: Path = Path("runs"), port: int = 8050, debug: bool = False, quiet: bool = True
):
    """Run the dashboard server.

    Args:
        runs_dir: Path to directory containing experiment runs
        port: Port number for the server
        debug: Whether to run in debug mode
        quiet: Whether to suppress HTTP request logging (default True)
    """
    # Suppress Flask/Werkzeug HTTP request logging to avoid cluttering console
    if quiet and not debug:
        import logging

        # Suppress werkzeug request logs
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

    app = create_app(runs_dir=runs_dir, port=port)

    logger.info(f"Starting PRISM Dashboard on http://localhost:{port}")
    logger.info(f"Monitoring experiments in: {runs_dir.absolute()}")

    try:
        app.run(debug=debug, host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        raise


if __name__ == "__main__":
    # Allow running directly with: python -m spids.web.dashboard
    import argparse

    parser = argparse.ArgumentParser(description="PRISM Dashboard")
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="Directory containing experiment runs"
    )
    parser.add_argument("--port", type=int, default=8050, help="Port number for the server")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    run_dashboard(runs_dir=args.runs_dir, port=args.port, debug=args.debug)
