# prism/web/callbacks/profiling.py
"""Profiling callbacks for PRISM dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
from dash import Input, Output, State, html
from loguru import logger


if TYPE_CHECKING:
    import dash

    from prism.web.server import DashboardServer


def register_profiling_callbacks(app: dash.Dash, server: DashboardServer) -> None:
    """Register profiling-related callbacks.

    Args:
        app: Dash application instance
        server: DashboardServer instance
    """

    @app.callback(
        Output("profile-file-dropdown", "options"),
        [Input("refresh-profiles-btn", "n_clicks")],
        prevent_initial_call=False,
    )
    def refresh_profile_files(n_clicks: int | None) -> list[dict[str, str]]:
        """Refresh available profile files."""
        from prism.web.layouts.profiling import get_profile_files

        return get_profile_files(server.runs_dir)

    @app.callback(
        [
            Output("profile-content", "style"),
            Output("profile-empty-state", "style"),
            Output("profile-summary-content", "children"),
            Output("operations-graph", "figure"),
            Output("memory-graph", "figure"),
            Output("bottlenecks-content", "children"),
            Output("profile-data-store", "data"),
        ],
        [Input("load-profile-btn", "n_clicks")],
        [State("profile-file-dropdown", "value")],
        prevent_initial_call=True,
    )
    def load_profile(
        n_clicks: int | None,
        profile_path: str | None,
    ) -> tuple[dict, dict, Any, go.Figure, go.Figure, Any, dict | None]:
        """Load and display profile data."""
        if not profile_path or not Path(profile_path).exists():
            # Return empty state
            return (
                {"display": "none"},  # Hide content
                {"display": "block"},  # Show empty state
                html.P("No profile loaded", className="text-muted"),
                go.Figure(),
                go.Figure(),
                html.P("No profile loaded", className="text-muted"),
                None,
            )

        try:
            from prism.profiling.analyzer import ProfileAnalyzer
            from prism.profiling.visualization.interactive import InteractiveProfilePlotter
            from prism.web.layouts.profiling import (
                create_bottlenecks_display,
                create_summary_card,
            )

            # Load profile
            analyzer = ProfileAnalyzer(Path(profile_path))

            # Create interactive plotter
            plotter = InteractiveProfilePlotter(analyzer)

            # Generate visualizations
            summary = create_summary_card(analyzer)
            operations_fig = plotter.create_operations_figure()
            memory_fig = plotter.create_memory_figure()
            bottlenecks = create_bottlenecks_display(analyzer)

            logger.info(f"Profile loaded: {profile_path}")

            return (
                {"display": "block"},  # Show content
                {"display": "none"},  # Hide empty state
                summary,
                operations_fig,
                memory_fig,
                bottlenecks,
                {"path": profile_path},
            )

        except Exception as e:
            logger.error(f"Error loading profile: {e}")
            return (
                {"display": "none"},
                {"display": "block"},
                html.P(f"Error: {e}", className="text-danger"),
                go.Figure(),
                go.Figure(),
                html.P(f"Error: {e}", className="text-danger"),
                None,
            )

    @app.callback(
        Output("call-graph-graph", "figure"),
        [
            Input("call-graph-type", "value"),
            Input("profile-data-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_call_graph(
        graph_type: str,
        profile_data: dict | None,
    ) -> go.Figure:
        """Update call graph visualization based on type selection."""
        if not profile_data or "path" not in profile_data:
            return go.Figure()

        try:
            from prism.profiling.analyzer import ProfileAnalyzer
            from prism.profiling.call_graph import CallGraphBuilder
            from prism.profiling.visualization.interactive import InteractiveProfilePlotter

            # Load profile
            analyzer = ProfileAnalyzer(Path(profile_data["path"]))
            plotter = InteractiveProfilePlotter(analyzer)

            # Build call graph
            builder = CallGraphBuilder()
            root = builder.build_from_regions(analyzer.data.region_times)

            # Generate visualization based on type
            if graph_type == "flame":
                return plotter.create_flame_graph(root)
            else:  # Default to sunburst
                return plotter.create_sunburst_figure(root)

        except Exception as e:
            logger.error(f"Error creating call graph: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {e}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig
