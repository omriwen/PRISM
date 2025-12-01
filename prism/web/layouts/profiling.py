# prism/web/layouts/profiling.py
"""Profiling layout components for PRISM dashboard."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dash_bootstrap_components as dbc
from dash import dcc, html


if TYPE_CHECKING:

    from prism.profiling.analyzer import ProfileAnalyzer


def create_profiling_layout() -> html.Div:
    """Create profiling tab layout.

    Returns:
        Dash HTML Div containing the profiling tab layout
    """
    return html.Div([
        # Profile file selection
        dbc.Row([
            dbc.Col([
                html.H4("Profile Analysis", className="mb-3"),
                dbc.InputGroup([
                    dbc.InputGroupText(html.I(className="fas fa-file-alt")),
                    dcc.Dropdown(
                        id="profile-file-dropdown",
                        placeholder="Select profile file...",
                        className="flex-grow-1",
                    ),
                ]),
            ], width=8),
            dbc.Col([
                dbc.Button(
                    [html.I(className="fas fa-sync-alt me-2"), "Refresh"],
                    id="refresh-profiles-btn",
                    color="secondary",
                    className="me-2 mt-4",
                ),
                dbc.Button(
                    [html.I(className="fas fa-chart-bar me-2"), "Load Profile"],
                    id="load-profile-btn",
                    color="primary",
                    className="mt-4",
                ),
            ], width=4, className="text-end"),
        ], className="mb-4"),

        # Profile content (hidden until profile is loaded)
        html.Div(id="profile-content", children=[
            # Summary card
            dbc.Card([
                dbc.CardHeader(html.H5("Summary", className="mb-0")),
                dbc.CardBody(id="profile-summary-content"),
            ], className="mb-4"),

            # Visualization tabs
            dbc.Tabs([
                dbc.Tab(label="Operations", tab_id="tab-operations", children=[
                    dcc.Graph(id="operations-graph", style={"height": "500px"}),
                ]),
                dbc.Tab(label="Memory", tab_id="tab-memory", children=[
                    dcc.Graph(id="memory-graph", style={"height": "400px"}),
                ]),
                dbc.Tab(label="Call Graph", tab_id="tab-call-graph", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Visualization Type:"),
                            dcc.Dropdown(
                                id="call-graph-type",
                                options=[
                                    {"label": "Sunburst Chart", "value": "sunburst"},
                                    {"label": "Flame Graph", "value": "flame"},
                                ],
                                value="sunburst",
                                clearable=False,
                            ),
                        ], width=3),
                    ], className="mb-3"),
                    dcc.Graph(id="call-graph-graph", style={"height": "600px"}),
                ]),
                dbc.Tab(label="Bottlenecks", tab_id="tab-bottlenecks", children=[
                    html.Div(id="bottlenecks-content", className="p-3"),
                ]),
            ], id="profile-tabs", active_tab="tab-operations"),
        ], style={"display": "none"}),

        # Empty state message
        html.Div(
            id="profile-empty-state",
            children=[
                html.Div([
                    html.I(className="fas fa-chart-area fa-4x text-muted mb-3"),
                    html.H5("No Profile Loaded", className="text-muted"),
                    html.P(
                        "Select a profile file and click 'Load Profile' to analyze training performance.",
                        className="text-muted",
                    ),
                ], className="text-center py-5"),
            ],
        ),

        # Store for profile data path
        dcc.Store(id="profile-data-store"),
    ])


def create_summary_card(analyzer: ProfileAnalyzer) -> dbc.Row:
    """Create summary statistics card content.

    Args:
        analyzer: ProfileAnalyzer instance with loaded data

    Returns:
        Dash Bootstrap Row with summary statistics
    """
    summary = analyzer.get_summary()

    return dbc.Row([
        dbc.Col([
            html.Div([
                html.H6("Total Samples", className="text-muted mb-1"),
                html.H4(f"{summary['total_samples']:,}", className="mb-0"),
            ], className="text-center p-3 border rounded"),
        ], width=2),
        dbc.Col([
            html.Div([
                html.H6("Total Epochs", className="text-muted mb-1"),
                html.H4(f"{summary['total_epochs']:,}", className="mb-0"),
            ], className="text-center p-3 border rounded"),
        ], width=2),
        dbc.Col([
            html.Div([
                html.H6("Total Time", className="text-muted mb-1"),
                html.H4(f"{summary['total_time_ms']:.1f} ms", className="mb-0"),
            ], className="text-center p-3 border rounded"),
        ], width=2),
        dbc.Col([
            html.Div([
                html.H6("Avg Epoch Time", className="text-muted mb-1"),
                html.H4(f"{summary['avg_epoch_time_ms']:.2f} ms", className="mb-0"),
            ], className="text-center p-3 border rounded"),
        ], width=2),
        dbc.Col([
            html.Div([
                html.H6("Peak GPU Memory", className="text-muted mb-1"),
                html.H4(f"{summary['peak_gpu_memory_mb']:.1f} MB", className="mb-0"),
            ], className="text-center p-3 border rounded"),
        ], width=2),
        dbc.Col([
            html.Div([
                html.H6("Memory Leak", className="text-muted mb-1"),
                html.H4(
                    "Yes" if summary['memory_leak_detected'] else "No",
                    className="mb-0 " + ("text-danger" if summary['memory_leak_detected'] else "text-success"),
                ),
            ], className="text-center p-3 border rounded"),
        ], width=2),
    ])


def create_bottlenecks_display(analyzer: ProfileAnalyzer) -> html.Div:
    """Create bottlenecks display content.

    Args:
        analyzer: ProfileAnalyzer instance with loaded data

    Returns:
        Dash HTML Div with bottleneck information
    """
    bottlenecks = analyzer.identify_bottlenecks()

    if not bottlenecks:
        return html.Div([
            html.Div([
                html.I(className="fas fa-check-circle fa-3x text-success mb-3"),
                html.H5("No Bottlenecks Detected", className="text-success"),
                html.P(
                    "Your training profile shows no significant performance bottlenecks.",
                    className="text-muted",
                ),
            ], className="text-center py-4"),
        ])

    # Create bottleneck cards
    cards = []
    for bottleneck in bottlenecks:
        # Color based on severity
        color_map = {
            "high": "danger",
            "medium": "warning",
            "low": "info",
        }
        color = color_map.get(bottleneck.severity, "secondary")

        cards.append(
            dbc.Card([
                dbc.CardHeader([
                    dbc.Badge(
                        bottleneck.severity.upper(),
                        color=color,
                        className="me-2",
                    ),
                    html.Strong(bottleneck.type.value.replace("_", " ").title()),
                ], className=f"border-{color}"),
                dbc.CardBody([
                    html.P(bottleneck.description, className="mb-2"),
                    html.Hr(className="my-2"),
                    html.Small([
                        html.I(className="fas fa-lightbulb me-2 text-warning"),
                        html.Strong("Recommendation: "),
                        bottleneck.recommendation,
                    ], className="text-muted"),
                    html.Div([
                        html.Small(
                            f"Impact: {bottleneck.impact_ms:.1f} ms",
                            className="text-muted",
                        ),
                    ], className="mt-2") if bottleneck.impact_ms > 0 else None,
                ]),
            ], className=f"mb-3 border-{color}")
        )

    return html.Div([
        html.H5(f"{len(bottlenecks)} Bottleneck(s) Detected", className="mb-3"),
        html.Div(cards),
    ])


def get_profile_files(runs_dir: Any) -> list[dict[str, str]]:
    """Get list of profile files from runs directory.

    Args:
        runs_dir: Path to runs directory

    Returns:
        List of options for dropdown [{label, value}]
    """
    from pathlib import Path

    runs_path = Path(runs_dir)
    profile_files = []

    # Search for profile files in runs directory
    for pattern in ["*.pt", "**/profile.pt", "**/profile.json"]:
        for profile_path in runs_path.glob(pattern):
            # Skip non-profile files
            if "checkpoint" in profile_path.name:
                continue

            # Create label from relative path
            try:
                rel_path = profile_path.relative_to(runs_path)
                label = str(rel_path)
            except ValueError:
                label = profile_path.name

            profile_files.append({
                "label": label,
                "value": str(profile_path),
            })

    return sorted(profile_files, key=lambda x: x["label"])
