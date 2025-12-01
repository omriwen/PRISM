"""
Interactive profiling visualizations using Plotly.

This module provides interactive, web-ready visualizations for PRISM training
profiler data. All plots are built using Plotly for rich interactivity including
zoom, pan, hover tooltips, and export capabilities.

Classes:
    InteractiveProfilePlotter: Main plotting class for creating interactive figures

Usage:
    from prism.profiling import ProfileAnalyzer
    from prism.profiling.visualization.interactive import InteractiveProfilePlotter
    from prism.profiling.call_graph import CallGraphBuilder

    analyzer = ProfileAnalyzer("profile.json")
    plotter = InteractiveProfilePlotter(analyzer)

    # Create various visualizations
    fig_timeline = plotter.create_timeline_figure()
    fig_memory = plotter.create_memory_figure()
    fig_ops = plotter.create_operations_figure()

    # Call graph visualizations
    builder = CallGraphBuilder()
    root = builder.build_from_regions(analyzer.data.region_times)
    fig_sunburst = plotter.create_sunburst_figure(root)
    fig_flame = plotter.create_flame_graph(root)

    # Display or save
    fig_timeline.show()
    fig_memory.write_html("memory_profile.html")

Notes:
    - All figures are interactive and support zoom/pan/export
    - Hover tooltips provide detailed information
    - Figures can be saved as HTML or static images (requires kaleido)
    - Memory plots handle missing data gracefully
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from prism.profiling.analyzer import ProfileAnalyzer
    from prism.profiling.call_graph import CallNode

import plotly.graph_objects as go


class InteractiveProfilePlotter:
    """
    Create interactive Plotly visualizations for profiling data.

    This class provides methods to generate various interactive plots for analyzing
    training performance, memory usage, and operation bottlenecks.

    Attributes:
        analyzer: ProfileAnalyzer instance containing profiling data
    """

    def __init__(self, analyzer: ProfileAnalyzer):
        """
        Initialize the plotter with a ProfileAnalyzer.

        Args:
            analyzer: ProfileAnalyzer instance with profiling data to visualize
        """
        self.analyzer = analyzer

    def create_timeline_figure(self) -> go.Figure:
        """
        Create timeline visualization of profiled operations.

        Generates a horizontal bar chart showing the sequence and duration of
        profiled operations over time. Operations are stacked to show their
        temporal overlap and duration.

        Returns:
            Plotly Figure with operation timeline

        Notes:
            - X-axis represents time in milliseconds
            - Y-axis shows operation names
            - Hover shows operation name, duration, and count
            - Operations are ordered by first occurrence
        """
        top_ops = self.analyzer.get_top_operations(n=20)

        if not top_ops:
            fig = go.Figure()
            fig.add_annotation(
                text="No operation data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 16},
            )
            return fig

        # Build cumulative timeline
        names = [op["name"] for op in top_ops]
        durations = [op["total_ms"] for op in top_ops]
        counts = [op["count"] for op in top_ops]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=names,
                x=durations,
                orientation="h",
                marker={"color": "steelblue"},
                hovertemplate="<b>%{y}</b><br>"
                + "Total time: %{x:.2f} ms<br>"
                + "Calls: %{customdata}<br>"
                + "<extra></extra>",
                customdata=counts,
            )
        )

        fig.update_layout(
            title="Operation Timeline (Top 20)",
            xaxis_title="Total Time (ms)",
            yaxis_title="Operation",
            yaxis={"autorange": "reversed"},  # Top operation at top
            height=max(400, len(names) * 25),
            hovermode="closest",
            template="plotly_white",
        )

        return fig

    def create_memory_figure(self) -> go.Figure:
        """
        Create interactive memory usage timeline.

        Plots GPU memory allocation and reservation over training samples.
        Shows both allocated memory (actual usage) and reserved memory (memory pool).

        Returns:
            Plotly Figure with memory timeline

        Notes:
            - Returns figure with annotation if no memory data available
            - Allocated memory shown as solid blue line
            - Reserved memory shown as dashed orange line
            - X-axis: sample index, Y-axis: memory in MB
            - Hover mode is "x unified" for easy comparison
        """
        memory_profile = self.analyzer.data.memory_profile

        if not memory_profile or not memory_profile.snapshots:
            fig = go.Figure()
            fig.add_annotation(
                text="No memory data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 16},
            )
            fig.update_layout(
                title="GPU Memory Profile",
                xaxis_title="Sample Index",
                yaxis_title="Memory (MB)",
                template="plotly_white",
            )
            return fig

        snapshots = memory_profile.snapshots
        sample_indices = [s.sample_idx for s in snapshots]
        allocated = [s.gpu_allocated_mb for s in snapshots]
        reserved = [s.gpu_reserved_mb for s in snapshots]

        fig = go.Figure()

        # Allocated memory trace
        fig.add_trace(
            go.Scatter(
                x=sample_indices,
                y=allocated,
                mode="lines",
                name="Allocated",
                line={"color": "steelblue", "width": 2},
                hovertemplate="Sample: %{x}<br>"
                + "Allocated: %{y:.2f} MB<br>"
                + "<extra></extra>",
            )
        )

        # Reserved memory trace
        fig.add_trace(
            go.Scatter(
                x=sample_indices,
                y=reserved,
                mode="lines",
                name="Reserved",
                line={"color": "orange", "width": 2, "dash": "dash"},
                hovertemplate="Sample: %{x}<br>"
                + "Reserved: %{y:.2f} MB<br>"
                + "<extra></extra>",
            )
        )

        # Add peak memory annotation if available
        if memory_profile.peak_memory_mb > 0:
            fig.add_hline(
                y=memory_profile.peak_memory_mb,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Peak: {memory_profile.peak_memory_mb:.1f} MB",
                annotation_position="right",
            )

        # Add leak warning if detected
        title = "GPU Memory Profile"
        if memory_profile.leak_detected:
            title += f" (⚠️ Leak: {memory_profile.leak_rate_mb_per_sample:.2f} MB/sample)"

        fig.update_layout(
            title=title,
            xaxis_title="Sample Index",
            yaxis_title="Memory (MB)",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            legend={
                "yanchor": "top",
                "y": 0.99,
                "xanchor": "left",
                "x": 0.01,
            },
        )

        return fig

    def create_operations_figure(self) -> go.Figure:
        """
        Create horizontal bar chart of top operations.

        Displays the top operations ranked by total execution time, with
        color-coding to indicate relative importance.

        Returns:
            Plotly Figure with top operations

        Notes:
            - Shows top 15 operations by default
            - Bars colored by percentage of total time
            - Hover shows operation name, time, count, and percentage
            - Y-axis ordered with slowest operation at top
        """
        top_ops = self.analyzer.get_top_operations(n=15)

        if not top_ops:
            fig = go.Figure()
            fig.add_annotation(
                text="No operation data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 16},
            )
            return fig

        total_time = sum(op["total_ms"] for op in top_ops)
        names = [op["name"] for op in top_ops]
        times = [op["total_ms"] for op in top_ops]
        counts = [op["count"] for op in top_ops]
        percentages = [
            (op["total_ms"] / total_time * 100) if total_time > 0 else 0
            for op in top_ops
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=names,
                x=times,
                orientation="h",
                marker={
                    "color": percentages,
                    "colorscale": "Reds",
                    "colorbar": {"title": "% of Total"},
                },
                hovertemplate="<b>%{y}</b><br>"
                + "Time: %{x:.2f} ms<br>"
                + "Calls: %{customdata[0]}<br>"
                + "Percentage: %{customdata[1]:.1f}%<br>"
                + "<extra></extra>",
                customdata=[[c, p] for c, p in zip(counts, percentages)],
            )
        )

        fig.update_layout(
            title="Top Operations by Time",
            xaxis_title="Total Time (ms)",
            yaxis_title="Operation",
            yaxis={"autorange": "reversed"},
            height=max(400, len(names) * 30),
            hovermode="closest",
            template="plotly_white",
        )

        return fig

    def create_sunburst_figure(self, root: CallNode) -> go.Figure:
        """
        Create hierarchical sunburst visualization of call graph.

        Displays the call hierarchy as a sunburst chart where each ring represents
        a level in the call stack and each segment's size corresponds to execution time.

        Args:
            root: Root CallNode from CallGraphBuilder.build_from_regions()

        Returns:
            Plotly Figure with sunburst chart

        Notes:
            - Inner rings are higher-level calls, outer rings are deeper calls
            - Segment size represents total time including children
            - Click segments to zoom in on subtrees
            - Hover shows operation name, time, and percentage
            - Uses branchvalues="total" for proper hierarchical sizing
        """
        from prism.profiling.call_graph import CallGraphBuilder

        builder = CallGraphBuilder()
        sunburst_data = builder.to_sunburst_data(root)

        fig = go.Figure()

        fig.add_trace(
            go.Sunburst(
                ids=sunburst_data["ids"],
                labels=sunburst_data["labels"],
                parents=sunburst_data["parents"],
                values=sunburst_data["values"],
                branchvalues="total",
                hovertemplate="<b>%{label}</b><br>"
                + "Time: %{value:.2f} ms<br>"
                + "Percentage: %{percentParent:.1f}%<br>"
                + "<extra></extra>",
                marker={"colorscale": "Blues", "cmid": root.total_time_ms / 2},
            )
        )

        fig.update_layout(
            title="Call Hierarchy (Sunburst)",
            height=600,
            template="plotly_white",
        )

        return fig

    def create_flame_graph(self, root: CallNode) -> go.Figure:
        """
        Create interactive flame graph visualization.

        Flame graphs show call stack depth on Y-axis and time on X-axis,
        with each bar representing a function call and its children stacked above.

        Args:
            root: Root CallNode from CallGraphBuilder.build_from_regions()

        Returns:
            Plotly Figure with flame graph

        Notes:
            - X-axis represents cumulative time
            - Y-axis represents call depth (stack depth)
            - Wider bars indicate longer execution time
            - Hover shows function name, self/total time, and percentage
            - Uses stacked horizontal bars for flame graph effect
        """
        from prism.profiling.call_graph import CallGraphBuilder

        builder = CallGraphBuilder()
        flame_data = builder.to_flame_graph_data(root)

        if not flame_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No call graph data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 16},
            )
            return fig

        # Group by depth for stacked bars
        max_depth = max(d["depth"] for d in flame_data)

        fig = go.Figure()

        # Create traces for each depth level
        for depth in range(max_depth + 1):
            depth_items = [d for d in flame_data if d["depth"] == depth]

            if not depth_items:
                continue

            # Create bars for this depth
            for item in depth_items:
                fig.add_trace(
                    go.Bar(
                        x=[item["width"]],
                        y=[depth],
                        base=[item["x"]],
                        orientation="h",
                        name=item["name"],
                        marker={"color": f"hsl({(depth * 60) % 360}, 70%, 60%)"},
                        hovertemplate=f"<b>{item['name']}</b><br>"
                        + f"Total time: {item['total_time']:.2f} ms<br>"
                        + f"Self time: {item['self_time']:.2f} ms<br>"
                        + f"Percentage: {item['percentage']:.1f}%<br>"
                        + "<extra></extra>",
                        showlegend=False,
                    )
                )

        fig.update_layout(
            title="Flame Graph (Call Stack)",
            xaxis_title="Time (ms)",
            yaxis_title="Call Depth",
            barmode="overlay",
            height=max(400, max_depth * 50 + 100),
            hovermode="closest",
            template="plotly_white",
            yaxis={"dtick": 1},
        )

        return fig
