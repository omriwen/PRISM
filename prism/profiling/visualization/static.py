"""
Module: prism.profiling.visualization.static
Purpose: Static profiling visualizations using Matplotlib
Dependencies: matplotlib, numpy, prism.visualization.base, prism.profiling.analyzer

Description:
    Provides static visualizations for profiling data including timing breakdowns,
    memory timelines, top operations, and bottleneck summaries. Extends BasePlotter
    for consistent styling and memory management.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from prism.visualization.base import BasePlotter


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from prism.profiling.analyzer import ProfileAnalyzer
    from prism.visualization.config import VisualizationConfig


class ProfilePlotter(BasePlotter):
    """Static profiling visualizations with Matplotlib.

    Creates static plots for training profiler data including timing analysis,
    memory tracking, and bottleneck identification.

    Parameters
    ----------
    analyzer : ProfileAnalyzer
        Analyzer with profiling data to visualize
    config : VisualizationConfig, optional
        Visualization configuration. If None, uses defaults.

    Examples
    --------
    >>> from prism.profiling.analyzer import ProfileAnalyzer
    >>> analyzer = ProfileAnalyzer("profile.json")
    >>> with ProfilePlotter(analyzer) as plotter:
    ...     fig = plotter.plot_timing_breakdown()
    ...     plotter.save("timing.png")
    ...     report = plotter.create_report("report.png")

    Notes
    -----
    - All plots handle empty data gracefully
    - Memory plots only shown if CUDA data available
    - Bottleneck table uses color coding for severity
    """

    def __init__(
        self,
        analyzer: ProfileAnalyzer,
        config: VisualizationConfig | None = None,
    ) -> None:
        """Initialize plotter with analyzer.

        Parameters
        ----------
        analyzer : ProfileAnalyzer
            Analyzer with profiling data
        config : VisualizationConfig, optional
            Visualization configuration
        """
        super().__init__(config)
        self.analyzer = analyzer

    def _create_figure(self) -> tuple[Figure, NDArray[np.object_]]:
        """Create default figure (single axis).

        Returns
        -------
        tuple[Figure, NDArray]
            Figure and axes array
        """
        fig, ax = plt.subplots(figsize=self.config.figure.figsize)
        return fig, np.asarray([ax])

    def plot(self, **kwargs: object) -> Figure:
        """Create default plot (timing breakdown).

        This is the abstract method implementation required by BasePlotter.
        By default, creates a timing breakdown plot.

        Parameters
        ----------
        **kwargs : dict
            Optional parameters (currently unused)

        Returns
        -------
        Figure
            Matplotlib figure handle
        """
        return self.plot_timing_breakdown()

    def plot_timing_breakdown(self) -> Figure:
        """Create horizontal bar chart of top operations by time.

        Shows the top operations by total time spent, sorted from highest to
        lowest. Useful for identifying performance hotspots.

        Returns
        -------
        Figure
            Matplotlib figure with timing breakdown

        Examples
        --------
        >>> fig = plotter.plot_timing_breakdown()
        >>> plt.show()
        """
        fig, ax = plt.subplots(figsize=self.config.figure.figsize)
        self._fig = fig
        self._axes = np.asarray([ax])

        top_ops = self.analyzer.get_top_operations(n=10)

        if not top_ops:
            ax.text(
                0.5,
                0.5,
                "No timing data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Extract data
        names = [op["name"] for op in top_ops]
        times_ms = [op["total_ms"] for op in top_ops]

        # Create horizontal bar chart (reversed for top-to-bottom ordering)
        y_positions = np.arange(len(names))
        bars = ax.barh(y_positions, times_ms, color="steelblue")

        # Add value labels
        for i, (bar, time_ms) in enumerate(zip(bars, times_ms)):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {time_ms:.1f} ms",
                va="center",
                fontsize=self.config.style.font_size - 1,
            )

        # Styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # Top operation at top
        ax.set_xlabel("Total Time (ms)")
        ax.set_title("Top Operations by Time")
        ax.grid(axis="x", alpha=self.config.style.grid_alpha)

        if self.config.figure.tight_layout:
            plt.tight_layout()

        return fig

    def plot_memory_timeline(self) -> Figure:
        """Create GPU memory timeline over training.

        Shows allocated GPU memory over sample indices with optional trend line
        if a memory leak is detected.

        Returns
        -------
        Figure
            Matplotlib figure with memory timeline

        Examples
        --------
        >>> fig = plotter.plot_memory_timeline()
        >>> plt.show()
        """
        fig, ax = plt.subplots(figsize=self.config.figure.figsize)
        self._fig = fig
        self._axes = np.asarray([ax])

        memory_profile = self.analyzer.data.memory_profile

        if not memory_profile or not memory_profile.snapshots:
            ax.text(
                0.5,
                0.5,
                "No memory data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Extract data
        sample_indices = [s.sample_idx for s in memory_profile.snapshots]
        allocated_mb = [s.gpu_allocated_mb for s in memory_profile.snapshots]

        # Plot memory usage
        ax.plot(
            sample_indices,
            allocated_mb,
            "o-",
            color="darkblue",
            linewidth=self.config.style.line_width,
            markersize=self.config.style.marker_size * 0.5,
            label="GPU Allocated",
        )

        # Add trend line if leak detected
        if memory_profile.leak_detected:
            x = np.array(sample_indices)
            y = np.array(allocated_mb)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)

            ax.plot(
                sample_indices,
                p(x),
                "r--",
                linewidth=self.config.style.line_width * 1.5,
                label=f"Leak: {memory_profile.leak_rate_mb_per_sample:.2f} MB/sample",
            )

        # Styling
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("GPU Memory (MB)")
        ax.set_title("GPU Memory Usage Over Training")
        ax.grid(alpha=self.config.style.grid_alpha)
        ax.legend()

        # Mark peak
        ax.axhline(
            y=memory_profile.peak_memory_mb,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label=f"Peak: {memory_profile.peak_memory_mb:.1f} MB",
        )

        if self.config.figure.tight_layout:
            plt.tight_layout()

        return fig

    def plot_top_operations(self, n: int = 10) -> Figure:
        """Create bar chart of top N operations by time.

        Alias for plot_timing_breakdown with configurable N.

        Parameters
        ----------
        n : int, default=10
            Number of top operations to show

        Returns
        -------
        Figure
            Matplotlib figure with top operations

        Examples
        --------
        >>> fig = plotter.plot_top_operations(n=5)
        >>> plt.show()
        """
        fig, ax = plt.subplots(figsize=self.config.figure.figsize)
        self._fig = fig
        self._axes = np.asarray([ax])

        top_ops = self.analyzer.get_top_operations(n=n)

        if not top_ops:
            ax.text(
                0.5,
                0.5,
                "No operation data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Extract data
        names = [op["name"] for op in top_ops]
        times_ms = [op["total_ms"] for op in top_ops]
        counts = [op["count"] for op in top_ops]

        # Create horizontal bar chart
        y_positions = np.arange(len(names))
        bars = ax.barh(y_positions, times_ms, color="steelblue")

        # Add value labels with call counts
        for i, (bar, time_ms, count) in enumerate(zip(bars, times_ms, counts)):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {time_ms:.1f} ms ({count} calls)",
                va="center",
                fontsize=self.config.style.font_size - 1,
            )

        # Styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Total Time (ms)")
        ax.set_title(f"Top {n} Operations")
        ax.grid(axis="x", alpha=self.config.style.grid_alpha)

        if self.config.figure.tight_layout:
            plt.tight_layout()

        return fig

    def plot_bottleneck_summary(self) -> Figure:
        """Create table of detected bottlenecks with color-coded severity.

        Displays bottlenecks as a formatted table with severity-based color
        coding (red=high, orange=medium, yellow=low).

        Returns
        -------
        Figure
            Matplotlib figure with bottleneck table

        Examples
        --------
        >>> fig = plotter.plot_bottleneck_summary()
        >>> plt.show()
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        self._fig = fig
        self._axes = np.asarray([ax])

        ax.axis("off")

        bottlenecks = self.analyzer.identify_bottlenecks()

        if not bottlenecks:
            ax.text(
                0.5,
                0.5,
                "No bottlenecks detected!",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=self.config.style.title_size,
                color="green",
                weight="bold",
            )
            return fig

        # Prepare table data
        headers = ["Severity", "Type", "Description", "Recommendation"]
        rows = []

        for b in bottlenecks:
            rows.append(
                [
                    b.severity.upper(),
                    b.type.value.replace("_", " ").title(),
                    b.description,
                    b.recommendation,
                ]
            )

        # Create table
        table = ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="left",
            colWidths=[0.12, 0.18, 0.35, 0.35],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(self.config.style.font_size - 1)
        table.scale(1, 2.5)

        # Color code by severity
        severity_colors = {
            "high": "#ffcccc",  # Light red
            "medium": "#ffd9b3",  # Light orange
            "low": "#ffffcc",  # Light yellow
        }

        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor("#4472C4")
            cell.set_text_props(weight="bold", color="white")

        # Color rows by severity
        for i, b in enumerate(bottlenecks, start=1):
            severity_cell = table[(i, 0)]
            severity_cell.set_facecolor(severity_colors.get(b.severity, "#ffffff"))
            severity_cell.set_text_props(weight="bold")

        ax.set_title(
            f"Performance Bottlenecks Detected ({len(bottlenecks)} issues)",
            pad=20,
            fontsize=self.config.style.title_size,
            weight="bold",
        )

        if self.config.figure.tight_layout:
            plt.tight_layout()

        return fig

    def create_report(self, output_path: Path | str) -> None:
        """Create comprehensive 2x2 multi-panel report and save to file.

        Generates a publication-ready report with four panels:
        - Top-left: Timing breakdown
        - Top-right: Memory timeline
        - Bottom-left: Top operations
        - Bottom-right: Bottleneck summary

        Parameters
        ----------
        output_path : Path or str
            Output file path for the report

        Examples
        --------
        >>> plotter.create_report("profile_report.png")
        """
        fig = plt.figure(figsize=(16, 12))
        self._fig = fig

        # Create 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Top-left: Timing breakdown
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_timing_on_axis(ax1)

        # Top-right: Memory timeline
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_memory_on_axis(ax2)

        # Bottom-left: Top operations
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_top_ops_on_axis(ax3, n=8)

        # Bottom-right: Bottleneck summary
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_bottlenecks_on_axis(ax4)

        # Add overall title
        summary = self.analyzer.get_summary()
        fig.suptitle(
            f"Training Profile Report - {summary['total_samples']} samples, "
            f"{summary['total_epochs']} epochs",
            fontsize=self.config.style.title_size + 2,
            weight="bold",
        )

        # Save
        output_path = Path(output_path)
        fig.savefig(
            output_path,
            dpi=self.config.figure.dpi,
            bbox_inches="tight",
        )

        print(f"Report saved to: {output_path}")

    def _plot_timing_on_axis(self, ax: plt.Axes) -> None:
        """Plot timing breakdown on given axis."""
        top_ops = self.analyzer.get_top_operations(n=8)

        if not top_ops:
            ax.text(0.5, 0.5, "No timing data", ha="center", va="center", transform=ax.transAxes)
            return

        names = [op["name"][:30] for op in top_ops]  # Truncate long names
        times_ms = [op["total_ms"] for op in top_ops]

        y_positions = np.arange(len(names))
        bars = ax.barh(y_positions, times_ms, color="steelblue")

        for bar, time_ms in zip(bars, times_ms):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {time_ms:.1f}",
                va="center",
                fontsize=9,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_title("Timing Breakdown", fontsize=12, weight="bold")
        ax.grid(axis="x", alpha=0.3)

    def _plot_memory_on_axis(self, ax: plt.Axes) -> None:
        """Plot memory timeline on given axis."""
        memory_profile = self.analyzer.data.memory_profile

        if not memory_profile or not memory_profile.snapshots:
            ax.text(0.5, 0.5, "No memory data", ha="center", va="center", transform=ax.transAxes)
            return

        sample_indices = [s.sample_idx for s in memory_profile.snapshots]
        allocated_mb = [s.gpu_allocated_mb for s in memory_profile.snapshots]

        ax.plot(sample_indices, allocated_mb, "o-", color="darkblue", linewidth=2, markersize=3)

        if memory_profile.leak_detected:
            x = np.array(sample_indices)
            y = np.array(allocated_mb)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(
                sample_indices,
                p(x),
                "r--",
                linewidth=2,
                label=f"Leak: {memory_profile.leak_rate_mb_per_sample:.2f} MB/sample",
            )
            ax.legend(fontsize=9)

        ax.set_xlabel("Sample Index", fontsize=10)
        ax.set_ylabel("GPU Memory (MB)", fontsize=10)
        ax.set_title("Memory Timeline", fontsize=12, weight="bold")
        ax.grid(alpha=0.3)

    def _plot_top_ops_on_axis(self, ax: plt.Axes, n: int = 8) -> None:
        """Plot top operations on given axis."""
        top_ops = self.analyzer.get_top_operations(n=n)

        if not top_ops:
            ax.text(0.5, 0.5, "No operation data", ha="center", va="center", transform=ax.transAxes)
            return

        names = [op["name"][:25] for op in top_ops]
        times_ms = [op["total_ms"] for op in top_ops]
        counts = [op["count"] for op in top_ops]

        y_positions = np.arange(len(names))
        bars = ax.barh(y_positions, times_ms, color="steelblue")

        for bar, time_ms, count in zip(bars, times_ms, counts):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {time_ms:.1f} ({count})",
                va="center",
                fontsize=9,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_title(f"Top {n} Operations", fontsize=12, weight="bold")
        ax.grid(axis="x", alpha=0.3)

    def _plot_bottlenecks_on_axis(self, ax: plt.Axes) -> None:
        """Plot bottleneck summary on given axis."""
        ax.axis("off")
        bottlenecks = self.analyzer.identify_bottlenecks()

        if not bottlenecks:
            ax.text(
                0.5,
                0.5,
                "No bottlenecks detected!",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="green",
                weight="bold",
            )
            return

        # Prepare compact table
        headers = ["Severity", "Issue", "Fix"]
        rows = []

        for b in bottlenecks:
            # Truncate long descriptions
            desc = b.description[:40] + "..." if len(b.description) > 40 else b.description
            fix = b.recommendation[:45] + "..." if len(b.recommendation) > 45 else b.recommendation
            rows.append([b.severity.upper(), desc, fix])

        table = ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="left",
            colWidths=[0.15, 0.40, 0.45],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.2)

        # Color code
        severity_colors = {"high": "#ffcccc", "medium": "#ffd9b3", "low": "#ffffcc"}

        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor("#4472C4")
            cell.set_text_props(weight="bold", color="white")

        for i, b in enumerate(bottlenecks, start=1):
            severity_cell = table[(i, 0)]
            severity_cell.set_facecolor(severity_colors.get(b.severity, "#ffffff"))
            severity_cell.set_text_props(weight="bold")

        ax.set_title("Bottlenecks", fontsize=12, weight="bold", pad=20)
