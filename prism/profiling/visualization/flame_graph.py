# prism/profiling/visualization/flame_graph.py
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from prism.profiling.call_graph import CallNode


def plot_flame_graph(root: CallNode, max_depth: int = 10) -> Figure:
    """Create static flame graph visualization.

    Parameters
    ----------
    root : CallNode
        Root node of the call graph hierarchy.
    max_depth : int, optional
        Maximum depth to visualize (default: 10).

    Returns
    -------
    Figure
        Matplotlib figure containing the flame graph.

    Notes
    -----
    - Rectangles are positioned with x based on time offset and y based on call depth
    - Width is proportional to total_time_ms / root.total_time_ms
    - Color is determined by depth using a plasma colormap
    - Text labels are shown for nodes wider than 5% of total width
    - Y-axis is inverted so root is at the top
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Track maximum depth encountered
    max_depth_encountered = 0

    # Colormap for depth-based coloring
    colormap = plt.cm.plasma

    def traverse(node: CallNode, depth: int, x_start: float, total_width: float) -> float:
        """Traverse call tree and draw rectangles.

        Parameters
        ----------
        node : CallNode
            Current node to render.
        depth : int
            Current depth in the call tree.
        x_start : float
            Starting x position (0 to 1 relative scale).
        total_width : float
            Total width available (0 to 1 relative scale).

        Returns
        -------
        float
            Updated x position after this node and its children.
        """
        nonlocal max_depth_encountered

        # Skip if max depth exceeded
        if depth >= max_depth:
            return x_start

        # Update max depth tracker
        max_depth_encountered = max(max_depth_encountered, depth)

        # Calculate width proportional to total time
        if root.total_time_ms > 0:
            width = (node.total_time_ms / root.total_time_ms) * total_width
        else:
            width = 0.0

        # Skip if width is negligible
        if width < 0.001:
            return x_start

        # Determine color based on depth
        color = colormap(depth / max(max_depth, 1))

        # Create rectangle
        rect = mpatches.Rectangle(
            (x_start, depth),
            width,
            1.0,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect)

        # Add text label if wide enough (>5% width)
        if width > 0.05 * total_width:
            label = f"{node.name}\n{node.percentage:.1f}%"
            ax.text(
                x_start + width / 2,
                depth + 0.5,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                clip_on=True,
            )

        # Recursively traverse children, sorted by total time (descending)
        child_x = x_start
        for child in sorted(node.children, key=lambda c: -c.total_time_ms):
            child_x = traverse(child, depth + 1, child_x, total_width)

        return x_start + width

    # Start traversal from root
    traverse(root, 0, 0.0, 1.0)

    # Configure axes
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, max_depth_encountered + 1.5)
    ax.set_xlabel("Relative Time", fontsize=10)
    ax.set_ylabel("Call Depth", fontsize=10)
    ax.set_title("Flame Graph - Call Hierarchy", fontsize=12, fontweight="bold")

    # Invert y-axis so root is at top
    ax.invert_yaxis()

    # Remove x-ticks (relative time scale)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

    # Set y-ticks to integer depths
    ax.set_yticks(range(max_depth_encountered + 1))

    # Tight layout
    plt.tight_layout()

    return fig
