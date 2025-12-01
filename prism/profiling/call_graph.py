# prism/profiling/call_graph.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CallNode:
    """Node in the call graph hierarchy."""

    name: str
    module: str | None = None
    self_time_ms: float = 0.0
    total_time_ms: float = 0.0
    call_count: int = 0
    children: list[CallNode] = field(default_factory=list)
    _root_time: float = field(default=0.0, repr=False)

    @property
    def percentage(self) -> float:
        """Percentage of total time."""
        if self._root_time == 0:
            return 0.0
        return self.total_time_ms / self._root_time * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "module": self.module,
            "self_time_ms": self.self_time_ms,
            "total_time_ms": self.total_time_ms,
            "call_count": self.call_count,
            "percentage": self.percentage,
            "children": [c.to_dict() for c in self.children],
        }


class CallGraphBuilder:
    """Build hierarchical call graph from profile data."""

    def build_from_regions(
        self,
        region_times: dict[str, list[float]],
    ) -> CallNode:
        """Build call graph from region timing data."""
        # Create root node
        total_time = sum(sum(t) for t in region_times.values())
        root = CallNode(
            name="root",
            total_time_ms=total_time,
            _root_time=total_time,
        )

        # Group by module path (e.g., "forward.conv1" -> forward -> conv1)
        for name, times in region_times.items():
            parts = name.split(".")
            current = root

            for i, part in enumerate(parts):
                # Find or create child
                child = next((c for c in current.children if c.name == part), None)
                if child is None:
                    child = CallNode(
                        name=part,
                        module=".".join(parts[: i + 1]),
                        _root_time=total_time,
                    )
                    current.children.append(child)

                child.call_count += len(times)
                if i == len(parts) - 1:
                    child.self_time_ms += sum(times)
                    child.total_time_ms += sum(times)
                else:
                    child.total_time_ms += sum(times)

                current = child

        return root

    def to_flame_graph_data(self, root: CallNode, depth: int = 0) -> list[dict]:
        """Convert to format suitable for flame graph rendering."""
        data = []

        def traverse(node: CallNode, d: int, x_offset: float) -> float:
            width = node.total_time_ms
            data.append({
                "name": node.name,
                "depth": d,
                "x": x_offset,
                "width": width,
                "self_time": node.self_time_ms,
                "total_time": node.total_time_ms,
                "percentage": node.percentage,
            })

            child_x = x_offset
            for child in sorted(node.children, key=lambda c: -c.total_time_ms):
                child_x = traverse(child, d + 1, child_x)

            return x_offset + width

        traverse(root, 0, 0)
        return data

    def to_sunburst_data(self, root: CallNode) -> dict:
        """Convert to Plotly sunburst format."""
        ids = []
        labels = []
        parents = []
        values = []

        def traverse(node: CallNode, parent_id: str = "") -> None:
            node_id = f"{parent_id}/{node.name}" if parent_id else node.name
            ids.append(node_id)
            labels.append(node.name)
            parents.append(parent_id)
            values.append(node.total_time_ms)

            for child in node.children:
                traverse(child, node_id)

        traverse(root)

        return {
            "ids": ids,
            "labels": labels,
            "parents": parents,
            "values": values,
        }
