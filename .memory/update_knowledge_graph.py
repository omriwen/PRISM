"""
Automated knowledge graph updater using AST parsing.

This tool scans Python source files and extracts:
- Class definitions (with type classification)
- Function definitions
- Import dependencies
- Inheritance relationships
- Protocol implementations

Usage:
    # Full scan of entire prism/ directory
    uv run python .memory/update_knowledge_graph.py --full

    # Incremental update (only files changed since last update)
    uv run python .memory/update_knowledge_graph.py --incremental

    # Update specific modules
    uv run python .memory/update_knowledge_graph.py --modules prism.core.instruments

    # Dry run (show changes without applying)
    uv run python .memory/update_knowledge_graph.py --dry-run

    # Show current state
    uv run python .memory/update_knowledge_graph.py --status

    # CI mode (exit non-zero if out of sync)
    uv run python .memory/update_knowledge_graph.py --ci-mode
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set


# Paths
REPO_ROOT = Path(__file__).parent.parent
MEMORY_DIR = REPO_ROOT / ".memory"
LAST_UPDATE_FILE = MEMORY_DIR / ".last_update"


@dataclass
class Entity:
    """Represents a component in the knowledge graph."""

    name: str
    entity_type: str  # Module, Class, Function, Protocol, Config, Pipeline, Propagator
    file_path: str
    line_number: int | None = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """Represents a relationship between entities."""

    from_entity: str
    to_entity: str
    relation_type: str  # uses, inherits_from, implements, configures, etc.


class PRISMASTVisitor(ast.NodeVisitor):
    """AST visitor to extract PRISM-specific patterns."""

    def __init__(self, file_path: Path, repo_root: Path):
        """Initialize visitor.

        Args:
            file_path: Path to file being analyzed
            repo_root: Root of repository
        """
        self.file_path = file_path.relative_to(repo_root)
        self.repo_root = repo_root
        self.entities: List[Entity] = []
        self.relations: List[Relation] = []
        self.current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class definitions."""
        # Classify class type
        entity_type = self._classify_class(node)

        # Extract base classes
        base_classes = [self._get_name(base) for base in node.bases]

        # Get docstring
        description = ast.get_docstring(node) or ""
        if description:
            # Take first line only
            description = description.split("\n")[0].strip()

        # Create entity
        entity = Entity(
            name=node.name,
            entity_type=entity_type,
            file_path=str(self.file_path),
            line_number=node.lineno,
            description=description,
            metadata={
                "base_classes": base_classes,
                "is_abstract": self._is_abstract(node),
            },
        )

        self.entities.append(entity)

        # Create inheritance relations
        for base in base_classes:
            if base and base != "object":
                self.relations.append(
                    Relation(
                        from_entity=node.name,
                        to_entity=base,
                        relation_type="inherits_from",
                    )
                )

        # Track current class for nested analysis
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function definitions (module-level only)."""
        # Only extract module-level functions
        if self.current_class is None:
            description = ast.get_docstring(node) or ""
            if description:
                description = description.split("\n")[0].strip()

            entity = Entity(
                name=node.name,
                entity_type="Function",
                file_path=str(self.file_path),
                line_number=node.lineno,
                description=description,
                metadata={"is_public": not node.name.startswith("_")},
            )
            self.entities.append(entity)

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Track import relations."""
        for alias in node.names:
            # Only track PRISM internal imports
            if alias.name.startswith("prism."):
                if self.current_class:
                    self.relations.append(
                        Relation(
                            from_entity=self.current_class,
                            to_entity=alias.name,
                            relation_type="depends_on",
                        )
                    )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from-import relations."""
        if node.module and node.module.startswith("prism."):
            if self.current_class:
                # Track which classes are imported
                for alias in node.names:
                    self.relations.append(
                        Relation(
                            from_entity=self.current_class,
                            to_entity=alias.name,
                            relation_type="uses",
                        )
                    )
        self.generic_visit(node)

    def _classify_class(self, node: ast.ClassDef) -> str:
        """Classify class into entity type.

        Args:
            node: AST ClassDef node

        Returns:
            Entity type string
        """
        name = node.name
        bases = [self._get_name(base) for base in node.bases]

        # Propagators
        if "Propagator" in name or "Propagator" in bases:
            return "Propagator"

        # Configs (dataclasses with 'Config' in name)
        if "Config" in name and self._has_decorator(node, "dataclass"):
            return "Config"

        # Protocols
        if self._is_abstract(node):
            return "Protocol"

        # Pipelines (classes with Runner, Trainer, or Orchestrator in name)
        if any(pattern in name for pattern in ["Runner", "Trainer", "Orchestrator"]):
            return "Pipeline"

        return "Class"

    def _is_abstract(self, node: ast.ClassDef) -> bool:
        """Check if class is abstract/protocol.

        Args:
            node: AST ClassDef node

        Returns:
            True if abstract
        """
        bases = [self._get_name(base) for base in node.bases]
        if "Protocol" in bases or "ABC" in bases:
            return True

        # Check for abstract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if self._has_decorator(item, "abstractmethod"):
                    return True

        return False

    def _has_decorator(self, node: ast.AST, decorator_name: str) -> bool:
        """Check if node has specific decorator.

        Args:
            node: AST node
            decorator_name: Name of decorator to check

        Returns:
            True if decorator present
        """
        if not hasattr(node, "decorator_list"):
            return False

        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == decorator_name:
                return True
            elif isinstance(decorator, ast.Attribute) and decorator.attr == decorator_name:
                return True

        return False

    def _get_name(self, node: ast.AST) -> str:
        """Extract name from AST node.

        Args:
            node: AST node

        Returns:
            Name string
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""


class KnowledgeGraphUpdater:
    """Extract PRISM components from Python source files."""

    def __init__(self, repo_root: Path):
        """Initialize updater.

        Args:
            repo_root: Root of repository
        """
        self.repo_root = repo_root
        self.entities: List[Entity] = []
        self.relations: List[Relation] = []

    def scan_file(self, file_path: Path) -> tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Tuple of (entities, relations)
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError as e:
            print(f"[!] Syntax error in {file_path}: {e}", file=sys.stderr)
            return [], []
        except UnicodeDecodeError as e:
            print(f"[!] Encoding error in {file_path}: {e}", file=sys.stderr)
            return [], []

        visitor = PRISMASTVisitor(file_path, self.repo_root)
        visitor.visit(tree)

        return visitor.entities, visitor.relations

    def scan_module(self, module_path: Path) -> None:
        """Scan entire module (package) recursively.

        Args:
            module_path: Path to module directory
        """
        # Find all Python files
        py_files = list(module_path.rglob("*.py"))

        for py_file in py_files:
            # Skip __pycache__
            if "__pycache__" in str(py_file):
                continue

            entities, relations = self.scan_file(py_file)
            self.entities.extend(entities)
            self.relations.extend(relations)

    def generate_summary(self) -> str:
        """Generate summary of discovered components.

        Returns:
            Summary string
        """
        # Count by type
        type_counts: dict[str, int] = {}
        for entity in self.entities:
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1

        # Format summary
        lines = ["[*] Discovered Components:", ""]
        for entity_type, count in sorted(type_counts.items()):
            lines.append(f"  {entity_type:15s}: {count:3d}")

        lines.append(f"\n  Relations: {len(self.relations)}")

        return "\n".join(lines)

    def generate_mcp_commands(self) -> List[str]:
        """Generate MCP memory tool commands to update graph.

        Returns:
            List of MCP command strings (pseudo-code)
        """
        commands = []

        # Generate create_entities commands (batch by 20)
        batch_size = 20
        for i in range(0, len(self.entities), batch_size):
            batch = self.entities[i : i + batch_size]
            entities_json = []

            for entity in batch:
                observations = [
                    f"File: {entity.file_path}",
                    f"Description: {entity.description}",
                ]

                if entity.line_number:
                    observations.insert(1, f"Line: {entity.line_number}")

                # Add metadata
                for key, value in entity.metadata.items():
                    if value:
                        observations.append(f"{key}: {value}")

                entities_json.append(
                    {
                        "type": "entity",
                        "name": entity.name,
                        "entityType": entity.entity_type,
                        "observations": observations,
                    }
                )

            commands.append(f"create_entities({entities_json})")

        # Generate create_relations commands (batch by 50)
        rel_batch_size = 50
        for i in range(0, len(self.relations), rel_batch_size):
            rel_batch = self.relations[i : i + rel_batch_size]
            relations_json = [
                {
                    "type": "relation",
                    "from": rel.from_entity,
                    "to": rel.to_entity,
                    "relationType": rel.relation_type,
                }
                for rel in rel_batch
            ]

            commands.append(f"create_relations({relations_json})")

        return commands


def get_current_commit() -> str:
    """Get current git commit hash.

    Returns:
        Git commit hash
    """
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%H"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_last_update_info() -> Dict[str, Any]:
    """Read last update metadata.

    Returns:
        Dict with last update info, or empty dict if not found
    """
    if not LAST_UPDATE_FILE.exists():
        return {}
    try:
        with open(LAST_UPDATE_FILE) as f:
            data: Dict[str, Any] = json.load(f)
            return data
    except (json.JSONDecodeError, IOError):
        return {}


def save_update_info(
    entity_count: int, relation_count: int, update_type: str, notes: str = ""
) -> None:
    """Save update metadata.

    Args:
        entity_count: Number of entities in graph
        relation_count: Number of relations in graph
        update_type: Type of update (full, incremental)
        notes: Optional notes about the update
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    info = {
        "last_commit": get_current_commit(),
        "last_update": datetime.now(timezone.utc).isoformat(),
        "entity_count": entity_count,
        "relation_count": relation_count,
        "schema_version": "1.0",
        "update_type": update_type,
        "notes": notes,
    }
    with open(LAST_UPDATE_FILE, "w") as f:
        json.dump(info, f, indent=2)


def get_changed_files(since_commit: str) -> Set[Path]:
    """Get Python files changed since a commit.

    Args:
        since_commit: Git commit hash to compare from

    Returns:
        Set of changed file paths (relative to repo root)
    """
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "diff",
            "--name-only",
            since_commit,
            "HEAD",
            "--",
            "prism/",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    changed = set()
    for line in result.stdout.strip().split("\n"):
        if line and line.endswith(".py"):
            changed.add(REPO_ROOT / line)

    return changed


def print_status() -> None:
    """Print current knowledge graph status."""
    info = get_last_update_info()

    if not info:
        print("[!] No knowledge graph update history found")
        print(f"   Memory file: {MEMORY_DIR / 'memory.jsonl'}")
        print("   Run with --full to create initial graph")
        return

    print("[*] Knowledge Graph Status")
    print("-" * 40)
    print(f"  Last commit:    {info.get('last_commit', 'unknown')[:12]}")
    print(f"  Last update:    {info.get('last_update', 'unknown')}")
    print(f"  Update type:    {info.get('update_type', 'unknown')}")
    print(f"  Entities:       {info.get('entity_count', 'unknown')}")
    print(f"  Relations:      {info.get('relation_count', 'unknown')}")
    print(f"  Schema version: {info.get('schema_version', 'unknown')}")

    # Check if update is needed
    current = get_current_commit()
    last = info.get("last_commit", "")

    if current != last:
        changed = get_changed_files(last)
        py_changed = [f for f in changed if f.suffix == ".py"]
        print(f"\n[!] {len(py_changed)} Python file(s) changed since last update")
        if py_changed and len(py_changed) <= 10:
            for f in sorted(py_changed):
                print(f"     - {f.relative_to(REPO_ROOT)}")
        print("\n   Run with --incremental to update")
    else:
        print("\n[OK] Graph is up to date with current commit")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update PRISM knowledge graph from source code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--full", action="store_true", help="Scan entire prism/ directory")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only scan files changed since last update",
    )
    parser.add_argument("--modules", nargs="+", help="Specific modules to scan")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--status", action="store_true", help="Show current graph status")
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="CI mode: exit non-zero if graph out of sync",
    )

    args = parser.parse_args()

    # Handle status command
    if args.status:
        print_status()
        return

    # Create updater
    updater = KnowledgeGraphUpdater(REPO_ROOT)

    # Scan based on mode
    if args.full:
        print("[*] Scanning entire prism/ directory...")
        updater.scan_module(REPO_ROOT / "prism")
        update_type = "full"
    elif args.incremental:
        info = get_last_update_info()
        if not info:
            print("[!] No previous update found. Running full scan instead.")
            updater.scan_module(REPO_ROOT / "prism")
            update_type = "full"
        else:
            last_commit = info.get("last_commit", "")
            changed = get_changed_files(last_commit)

            if not changed:
                print("[OK] No Python files changed since last update")
                return

            print(f"[*] Scanning {len(changed)} changed file(s)...")
            for f in changed:
                if f.exists():
                    entities, relations = updater.scan_file(f)
                    updater.entities.extend(entities)
                    updater.relations.extend(relations)
            update_type = "incremental"
    elif args.modules:
        for module in args.modules:
            module_path = REPO_ROOT / "prism" / module.replace(".", "/")
            print(f"[*] Scanning {module}...")
            updater.scan_module(module_path)
        update_type = "modules"
    else:
        print("[!] Must specify --full, --incremental, or --modules", file=sys.stderr)
        print("   Use --status to see current state")
        sys.exit(1)

    # Print summary
    print(updater.generate_summary())

    # Generate commands
    if args.dry_run or args.ci_mode:
        print("\n[*] MCP Commands (not executed):")
        commands = updater.generate_mcp_commands()
        print(f"Would execute {len(commands)} command(s)")

        if args.ci_mode:
            # Check if there are changes to sync
            current = get_current_commit()
            info = get_last_update_info()
            last = info.get("last_commit", "")

            if current != last:
                changed = get_changed_files(last) if last else set()
                if changed:
                    print(f"\n[!] Graph is out of sync: {len(changed)} file(s) changed")
                    sys.exit(1)

            print("\n[OK] Knowledge graph validation passed")
            sys.exit(0)
    else:
        print("\n[*] Graph changes detected. To apply:")
        print("   1. Use Claude Code with MCP memory tools")
        print("   2. Or run: /update-knowledge-graph")
        print("\n[>] The slash command will handle MCP updates automatically")

        # Save state for tracking (but don't update the graph - that's done via MCP)
        if not args.dry_run:
            save_update_info(
                entity_count=len(updater.entities),
                relation_count=len(updater.relations),
                update_type=update_type,
                notes=f"Scanned {len(updater.entities)} entities",
            )
            print("\n[OK] Updated .last_update with current state")


if __name__ == "__main__":
    main()
