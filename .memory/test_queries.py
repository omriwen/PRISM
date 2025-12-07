"""Test suite for knowledge graph validation.

Validates the knowledge graph structure (not specific content).
Content is discovered by the AST scanner - these tests only verify
the graph file is valid and has the expected structure.

Usage:
    uv run python .memory/test_queries.py
    uv run python .memory/test_queries.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
MEMORY_FILE = REPO_ROOT / ".memory" / "memory.jsonl"


class KnowledgeGraphTester:
    """Test suite for knowledge graph structure validation."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.graph_data = self._load_graph()
        self.passed = 0
        self.failed = 0
        self.parse_errors: list[str] = []

    def _load_graph(self) -> dict[str, Any]:
        """Load graph from MCP memory server JSONL format.

        Format: One JSON object per line, each with a "type" field:
        - Entities: {"type": "entity", "name": "...", "entityType": "...", "observations": [...]}
        - Relations: {"type": "relation", "from": "...", "to": "...", "relationType": "..."}
        """
        entities: dict[str, dict] = {}
        relations: list[dict] = []

        if not MEMORY_FILE.exists():
            return {"entities": entities, "relations": relations}

        try:
            with open(MEMORY_FILE, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        obj_type = obj.get("type", "")
                        if obj_type == "entity" or "entityType" in obj:
                            entities[obj["name"]] = obj
                        elif obj_type == "relation" or "relationType" in obj:
                            relations.append(obj)
                    except json.JSONDecodeError as e:
                        self.parse_errors.append(f"Line {line_num}: {e}")
        except IOError as e:
            self.parse_errors.append(f"File read error: {e}")

        return {"entities": entities, "relations": relations}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"    {msg}")

    def run_test(self, name: str, test_fn: callable) -> None:
        try:
            if test_fn():
                print(f"  PASS: {name}")
                self.passed += 1
            else:
                print(f"  FAIL: {name}")
                self.failed += 1
        except Exception as e:
            print(f"  ERROR: {name} - {e}")
            self.failed += 1

    def run_all_tests(self) -> int:
        print("Knowledge Graph Structure Tests")
        print("=" * 40)

        # File tests
        print("\nFile:")
        self.run_test("memory.jsonl exists", lambda: MEMORY_FILE.exists())
        self.run_test("No JSON parse errors", lambda: len(self.parse_errors) == 0)
        if self.parse_errors and self.verbose:
            for err in self.parse_errors[:5]:
                print(f"    {err}")

        # Structure tests
        print("\nStructure:")
        entity_count = len(self.graph_data["entities"])
        relation_count = len(self.graph_data["relations"])

        self.run_test(f"Has entities (found: {entity_count})", lambda: entity_count > 0)
        self.run_test(f"Has relations (found: {relation_count})", lambda: relation_count > 0)

        # Entity structure tests
        print("\nEntity Structure:")
        entities_have_names = all(
            "name" in e for e in self.graph_data["entities"].values()
        )
        entities_have_types = all(
            "entityType" in e for e in self.graph_data["entities"].values()
        )
        entities_have_observations = all(
            "observations" in e and isinstance(e["observations"], list)
            for e in self.graph_data["entities"].values()
        )

        self.run_test("All entities have 'name'", lambda: entities_have_names)
        self.run_test("All entities have 'entityType'", lambda: entities_have_types)
        self.run_test("All entities have 'observations' list", lambda: entities_have_observations)

        # Relation structure tests
        print("\nRelation Structure:")
        relations_have_from = all(
            "from" in r for r in self.graph_data["relations"]
        )
        relations_have_to = all(
            "to" in r for r in self.graph_data["relations"]
        )
        relations_have_type = all(
            "relationType" in r for r in self.graph_data["relations"]
        )

        self.run_test("All relations have 'from'", lambda: relations_have_from)
        self.run_test("All relations have 'to'", lambda: relations_have_to)
        self.run_test("All relations have 'relationType'", lambda: relations_have_type)

        # Summary
        if self.verbose:
            print("\nEntity Types Found:")
            entity_types: dict[str, int] = {}
            for e in self.graph_data["entities"].values():
                t = e.get("entityType", "unknown")
                entity_types[t] = entity_types.get(t, 0) + 1
            for t, count in sorted(entity_types.items()):
                print(f"    {t}: {count}")

            print("\nRelation Types Found:")
            relation_types: dict[str, int] = {}
            for r in self.graph_data["relations"]:
                t = r.get("relationType", "unknown")
                relation_types[t] = relation_types.get(t, 0) + 1
            for t, count in sorted(relation_types.items()):
                print(f"    {t}: {count}")

        print("\n" + "=" * 40)
        total = self.passed + self.failed
        print(f"Results: {self.passed}/{total} passed")

        return 0 if self.failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test knowledge graph structure (not specific content)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    tester = KnowledgeGraphTester(verbose=args.verbose)
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
