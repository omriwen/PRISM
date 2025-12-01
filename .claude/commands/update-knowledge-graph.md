# Update Knowledge Graph

Update the PRISM knowledge graph with changes from the codebase.

## Instructions

You are updating the PRISM knowledge graph stored in `.memory/memory.jsonl`. This graph enables 10x faster component lookups for AI agents navigating the codebase.

### Step 1: Check Current State

First, run the status check to see what needs updating:

```bash
uv run python tools/update_knowledge_graph.py --status
```

### Step 2: Scan for Changes

If there are changes since the last update, run an incremental scan:

```bash
uv run python tools/update_knowledge_graph.py --incremental --dry-run
```

This will show:
- Which files have changed
- What entities were discovered
- What relations exist

### Step 3: Update the Graph

Based on the scan results, update the knowledge graph using MCP memory tools:

1. **For NEW entities** discovered in the scan, use `mcp__memory__create_entities` to add them
2. **For MODIFIED entities**, use `mcp__memory__add_observations` to update their observations
3. **For DELETED entities** (files removed), use `mcp__memory__delete_entities` to remove them
4. **For NEW relations**, use `mcp__memory__create_relations` to add them

### Step 4: Verify and Save State

After updates:

1. Verify the graph with `mcp__memory__read_graph`
2. Update the `.memory/.last_update` file with the current git commit hash and entity counts

### Entity Types to Track

- **Module**: Python packages (prism.core, prism.models, etc.)
- **Class**: Component implementations (Telescope, MeasurementSystem)
- **Config**: Configuration dataclasses (TelescopeConfig, etc.)
- **Pipeline**: Orchestrators (PRISMRunner, PRISMTrainer)
- **Propagator**: Physics engines (FresnelPropagator, etc.)
- **Protocol**: Abstract base classes (Instrument, Target)
- **Function**: Key factory functions (create_instrument, get_scenario_preset)

### Relation Types

- `defines`: Module defines Class/Function
- `uses`: Component uses another component
- `inherits_from`: Class inheritance
- `configures`: Config configures Component
- `orchestrates`: Pipeline orchestrates components
- `provides`: Scenario provides Config

### Example Entity Format

```python
{
    "name": "ClassName",
    "entityType": "Class",
    "observations": [
        "Type: Class",
        "Path: prism/core/module.py:42",
        "Description: Brief description",
        "Inherits from: BaseClass",
        "Uses: DependencyA, DependencyB"
    ]
}
```

## Quick Mode (Full Rescan)

For a complete refresh of the knowledge graph:

```bash
uv run python tools/update_knowledge_graph.py --full --dry-run
```

Then recreate the entire graph using MCP tools.

## Notes

- The graph schema is documented in `docs/knowledge-graph-schema.md`
- Always update `.memory/.last_update` after changes
- The memory file is tracked in git for sharing across developers
