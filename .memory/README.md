# PRISM Knowledge Graph Memory

This directory contains the persistent knowledge graph for the PRISM project, used by AI agents for fast navigation and component discovery.

## Files

- `memory.jsonl` - Knowledge graph data (JSONL format, used by MCP Memory server)
- `.last_update` - Metadata about the last graph update (git commit hash, timestamp)

## Purpose

The knowledge graph provides **10x faster component lookups** compared to grep/glob for:
- Architectural queries: "What uses Telescope?", "Where is component X?"
- Dependency analysis: "What does MeasurementSystem depend on?"
- Type-based queries: "Show all pipelines", "List all configs"

## Schema

See [docs/knowledge-graph-schema.md](../docs/knowledge-graph-schema.md) for the full schema.

**Entity Types:** Module, Class, Function, Protocol, Config, Pipeline, Propagator

**Relation Types:** defines, uses, inherits_from, implements, configures, orchestrates, depends_on

## AI Agent Usage

AI agents should query this graph using MCP memory tools:
```python
# Search for components
mcp__memory__search_nodes("Telescope")

# Get detailed info
mcp__memory__open_nodes(["Telescope", "TelescopeConfig"])

# Read entire graph
mcp__memory__read_graph()
```

## Updates

Run the update command when the codebase changes:
```bash
# Via slash command (recommended)
/update-knowledge-graph

# Or manually
python tools/update_knowledge_graph.py --incremental
```

## Tracked in Git

This directory is intentionally tracked in git so the knowledge graph is available to anyone pulling the repo. The MCP memory server reads from `memory.jsonl` at the configured path.
