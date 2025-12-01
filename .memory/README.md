# PRISM Knowledge Graph

Knowledge graph for AI agent navigation and component discovery.

## Files

- `memory.jsonl` - Graph data (JSONL format, MCP Memory server compatible)
- `.last_update` - Update metadata (commit hash, timestamp)
- `update_knowledge_graph.py` - AST-based graph updater
- `validate_graph.py` - Sync validation for CI
- `test_queries.py` - Graph validation tests

## AI Agent Usage

AI agents query via MCP memory tools:
```python
mcp__memory__search_nodes("Telescope")
mcp__memory__open_nodes(["Telescope", "TelescopeConfig"])
mcp__memory__read_graph()
```

## Graph Contents

| Type | Count | Examples |
|------|-------|----------|
| Classes | 21 | Telescope, Microscope, MeasurementSystem |
| Configs | 9 | TelescopeConfig, MicroscopeConfig |
| Modules | 12 | prism.core, prism.models |
| Pipelines | 2 | PRISMRunner, PRISMTrainer |
| Propagators | 4 | FresnelPropagator, FraunhoferPropagator |

**Total: 60 entities, 60 relations**

## Maintenance

Update after modifying `prism/` Python files:
```bash
uv run python .memory/update_knowledge_graph.py --full
```

Validate:
```bash
uv run python .memory/test_queries.py
uv run python .memory/validate_graph.py
```

## Schema

**Entity Types:** Module, Class, Function, Protocol, Config, Pipeline, Propagator

**Relation Types:** defines, uses, inherits_from, implements, configures, orchestrates, depends_on
