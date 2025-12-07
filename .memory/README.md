# PRISM Knowledge Graph

Knowledge graph enabling AI agents to navigate the codebase **faster and more accurately** than using bash commands (grep, find, etc.).

Instead of searching through files, agents can query semantic relationships directly:
- **What exists:** Classes, functions, configs, modules
- **How they relate:** Inheritance, usage, configuration relationships
- **Where to find them:** File paths and line numbers

## Prerequisites

This knowledge graph requires the **MCP Memory server** to be configured. Add this to your `.mcp.json`:

**Linux/macOS:**
```json
{
  "mcpServers": {
    "memory": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {
        "MEMORY_FILE_PATH": "/absolute/path/to/PRISM/.memory/memory.jsonl"
      }
    }
  }
}
```

**Windows:**
```json
{
  "mcpServers": {
    "memory": {
      "type": "stdio",
      "command": "cmd",
      "args": ["/c", "npx", "-y", "@modelcontextprotocol/server-memory"],
      "env": {
        "MEMORY_FILE_PATH": "C:\\Users\\YourName\\PRISM\\.memory\\memory.jsonl"
      }
    }
  }
}
```

> **Windows Note:** You must wrap `npx` (and `uvx`) commands with `cmd /c` for MCP servers to start correctly. Without this wrapper, servers fail with "Connection closed" errors.

**Important:** `MEMORY_FILE_PATH` must be an **absolute path** to `.memory/memory.jsonl` in your local clone.

See [Claude Code MCP documentation](https://docs.anthropic.com/en/docs/claude-code) for general MCP setup.

## Files

- `memory.jsonl` - Graph data (JSONL format, MCP Memory server compatible)
- `.last_update` - Update metadata (commit hash, timestamp)
- `update_knowledge_graph.py` - AST-based graph updater and sync checker
- `test_queries.py` - Graph structure validation tests

## JSONL Format (IMPORTANT)

The MCP Memory server requires a specific JSONL format with a **`type` field** on each line:

**Entity format:**
```json
{"type": "entity", "name": "ClassName", "entityType": "Class", "observations": ["Path: file.py", "Description: ..."]}
```

**Relation format:**
```json
{"type": "relation", "from": "ChildClass", "to": "ParentClass", "relationType": "inherits_from"}
```

**Key requirements:**
- One JSON object per line (JSONL, not a single JSON object)
- Each line MUST have `"type": "entity"` or `"type": "relation"`
- Use Unix line endings (LF), not Windows (CRLF)
- Without the `type` field, the MCP server will return empty results

## First Time Setup

If `memory.jsonl` doesn't exist yet:

1. **Scan the codebase** to discover entities:
   ```bash
   uv run python .memory/update_knowledge_graph.py --full --dry-run
   ```

2. **Review the output** - it shows discovered classes, functions, and relations

3. **Create entities** using MCP tools based on scanner output:
   ```python
   mcp__memory__create_entities([...])  # Entities from scanner
   mcp__memory__create_relations([...]) # Relations from scanner
   ```

4. **Save state** (creates `.last_update`):
   ```bash
   uv run python .memory/update_knowledge_graph.py --full
   ```

## Update Strategy

**Automatic:** Claude updates the graph during conversations when Python files in `prism/` are modified.

**Manual:** Use the scripts for bulk updates:
```bash
uv run python .memory/update_knowledge_graph.py --incremental  # Changed files only
uv run python .memory/update_knowledge_graph.py --full         # Full rescan
```

**Validation:** A pre-commit hook runs `update_knowledge_graph.py --status` to warn (not block) if out of sync.

## AI Agent Usage

AI agents query via MCP memory tools:
```python
mcp__memory__search_nodes("<query>")      # Search for entities by name
mcp__memory__open_nodes(["<name>", ...])  # Get details on specific entities
mcp__memory__read_graph()                  # Read entire graph
```

## Manual Maintenance (Optional)

These scripts are available for manual use when needed:

```bash
# Check sync status
uv run python .memory/update_knowledge_graph.py --status

# Full rescan
uv run python .memory/update_knowledge_graph.py --full

# Validate graph structure
uv run python .memory/test_queries.py
```

## Schema

Entity types and relations are discovered automatically by `update_knowledge_graph.py` through AST analysis. Run with `--full --dry-run` to see what the scanner finds in the current codebase.
