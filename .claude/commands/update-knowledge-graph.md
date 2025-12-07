# Update Knowledge Graph

Incrementally update the knowledge graph with recent changes.

## Purpose

The knowledge graph enables AI agents to navigate the codebase **faster and more accurately** than using bash commands (grep, find, etc.). Keeping it in sync ensures agents have current, reliable information about:
- **What exists:** Classes, functions, configs, modules
- **How they relate:** Inheritance, usage, configuration relationships
- **Where to find them:** File paths and line numbers

## When to Use

- After adding, modifying, or removing Python files in `prism/`
- When `.last_update` shows the graph is behind the current commit
- For routine maintenance

For first-time setup or a complete refresh, use `/create-knowledge-graph` instead.

## Instructions

### Step 1: Check What Changed

```bash
uv run python .memory/update_knowledge_graph.py --status
```

This compares the current commit against `.last_update` and shows which files changed.

### Step 2: Scan Changed Files

```bash
uv run python .memory/update_knowledge_graph.py --incremental --dry-run
```

Review the output - it shows entities discovered in the changed files.

### Step 3: Update the Graph

Based on the scan results. **IMPORTANT:** Include the `type` field in all entities/relations:

**For new/modified entities:**
```python
mcp__memory__create_entities([
    {"type": "entity", "name": "...", "entityType": "...", "observations": [...]}
])
mcp__memory__add_observations(...)        # Updates to existing
```

**For deleted entities** (if files were removed):
```python
mcp__memory__delete_entities([...])
```

**For new relations:**
```python
mcp__memory__create_relations([
    {"type": "relation", "from": "...", "to": "...", "relationType": "..."}
])
```

### Step 4: Save State

```bash
uv run python .memory/update_knowledge_graph.py --incremental
```

This updates `.last_update` with the current commit hash.

### Step 5: Verify

```bash
uv run python .memory/test_queries.py
```

## Notes

- Uses `.last_update` to track sync state
- Only scans files changed since last update
- See `.memory/README.md` for MCP server prerequisites and **JSONL format details**
- **Goal:** Keep the graph accurate so AI agents can answer questions like "what uses X?" or "where is Y defined?" without grep/find
- **Keep docs in sync:** When making significant changes (new modules, renamed components, architectural changes), consider updating the "Essential Files" tree in `AI_ASSISTANT_GUIDE.md` if the changes affect the high-level project structure

## JSONL Format Requirements

The MCP Memory server requires the `type` field on each line:

```jsonl
{"type": "entity", "name": "MyClass", "entityType": "Class", "observations": ["Path: file.py"]}
{"type": "relation", "from": "MyClass", "to": "BaseClass", "relationType": "inherits_from"}
```

Without the `type` field, the MCP server returns empty results.
