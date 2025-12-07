# Create Knowledge Graph

Initialize a new knowledge graph from scratch, or recreate it fresh after major changes.

## Purpose

The knowledge graph enables AI agents to navigate the codebase **faster and more accurately** than using bash commands (grep, find, etc.). Instead of searching through files, agents can query semantic relationships between components directly.

A well-structured graph should capture:
- **What exists:** Classes, functions, configs, modules
- **How they relate:** Inheritance, usage, configuration relationships
- **Where to find them:** File paths and line numbers

## When to Use

- **First time setup:** `memory.jsonl` doesn't exist yet
- **Fresh start:** After major refactoring, cleanup, or when the graph has drifted too far

This command creates a completely new graph. For incremental updates, use `/update-knowledge-graph` instead.

## Instructions

### Step 1: Scan the Codebase

Run a full scan to discover all entities:

```bash
uv run python .memory/update_knowledge_graph.py --full --dry-run
```

Review the output - it shows all discovered classes, functions, and relations.

### Step 2: Clear Existing Graph (if recreating)

If `memory.jsonl` already exists and you want a fresh start:

```python
# Read current graph to see what will be replaced
mcp__memory__read_graph()

# Delete all existing entities (get names from read_graph output)
mcp__memory__delete_entities(["<entity1>", "<entity2>", ...])
```

### Step 3: Create Entities

Use the scanner output to create entities. **IMPORTANT:** Include the `type` field:

```python
mcp__memory__create_entities([
    {"type": "entity", "name": "<name>", "entityType": "<type>", "observations": [...]},
    # ... entities from scanner output
])
```

### Step 4: Create Relations

**IMPORTANT:** Include the `type` field:

```python
mcp__memory__create_relations([
    {"type": "relation", "from": "<entity>", "to": "<entity>", "relationType": "<type>"},
    # ... relations from scanner output
])
```

### Step 5: Save State

Run without `--dry-run` to create `.last_update`:

```bash
uv run python .memory/update_knowledge_graph.py --full
```

### Step 6: Verify

```bash
uv run python .memory/test_queries.py --verbose
```

## Notes

- Entity types and relations are discovered by the AST scanner
- See `.memory/README.md` for MCP server prerequisites and **JSONL format details**
- **Goal:** Structure the graph so AI agents can answer questions like "what uses X?" or "where is Y defined?" without grep/find
- **Keep docs in sync:** When making significant changes (new modules, renamed components, architectural changes), consider updating the "Essential Files" tree in `AI_ASSISTANT_GUIDE.md` if the changes affect the high-level project structure

## JSONL Format Requirements

The MCP Memory server requires this exact format:

```jsonl
{"type": "entity", "name": "MyClass", "entityType": "Class", "observations": ["Path: file.py"]}
{"type": "relation", "from": "MyClass", "to": "BaseClass", "relationType": "inherits_from"}
```

**Critical:**
- Each line is a separate JSON object (JSONL format)
- The `"type"` field is **required** - without it, the MCP server returns empty results
- Use Unix line endings (LF), not Windows (CRLF)
