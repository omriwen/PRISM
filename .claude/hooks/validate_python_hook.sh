#!/bin/bash
# validate_python_hook.sh - PreToolUse hook to enforce uv run python requirement
# Blocks direct python/python3 commands while allowing uv run python

# Read JSON input from stdin
json_input=$(cat)

# Extract the command being executed
command=$(echo "$json_input" | jq -r '.tool_input.command // empty')

# If no command found, allow it
if [ -z "$command" ]; then
    exit 0
fi

# Block direct python/python3 commands (but allow uv run python)
if echo "$command" | grep -qE '^(python|python3)(\s|$)'; then
    cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Direct python/python3 commands are blocked. Use 'uv run python' instead.\n\nForbidden: python script.py\nAllowed: uv run python script.py\n\nForbidden: python3 -m pytest\nAllowed: uv run pytest"
  }
}
EOF
    exit 0
fi

# Allow everything else
exit 0
