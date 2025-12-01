#!/bin/bash

# format_python_hook.sh - Wrapper script for Claude Code Stop hook
# Parses the transcript to find all modified Python files and formats them once at the end

# Read JSON from stdin
json_input=$(cat)

# Extract transcript path
transcript_path=$(echo "$json_input" | jq -r '.transcript_path // empty')

# If transcript path is not found, exit silently
if [ -z "$transcript_path" ] || [ "$transcript_path" = "null" ] || [ ! -f "$transcript_path" ]; then
    exit 0
fi

# Extract project directory for resolving relative paths
project_dir=$(echo "$json_input" | jq -r '.cwd // empty')
if [ -z "$project_dir" ]; then
    project_dir="$CLAUDE_PROJECT_DIR"
fi

# Parse transcript to find all Edit/Write tool calls and collect Python file paths
# Handle JSONL format (one JSON object per line)
python_files=$(jq -r --slurp '
  # Process all lines, extract file paths from Edit/Write operations
  [.[] |
    select(.type == "tool_call" and (.tool_name == "Edit" or .tool_name == "Write")) |
    (.tool_input.file_path // .tool_input.filePath // .tool_response.filePath // .tool_response.file_path // empty) |
    select(. != null and . != "" and . != "null") |
    select(endswith(".py"))
  ] | unique | .[]
' "$transcript_path" 2>/dev/null)

# Alternative: if slurp doesn't work, try inputs (for streaming JSONL)
if [ -z "$python_files" ]; then
  python_files=$(jq -r '
    [inputs |
      select(.type == "tool_call" and (.tool_name == "Edit" or .tool_name == "Write")) |
      (.tool_input.file_path // .tool_input.filePath // .tool_response.filePath // .tool_response.file_path // empty) |
      select(. != null and . != "" and . != "null") |
      select(endswith(".py"))
    ] | unique | .[]
  ' "$transcript_path" 2>/dev/null)
fi

# If no Python files were found, exit silently
if [ -z "$python_files" ]; then
    exit 0
fi

# Convert newline-separated files to array and make paths absolute if needed
file_list=()
while IFS= read -r file; do
    if [ -n "$file" ]; then
        # Make path absolute if it's relative (check if it starts with /)
        if [[ "$file" != /* ]]; then
            file="$project_dir/$file"
        fi
        # Only add if file exists
        if [ -f "$file" ]; then
            file_list+=("$file")
        fi
    fi
done <<< "$python_files"

# If we have files to format, run format_python.sh with all of them
if [ ${#file_list[@]} -gt 0 ]; then
    ~/bin/format_python.sh "${file_list[@]}" 2>/dev/null || true
fi

exit 0
