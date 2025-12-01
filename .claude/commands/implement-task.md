# Implement Plan Task(s)

Complete specified task(s) from the current implementation plan and update the plan file accordingly.

## Arguments

`$ARGUMENTS` should contain task number(s) in one of these formats:
- Single task: `1.1`
- Multiple tasks (comma-separated): `1.1, 1.2, 1.3`
- Task range: `1.1-1.3`
- Mixed: `1.1, 2.1-2.3, 3.1`

## Instructions

### Step 1: Identify the Relevant Plan File

Determine which plan file to use:

1. **Check context**: Look for a recently discussed or opened plan file in the conversation
2. **Check branch name**: Match branch name to plan file (e.g., `feature/optical-catalog-and-fpm` -> files mentioning "optical" or "fpm")
3. **List available plans**: If ambiguous, list files in `docs/plans/` and ask user to specify

If a plan file is already open or was recently discussed, use that one.

### Step 2: Parse Task Numbers

Parse `$ARGUMENTS` into a list of task numbers:
- `1.1` -> `["1.1"]`
- `1.1, 1.2, 1.3` -> `["1.1", "1.2", "1.3"]`
- `1.1-1.3` -> `["1.1", "1.2", "1.3"]`
- `2.1-2.3, 3.1` -> `["2.1", "2.2", "2.3", "3.1"]`

### Step 3: Read the Plan

Read the plan file and locate:
1. **Progress Tracking Table**: Find current status of requested tasks
2. **Task Details**: Find the detailed implementation instructions for each task in the "Implementation Phases" section

### Step 4: Validate Tasks

Check that:
- All requested task numbers exist in the plan
- Tasks are not already marked as "Complete" (warn if so)
- Dependencies are satisfied (earlier tasks should be complete or in progress)

If dependencies are not met, warn the user but proceed if they confirm.

### Step 5: Implement Each Task

For each task (in numerical order):

1. **Read task details** from the plan's Implementation Phases section
2. **Create a todo item** using TodoWrite tool for tracking
3. **Implement the task** following the detailed instructions:
   - Create new files as specified
   - Modify existing files as specified
   - Follow any code snippets provided in the plan
   - Run tests if mentioned
4. **Mark todo as complete** when done

### Step 6: Update the Plan File

After completing each task, update the Progress Tracking table:

**Change status from:**
```
| Phase | Task | Description | Status |
|-------|------|-------------|--------|
| 1 | 1.1 | [Description] | Pending |
```

**To:**
```
| Phase | Task | Description | Status |
|-------|------|-------------|--------|
| 1 | 1.1 | [Description] | Complete |
```

Use the Edit tool to replace the status column value.

### Step 7: Report Results

After all tasks are complete, provide a summary:

```
## Tasks Completed

| Task | Description | Files Modified |
|------|-------------|----------------|
| 1.1 | [Description] | `path/to/file.py` |
| 1.2 | [Description] | `path/to/other.py` |

## Plan Updated

Progress tracking table in `docs/plans/[plan-name].md` has been updated.

## Next Steps

Remaining tasks in current phase:
- Task X.X: [Description] (Pending)
- Task X.X: [Description] (Pending)
```

## Example Usage

```bash
# Implement a single task
/implement-task 1.1

# Implement multiple specific tasks
/implement-task 1.1, 1.2, 1.3

# Implement a range of tasks
/implement-task 2.1-2.5

# Implement all tasks in a phase (if supported by plan structure)
/implement-task phase 1

# Mixed format
/implement-task 1.1, 2.1-2.3, 3.1
```

## Important Notes

- **Follow the plan exactly**: The plan contains detailed instructions for each task. Follow them precisely.
- **Use code snippets**: If the plan includes code snippets, use them as the implementation basis.
- **Test as you go**: If the task includes test instructions, run them.
- **Update status immediately**: Update the plan file after each task, not at the end.
- **Handle failures**: If a task fails, update status to "Failed" and explain why in the report.
- **Preserve formatting**: When editing the plan file, preserve markdown formatting exactly.

## Status Values

Use these status values in the Progress Tracking table:
- `Pending` - Task not started
- `In Progress` - Task currently being worked on
- `Complete` - Task finished successfully
- `Failed` - Task attempted but failed
- `Skipped` - Task intentionally skipped (explain why)
- `Blocked` - Task cannot proceed due to external dependency

## Error Handling

If unable to complete a task:
1. Update plan status to `Failed` or `Blocked`
2. Add a note explaining the issue
3. Continue with remaining tasks if possible
4. Report all failures in the summary
