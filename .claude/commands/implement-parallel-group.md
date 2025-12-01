# Implement Parallel Task Group

Implement all tasks in a parallel execution group from the current implementation plan, running independent tasks concurrently using multiple agents.

## Arguments

`$ARGUMENTS` should specify the parallel group to implement:
- Group number: `1`, `2`, `3`, etc.
- Group name: `Foundation`, `Core Functions`, `Testing`, etc.
- Full reference: `group 1`, `parallel group 2`, etc.

## Instructions

### Step 1: Identify the Relevant Plan File

Determine which plan file to use:

1. **Check context**: Look for a recently discussed or opened plan file in the conversation
2. **Check branch name**: Match branch name to plan file
3. **List available plans**: If ambiguous, list files in `docs/plans/` and ask user to specify

### Step 2: Parse Group Identifier

Parse `$ARGUMENTS` to identify the target group:
- `1` or `group 1` -> Parallel Group 1
- `Foundation` -> Match by group name
- `2` or `Core Functions` -> Parallel Group 2

### Step 3: Read the Plan and Locate Group

Read the plan file and find the "Parallel Implementation Workflow" section:

1. Locate the **Parallel Execution Groups** subsection
2. Find the specified group (by number or name)
3. Extract the task list from the group's table:

```markdown
#### Group 1: Foundation (No Dependencies)

**Can be done simultaneously:**

| Task | File | Description |
|------|------|-------------|
| **1.1** | `path/to/file1.py` | Description 1 |
| **2.1** | `path/to/file2.py` | Description 2 |
| **3.1** | `path/to/file3.py` | Description 3 |
```

### Step 4: Check Prerequisites

Before implementing the group:

1. **Check dependencies**: Verify prerequisite groups are complete
   - "Depends on Group 1" means Group 1 tasks should be Complete
2. **Check task status**: Verify tasks are not already Complete
3. **Warn if blocked**: If dependencies not met, warn user and ask to confirm

### Step 5: Implement Tasks in Parallel

**CRITICAL**: Use the Task tool to launch multiple agents simultaneously for independent tasks.

For each task in the group, spawn a parallel agent:

```
<Task tool call 1: Task 1.1>
<Task tool call 2: Task 2.1>
<Task tool call 3: Task 3.1>
```

**All Task tool calls MUST be in a single message** to enable true parallel execution.

Each agent should:
1. Read the detailed task instructions from the "Implementation Phases" section
2. Implement the task following the plan exactly
3. Report back what was done and any files created/modified

**Agent Prompt Template**:
```
Implement Task X.X from the implementation plan.

Plan file: docs/plans/[plan-name].md

Task details from plan:
[Copy the full task section from Implementation Phases]

Instructions:
1. Read the task details carefully
2. Implement exactly as specified in the plan
3. Create/modify the files mentioned
4. If code snippets are provided, use them as the basis
5. Report back:
   - Files created
   - Files modified
   - Any issues encountered
   - Confirmation of completion

Do NOT update the plan file - the main agent will do that.
```

### Step 6: Collect Results

After all parallel agents complete:

1. Collect results from each agent
2. Verify each task was completed successfully
3. Note any failures or issues

### Step 7: Update the Plan File

Update the Progress Tracking table for ALL completed tasks:

**Before:**
```markdown
| Phase | Task | Description | Status |
|-------|------|-------------|--------|
| 1 | 1.1 | Fix notebook FOV | Pending |
| 2 | 2.1 | Create InputMode enum | Pending |
| 3 | 3.1 | Add padding_factor | Pending |
```

**After:**
```markdown
| Phase | Task | Description | Status |
|-------|------|-------------|--------|
| 1 | 1.1 | Fix notebook FOV | Complete |
| 2 | 2.1 | Create InputMode enum | Complete |
| 3 | 3.1 | Add padding_factor | Complete |
```

Use the Edit tool with `replace_all=false` for each status update.

### Step 8: Report Results

Provide a comprehensive summary:

```markdown
## Parallel Group [N]: [Name] - Completed

### Tasks Implemented (in parallel)

| Task | Description | Status | Files |
|------|-------------|--------|-------|
| 1.1 | Fix notebook FOV | Complete | `notebook.ipynb` |
| 2.1 | Create InputMode enum | Complete | `input_handling.py` |
| 3.1 | Add padding_factor | Complete | `microscope_forward.py` |

### Execution Summary
- **Tasks in group**: 3
- **Completed**: 3
- **Failed**: 0
- **Parallel agents used**: 3

### Plan Updated
Progress tracking table in `docs/plans/[plan-name].md` updated.

### Next Group
**Group [N+1]: [Name]** is now unblocked.
Tasks ready for implementation:
- Task X.X: [Description]
- Task Y.Y: [Description]
```

## Example Usage

```bash
# Implement Group 1 (Foundation)
/implement-parallel-group 1

# Implement by group name
/implement-parallel-group Foundation

# Implement Group 2 (Core Functions)
/implement-parallel-group 2

# Implement testing group
/implement-parallel-group Testing
```

## Important Notes

### Maximizing Parallelization

- **All agents must be spawned in a single message** - this is required for true parallel execution
- Use `subagent_type="general-purpose"` for implementation tasks
- Each agent works independently on its assigned task
- The main agent coordinates and updates the plan

### Handling Failures

If an agent fails:
1. Mark that specific task as `Failed` in the plan
2. Continue with other successful tasks
3. Report failure details in summary
4. Suggest re-running the failed task with `/implement-task X.X`

### Dependencies Between Tasks in Same Group

Tasks in the same parallel group should have NO dependencies on each other. If the plan is correct:
- All tasks in a group can run simultaneously
- No task needs output from another task in the same group
- Each task modifies different files (or different parts of the same file)

If you detect a dependency issue, warn the user and suggest running tasks sequentially instead.

### File Conflicts

If multiple tasks modify the same file:
1. Warn about potential conflicts
2. Consider running those tasks sequentially
3. Or ensure they modify different sections of the file

## Status Values

Use these status values:
- `Pending` - Not started
- `In Progress` - Currently running (set before spawning agents)
- `Complete` - Successfully finished
- `Failed` - Agent reported failure
- `Blocked` - Dependencies not met
