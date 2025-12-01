# Finalize Implementation Plan

Finalize your implementation plan and save it as a detailed markdown file in docs/plans/.

## Instructions

The user wants you to finalize an implementation plan for: $ARGUMENTS

Follow these steps:

1. **Analyze the Plan Context**
   - Review any previous discussion or planning in this conversation
   - Identify the main goal, problem statement, and key design decisions
   - Determine appropriate phases and tasks

2. **Generate a Descriptive Filename**
   - Use **lowercase with hyphens** (kebab-case) - following existing docs/plans/ patterns
   - Be descriptive but concise
   - End with `.md`
   - Examples based on existing plans:
     - `adaptive-convergence-plan.md`
     - `microscope-lens-forward-model.md`
     - `unified-instruments-api-refactoring.md`

3. **Write the Plan Document**

   Use this structure (adapt sections as needed):

   ```markdown
   # [Plan Title] Implementation Plan

   **Document**: Implementation Plan
   **Created**: [Today's Date YYYY-MM-DD]
   **Status**: Planning
   **Branch**: `feature/[suggested-branch-name]`

   ---

   ## Executive Summary

   [2-3 sentences describing what this plan accomplishes]

   ### Key Design Decisions

   | Decision | Choice | Rationale |
   |----------|--------|-----------|
   | ... | ... | ... |

   ### Guiding Principle

   > [One-sentence principle that guides implementation choices]

   ---

   ## Problem Statement

   ### Current State
   [Describe what exists now and its limitations]

   ### Desired State
   [Describe the target outcome]

   ---

   ## Architecture

   ### Overview
   [High-level architecture diagram or description]

   ### Key Components
   [List and describe main components]

   ---

   ## Progress Tracking

   | Phase | Task | Description | Status |
   |-------|------|-------------|--------|
   | 1 | 1.1 | [Task description] | Pending |
   | ... | ... | ... | ... |

   ---

   ## Implementation Phases

   ### Phase 1: [Phase Name]

   **Goal**: [What this phase accomplishes]

   #### Task 1.1: [Task Name]

   **File(s)**: `path/to/file.py`

   [Detailed description of what to do]

   ```python
   # Code snippets if helpful
   ```

   #### Task 1.2: [Task Name]
   ...

   ---

   ### Phase 2: [Phase Name]
   ...

   ---

   ## Parallel Implementation Workflow

   This section defines how tasks can be parallelized to minimize total implementation time.

   ### Task Dependency Graph

   ```
   [ASCII diagram showing task dependencies]

   Example:
                       ┌─────────────────────────────────────────────┐
                       │            PARALLEL GROUP 1                  │
                       │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
                       │  │ Task 1.1 │  │ Task 2.1 │  │ Task 3.1 │   │
                       │  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
                       └───────┼─────────────┼─────────────┼─────────┘
                               │             │             │
                               └──────┬──────┴──────┬──────┘
                                      ▼             ▼
                       ┌─────────────────────────────────────────────┐
                       │            PARALLEL GROUP 2                  │
                       │  ┌──────────┐  ┌──────────┐                 │
                       │  │ Task 1.2 │  │ Task 2.2 │                 │
                       │  └────┬─────┘  └────┬─────┘                 │
                       └───────┼─────────────┼───────────────────────┘
                               │             │
                               └──────┬──────┘
                                      ▼
                       ┌─────────────────────────────────────────────┐
                       │            SEQUENTIAL FINAL                  │
                       │  ┌──────────┐ ──▶ ┌──────────┐              │
                       │  │ Task 4.1 │     │ Task 4.2 │              │
                       │  └──────────┘     └──────────┘              │
                       └─────────────────────────────────────────────┘
   ```

   ### Parallel Execution Groups

   Maximize parallelization by grouping tasks that have no dependencies on each other.

   #### Group 1: [Group Name] (No Dependencies)

   **Can be done simultaneously:**

   | Task | File | Description |
   |------|------|-------------|
   | **X.X** | `path/to/file.py` | [Description] |
   | **Y.Y** | `path/to/other.py` | [Description] |

   **Rationale**: [Why these tasks can run in parallel]

   ---

   #### Group 2: [Group Name] (Depends on Group 1)

   **Can be done simultaneously:**

   | Task | File | Description |
   |------|------|-------------|
   | **X.X** | `path/to/file.py` | [Description] |
   | **Y.Y** | `path/to/other.py` | [Description] |

   **Rationale**: [Why these tasks can run in parallel]

   ---

   [Additional groups as needed...]

   ---

   ### Sequential Workflow (Single Developer)

   For a single developer, the optimal order that minimizes context switching:

   ```
   Block 1: Foundation
   ├── Task X.X: [Name]
   ├── Task Y.Y: [Name]
   └── Task Z.Z: [Name]

   Block 2: Core Implementation
   ├── Task X.X: [Name]
   └── Task Y.Y: [Name]

   Block 3: Integration & Testing
   ├── Task X.X: [Name]
   ├── Task Y.Y: [Name]
   └── Task Z.Z: [Name]
   ```

   ---

   ## Summary

   ### Files to Create

   | File | Lines (est.) | Purpose |
   |------|--------------|---------|
   | `path/to/new_file.py` | ~100 | [Purpose] |

   ### Files to Modify

   | File | Changes |
   |------|---------|
   | `path/to/existing.py` | [What changes] |

   ### Total Estimated Effort

   | Phase | Description | LOC |
   |-------|-------------|-----|
   | Phase 1 | [Name] | ~XXX |
   | Phase 2 | [Name] | ~XXX |
   | **Total** | | **~XXX** |

   ---

   ## Validation Checklist

   - [ ] All existing tests pass
   - [ ] New unit tests pass
   - [ ] Integration tests pass
   - [ ] Documentation updated
   - [ ] [Additional validation items specific to this plan]
   ```

4. **Ensure Quality**
   - Plan must be **self-contained** (someone unfamiliar can follow it)
   - Include **code snippets** for complex implementations
   - Each task should be **specific and actionable**
   - Phases should have **clear goals and dependencies**
   - Include **validation/testing tasks**
   - **Parallel workflow must maximize parallelization** - group as many independent tasks together as possible

5. **Save the File**
   - Save to `docs/plans/[filename].md`
   - Use the Write tool to create the file

6. **Report to User**
   - Show the filename that was created
   - Provide a brief summary of the plan structure
   - List the phases and task count
   - Highlight the maximum parallelization achieved (e.g., "Up to 4 tasks can run in parallel in Group 1")

## Example Usage

```
/finalize-plan dark mode toggle feature
/finalize-plan database connection pooling optimization
/finalize-plan REST API to GraphQL migration
```

## Notes

- If the plan topic is unclear, ask clarifying questions first
- For complex features, create detailed dependency graphs
- Reference existing patterns and conventions from the codebase
- Consider backward compatibility when modifying existing code
- Include estimated lines of code (LOC) where helpful for scoping
- **Maximize parallelization** - analyze dependencies carefully to identify all tasks that can run simultaneously
