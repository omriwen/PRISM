# Refactor Status

Show the current status of the architecture refactoring defined in REFACTORING_DESIGN_DOCUMENT.md.

## Instructions

### Step 1: Read the Refactoring Design Document

Read `REFACTORING_DESIGN_DOCUMENT.md` to understand the target architecture and phases.

### Step 2: Check Component Existence

Check which target architecture components exist:

#### Phase 1: Foundation Components
| Component | File | Expected Class/Content |
|-----------|------|------------------------|
| AbstractRunner | `prism/core/runner/base.py` | `class AbstractRunner(ABC)` with `run()` template method |
| AbstractTrainer | `prism/core/trainers/base.py` | `class AbstractTrainer(ABC)` with `train()` method |
| ProgressiveTrainer | `prism/core/trainers/progressive.py` | `class ProgressiveTrainer(AbstractTrainer)` |
| PRISMRunner | `prism/core/runner/prism_runner.py` | `class PRISMRunner(AbstractRunner)` |
| TrainingConfig | `prism/core/trainers/config.py` | `@dataclass class TrainingConfig` |
| Result types | `prism/core/runner/results.py` | `ExperimentResult`, `TrainingResult` dataclasses |

#### Phase 2: MoPIE Integration
| Component | File | Expected Class/Content |
|-----------|------|------------------------|
| EpochalTrainer | `prism/core/trainers/epochal.py` | `class EpochalTrainer(AbstractTrainer)` |
| MoPIERunner | `prism/core/runner/mopie_runner.py` | `class MoPIERunner(AbstractRunner)` |
| Trainer mixins | `prism/core/trainers/mixins.py` | `CheckpointMixin`, `MetricsMixin` |

#### Phase 3: Unification
| Component | File | Expected Class/Content |
|-----------|------|------------------------|
| Entry points | `prism/cli/entry_points.py` | `run_prism()`, `run_mopie()` functions |
| Runner factory | `prism/core/runner/factory.py` | `create_runner(algorithm: str)` function |

#### Phase 4: Testing & Documentation
| Component | File | Expected Content |
|-----------|------|------------------|
| Runner tests | `tests/core/test_runners.py` | Test classes for all runners |
| Trainer tests | `tests/core/test_trainers.py` | Test classes for all trainers |

#### Legacy Files (should exist until refactoring complete)
| File | Status After Refactoring |
|------|-------------------------|
| `main.py` | Replaced by `prism run prism` CLI |
| `main_mopie.py` | Replaced by `prism run mopie` CLI |
| `main_run_from_checkpoint.py` | Replaced by `prism resume` CLI |

### Step 3: Analyze Each Phase

For each phase, determine status:
- **Not Started**: No files exist
- **In Progress**: Some files exist, incomplete implementation
- **Complete**: All files exist with correct classes/content
- **Blocked**: Dependencies not met

### Step 4: Generate Status Report

Output a comprehensive status report:

```markdown
## Refactoring Status Report

### Executive Summary
- **Overall Progress**: X/4 phases complete
- **Current Phase**: Phase N - [Name]
- **Blockers**: [List any blockers]

### Phase 1: Foundation
**Status**: [Not Started | In Progress | Complete]

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| AbstractRunner | `prism/core/runner/base.py` | [Exists/Missing] | [Notes] |
| ... | ... | ... | ... |

### Phase 2: MoPIE Integration
**Status**: [Not Started | In Progress | Complete | Blocked]
...

### Phase 3: Unification
**Status**: [Not Started | In Progress | Complete | Blocked]
...

### Phase 4: Testing & Documentation
**Status**: [Not Started | In Progress | Complete | Blocked]
...

### Legacy File Status
| File | Current State | Migration Status |
|------|---------------|------------------|
| main.py | [Exists] | [Not migrated / Partially migrated / Fully migrated] |
| ... | ... | ... |

### Next Recommended Tasks
Based on current progress, the next tasks are:
1. [Task description]
2. [Task description]
3. [Task description]

### Code Quality Metrics
- **New code LOC**: [Count lines in prism/core/runner/, prism/core/trainers/]
- **Test coverage**: [If available]
- **Type coverage**: [Run mypy and report]
```

### Step 5: Check for Migration Blockers

Identify any issues blocking progress:
- Missing dependencies
- Circular imports
- Failing tests
- Incomplete interfaces

## Example Usage

```bash
# Check refactoring progress
/refactor-status
```

## Important Notes

- This command is read-only; it doesn't modify any files
- Run this before starting refactoring work to understand current state
- Run after completing tasks to verify progress
- Use with `/implement-task` to work through the refactoring phases
