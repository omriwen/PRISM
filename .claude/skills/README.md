## PRISM Development Skills

This directory contains custom skills to support PRISM development and code quality.

## Available Skills

### Phase 1: Foundation Skills

1. **module-extractor** ⭐⭐⭐
   - Extract classes/functions from one Python file to another
   - Automatically manage imports and dependencies
   - Use ~20+ times during Phase 2 restructuring

2. **import-updater** ⭐⭐⭐
   - Update import statements across codebase after module moves
   - Handle relative vs absolute imports
   - Organize imports following Python standards

3. **git-commit-maker** ⭐⭐⭐
   - Create conventional commit messages
   - Follow best practices for commit history
   - Use ~50+ times throughout refactoring

4. **code-formatter** ⭐⭐
   - Format code with ruff (check + format)
   - Set up pre-commit hooks
   - Ensure consistent code style

### Phase 3: Specialized Skills

5. **type-hint-adder** ⭐⭐⭐
   - Add comprehensive type hints to functions
   - Special support for PyTorch tensor types
   - Use ~100+ times across all modules

6. **unit-test-generator** ⭐⭐⭐
   - Generate pytest unit tests with fixtures
   - Create parametrized tests for edge cases
   - Use ~30+ times for test coverage

7. **docstring-formatter**
   - Convert docstrings to NumPy/Sphinx style
   - Add Parameters, Returns, Examples sections
   - Ensure documentation completeness

### PRISM-Specific Skills

8. **torch-shape-validator** ⭐⭐
   - Add tensor shape documentation and validation
   - PyTorch-specific shape comments
   - Runtime assertions for debugging

### Additional Skills

9. **dead-code-finder**
   - Identify commented code, unused imports, unreachable code
   - Use vulture and AST analysis
   - Phase 1 cleanup tasks

## Using Skills

Skills are invoked automatically by Claude Code when relevant, or you can explicitly request them:

```
User: "Extract the Grid class from prism/core/grid.py to a new module"
Claude: [Uses module-extractor skill]

User: "Add type hints to prism/core/telescope.py"
Claude: [Uses type-hint-adder skill]

User: "Format all Python files in prism/"
Claude: [Uses code-formatter skill]
```

## Skill Priority

### Must-Have (Use 10+ times)
- module-extractor
- type-hint-adder
- unit-test-generator
- import-updater
- git-commit-maker

### Should-Have (High value)
- code-formatter
- torch-shape-validator
- docstring-formatter

### Nice-to-Have
- dead-code-finder
- Other specialized skills as needed

## Related Documentation

- See `REFACTORING_DESIGN_DOCUMENT.md` for architecture refactoring plans
- See `AI_ASSISTANT_GUIDE.md` for PRISM project-specific development guidelines

## Implementation Status

All skills have been created:
- ✅ module-extractor (template - needs content)
- ✅ import-updater
- ✅ git-commit-maker
- ✅ code-formatter
- ✅ type-hint-adder
- ✅ unit-test-generator
- ✅ torch-shape-validator
- ✅ docstring-formatter
- ✅ dead-code-finder
- ✅ loop-vectorizer
- ✅ memory-leak-detector
- ✅ complex-tensor-handler (PRISM-specific)

## Contributing

When creating new skills:
1. Use the `skill-creator` skill or manually create a SKILL.md file
2. Follow the template structure with YAML frontmatter
3. Write in imperative/infinitive form
4. Include concrete examples
5. Test the skill on real refactoring tasks

## Support

For skill creation guidance, invoke the `skill-creator` skill:
```
User: "Help me create a new skill for [task]"
```
