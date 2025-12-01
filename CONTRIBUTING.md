# Contributing to PRISM

Thank you for your interest in contributing to PRISM!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/PRISM.git`
3. Install dependencies: `uv sync`
4. Create a feature branch: `git checkout -b feature/your-feature`

## Development Setup

```bash
# Install all dependencies including dev tools
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run tests
uv run pytest tests/ -v

# Check code quality
uv run ruff check prism/
uv run ruff format prism/
uv run mypy prism/
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Run `ruff format` before committing

## Testing

- Add tests for new functionality
- Ensure all tests pass: `uv run pytest tests/ -v`
- Aim for good test coverage

## Pull Requests

1. Update documentation if needed
2. Add tests for new features
3. Ensure CI passes
4. Request review from maintainers

## Questions?

Open an issue on [GitHub](https://github.com/omri/PRISM/issues) for questions or discussions.
