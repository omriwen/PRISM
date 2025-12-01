# PRISM API Documentation

This directory contains the Sphinx-based API documentation for PRISM.

## Building the Documentation

### Prerequisites

Install Sphinx and dependencies:

```bash
cd ../..
uv sync  # This includes dev dependencies with Sphinx
```

### Build HTML Documentation

```bash
cd docs/api
make html
```

The generated HTML files will be in `_build/html/`.

### View Documentation Locally

```bash
# Option 1: Open in browser
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows

# Option 2: Use Python's HTTP server
cd _build/html
python -m http.server 8000
# Then navigate to http://localhost:8000
```

### Clean Build Files

```bash
make clean
```

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation index
- `modules/` - Module-specific documentation (.rst files)
  - `core.rst` - Core components (Telescope, Aggregators, Trainers)
  - `models.rst` - Model architectures and loss functions
  - `utils.rst` - Utility functions (sampling, image processing, metrics, etc.)
  - `config.rst` - Configuration management

## Adding New Documentation

1. Create or update `.rst` files in `modules/`
2. Add references to new modules using the `.. automodule::` directive
3. Update `index.rst` if adding new top-level sections
4. Rebuild: `make clean && make html`

## Extensions Used

- `sphinx.ext.autodoc` - Auto-generate docs from docstrings
- `sphinx.ext.autosummary` - Generate summary tables
- `sphinx.ext.napoleon` - Support for NumPy/Google docstrings
- `sphinx.ext.viewcode` - Add links to source code
- `sphinx.ext.intersphinx` - Link to other documentation
- `sphinx_autodoc_typehints` - Include type hints in documentation

## Docstring Style

PRISM uses NumPy-style docstrings. Example:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Short description.

    Longer description explaining the function.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2

    Returns
    -------
    bool
        Description of return value
    """
    pass
```

## Troubleshooting

### Import Errors

If you see warnings about failed imports:
1. Ensure the module exists in the codebase
2. Check that the module path in `.rst` files is correct
3. Make sure all dependencies are installed

### Formatting Warnings

Sphinx is strict about RST formatting. Common issues:
- Missing blank lines before/after code blocks
- Incorrect indentation in docstrings
- Missing colons after section headers
