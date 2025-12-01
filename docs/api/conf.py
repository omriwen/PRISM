"""Sphinx configuration for PRISM API documentation."""

import sys
from pathlib import Path


# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Project information
project = "PRISM"
copyright = "2025, Omri"
author = "Omri"
release = "0.3.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

autosummary_generate = True

# Napoleon settings (NumPy/Google docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# HTML theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Suppress warnings for missing references
suppress_warnings = ["ref.python"]
