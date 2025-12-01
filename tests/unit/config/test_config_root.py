"""Tests for configuration dataclasses."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pytest

from prism.config.base import TelescopeConfig


def test_pattern_fn_configuration():
    """Test new pattern_fn field."""
    # Custom pattern
    config = TelescopeConfig(pattern_fn="/path/to/pattern.py")
    assert config.pattern_fn == "/path/to/pattern.py"

    # Builtin pattern
    config = TelescopeConfig(pattern_fn="builtin:fermat")
    assert config.pattern_fn == "builtin:fermat"


def test_legacy_pattern_flags_deprecated():
    """Test legacy flags show deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = TelescopeConfig(fermat_sample=True)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()

        # Should auto-convert
        assert config.pattern_fn == "builtin:fermat"


def test_default_pattern():
    """Test default pattern is random."""
    config = TelescopeConfig()
    assert config.pattern_fn == "builtin:random"


def test_star_sample_auto_conversion():
    """Test star_sample flag auto-converts to pattern_fn."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        config = TelescopeConfig(star_sample=True)
        assert config.pattern_fn == "builtin:star"


def test_pattern_fn_overrides_legacy():
    """Test that pattern_fn takes precedence over legacy flags."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        # When pattern_fn is set, legacy flags shouldn't override it
        config = TelescopeConfig(pattern_fn="builtin:random", fermat_sample=True)
        # pattern_fn should remain unchanged
        assert config.pattern_fn == "builtin:random"


def test_builtin_pattern_validation():
    """Test validation of builtin pattern names."""
    from prism.config.base import PhysicsConfig, PRISMConfig, TelescopeConfig

    # Valid builtin pattern
    config = PRISMConfig(
        physics=PhysicsConfig(obj_name="europa"),
        telescope=TelescopeConfig(pattern_fn="builtin:fermat"),
    )
    config.validate()  # Should not raise

    # Invalid builtin pattern
    config = PRISMConfig(
        physics=PhysicsConfig(obj_name="europa"),
        telescope=TelescopeConfig(pattern_fn="builtin:invalid"),
    )
    with pytest.raises(ValueError, match="Unknown builtin pattern"):
        config.validate()


def test_pattern_file_validation():
    """Test validation of pattern file paths."""
    from prism.config.base import PhysicsConfig, PRISMConfig, TelescopeConfig

    # Non-existent file
    config = PRISMConfig(
        physics=PhysicsConfig(obj_name="europa"),
        telescope=TelescopeConfig(pattern_fn="/nonexistent/pattern.py"),
    )
    with pytest.raises(ValueError, match="Pattern file not found"):
        config.validate()

    # Non-python file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        temp_path = f.name

    try:
        config = PRISMConfig(
            physics=PhysicsConfig(obj_name="europa"),
            telescope=TelescopeConfig(pattern_fn=temp_path),
        )
        with pytest.raises(ValueError, match="must be a .py file"):
            config.validate()
    finally:
        Path(temp_path).unlink()

    # Valid python file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def generate_pattern(config):\n    pass\n")
        temp_path = f.name

    try:
        config = PRISMConfig(
            physics=PhysicsConfig(obj_name="europa"),
            telescope=TelescopeConfig(pattern_fn=temp_path),
        )
        config.validate()  # Should not raise
    finally:
        Path(temp_path).unlink()


def test_both_legacy_flags_raises_error():
    """Test that using both star_sample and fermat_sample raises error."""
    from prism.config.base import PhysicsConfig, PRISMConfig, TelescopeConfig

    config = PRISMConfig(
        physics=PhysicsConfig(obj_name="europa"),
        telescope=TelescopeConfig(star_sample=True, fermat_sample=True),
    )
    with pytest.raises(ValueError, match="Cannot use both"):
        config.validate()
