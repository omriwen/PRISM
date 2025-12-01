"""
Config Inheritance Tests for SPIDS Configuration

Tests the 'extends' keyword, path resolution, and deep merging.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prism.config.loader import _deep_merge_dicts, _resolve_config_path, load_config


class TestConfigInheritanceBasic:
    """Test basic config inheritance functionality."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create temporary config directory."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        return config_dir

    @pytest.fixture
    def parent_config(self, temp_config_dir):
        """Create a parent config file."""
        parent = temp_config_dir / "parent.yaml"
        parent.write_text(
            """
telescope:
  n_samples: 100
  fermat_sample: true
training:
  max_epochs: 10
  lr: 0.001
"""
        )
        return parent

    @pytest.fixture
    def child_config(self, temp_config_dir, parent_config):
        """Create a child config that extends parent."""
        child = temp_config_dir / "child.yaml"
        child.write_text(
            """
extends: parent.yaml
telescope:
  n_samples: 200
physics:
  obj_name: europa
"""
        )
        return child

    def test_extends_single_level(self, child_config):
        """Test single-level inheritance."""
        config = load_config(str(child_config))
        # Child overrides
        assert config.telescope.n_samples == 200
        # Parent values inherited
        assert config.telescope.fermat_sample
        assert config.training.max_epochs == 10
        # Child additions
        assert config.physics.obj_name == "europa"

    def test_extends_override_value(self, child_config):
        """Test child overrides parent value."""
        config = load_config(str(child_config))
        assert config.telescope.n_samples == 200  # Override

    def test_extends_preserves_parent(self, child_config):
        """Test non-overridden parent values preserved."""
        config = load_config(str(child_config))
        assert config.training.lr == 0.001  # From parent
        assert config.telescope.fermat_sample  # From parent

    def test_extends_deep_merge(self, child_config):
        """Test deep merge of nested dicts."""
        config = load_config(str(child_config))
        # telescope section has values from both parent and child
        assert config.telescope.n_samples == 200  # Child
        assert config.telescope.fermat_sample  # Parent


class TestConfigInheritancePaths:
    """Test config path resolution."""

    def test_resolve_relative_path(self, tmp_path):
        """Test resolving relative path."""
        # Create config files
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        parent = config_dir / "parent.yaml"
        parent.write_text("telescope:\n  n_samples: 100\nphysics:\n  obj_name: europa")

        # Resolve relative path from config_dir
        resolved = _resolve_config_path("parent.yaml", config_dir)
        assert resolved == parent.resolve()

    def test_resolve_nonexistent_fails(self):
        """Test helpful error when config doesn't exist."""
        with pytest.raises((FileNotFoundError, ValueError)):
            _resolve_config_path("nonexistent.yaml", Path("."))


class TestConfigInheritanceMultiLevel:
    """Test multi-level inheritance chains."""

    @pytest.fixture
    def three_level_configs(self, tmp_path):
        """Create A → B → C inheritance chain."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        # Level 1: Base
        base = config_dir / "base.yaml"
        base.write_text(
            """
telescope:
  n_samples: 50
  snr: 30
training:
  lr: 0.01
physics:
  obj_name: europa
"""
        )

        # Level 2: Middle (extends base)
        middle = config_dir / "middle.yaml"
        middle.write_text(
            """
extends: base.yaml
telescope:
  n_samples: 100
training:
  max_epochs: 20
"""
        )

        # Level 3: Top (extends middle)
        top = config_dir / "top.yaml"
        top.write_text(
            """
extends: middle.yaml
telescope:
  n_samples: 200
physics:
  obj_name: titan
"""
        )

        return {"base": base, "middle": middle, "top": top}

    def test_multi_level_inheritance(self, three_level_configs):
        """Test A → B → C inheritance chain."""
        config = load_config(str(three_level_configs["top"]))

        # Top level override
        assert config.telescope.n_samples == 200
        # Middle level override
        assert config.training.max_epochs == 20
        # Base level values
        assert config.telescope.snr == 30
        assert config.training.lr == 0.01
        # Top level addition
        assert config.physics.obj_name == "titan"

    def test_multi_level_override_order(self, three_level_configs):
        """Test rightmost (child) has highest priority."""
        config = load_config(str(three_level_configs["top"]))
        # n_samples defined in all 3 levels, top should win
        assert config.telescope.n_samples == 200  # From top, not 100 or 50


class TestConfigInheritanceEdgeCases:
    """Test edge cases and error handling."""

    def test_circular_inheritance_detected(self, tmp_path):
        """Test circular inheritance raises error."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        # Create A → B → A circular reference
        a = config_dir / "a.yaml"
        b = config_dir / "b.yaml"

        a.write_text("extends: b.yaml\nvalue: a\nphysics:\n  obj_name: europa")
        b.write_text("extends: a.yaml\nvalue: b\nphysics:\n  obj_name: europa")

        with pytest.raises(ValueError, match="[Cc]ircular"):
            load_config(str(a))

    def test_missing_parent_file(self, tmp_path):
        """Test missing parent file raises error."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        child = config_dir / "child.yaml"
        child.write_text("extends: nonexistent.yaml\nphysics:\n  obj_name: europa")

        with pytest.raises(FileNotFoundError):
            load_config(str(child))

    def test_extends_with_none_values(self, tmp_path):
        """Test None values handled correctly."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        parent = config_dir / "parent.yaml"
        parent.write_text(
            """
telescope:
  n_samples: 100
  snr: null
physics:
  obj_name: europa
"""
        )

        child = config_dir / "child.yaml"
        child.write_text(
            """
extends: parent.yaml
telescope:
  snr: 40
"""
        )

        config = load_config(str(child))
        assert config.telescope.snr == 40  # Child overrides None


class TestDeepMergeDicts:
    """Test deep dictionary merging utility."""

    def test_deep_merge_simple(self):
        """Test simple merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = _deep_merge_dicts(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self):
        """Test nested dict merge."""
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"b": 99}}

        result = _deep_merge_dicts(base, override)
        assert result == {"a": {"b": 99, "c": 2}}

    def test_deep_merge_deep_nesting(self):
        """Test deeply nested merge."""
        base = {"level1": {"level2": {"level3": {"a": 1, "b": 2}}}}
        override = {"level1": {"level2": {"level3": {"a": 99}}}}

        result = _deep_merge_dicts(base, override)
        assert result["level1"]["level2"]["level3"]["a"] == 99
        assert result["level1"]["level2"]["level3"]["b"] == 2

    def test_deep_merge_preserves_base(self):
        """Test that merge doesn't modify original base dict."""
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}

        result = _deep_merge_dicts(base, override)

        # Original base unchanged
        assert base["a"]["b"] == 1
        # Result has override
        assert result["a"]["b"] == 2
