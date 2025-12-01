"""
Backward Compatibility Tests for SPIDS Configuration System

Ensures all pre-existing usage patterns work unchanged after UX improvements.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from prism.config import PRISMConfig, load_config, merge_config_with_args


class TestBackwardCompatibilityConfigs:
    """Verify all legacy config files still load correctly."""

    def test_legacy_config_default_yaml(self):
        """Test configs/default.yaml loads (pre-refactor config)."""
        config = load_config("configs/default.yaml")
        assert isinstance(config, PRISMConfig)
        assert config.image.image_size == 1024

    def test_legacy_config_quick_test_yaml(self):
        """Test configs/quick_test.yaml loads."""
        config = load_config("configs/quick_test.yaml")
        assert config.telescope.n_samples == 64
        assert not config.save_data

    def test_legacy_config_production_europa_yaml(self):
        """Test configs/production_europa.yaml loads."""
        config = load_config("configs/production_europa.yaml")
        assert config.physics.obj_name == "europa"
        assert config.telescope.n_samples == 240

    def test_legacy_config_mopie_example_yaml(self):
        """Test configs/mopie_example.yaml loads with Mo-PIE config."""
        config = load_config("configs/mopie_example.yaml")
        assert config.mopie is not None
        assert config.mopie.lr_obj == 1.0

    def test_legacy_config_point_source_yaml(self):
        """Test configs/point_source_test.yaml loads."""
        config = load_config("configs/point_source_test.yaml")
        assert config.point_source.is_point_source


class TestBackwardCompatibilityCLI:
    """Verify legacy CLI patterns still work."""

    def run_main_cli(self, *args):
        """Helper: Run main.py with args, capture output."""
        result = subprocess.run(
            [sys.executable, "main.py", *args], capture_output=True, text=True, timeout=10
        )
        return result

    def test_legacy_cli_basic_pattern(self):
        """Test: python main.py --obj_name europa --n_samples 100 --fermat"""
        result = self.run_main_cli(
            "--obj_name",
            "europa",
            "--n_samples",
            "100",
            "--fermat",
            "--show-config",  # Don't actually run, just show config
        )
        assert result.returncode == 0
        assert "europa" in result.stdout

    def test_legacy_cli_with_config_file(self):
        """Test: python main.py --config configs/default.yaml"""
        result = self.run_main_cli("--config", "configs/default.yaml", "--show-config")
        assert result.returncode == 0

    def test_legacy_cli_debug_flag(self):
        """Test: --debug flag (sets save_data=False)"""
        result = self.run_main_cli("--obj_name", "europa", "--debug", "--show-config")
        assert result.returncode == 0
        assert "save_data" in result.stdout.lower()


class TestBackwardCompatibilityPriority:
    """Verify priority order: CLI > Config > Defaults still works."""

    def test_priority_cli_overrides_config(self):
        """Verify explicit CLI args override config file values."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--config",
                "configs/quick_test.yaml",  # n_samples=64
                "--n_samples",
                "200",  # Override to 200
                "--show-config",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "200" in result.stdout  # CLI value wins

    def test_priority_config_overrides_defaults(self):
        """Verify config file values override argparse defaults."""
        import argparse

        from prism.config import load_config

        # Load config with n_samples=64
        config = load_config("configs/quick_test.yaml")

        # Create args with default n_samples=200
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_samples", type=int, default=200)
        args = parser.parse_args([])  # No CLI args

        # Merge should preserve config value (64), not default (200)
        merged = merge_config_with_args(config, args, cli_provided_args=set())
        assert merged.n_samples == 64  # Config wins


class TestBackwardCompatibilityAPI:
    """Verify programmatic API unchanged."""

    def test_legacy_load_config_signature(self):
        """Verify load_config(path) signature unchanged."""
        import inspect

        from prism.config import load_config

        sig = inspect.signature(load_config)
        # Should accept path as string
        params = list(sig.parameters.keys())
        assert len(params) >= 1  # At least one parameter

    def test_legacy_save_config_signature(self):
        """Verify save_config(config, path) signature."""
        import inspect

        from prism.config import save_config

        sig = inspect.signature(save_config)
        params = list(sig.parameters.keys())
        assert "config" in params
        assert "output_path" in params

    def test_legacy_args_to_config(self):
        """Verify args_to_config() still works."""
        import argparse

        from prism.config import args_to_config

        parser = argparse.ArgumentParser()
        parser.add_argument("--obj_name", default="europa")
        parser.add_argument("--name", default="test")
        args = parser.parse_args([])

        config = args_to_config(args)
        assert config.physics.obj_name == "europa"

    def test_legacy_dataclass_imports(self):
        """Verify all config classes still importable."""
        try:
            from prism.config import (  # noqa: F401
                ImageConfig,
                ModelConfig,
                MoPIEConfig,
                PhysicsConfig,
                PointSourceConfig,
                PRISMConfig,
                TelescopeConfig,
                TrainingConfig,
            )

            # All imports successful
            assert True
        except ImportError as e:
            pytest.fail(f"Config class import failed: {e}")
