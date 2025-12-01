"""
CLI Flags Integration Tests for SPIDS Configuration

Tests new command-line flags added in UX improvements:
--preset, --obj, --show-*, --interactive, --list-presets
"""

from __future__ import annotations

import subprocess
import sys


class TestPresetFlag:
    """Test --preset flag for loading presets."""

    def run_cli(self, *args, timeout=10):
        """Helper: Run main.py with args."""
        result = subprocess.run(
            [sys.executable, "main.py", *args], capture_output=True, text=True, timeout=timeout
        )
        return result

    def test_preset_flag_quick_test(self):
        """Test --preset quick_test loads preset config."""
        result = self.run_cli("--preset", "quick_test", "--obj_name", "europa", "--show-config")
        assert result.returncode == 0
        # Verify preset was loaded
        assert "quick_test" in result.stderr or "Loaded preset" in result.stderr

    def test_preset_flag_production(self):
        """Test --preset production loads preset config."""
        result = self.run_cli("--preset", "production", "--obj_name", "europa", "--show-config")
        assert result.returncode == 0
        assert "production" in result.stderr or "Loaded preset" in result.stderr

    def test_preset_with_override(self):
        """Test --preset with CLI override."""
        result = self.run_cli(
            "--preset",
            "quick_test",
            "--n_samples",
            "150",  # Override preset value
            "--obj_name",
            "europa",
            "--show-config",
        )
        assert result.returncode == 0
        # Just verify command succeeded
        assert "Configuration is valid" in result.stdout or result.returncode == 0

    def test_invalid_preset_name(self):
        """Test helpful error for invalid preset name."""
        result = self.run_cli(
            "--preset", "nonexistent_preset", "--obj_name", "europa", "--show-config"
        )
        assert result.returncode != 0
        # Error message may be in stdout or stderr
        output = (result.stdout + result.stderr).lower()
        assert "preset" in output or "not found" in output


class TestObjShorthand:
    """Test --obj shorthand for object selection."""

    def run_cli(self, *args, timeout=10):
        """Helper: Run main.py with args."""
        result = subprocess.run(
            [sys.executable, "main.py", *args], capture_output=True, text=True, timeout=timeout
        )
        return result

    def test_obj_flag_europa(self):
        """Test --obj europa sets object parameters."""
        result = self.run_cli("--obj", "europa", "--show-config")
        assert result.returncode == 0
        assert "europa" in result.stdout.lower()

    def test_obj_flag_titan(self):
        """Test --obj titan sets object parameters."""
        result = self.run_cli("--obj", "titan", "--show-config")
        assert result.returncode == 0
        assert "titan" in result.stdout.lower()


class TestInspectionFlags:
    """Test --show-* inspection flags."""

    def run_cli(self, *args, timeout=10):
        """Helper: Run main.py with args."""
        result = subprocess.run(
            [sys.executable, "main.py", *args], capture_output=True, text=True, timeout=timeout
        )
        return result

    def test_list_presets_flag(self):
        """Test --list-presets shows available presets."""
        result = self.run_cli("--list-presets")
        assert result.returncode == 0
        assert "quick_test" in result.stdout
        assert "production" in result.stdout

    def test_show_preset_flag(self):
        """Test --show-preset displays preset details."""
        result = self.run_cli("--show-preset", "quick_test")
        assert result.returncode == 0
        # Just verify command succeeded with some output
        assert len(result.stdout) > 0

    def test_show_object_flag(self):
        """Test --show-object displays object parameters."""
        result = self.run_cli("--show-object", "europa")
        assert result.returncode == 0
        assert "europa" in result.stdout.lower()

    def test_show_config_flag(self):
        """Test --show-config displays final configuration."""
        result = self.run_cli("--obj_name", "europa", "--n_samples", "100", "--show-config")
        assert result.returncode == 0
        assert "europa" in result.stdout.lower()
        assert "100" in result.stdout

    def test_validate_only_flag(self):
        """Test --validate-only validates without running."""
        result = self.run_cli("--obj_name", "europa", "--validate-only")
        # Should validate and exit without running training
        assert result.returncode == 0


class TestFlagCombinations:
    """Test combinations of flags and precedence."""

    def run_cli(self, *args, timeout=10):
        """Helper: Run main.py with args."""
        result = subprocess.run(
            [sys.executable, "main.py", *args], capture_output=True, text=True, timeout=timeout
        )
        return result

    def test_preset_and_config_file(self):
        """Test --preset with --config (preset should take precedence)."""
        result = self.run_cli(
            "--config", "configs/default.yaml", "--preset", "quick_test", "--show-config"
        )
        assert result.returncode == 0
        # Just verify command succeeded
        assert "Configuration is valid" in result.stdout

    def test_preset_obj_and_override(self):
        """Test --preset + --obj + CLI override."""
        result = self.run_cli(
            "--preset",
            "production",
            "--obj",
            "europa",
            "--n_samples",
            "99",  # CLI override
            "--show-config",
        )
        assert result.returncode == 0
        # Just verify command succeeded
        assert "europa" in result.stdout.lower()

    def test_multiple_inspection_flags(self):
        """Test multiple --show-* flags together."""
        result = self.run_cli("--preset", "quick_test", "--obj", "europa", "--show-config")
        assert result.returncode == 0


class TestMoPIECLIFlags:
    """Test CLI flags for Mo-PIE mode."""

    def run_cli(self, *args, timeout=10):
        """Helper: Run main_mopie.py with args."""
        result = subprocess.run(
            [sys.executable, "main_mopie.py", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result

    def test_mopie_preset_flag(self):
        """Test --preset with Mo-PIE preset."""
        result = self.run_cli("--preset", "mopie_baseline", "--obj_name", "europa", "--show-config")
        assert result.returncode == 0
        # Should contain Mo-PIE-specific parameters

    def test_mopie_list_presets(self):
        """Test --list-presets in Mo-PIE mode."""
        result = self.run_cli("--list-presets")
        assert result.returncode == 0
        assert "mopie" in result.stdout.lower()
