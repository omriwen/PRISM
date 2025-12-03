"""Integration tests for natural language configuration."""
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    subprocess.run(["which", "ollama"], capture_output=True).returncode != 0,
    reason="Ollama not installed",
)
class TestNaturalLanguageIntegration:
    """Integration tests requiring Ollama."""

    def test_show_parse_only(self):
        """Test --show-parse-only flag exits cleanly."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--instruction",
                "train europa with lr 0.01",
                "--show-parse-only",
                "--auto-confirm",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0
        assert "europa" in result.stdout or "lr" in result.stdout
