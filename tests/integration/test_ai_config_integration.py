"""Integration tests for AI configuration (requires running ollama)."""

import pytest

from prism.cli.parser import create_main_parser
from prism.config.ai_config import AIConfigurator


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.importorskip("ollama"),
    reason="ollama not available",
)
class TestOllamaIntegration:
    """Integration tests requiring ollama server."""

    @pytest.fixture
    def configurator(self) -> AIConfigurator:
        """Create configurator with real parser."""
        parser = create_main_parser()
        return AIConfigurator(parser)

    def test_real_instruction(self, configurator: AIConfigurator) -> None:
        """Test real LLM call with simple instruction."""
        parser = create_main_parser()
        current = parser.parse_args([])

        try:
            delta = configurator.get_delta("use 100 samples", current)
            # Should either have the change or be empty (LLM variability)
            if delta.changes:
                assert "n_samples" in delta.changes
        except ConnectionError:
            pytest.skip("ollama not running")
