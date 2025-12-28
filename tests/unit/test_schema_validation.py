"""Unit tests for Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from yourbench.conf.schema import (
    ModelConfig,
    ChunkingConfig,
    CrossDocConfig,
    SummarizationConfig,
    CitationFilteringConfig,
)


class TestSchemaValidation:
    """Tests for config schema validation rules."""

    def test_summarization_valid_and_invalid(self):
        """SummarizationConfig validates token constraints."""
        cfg = SummarizationConfig(max_tokens=1000, token_overlap=100)
        assert cfg.max_tokens == 1000

        with pytest.raises(ValidationError, match="max_tokens must be > 0"):
            SummarizationConfig(max_tokens=0)
        with pytest.raises(ValidationError, match="token_overlap must be >= 0"):
            SummarizationConfig(token_overlap=-1)
        with pytest.raises(ValidationError, match="token_overlap.*must be < max_tokens"):
            SummarizationConfig(max_tokens=100, token_overlap=100)

    def test_chunking_valid_and_invalid(self):
        """ChunkingConfig validates heading level constraints."""
        cfg = ChunkingConfig(h_min=2, h_max=5)
        assert cfg.h_min == 2

        with pytest.raises(ValidationError, match="h_min must be >= 1"):
            ChunkingConfig(h_min=0)
        with pytest.raises(ValidationError, match="h_max.*must be >= h_min"):
            ChunkingConfig(h_min=5, h_max=2)

    def test_cross_doc_valid_and_invalid(self):
        """CrossDocConfig validates document count constraints."""
        cfg = CrossDocConfig(num_docs_per_combination=[2, 5])
        assert cfg.max_combinations == 100

        with pytest.raises(ValidationError, match="2 elements"):
            CrossDocConfig(num_docs_per_combination=[2])
        with pytest.raises(ValidationError, match="must be >= 2"):
            CrossDocConfig(num_docs_per_combination=[1, 5])
        with pytest.raises(ValidationError, match="must be >="):
            CrossDocConfig(num_docs_per_combination=[5, 2])

    def test_model_config_valid_and_invalid(self):
        """ModelConfig validates concurrency constraints."""
        cfg = ModelConfig(model_name="test", max_concurrent_requests=16)
        assert cfg.max_concurrent_requests == 16

        with pytest.raises(ValidationError, match="max_concurrent_requests must be >= 1"):
            ModelConfig(max_concurrent_requests=0)

    def test_citation_filtering_valid_and_invalid(self):
        """CitationFilteringConfig validates alpha/beta range."""
        cfg = CitationFilteringConfig(alpha=0.5, beta=0.5)
        assert cfg.alpha == 0.5

        with pytest.raises(ValidationError, match="alpha must be in"):
            CitationFilteringConfig(alpha=1.5)
        with pytest.raises(ValidationError, match="beta must be in"):
            CitationFilteringConfig(beta=-0.1)
