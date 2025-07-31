"""
Unit tests for configuration module.
"""
import pytest
import os
from unittest.mock import patch

from src.config.settings import AppConfig, ModelConfig, ProcessingConfig, QueryConfig


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.embedding_model == "BAAI/bge-small-en-v1.5"
        assert config.llm_model == "models/gemini-1.5-flash"
        assert config.temperature == 0.2
    
    @patch.dict(os.environ, {'API_KEY': 'test_key'})
    def test_api_key_from_env(self):
        """Test API key loading from environment."""
        config = ModelConfig()
        assert config.api_key == 'test_key'
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError):
                ModelConfig()


class TestProcessingConfig:
    """Test ProcessingConfig class."""
    
    def test_default_values(self):
        """Test default processing configuration."""
        config = ProcessingConfig()
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100
        assert config.use_ocr is True
        assert config.crawl_links is True
        assert config.max_links_to_crawl == 5


class TestQueryConfig:
    """Test QueryConfig class."""
    
    def test_default_values(self):
        """Test default query configuration."""
        config = QueryConfig()
        assert config.similarity_top_k == 5
        assert config.similarity_cutoff == 0.7


class TestAppConfig:
    """Test AppConfig class."""
    
    @patch.dict(os.environ, {'API_KEY': 'test_key'})
    def test_default_config(self):
        """Test default app configuration."""
        config = AppConfig.default()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.query, QueryConfig)
        assert config.log_level == "INFO"
        assert config.log_file == "app.log"
