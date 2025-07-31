"""
Configuration settings for the Document QA System with Gemini 2.5 Flash support.
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for AI models."""
    embedding_model: str = "models/embedding-001"  # Proper Gemini embedding model
    llm_model: str = "models/gemini-2.5-flash"     # Updated to use Gemini 2.5 Flash by default
    temperature: float = 0.2
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("API_KEY")
            if not self.api_key:
                raise EnvironmentError("Missing API_KEY in environment variables or .env file.")
        os.environ["GOOGLE_API_KEY"] = self.api_key

@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    chunk_size: int = 1024
    chunk_overlap: int = 100
    use_ocr: bool = True
    crawl_links: bool = True
    max_links_to_crawl: int = 5
    structured_data_threshold: int = 5

@dataclass
class QueryConfig:
    """Configuration for query processing."""
    similarity_top_k: int = 5
    similarity_cutoff: float = 0.7

@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    processing: ProcessingConfig
    query: QueryConfig
    log_level: str = "INFO"
    log_file: str = "app.log"
    
    @classmethod
    def default(cls) -> 'AppConfig':
        """Create default configuration."""
        return cls(
            model=ModelConfig(),
            processing=ProcessingConfig(),
            query=QueryConfig()
        )

# Global configuration instance
config = AppConfig.default()
