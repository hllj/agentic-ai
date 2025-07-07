"""
Configuration module for the Agentic AI application.

This module provides centralized configuration management for all components
including API keys, model settings, and application parameters.
"""

import os
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()


@dataclass
class OpenAIConfig:
    """OpenAI specific configuration settings."""
    
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("OPENAI_MAX_TOKENS", "4000")))
    timeout: int = field(default_factory=lambda: int(os.getenv("OPENAI_TIMEOUT", "60")))


@dataclass
class VectorDBConfig:
    """Vector database configuration settings."""
    
    provider: str = field(default_factory=lambda: os.getenv("VECTOR_DB_PROVIDER", "chroma"))
    persist_directory: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "documents"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))


@dataclass
class MemoryConfig:
    """Memory system configuration settings."""
    
    buffer_size: int = field(default_factory=lambda: int(os.getenv("MEMORY_BUFFER_SIZE", "10")))
    summary_max_tokens: int = field(default_factory=lambda: int(os.getenv("MEMORY_SUMMARY_MAX_TOKENS", "1000")))
    vector_memory_k: int = field(default_factory=lambda: int(os.getenv("MEMORY_VECTOR_K", "5")))


@dataclass
class ApplicationConfig:
    """Main application configuration."""
    
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    data_directory: str = field(default_factory=lambda: os.getenv("DATA_DIRECTORY", "./data"))
    
    # Component configurations
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


# Global configuration instance
config = ApplicationConfig()


def get_config() -> ApplicationConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> None:
    """Update configuration with new values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)


def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration as dictionary."""
    return {
        "api_key": config.openai.api_key,
        "model": config.openai.model,
        "temperature": config.openai.temperature,
        "max_tokens": config.openai.max_tokens,
        "timeout": config.openai.timeout,
    }


def get_vector_db_config() -> Dict[str, Any]:
    """Get vector database configuration as dictionary."""
    return {
        "provider": config.vector_db.provider,
        "persist_directory": config.vector_db.persist_directory,
        "collection_name": config.vector_db.collection_name,
        "embedding_model": config.vector_db.embedding_model,
        "chunk_size": config.vector_db.chunk_size,
        "chunk_overlap": config.vector_db.chunk_overlap,
    }


def get_memory_config() -> Dict[str, Any]:
    """Get memory configuration as dictionary."""
    return {
        "buffer_size": config.memory.buffer_size,
        "summary_max_tokens": config.memory.summary_max_tokens,
        "vector_memory_k": config.memory.vector_memory_k,
    }


# Validation function to ensure all required settings are present
def validate_config() -> None:
    """Validate that all required configuration is present."""
    try:
        # Test OpenAI API key
        if not config.openai.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        # Create data directory if it doesn't exist
        os.makedirs(config.data_directory, exist_ok=True)
        os.makedirs(config.vector_db.persist_directory, exist_ok=True)
        
        print("✅ Configuration validation passed")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        raise


if __name__ == "__main__":
    # Print current configuration for debugging
    print("Current Configuration:")
    print(f"Environment: {config.environment}")
    print(f"Debug: {config.debug}")
    print(f"Log Level: {config.log_level}")
    print(f"OpenAI Model: {config.openai.model}")
    print(f"Vector DB Provider: {config.vector_db.provider}")
    print(f"Data Directory: {config.data_directory}")
    
    # Validate configuration
    validate_config()
