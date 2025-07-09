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
class PostgresConfig:
    """PostgreSQL database configuration for conversation history and user preferences."""
    
    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    database: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "agentic_ai"))
    username: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", ""))
    password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    url: str = field(default_factory=lambda: os.getenv("POSTGRES_URL", ""))
    pool_size: int = field(default_factory=lambda: int(os.getenv("POSTGRES_POOL_SIZE", "5")))
    max_overflow: int = field(default_factory=lambda: int(os.getenv("POSTGRES_MAX_OVERFLOW", "10")))

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
class DatabaseConfig:
    """Database configuration for table management and migrations."""
    
    auto_create_tables: bool = field(default_factory=lambda: os.getenv("AUTO_CREATE_TABLES", "true").lower() == "true")
    drop_tables_on_start: bool = field(default_factory=lambda: os.getenv("DROP_TABLES_ON_START", "false").lower() == "true")
    enable_migrations: bool = field(default_factory=lambda: os.getenv("ENABLE_MIGRATIONS", "true").lower() == "true")

@dataclass
class MemoryConfig:
    """Memory system configuration settings."""
    
    buffer_size: int = field(default_factory=lambda: int(os.getenv("MEMORY_BUFFER_SIZE", "10")))
    summary_max_tokens: int = field(default_factory=lambda: int(os.getenv("MEMORY_SUMMARY_MAX_TOKENS", "1000")))
    vector_memory_k: int = field(default_factory=lambda: int(os.getenv("MEMORY_VECTOR_K", "5")))
    use_postgres: bool = field(default_factory=lambda: os.getenv("USE_POSTGRES_MEMORY", "true").lower() == "true")

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
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)


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
        "use_postgres": config.memory.use_postgres,
    }


def get_postgres_config() -> Dict[str, Any]:
    """Get PostgreSQL configuration as dictionary."""
    return {
        "host": config.postgres.host,
        "port": config.postgres.port,
        "database": config.postgres.database,
        "username": config.postgres.username,
        "password": config.postgres.password,
        "url": config.postgres.url,
        "pool_size": config.postgres.pool_size,
        "max_overflow": config.postgres.max_overflow,
    }


def get_database_config() -> Dict[str, Any]:
    """Get database management configuration as dictionary."""
    return {
        "auto_create_tables": config.database.auto_create_tables,
        "drop_tables_on_start": config.database.drop_tables_on_start,
        "enable_migrations": config.database.enable_migrations,
    }


def get_database_url() -> str:
    """Get database URL for SQLAlchemy."""
    if config.postgres.url:
        return config.postgres.url
    else:
        return (
            f"postgresql://{config.postgres.username}:"
            f"{config.postgres.password}@"
            f"{config.postgres.host}:{config.postgres.port}/"
            f"{config.postgres.database}"
        )


# Validation function to ensure all required settings are present
def validate_config() -> None:
    """Validate that all required configuration is present."""
    try:
        # Test OpenAI API key
        if not config.openai.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        # Validate PostgreSQL configuration
        if config.memory.use_postgres:
            if not config.postgres.url:
                if not all([config.postgres.host, config.postgres.database, 
                           config.postgres.username, config.postgres.password]):
                    raise ValueError("PostgreSQL configuration is incomplete. Either provide POSTGRES_URL or all individual parameters (host, database, username, password)")
        
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
