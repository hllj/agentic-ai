# Lecture 2: Configuration and Environment Setup

## Learning Objectives
- Set up a complete development environment for LLM applications
- Understand configuration management and best practices
- Configure OpenAI API and vector database connections
- Implement proper environment variable handling and security

## 1. Development Environment Setup

### 1.1 Prerequisites
- Python 3.11 or higher
- Git for version control
- Virtual environment manager (we're using `uv`)
- OpenAI API key
- (Optional) Vector database service account

### 1.2 Project Structure Overview
```
agentic-ai/
├── src/                    # Core application code
│   ├── configuration.py    # Configuration management
│   ├── states.py          # State definitions
│   ├── prompts.py         # Prompt templates
│   ├── utils.py           # Utility functions
│   ├── graph.py           # Workflow orchestration
│   ├── nodes/             # Processing nodes
│   └── tools/             # Available tools
├── docs/                  # Documentation and learning materials
├── tests/                 # Test suite
├── main.py               # Application entry point
├── pyproject.toml        # Project dependencies
└── .env                  # Environment variables (not in git)
```

## 2. Configuration Management Architecture

### 2.1 Configuration Classes
Our configuration system uses dataclasses for type safety and validation:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 30

@dataclass
class VectorDBConfig:
    provider: str = "chroma"  # chroma, pinecone, weaviate
    collection_name: str = "default"
    persist_directory: Optional[str] = None
    # Provider-specific settings
    api_key: Optional[str] = None
    environment: Optional[str] = None
    index_name: Optional[str] = None

@dataclass
class MemoryConfig:
    max_messages: int = 100
    summary_threshold: int = 20
    persist_path: Optional[str] = None
```

### 2.2 Environment Variable Mapping
```python
def get_config() -> Configuration:
    """Load configuration from environment variables with fallbacks."""
    return Configuration(
        openai=OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")) if os.getenv("OPENAI_MAX_TOKENS") else None,
            timeout=int(os.getenv("OPENAI_TIMEOUT", "30"))
        ),
        vector_db=VectorDBConfig(
            provider=os.getenv("VECTOR_DB_PROVIDER", "chroma"),
            collection_name=os.getenv("VECTOR_DB_COLLECTION", "default"),
            persist_directory=os.getenv("VECTOR_DB_PERSIST_DIR"),
            api_key=os.getenv("VECTOR_DB_API_KEY"),
            environment=os.getenv("VECTOR_DB_ENVIRONMENT"),
            index_name=os.getenv("VECTOR_DB_INDEX_NAME")
        ),
        memory=MemoryConfig(
            max_messages=int(os.getenv("MEMORY_MAX_MESSAGES", "100")),
            summary_threshold=int(os.getenv("MEMORY_SUMMARY_THRESHOLD", "20")),
            persist_path=os.getenv("MEMORY_PERSIST_PATH")
        )
    )
```

## 3. Setting Up Your Environment

### 3.1 Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd agentic-ai

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
uv pip install -e .
```

### 3.2 Environment Variables Configuration

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your settings
nano .env
```

Example `.env` file:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=1000
OPENAI_TIMEOUT=30

# Vector Database Configuration (ChromaDB - local)
VECTOR_DB_PROVIDER=chroma
VECTOR_DB_COLLECTION=agentic_ai_collection
VECTOR_DB_PERSIST_DIR=./data/chroma

# Alternative: Pinecone Configuration
# VECTOR_DB_PROVIDER=pinecone
# VECTOR_DB_API_KEY=your_pinecone_api_key
# VECTOR_DB_ENVIRONMENT=your_pinecone_environment
# VECTOR_DB_INDEX_NAME=your_index_name

# Memory Configuration
MEMORY_MAX_MESSAGES=100
MEMORY_SUMMARY_THRESHOLD=20
MEMORY_PERSIST_PATH=./data/memory

# Application Settings
LOG_LEVEL=INFO
DEBUG_MODE=false
```

### 3.3 Security Best Practices

1. **Never commit API keys to version control**
2. **Use environment variables for all sensitive data**
3. **Implement validation for configuration values**
4. **Use different configurations for development/production**

```python
def validate_config(config: Configuration) -> None:
    """Validate configuration and raise errors for invalid settings."""
    if not config.openai.api_key:
        raise ValueError("OpenAI API key is required")
    
    if config.openai.temperature < 0 or config.openai.temperature > 2:
        raise ValueError("OpenAI temperature must be between 0 and 2")
    
    if config.memory.max_messages < 1:
        raise ValueError("Memory max_messages must be positive")
    
    # Add more validation as needed
```

## 4. Testing Your Configuration

### 4.1 Configuration Validation Script
```python
# test_config.py
from src.configuration import get_config, validate_config

def test_configuration():
    """Test that configuration loads correctly."""
    try:
        config = get_config()
        validate_config(config)
        print("✅ Configuration loaded successfully!")
        print(f"OpenAI Model: {config.openai.model}")
        print(f"Vector DB Provider: {config.vector_db.provider}")
        print(f"Memory Max Messages: {config.memory.max_messages}")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_configuration()
```

### 4.2 OpenAI Connection Test
```python
# test_openai.py
import openai
from src.configuration import get_config

def test_openai_connection():
    """Test OpenAI API connection."""
    config = get_config()
    
    try:
        client = openai.OpenAI(api_key=config.openai.api_key)
        
        # Simple test call
        response = client.chat.completions.create(
            model=config.openai.model,
            messages=[{"role": "user", "content": "Say 'Hello, Configuration!'"}],
            max_tokens=10
        )
        
        print("✅ OpenAI connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

if __name__ == "__main__":
    test_openai_connection()
```

## 5. Advanced Configuration Patterns

### 5.1 Environment-Specific Configurations
```python
@dataclass
class Configuration:
    openai: OpenAIConfig
    vector_db: VectorDBConfig
    memory: MemoryConfig
    environment: str = "development"  # development, staging, production
    
    def is_development(self) -> bool:
        return self.environment == "development"
    
    def is_production(self) -> bool:
        return self.environment == "production"
```

### 5.2 Configuration Factory Pattern
```python
class ConfigurationFactory:
    @staticmethod
    def create_development_config() -> Configuration:
        """Create configuration optimized for development."""
        return Configuration(
            openai=OpenAIConfig(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model="gpt-3.5-turbo",  # Cheaper for development
                temperature=0.8,  # More creative for testing
                max_tokens=500  # Lower limits for cost control
            ),
            # ... other development-specific settings
            environment="development"
        )
    
    @staticmethod
    def create_production_config() -> Configuration:
        """Create configuration optimized for production."""
        return Configuration(
            openai=OpenAIConfig(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model="gpt-4",  # Better quality for production
                temperature=0.3,  # More deterministic
                max_tokens=2000  # Higher limits for full responses
            ),
            # ... other production-specific settings
            environment="production"
        )
```

## 6. Working with Different Vector Databases

### 6.1 ChromaDB (Local Development)
```python
# Configuration for local ChromaDB
VECTOR_DB_PROVIDER=chroma
VECTOR_DB_COLLECTION=my_collection
VECTOR_DB_PERSIST_DIR=./data/chroma
```

### 6.2 Pinecone (Production)
```python
# Configuration for Pinecone
VECTOR_DB_PROVIDER=pinecone
VECTOR_DB_API_KEY=your_pinecone_key
VECTOR_DB_ENVIRONMENT=us-west1-gcp
VECTOR_DB_INDEX_NAME=agentic-ai-index
```

### 6.3 Weaviate (Enterprise)
```python
# Configuration for Weaviate
VECTOR_DB_PROVIDER=weaviate
VECTOR_DB_API_KEY=your_weaviate_key
VECTOR_DB_URL=https://your-cluster.weaviate.network
```

## 7. Configuration Best Practices

1. **Use type hints and dataclasses** for configuration structures
2. **Implement validation** for all configuration values
3. **Provide sensible defaults** where appropriate
4. **Document all configuration options** clearly
5. **Use environment-specific configurations** for different deployment stages
6. **Keep sensitive data in environment variables** only
7. **Test your configuration** regularly in different environments

## 8. Troubleshooting Common Issues

### 8.1 Missing API Keys
```bash
Error: OpenAI API key is required
Solution: Set OPENAI_API_KEY in your .env file
```

### 8.2 Invalid Configuration Values
```bash
Error: OpenAI temperature must be between 0 and 2
Solution: Check OPENAI_TEMPERATURE value in .env
```

### 8.3 Import Errors
```bash
Error: No module named 'src'
Solution: Install the package in development mode: uv pip install -e .
```

## 9. Next Steps

In the next lecture, we'll:
- Implement our first context-aware workflow
- Create a simple RAG (Retrieval-Augmented Generation) system
- Explore state management patterns
- Build a working chatbot with memory

## Hands-On Exercise

1. **Set up your environment** following the steps above
2. **Create your `.env` file** with your OpenAI API key
3. **Run the configuration tests** to verify everything works
4. **Try different configuration values** and see how they affect the system
5. **Explore the main application**: `python main.py --help`
