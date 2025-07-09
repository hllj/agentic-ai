# Agentic AI with LangGraph

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A comprehensive repository for building context-aware, multi-agent AI applications using LangGraph and modern LLM orchestration patterns.

## 🚀 Overview

This repository provides a complete learning path and practical implementation guide for developing sophisticated agentic AI systems. From basic context-aware applications to complex multi-agent workflows, learn to build production-ready AI agents that can reason, collaborate, and adapt.

## 📚 Table of Contents

- [Module Structure](#module-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Breakdown](#module-breakdown)
- [Examples](#examples)
- [Contributing](#contributing)
- [Resources](#resources)
- [License](#license)

## 🏗️ Module Structure

This repository is organized into 7 comprehensive modules, each building upon the previous:

```
agentic-ai/
├── docs/
│   ├── module-01-context-aware-llm/     # Context-aware applications & MCP
│   ├── module-02-vector-databases/      # Vector storage & retrieval
│   ├── module-03-multi-agent/          # Collaborative agent systems
│   ├── module-04-agentic-workflows/    # LangGraph fundamentals
│   ├── module-05-design-patterns/      # ReAct, Reflection, CodeAct
│   ├── module-06-interoperability/     # Cross-platform agent communication
│   ├── module-07-observability/        # Monitoring & debugging
│   └── tutorials/                      # Additional documentation
├── examples/                           # Practical implementations
├── tests/                             # Test suites
├── requirements.txt                   # Python dependencies
├── .env.example                      # Environment variables template
└── README.md                         # This file
```

## 🔧 Prerequisites

- **Python**: 3.9 or higher
- **API Keys**: OpenAI
- **Basic Knowledge**: Python programming, REST APIs
- **Optional**: Docker for containerized deployments

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/agentic-ai.git
cd agentic-ai
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

# Agentic AI Application

## Environment Setup

### 1. Environment Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Then edit `.env` with your actual values:

#### Required Configuration

- **OPENAI_API_KEY**: Your OpenAI API key
- **POSTGRES_***: PostgreSQL database connection details

#### Optional Configuration

- **VECTOR_DB_PROVIDER**: Vector database provider (default: chroma)
- **MEMORY_***: Memory system settings
- **DEBUG**: Enable debug mode (default: false)

### 2. Database Setup

#### PostgreSQL Setup

1. Install PostgreSQL locally or use a cloud provider
2. Create a database:
   ```sql
   CREATE DATABASE agentic_ai;
   ```
3. Update your `.env` file with database credentials

#### Database Tables

Tables will be created automatically when you first run the application if `AUTO_CREATE_TABLES=true` (default).

### 3. Vector Database Setup

ChromaDB will automatically create the necessary files in the directory specified by `CHROMA_PERSIST_DIR`.

### 4. Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Validate configuration
python -m src.configuration

# Initialize database
python -c "from src.database import init_database; init_database()"

# Run your application
python main.py
```

## Configuration Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | Yes |
| `POSTGRES_HOST` | PostgreSQL host | localhost | Yes* |
| `POSTGRES_USER` | PostgreSQL username | - | Yes* |
| `POSTGRES_PASSWORD` | PostgreSQL password | - | Yes* |
| `POSTGRES_DB` | PostgreSQL database name | agentic_ai | Yes* |
| `POSTGRES_URL` | Complete PostgreSQL URL | - | Alternative* |
| `CHROMA_PERSIST_DIR` | ChromaDB storage directory | ./data/chroma | No |
| `EMBEDDING_MODEL` | OpenAI embedding model | text-embedding-3-small | No |
| `MEMORY_BUFFER_SIZE` | Conversation memory buffer size | 10 | No |
| `USE_POSTGRES_MEMORY` | Enable PostgreSQL memory storage | true | No |

\* Either provide individual PostgreSQL parameters or `POSTGRES_URL`

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Resources

### Documentation
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

### Research Papers
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

### Community
- [LangChain Discord](https://discord.gg/langchain)
- [GitHub Discussions](https://github.com/hllj/agentic-ai/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LangChain team for the foundational framework
- LangGraph contributors for workflow orchestration
- The broader AI/ML community for continuous innovation

---

**Ready to build the future of AI agents?** Start with [Module I](./docs/module-01-context-aware-llm/) and begin your journey into agentic AI! 🚀

## TODO

- Update placeholder for all nodes.
- Update magic number in configuration.
- Update for each workflow.
- Setup context: ChromaDB, User Profile, Memory.