# Module 1: Building Context-Aware LLM Applications and MCP

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Understand and implement Model I/O patterns with prompts, responses, and parsers
- Build retrieval chains using various loaders and retrievers
- Implement different types of memory systems (buffer, summarization, vector-backed)
- Combine modules into coherent, state-aware workflows
- Get introduced to Model Context Protocol (MCP)
- Create hands-on applications demonstrating context-aware capabilities

## 📚 Module Overview

This module focuses on building LLM applications that can maintain and effectively use context over time. We'll explore the fundamental components that make applications "context-aware" and learn how to orchestrate them using LangChain and LangGraph.

### Key Components Covered:

1. **Model I/O Architecture**
   - Prompt engineering and templates
   - Response parsing and validation
   - Output formatting and structure

2. **Retrieval Systems**
   - Document loaders for various sources
   - Text splitters and chunking strategies
   - Vector retrievers and similarity search

3. **Memory Systems**
   - Buffer memory for recent context
   - Summarization memory for long conversations
   - Vector-backed memory for semantic retrieval

4. **Model Context Protocol (MCP)**
   - Introduction to MCP architecture
   - Building MCP-compatible components
   - Context sharing between models

## 🛠️ Technical Stack

- **LangChain**: Core framework for LLM applications
- **LangGraph**: Workflow orchestration and state management
- **OpenAI**: Primary LLM provider
- **Chroma**: Vector database for embeddings
- **Pydantic**: Data validation and parsing

## 📁 Module Structure

```
module-01-context-aware-llm/
├── README.md                    # This file
├── lecture/                     # Lecture materials
│   ├── 01-model-io.md          # Model I/O concepts
│   ├── 02-retrieval-chains.md  # Retrieval patterns
│   ├── 03-memory-systems.md    # Memory implementations
│   └── 04-mcp-intro.md         # Model Context Protocol
├── exercises/                   # Practical exercises
│   ├── exercise-01-basic-io.py
│   ├── exercise-02-retrieval.py
│   ├── exercise-03-memory.py
│   └── exercise-04-mcp.py
├── examples/                    # Complete implementations
│   ├── context_aware_chatbot/
│   ├── document_qa_system/
│   └── memory_enhanced_agent/
└── resources/                   # Additional materials
    ├── sample_data/
    ├── templates/
    └── reference_implementations/
```

## 🚀 Getting Started

### Prerequisites

Before starting this module, ensure you have:

1. Python 3.9+ installed
2. API keys for OpenAI
3. Basic understanding of Python and LLMs

### Setup

1. **Install dependencies**:
```bash
# From the project root
pip install -e .
```

2. **Set up environment variables**:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys:
OPENAI_API_KEY=your_openai_key_here
```

3. **Verify installation**:
```bash
python -c "import langchain, langgraph; print('Setup complete!')"
```

## 📖 Learning Path

### Session 1: Model I/O Fundamentals (45 minutes)
- **Lecture**: Understanding prompts, responses, and parsers
- **Discussion**: Best practices for prompt engineering
- **Exercise**: Build a structured output parser

### Session 2: Retrieval Chains (45 minutes)
- **Lecture**: Document loaders and retrieval strategies
- **Discussion**: Chunking strategies and their trade-offs
- **Exercise**: Implement a document Q&A system

### Session 3: Memory Systems (45 minutes)
- **Lecture**: Types of memory and their use cases
- **Discussion**: Memory management strategies
- **Exercise**: Build a conversation with memory

### Session 4: MCP Introduction (45 minutes)
- **Lecture**: Model Context Protocol overview
- **Discussion**: Context sharing patterns
- **Exercise**: Create an MCP-compatible component

## 🎯 Hands-on Exercises

### Exercise 1: Basic Model I/O
Create a structured conversation system with proper input validation and output parsing.

### Exercise 2: Document Retrieval
Build a document question-answering system with semantic search capabilities.

### Exercise 3: Memory Integration
Implement a chatbot that remembers conversation history and user preferences.

### Exercise 4: MCP Implementation
Create a simple MCP server that provides context to multiple models.

## 🔍 Assessment Criteria

By the end of this module, you should be able to demonstrate:

1. **Technical Proficiency**:
   - Implement all major memory types
   - Build working retrieval chains
   - Create proper prompt templates
   - Handle errors and edge cases

2. **Design Understanding**:
   - Choose appropriate memory types for use cases
   - Design efficient retrieval strategies
   - Structure context-aware workflows

3. **Best Practices**:
   - Follow LangChain/LangGraph patterns
   - Implement proper error handling
   - Write maintainable, documented code

## 📚 Additional Resources

### Documentation
- [LangChain Memory Guide](https://python.langchain.com/docs/modules/memory/)
- [LangChain Retrieval Guide](https://python.langchain.com/docs/modules/data_connection/)
- [Model Context Protocol Spec](https://modelcontextprotocol.io/)

### Research Papers
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [In-Context Learning and Prompt Engineering](https://arxiv.org/abs/2301.00234)

### Community Resources
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [Vector Database Comparison](https://github.com/chroma-core/chroma)

## 🎉 Next Steps

After completing this module, you'll be ready for:
- **Module 2**: Vector Databases and Advanced Retrieval
- **Module 3**: Multi-Agent Collaboration Systems
- **Module 4**: LangGraph Workflow Orchestration

---

**Ready to build context-aware applications?** Start with the [lecture materials](./lecture/) and then dive into the [hands-on exercises](./exercises/)! 🚀
