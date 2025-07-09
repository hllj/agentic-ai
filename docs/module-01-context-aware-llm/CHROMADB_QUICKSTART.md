# ChromaDB Quickstart Guide

This guide will help you get started with ChromaDB using LangChain in your agentic AI project.

## Prerequisites

1. **Environment Setup**: Create a `.env` file in the project root with your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

2. **Install Dependencies**: 
   ```bash
   uv sync
   ```

## Running the Quickstart

### Basic Demonstration
Run the complete demonstration:
```bash
python quickstart_chromadb.py
```

### Interactive Mode
Start in interactive search mode:
```bash
python quickstart_chromadb.py --interactive
```

## What the Quickstart Demonstrates

### 1. Document Creation and Indexing
- Creates 8 sample documents about AI, ML, vector databases, ChromaDB, LangChain, RAG, NLP, and embeddings
- Automatically chunks documents based on your configuration
- Stores documents with metadata (title, category, author, difficulty)
- Generates embeddings using OpenAI's text-embedding-3-small model

### 2. Similarity Search
- Performs semantic similarity search on indexed documents
- Shows top-k results for various queries
- Demonstrates search with relevance scores
- Includes filtered search by metadata (e.g., category)

### 3. Retriever Interface
- Shows how to use ChromaDB as a LangChain retriever
- Useful for RAG (Retrieval-Augmented Generation) applications
- Configurable search parameters

### 4. Interactive Search
- Real-time search interface
- Test your own queries against the indexed documents
- Great for experimentation and understanding

## Sample Documents Included

The quickstart includes documents covering:
- **AI Fundamentals**: Introduction to artificial intelligence
- **Machine Learning**: ML basics and techniques
- **Vector Databases**: How vector databases work
- **ChromaDB**: Overview of ChromaDB features
- **LangChain**: Framework for LLM applications
- **RAG Systems**: Retrieval-Augmented Generation
- **NLP**: Natural Language Processing fundamentals
- **Embeddings**: Understanding vector embeddings

## Configuration

The quickstart uses your existing configuration from `src/configuration.py`:

- **Persist Directory**: `./data/chroma` (configurable via `CHROMA_PERSIST_DIR`)
- **Collection Name**: `documents` (configurable via `CHROMA_COLLECTION_NAME`)
- **Embedding Model**: `text-embedding-3-small` (configurable via `EMBEDDING_MODEL`)
- **Chunk Size**: `1000` characters (configurable via `CHUNK_SIZE`)
- **Chunk Overlap**: `200` characters (configurable via `CHUNK_OVERLAP`)

## Example Queries to Try

Once you run the quickstart, try these queries:

1. **"What is artificial intelligence?"**
   - Should return the AI introduction document

2. **"How do vector databases work?"**
   - Should return vector database and ChromaDB documents

3. **"Tell me about machine learning"**
   - Should return ML-related documents

4. **"What is RAG and how does it work?"**
   - Should return RAG and related documents

5. **"embedding and vector similarity"**
   - Should return documents about embeddings and vector operations

## Filtered Search Examples

The quickstart demonstrates filtered search:
- Filter by category: `technology`, `database`, `framework`
- Filter by difficulty: `beginner`, `intermediate`, `advanced`
- Filter by author: Various expert names

## File Structure

```
quickstart_chromadb.py          # Main quickstart script
src/
├── configuration.py            # Configuration management
├── vectordb/
│   └── chroma_client.py       # ChromaDB client wrapper
data/
└── chroma/                    # ChromaDB persistent storage
```

## Next Steps

After running the quickstart:

1. **Add Your Own Documents**: Modify the `SAMPLE_DOCUMENTS` list or create functions to load your own documents
2. **Build a RAG Application**: Use the retriever interface to build question-answering systems
3. **Experiment with Different Models**: Try different embedding models in your configuration
4. **Scale Up**: Add more documents and test performance
5. **Integration**: Integrate with your existing LangGraph workflows

## Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**
   ```
   Error: OPENAI_API_KEY is required
   ```
   Solution: Add your API key to the `.env` file

2. **ChromaDB Import Error**
   ```
   ModuleNotFoundError: No module named 'chromadb'
   ```
   Solution: Run `uv sync` to install dependencies

3. **Permission Errors**
   ```
   Permission denied: ./data/chroma
   ```
   Solution: Ensure the data directory is writable

4. **Network Issues**
   ```
   Connection timeout to OpenAI API
   ```
   Solution: Check your internet connection and API key

### Getting Help

If you encounter issues:
1. Check the logs for detailed error messages
2. Verify your `.env` file has the correct API key
3. Ensure all dependencies are installed with `uv sync`
4. Check that the data directory is writable

## Advanced Usage

### Custom Document Loading

```python
# Add your own documents
documents = [
    Document(
        page_content="Your document content here",
        metadata={"title": "My Document", "category": "custom"}
    )
]
doc_ids = chroma_client.add_documents(documents)
```

### Custom Search Parameters

```python
# Search with custom parameters
results = chroma_client.similarity_search(
    query="your query",
    k=10,  # More results
    filter={"category": "technology"}  # Filter by metadata
)
```

### Using as a Retriever

```python
# Get retriever for RAG applications
retriever = chroma_client.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Use with LangChain chains
docs = retriever.get_relevant_documents("your query")
```
