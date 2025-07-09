#!/usr/bin/env python3
"""
ChromaDB Quickstart with LangChain

This script demonstrates how to use ChromaDB with LangChain for:
1. Creating and indexing documents
2. Performing similarity searches
3. Retrieving relevant documents based on queries

Prerequisites:
- Set up your .env file with OPENAI_API_KEY
- Install dependencies: uv sync
"""

import os
import sys
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.vectordb.chroma_client import get_chroma_client
from src.configuration import get_config, validate_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample documents to demonstrate ChromaDB functionality
SAMPLE_DOCUMENTS = [
    {
        "content": """
        Artificial Intelligence (AI) is a branch of computer science that aims to create 
        intelligent machines that can think, learn, and adapt like humans. AI systems use 
        various techniques including machine learning, deep learning, natural language 
        processing, and computer vision to solve complex problems.
        """,
        "metadata": {
            "title": "Introduction to AI",
            "category": "technology",
            "author": "AI Expert",
            "difficulty": "beginner"
        }
    },
    {
        "content": """
        Machine Learning (ML) is a subset of artificial intelligence that enables computers 
        to learn and improve from experience without being explicitly programmed. ML algorithms 
        use statistical methods to analyze data, identify patterns, and make predictions or 
        decisions. Common types include supervised learning, unsupervised learning, and 
        reinforcement learning.
        """,
        "metadata": {
            "title": "Machine Learning Basics",
            "category": "technology",
            "author": "ML Researcher",
            "difficulty": "intermediate"
        }
    },
    {
        "content": """
        Vector databases are specialized databases designed to store and query high-dimensional 
        vector embeddings. They are essential for applications like semantic search, 
        recommendation systems, and RAG (Retrieval-Augmented Generation) systems. Vector 
        databases use similarity search algorithms like cosine similarity or euclidean distance 
        to find relevant documents.
        """,
        "metadata": {
            "title": "Vector Databases Explained",
            "category": "database",
            "author": "Database Expert",
            "difficulty": "intermediate"
        }
    },
    {
        "content": """
        ChromaDB is an open-source vector database that makes it easy to build AI applications 
        with embeddings. It features a simple Python API, supports multiple embedding models, 
        and provides efficient similarity search capabilities. ChromaDB can be used as an 
        in-memory database or with persistent storage.
        """,
        "metadata": {
            "title": "ChromaDB Overview",
            "category": "database",
            "author": "ChromaDB Team",
            "difficulty": "beginner"
        }
    },
    {
        "content": """
        LangChain is a framework for developing applications powered by language models. 
        It provides tools for chaining together different components like prompt templates, 
        language models, and output parsers. LangChain also includes integrations with 
        various vector databases, APIs, and other tools commonly used in AI applications.
        """,
        "metadata": {
            "title": "LangChain Framework",
            "category": "framework",
            "author": "LangChain Developers",
            "difficulty": "intermediate"
        }
    },
    {
        "content": """
        Retrieval-Augmented Generation (RAG) is a technique that combines the power of 
        large language models with external knowledge sources. RAG systems first retrieve 
        relevant information from a knowledge base (usually stored in a vector database), 
        then use that information to generate accurate and contextually relevant responses.
        """,
        "metadata": {
            "title": "RAG Systems",
            "category": "technology",
            "author": "RAG Specialist",
            "difficulty": "advanced"
        }
    },
    {
        "content": """
        Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
        between computers and human language. NLP techniques enable computers to understand, 
        interpret, and generate human language in a valuable way. Common NLP tasks include 
        sentiment analysis, named entity recognition, and text summarization.
        """,
        "metadata": {
            "title": "NLP Fundamentals",
            "category": "technology",
            "author": "NLP Expert",
            "difficulty": "intermediate"
        }
    },
    {
        "content": """
        Embeddings are numerical representations of text, images, or other data that capture 
        semantic meaning in high-dimensional space. Similar items have similar embeddings, 
        making them useful for similarity search, clustering, and recommendation systems. 
        Modern embedding models like OpenAI's text-embedding-3-small produce high-quality 
        embeddings for various applications.
        """,
        "metadata": {
            "title": "Understanding Embeddings",
            "category": "technology",
            "author": "Embedding Expert",
            "difficulty": "intermediate"
        }
    }
]

class ChromaDBQuickstart:
    """ChromaDB quickstart demonstration class."""
    
    def __init__(self):
        """Initialize the quickstart with configuration and ChromaDB client."""
        self.config = get_config()
        self.chroma_client = get_chroma_client()
        
    def display_config(self):
        """Display current configuration."""
        print("\n" + "="*60)
        print("CHROMADB QUICKSTART CONFIGURATION")
        print("="*60)
        print(f"📁 Persist Directory: {self.config.vector_db.persist_directory}")
        print(f"🗂️  Collection Name: {self.config.vector_db.collection_name}")
        print(f"🤖 Embedding Model: {self.config.vector_db.embedding_model}")
        print(f"📏 Chunk Size: {self.config.vector_db.chunk_size}")
        print(f"🔄 Chunk Overlap: {self.config.vector_db.chunk_overlap}")
        print(f"🔍 Vector Memory K: {self.config.memory.vector_memory_k}")
        print("="*60)
        
    def check_collection_status(self):
        """Check and display collection status."""
        try:
            collection_info = self.chroma_client.get_collection_info()
            print(f"\n📊 Collection Status:")
            print(f"   Name: {collection_info['name']}")
            print(f"   Document Count: {collection_info['count']}")
            print(f"   Metadata: {collection_info.get('metadata', {})}")
            return collection_info['count']
        except Exception as e:
            logger.error(f"Error checking collection status: {e}")
            return 0
    
    def add_sample_documents(self):
        """Add sample documents to ChromaDB."""
        print("\n🔄 Adding sample documents to ChromaDB...")
        
        # Convert sample documents to LangChain Document objects
        documents = [
            Document(
                page_content=doc["content"].strip(),
                metadata=doc["metadata"]
            )
            for doc in SAMPLE_DOCUMENTS
        ]
        
        try:
            # Add documents to ChromaDB
            doc_ids = self.chroma_client.add_documents(documents)
            print(f"✅ Successfully added {len(doc_ids)} documents to ChromaDB")
            print(f"   Document IDs: {doc_ids[:3]}..." if len(doc_ids) > 3 else f"   Document IDs: {doc_ids}")
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []
    
    def perform_similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Perform similarity search and display results."""
        print(f"\n🔍 Performing similarity search for: '{query}'")
        print(f"   Retrieving top {k} results...")
        
        try:
            # Perform similarity search
            results = self.chroma_client.similarity_search(query, k=k)
            
            print(f"\n📋 Search Results ({len(results)} found):")
            print("-" * 80)
            
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. {doc.metadata.get('title', 'Untitled')}")
                print(f"   Category: {doc.metadata.get('category', 'Unknown')}")
                print(f"   Author: {doc.metadata.get('author', 'Unknown')}")
                print(f"   Difficulty: {doc.metadata.get('difficulty', 'Unknown')}")
                print(f"   Content: {doc.page_content[:200]}...")
                
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    def perform_similarity_search_with_scores(self, query: str, k: int = 3):
        """Perform similarity search with relevance scores."""
        print(f"\n🎯 Performing similarity search with scores for: '{query}'")
        
        try:
            # Perform similarity search with scores
            results = self.chroma_client.similarity_search_with_score(query, k=k)
            
            print(f"\n📊 Search Results with Scores ({len(results)} found):")
            print("-" * 80)
            
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n{i}. {doc.metadata.get('title', 'Untitled')} (Score: {score:.4f})")
                print(f"   Category: {doc.metadata.get('category', 'Unknown')}")
                print(f"   Content: {doc.page_content[:150]}...")
                
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {e}")
            return []
    
    def demonstrate_filtered_search(self):
        """Demonstrate filtered search capabilities."""
        print("\n🔽 Demonstrating filtered search...")
        
        # Search with category filter
        query = "machine learning"
        filter_condition = {"category": "technology"}
        
        print(f"Query: '{query}' with filter: {filter_condition}")
        
        try:
            results = self.chroma_client.similarity_search(
                query=query,
                k=5,
                filter=filter_condition
            )
            
            print(f"\n📋 Filtered Search Results ({len(results)} found):")
            print("-" * 80)
            
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. {doc.metadata.get('title', 'Untitled')}")
                print(f"   Category: {doc.metadata.get('category', 'Unknown')}")
                print(f"   Content: {doc.page_content[:100]}...")
                
        except Exception as e:
            logger.error(f"Error performing filtered search: {e}")
    
    def demonstrate_retriever_interface(self):
        """Demonstrate using ChromaDB as a retriever."""
        print("\n🔄 Demonstrating retriever interface...")
        
        try:
            # Get retriever
            retriever = self.chroma_client.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Use retriever to get relevant documents
            query = "What is vector database?"
            docs = retriever.get_relevant_documents(query)
            
            print(f"🔍 Retriever Query: '{query}'")
            print(f"📋 Retrieved {len(docs)} documents:")
            
            for i, doc in enumerate(docs, 1):
                print(f"\n{i}. {doc.metadata.get('title', 'Untitled')}")
                print(f"   Content: {doc.page_content[:100]}...")
                
        except Exception as e:
            logger.error(f"Error demonstrating retriever interface: {e}")
    
    def run_interactive_search(self):
        """Run interactive search session."""
        print("\n🎮 Interactive Search Mode")
        print("Enter your search queries (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                query = input("\n🔍 Search Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Perform search
                results = self.chroma_client.similarity_search(query, k=2)
                
                if results:
                    print(f"\n📋 Top {len(results)} results:")
                    for i, doc in enumerate(results, 1):
                        print(f"\n{i}. {doc.metadata.get('title', 'Untitled')}")
                        print(f"   {doc.page_content[:100]}...")
                else:
                    print("No results found.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def cleanup_demonstration(self):
        """Demonstrate cleanup operations."""
        print("\n🧹 Cleanup Operations")
        print("Note: This is for demonstration only. Uncomment to actually clear data.")
        
        # Show how to clear collection (commented out for safety)
        print("To clear all documents:")
        print("   self.chroma_client.clear_collection()")
        
        # Show current collection info
        self.check_collection_status()
    
    def run_full_demonstration(self):
        """Run the complete ChromaDB demonstration."""
        print("🚀 Starting ChromaDB Quickstart Demonstration")
        
        # Display configuration
        self.display_config()
        
        # Check initial collection status
        initial_count = self.check_collection_status()
        
        # Add sample documents if collection is empty
        if initial_count == 0:
            self.add_sample_documents()
        else:
            print(f"\n📚 Collection already contains {initial_count} documents. Skipping document addition.")
        
        # Demonstrate various search capabilities
        queries = [
            "What is artificial intelligence?",
            "How do vector databases work?",
            "Tell me about machine learning",
            "What is RAG and how does it work?"
        ]
        
        for query in queries:
            self.perform_similarity_search(query, k=2)
        
        # Demonstrate search with scores
        self.perform_similarity_search_with_scores("embedding and vector similarity", k=2)
        
        # Demonstrate filtered search
        self.demonstrate_filtered_search()
        
        # Demonstrate retriever interface
        self.demonstrate_retriever_interface()
        
        # Show final collection status
        self.check_collection_status()
        
        # Cleanup demonstration
        self.cleanup_demonstration()
        
        print("\n✅ ChromaDB Quickstart Demonstration Complete!")
        print("\nNext steps:")
        print("1. Try the interactive search mode")
        print("2. Experiment with your own documents")
        print("3. Explore different embedding models")
        print("4. Build a RAG application using this setup")


def main():
    """Main entry point for the ChromaDB quickstart."""
    try:
        # Validate configuration
        print("🔧 Validating configuration...")
        validate_config()
        
        # Create quickstart instance
        quickstart = ChromaDBQuickstart()
        
        # Check if user wants interactive mode
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            quickstart.run_interactive_search()
        else:
            # Run full demonstration
            quickstart.run_full_demonstration()
            
            # Offer interactive mode
            print("\n🎮 Would you like to try interactive search mode? (y/n): ", end="")
            response = input().strip().lower()
            if response in ['y', 'yes']:
                quickstart.run_interactive_search()
        
    except Exception as e:
        logger.error(f"Error running ChromaDB quickstart: {e}")
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a .env file with OPENAI_API_KEY")
        print("2. Install dependencies: uv sync")
        print("3. Check that ChromaDB is properly installed")
        sys.exit(1)


if __name__ == "__main__":
    main()