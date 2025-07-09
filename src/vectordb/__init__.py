"""
Vector database package for ChromaDB operations.

This package provides vector storage and retrieval capabilities
using ChromaDB with LangChain integration.
"""

from .chroma_client import ChromaDBClient, chroma_client, get_chroma_client

__all__ = [
    "ChromaDBClient",
    "chroma_client", 
    "get_chroma_client"
]
