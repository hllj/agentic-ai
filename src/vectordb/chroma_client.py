"""
ChromaDB client for vector database operations.

This module provides ChromaDB integration with LangChain
for document embeddings and similarity search.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

from ..configuration import get_config

logger = logging.getLogger(__name__)

class ChromaDBClient:
    """ChromaDB client for vector operations."""
    
    def __init__(self):
        self.config = get_config()
        self.embeddings = OpenAIEmbeddings(
            model=self.config.vector_db.embedding_model,
            openai_api_key=self.config.openai.api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.vector_db.chunk_size,
            chunk_overlap=self.config.vector_db.chunk_overlap,
            length_function=len,
        )
        self.vectorstore: Optional[Chroma] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client
            client_settings = Settings(
                persist_directory=self.config.vector_db.persist_directory,
                is_persistent=True,
            )
            
            self.vectorstore = Chroma(
                collection_name=self.config.vector_db.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.config.vector_db.persist_directory,
                client_settings=client_settings,
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.config.vector_db.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """Add documents to the vector store."""
        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add to vectorstore
            doc_ids = self.vectorstore.add_documents(documents=chunks, ids=ids)
            
            logger.info(f"Added {len(chunks)} document chunks to ChromaDB")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> List[str]:
        """Add texts to the vector store."""
        try:
            # Split texts into chunks
            documents = [Document(page_content=text, metadata=metadata or {}) 
                        for text, metadata in zip(texts, metadatas or [{}] * len(texts))]
            
            chunks = self.text_splitter.split_documents(documents)
            
            # Extract texts and metadatas from chunks
            chunk_texts = [chunk.page_content for chunk in chunks]
            chunk_metadatas = [chunk.metadata for chunk in chunks]
            
            # Add to vectorstore
            doc_ids = self.vectorstore.add_texts(
                texts=chunk_texts, 
                metadatas=chunk_metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(chunk_texts)} text chunks to ChromaDB")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to add texts to ChromaDB: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform similarity search."""
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.debug(f"Similarity search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Perform similarity search with relevance scores."""
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.debug(f"Similarity search with score returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search with score: {e}")
            raise
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        try:
            self.vectorstore.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection = self.vectorstore._collection
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Get all document IDs
            collection = self.vectorstore._collection
            all_docs = collection.get()
            
            if all_docs['ids']:
                collection.delete(ids=all_docs['ids'])
                logger.info("Cleared all documents from ChromaDB collection")
            else:
                logger.info("Collection is already empty")
                
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict[str, Any]] = None):
        """Return vectorstore as a retriever."""
        search_kwargs = search_kwargs or {"k": self.config.memory.vector_memory_k}
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

# Global ChromaDB client instance
chroma_client = ChromaDBClient()

def get_chroma_client() -> ChromaDBClient:
    """Get the global ChromaDB client instance."""
    return chroma_client
