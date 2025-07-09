"""
Conversation memory system using PostgreSQL and ChromaDB.

This module provides conversation memory management using
LangChain memory classes with PostgreSQL for persistence
and ChromaDB for vector-based memory retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID

from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from ..configuration import get_config
from ..database.connection import get_db_session
from ..database.models import User, Conversation, Message, ConversationSummary
from ..vectordb.chroma_client import get_chroma_client

logger = logging.getLogger(__name__)

class ConversationMemoryManager:
    """Manages conversation memory using PostgreSQL and ChromaDB."""
    
    def __init__(self, user_id: UUID, conversation_id: Optional[UUID] = None):
        self.config = get_config()
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.chroma_client = get_chroma_client()
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model=self.config.openai.model,
            temperature=self.config.openai.temperature,
            openai_api_key=self.config.openai.api_key
        )
        
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=self.config.memory.summary_max_tokens,
            return_messages=True
        )
        
        # Load existing conversation if provided
        if self.conversation_id:
            self._load_conversation_history()
    
    def _load_conversation_history(self):
        """Load conversation history from PostgreSQL."""
        try:
            with get_db_session() as session:
                messages = session.query(Message).filter(
                    Message.conversation_id == self.conversation_id
                ).order_by(Message.created_at).all()
                
                # Convert to LangChain messages
                for msg in messages:
                    if msg.role == "user":
                        self.memory.chat_memory.add_user_message(msg.content)
                    elif msg.role == "assistant":
                        self.memory.chat_memory.add_ai_message(msg.content)
                
                logger.info(f"Loaded {len(messages)} messages from conversation {self.conversation_id}")
                
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
            raise
    
    def add_message(self, content: str, role: str, message_type: str = "text", metadata: Optional[Dict[str, Any]] = None):
        """Add a message to both memory and database."""
        try:
            # Add to LangChain memory
            if role == "user":
                self.memory.chat_memory.add_user_message(content)
            elif role == "assistant":
                self.memory.chat_memory.add_ai_message(content)
            
            # Save to PostgreSQL if using postgres
            if self.config.memory.use_postgres:
                self._save_message_to_db(content, role, message_type, metadata)
            
            # Add to vector memory for retrieval
            self._add_to_vector_memory(content, role, metadata)
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise
    
    def _save_message_to_db(self, content: str, role: str, message_type: str, metadata: Optional[Dict[str, Any]]):
        """Save message to PostgreSQL database."""
        try:
            with get_db_session() as session:
                # Ensure conversation exists
                if not self.conversation_id:
                    conversation = Conversation(
                        user_id=self.user_id,
                        title=f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
                    )
                    session.add(conversation)
                    session.flush()
                    self.conversation_id = conversation.id
                
                # Create message
                message = Message(
                    conversation_id=self.conversation_id,
                    user_id=self.user_id,
                    content=content,
                    role=role,
                    message_type=message_type,
                    metadata=metadata or {}
                )
                session.add(message)
                
                logger.debug(f"Saved message to database: {role}")
                
        except Exception as e:
            logger.error(f"Failed to save message to database: {e}")
            raise
    
    def _add_to_vector_memory(self, content: str, role: str, metadata: Optional[Dict[str, Any]]):
        """Add message to ChromaDB for vector-based retrieval."""
        try:
            # Create document with enhanced metadata
            doc_metadata = {
                "user_id": str(self.user_id),
                "conversation_id": str(self.conversation_id) if self.conversation_id else "",
                "role": role,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            # Add to vector store
            self.chroma_client.add_texts(
                texts=[content],
                metadatas=[doc_metadata],
                ids=[f"{self.conversation_id}_{role}_{datetime.utcnow().timestamp()}"]
            )
            
        except Exception as e:
            logger.error(f"Failed to add to vector memory: {e}")
            # Don't raise here as vector memory is supplementary
    
    def get_relevant_context(self, query: str, k: int = None) -> List[Document]:
        """Get relevant context from vector memory."""
        try:
            k = k or self.config.memory.vector_memory_k
            
            # Search for relevant messages
            filter_criteria = {
                "user_id": str(self.user_id)
            }
            
            if self.conversation_id:
                filter_criteria["conversation_id"] = str(self.conversation_id)
            
            results = self.chroma_client.similarity_search(
                query=query,
                k=k,
                filter=filter_criteria
            )
            
            logger.debug(f"Retrieved {len(results)} relevant context items")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get relevant context: {e}")
            return []
    
    def get_conversation_summary(self) -> str:
        """Get conversation summary from memory."""
        try:
            summary = self.memory.predict_new_summary(
                messages=self.memory.chat_memory.messages,
                existing_summary=""
            )
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return ""
    
    def save_conversation_summary(self, summary: str = None):
        """Save conversation summary to database."""
        try:
            if not summary:
                summary = self.get_conversation_summary()
            
            if not self.conversation_id:
                return
            
            with get_db_session() as session:
                conversation_summary = ConversationSummary(
                    conversation_id=self.conversation_id,
                    summary_text=summary,
                    message_count=len(self.memory.chat_memory.messages)
                )
                session.add(conversation_summary)
                
                logger.info(f"Saved conversation summary for {self.conversation_id}")
                
        except Exception as e:
            logger.error(f"Failed to save conversation summary: {e}")
            raise
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables for use in prompts."""
        try:
            return self.memory.load_memory_variables({})
        except Exception as e:
            logger.error(f"Failed to get memory variables: {e}")
            return {}
    
    def clear_memory(self):
        """Clear memory buffer."""
        try:
            self.memory.clear()
            logger.info("Memory buffer cleared")
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            raise
    
    def get_conversation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history from database."""
        try:
            if not self.conversation_id:
                return []
            
            with get_db_session() as session:
                messages = session.query(Message).filter(
                    Message.conversation_id == self.conversation_id
                ).order_by(Message.created_at.desc()).limit(limit).all()
                
                return [
                    {
                        "id": str(msg.id),
                        "content": msg.content,
                        "role": msg.role,
                        "message_type": msg.message_type,
                        "metadata": msg.metadata,
                        "created_at": msg.created_at.isoformat()
                    }
                    for msg in reversed(messages)
                ]
                
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

def create_memory_manager(user_id: UUID, conversation_id: Optional[UUID] = None) -> ConversationMemoryManager:
    """Factory function to create memory manager."""
    return ConversationMemoryManager(user_id=user_id, conversation_id=conversation_id)
