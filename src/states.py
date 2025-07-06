"""
State management module for LangGraph workflows.

This module defines the various state classes used throughout the application
for managing conversation context, memory, and workflow orchestration.
"""

from typing import List, Dict, Any, Optional, Union, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime
import operator


class MessagesState(TypedDict):
    """Basic state for managing conversation messages."""
    messages: Annotated[List[Dict[str, Any]], operator.add]


class ContextAwareState(TypedDict):
    """Extended state for context-aware applications."""
    messages: Annotated[List[Dict[str, Any]], operator.add]
    context: Dict[str, Any]
    memory: Dict[str, Any]
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: Optional[datetime]


class RetrievalState(TypedDict):
    """State for retrieval-augmented generation workflows."""
    messages: Annotated[List[Dict[str, Any]], operator.add]
    query: str
    retrieved_docs: List[Dict[str, Any]]
    context: str
    sources: List[str]
    retrieval_metadata: Dict[str, Any]


class MemoryEnhancedState(TypedDict):
    """State with comprehensive memory management."""
    messages: Annotated[List[Dict[str, Any]], operator.add]
    buffer_memory: List[Dict[str, Any]]
    summary_memory: str
    vector_memory_ids: List[str]
    memory_metadata: Dict[str, Any]
    user_profile: Dict[str, Any]


class WorkflowState(TypedDict):
    """Comprehensive state for complex workflows."""
    messages: Annotated[List[Dict[str, Any]], operator.add]
    current_step: str
    workflow_data: Dict[str, Any]
    intermediate_results: Dict[str, Any]
    error_info: Optional[Dict[str, Any]]
    retry_count: int


@dataclass
class ConversationMemory:
    """Data class for managing conversation memory."""
    
    buffer: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    vector_store_keys: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    context_window: int = 10
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the buffer memory."""
        self.buffer.append(message)
        
        # Keep only the last N messages in buffer
        if len(self.buffer) > self.context_window:
            self.buffer = self.buffer[-self.context_window:]
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent messages."""
        return self.buffer[-count:] if self.buffer else []
    
    def clear_buffer(self) -> None:
        """Clear the buffer memory."""
        self.buffer.clear()
    
    def update_summary(self, new_summary: str) -> None:
        """Update the conversation summary."""
        self.summary = new_summary
    
    def add_vector_key(self, key: str) -> None:
        """Add a vector store key for semantic memory."""
        if key not in self.vector_store_keys:
            self.vector_store_keys.append(key)


@dataclass
class UserContext:
    """Data class for managing user-specific context."""
    
    user_id: str
    session_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    
    def update_last_active(self) -> None:
        """Update the last active timestamp."""
        self.last_active = datetime.now()
    
    def add_interaction(self, interaction: Dict[str, Any]) -> None:
        """Add an interaction to the history."""
        interaction["timestamp"] = datetime.now()
        self.interaction_history.append(interaction)
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference with optional default."""
        return self.preferences.get(key, default)
    
    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference."""
        self.preferences[key] = value


@dataclass
class DocumentContext:
    """Data class for managing document retrieval context."""
    
    query: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity_scores: List[float] = field(default_factory=list)
    retrieval_method: str = "similarity"
    
    def add_document(self, document: Dict[str, Any], score: float = 0.0) -> None:
        """Add a retrieved document with its similarity score."""
        self.documents.append(document)
        self.similarity_scores.append(score)
    
    def get_top_documents(self, k: int = 5) -> List[Dict[str, Any]]:
        """Get the top k most relevant documents."""
        if not self.documents:
            return []
        
        # Sort by similarity score if available
        if self.similarity_scores and len(self.similarity_scores) == len(self.documents):
            sorted_docs = sorted(
                zip(self.documents, self.similarity_scores),
                key=lambda x: x[1],
                reverse=True
            )
            return [doc for doc, _ in sorted_docs[:k]]
        
        # Otherwise return first k documents
        return self.documents[:k]
    
    def format_context(self) -> str:
        """Format documents into a context string."""
        if not self.documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(self.documents[:5]):  # Limit to top 5
            content = doc.get("content", str(doc))
            source = doc.get("source", f"Document {i+1}")
            context_parts.append(f"Source: {source}\nContent: {content}\n")
        
        return "\n".join(context_parts)


def create_base_state(
    messages: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create a base state dictionary with common fields."""
    return {
        "messages": messages or [],
        "timestamp": datetime.now(),
        **kwargs
    }


def create_context_aware_state(
    messages: Optional[List[Dict[str, Any]]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> ContextAwareState:
    """Create a context-aware state with user and session information."""
    return ContextAwareState(
        messages=messages or [],
        context=kwargs.get("context", {}),
        memory=kwargs.get("memory", {}),
        user_id=user_id,
        session_id=session_id,
        timestamp=datetime.now()
    )


def create_retrieval_state(
    query: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> RetrievalState:
    """Create a retrieval state for RAG workflows."""
    return RetrievalState(
        messages=messages or [],
        query=query,
        retrieved_docs=kwargs.get("retrieved_docs", []),
        context=kwargs.get("context", ""),
        sources=kwargs.get("sources", []),
        retrieval_metadata=kwargs.get("retrieval_metadata", {})
    )


def merge_states(state1: Dict[str, Any], state2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two state dictionaries, handling message lists specially."""
    merged = state1.copy()
    
    for key, value in state2.items():
        if key == "messages" and key in merged:
            # Combine message lists
            merged[key] = merged[key] + value
        else:
            merged[key] = value
    
    return merged


def extract_messages_content(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract content strings from message dictionaries."""
    content = []
    for msg in messages:
        if isinstance(msg, dict):
            if "content" in msg:
                content.append(str(msg["content"]))
            elif "text" in msg:
                content.append(str(msg["text"]))
            else:
                content.append(str(msg))
        else:
            content.append(str(msg))
    
    return content
