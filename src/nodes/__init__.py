"""
Core processing nodes for LangGraph workflows.

This module contains the fundamental processing nodes that can be used
in various workflow configurations for context-aware LLM applications.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Try to import LangChain components
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    HumanMessage = None
    AIMessage = None
    SystemMessage = None

from ..configuration import get_config
from ..prompts import (
    build_conversation_prompt,
    build_rag_prompt,
    build_memory_context_prompt
)


logger = logging.getLogger(__name__)


# =====================================
# INPUT PROCESSING NODES
# =====================================

def process_user_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process and validate user input."""
    messages = state.get("messages", [])
    
    if not messages:
        return {
            "error_info": {
                "message": "No input messages provided",
                "timestamp": datetime.now()
            }
        }
    
    # Get the latest message
    latest_message = messages[-1] if messages else {}
    user_input = latest_message.get("content", "")
    
    # Basic input validation and cleaning
    if not user_input.strip():
        return {
            "error_info": {
                "message": "Empty input provided",
                "timestamp": datetime.now()
            }
        }
    
    # Process and enhance the input
    processed_input = {
        "content": user_input.strip(),
        "timestamp": datetime.now(),
        "word_count": len(user_input.split()),
        "character_count": len(user_input)
    }
    
    # Update context with processed input
    current_context = state.get("context", {})
    updated_context = {
        **current_context,
        "processed_input": processed_input,
        "last_updated": datetime.now()
    }
    
    return {
        "context": updated_context
    }


def extract_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract user intent from input."""
    context = state.get("context", {})
    processed_input = context.get("processed_input", {})
    user_content = processed_input.get("content", "")
    
    # Simple intent classification (would use NLP models in practice)
    intent_keywords = {
        "question": ["what", "how", "why", "when", "where", "who", "?"],
        "request": ["please", "can you", "could you", "would you"],
        "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
        "goodbye": ["bye", "goodbye", "see you", "farewell"],
        "help": ["help", "assist", "support", "guidance"]
    }
    
    detected_intent = "general"
    confidence = 0.0
    
    user_lower = user_content.lower()
    for intent, keywords in intent_keywords.items():
        matches = sum(1 for keyword in keywords if keyword in user_lower)
        if matches > 0:
            intent_confidence = matches / len(keywords)
            if intent_confidence > confidence:
                detected_intent = intent
                confidence = intent_confidence
    
    # Update context with intent information
    current_context = state.get("context", {})
    updated_context = {
        **current_context,
        "intent": {
            "type": detected_intent,
            "confidence": confidence,
            "detected_at": datetime.now()
        }
    }
    
    return {
        "context": updated_context
    }


# =====================================
# CONTEXT RETRIEVAL NODES
# =====================================

def retrieve_conversation_history(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve relevant conversation history."""
    messages = state.get("messages", [])
    session_id = state.get("session_id")
    user_id = state.get("user_id")
    
    # In practice, this would query a database or memory store
    # For now, we'll use the messages already in state
    
    # Get conversation history (excluding the current message)
    conversation_history = messages[:-1] if len(messages) > 1 else []
    
    # Limit to recent conversation (last 10 exchanges)
    max_history = 10
    if len(conversation_history) > max_history:
        conversation_history = conversation_history[-max_history:]
    
    # Calculate conversation statistics
    history_stats = {
        "total_messages": len(conversation_history),
        "user_messages": len([m for m in conversation_history if m.get("role") == "user"]),
        "assistant_messages": len([m for m in conversation_history if m.get("role") == "assistant"]),
        "conversation_length": sum(len(m.get("content", "")) for m in conversation_history)
    }
    
    # Update context
    current_context = state.get("context", {})
    updated_context = {
        **current_context,
        "conversation_history": conversation_history,
        "conversation_stats": history_stats,
        "history_retrieved_at": datetime.now()
    }
    
    return {
        "context": updated_context
    }


def retrieve_user_profile(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve user profile and preferences."""
    user_id = state.get("user_id")
    session_id = state.get("session_id")
    
    # Placeholder for user profile retrieval
    # In practice, this would query a user database
    
    default_profile = {
        "user_id": user_id,
        "preferences": {
            "response_style": "helpful",
            "detail_level": "moderate",
            "language": "english"
        },
        "interaction_count": 0,
        "last_seen": datetime.now(),
        "topics_of_interest": [],
        "communication_style": "friendly"
    }
    
    # Update context with user profile
    current_context = state.get("context", {})
    updated_context = {
        **current_context,
        "user_profile": default_profile,
        "profile_retrieved_at": datetime.now()
    }
    
    return {
        "context": updated_context
    }


# =====================================
# DOCUMENT RETRIEVAL NODES
# =====================================

def search_knowledge_base(state: Dict[str, Any]) -> Dict[str, Any]:
    """Search knowledge base for relevant documents."""
    query = state.get("query", "")
    context = state.get("context", {})
    
    if not query and context:
        # Extract query from processed input
        processed_input = context.get("processed_input", {})
        query = processed_input.get("content", "")
    
    if not query:
        return {
            "retrieved_docs": [],
            "sources": [],
            "retrieval_metadata": {
                "error": "No query provided for knowledge base search",
                "timestamp": datetime.now()
            }
        }
    
    # Placeholder for vector database search
    # In practice, this would use a vector database like Chroma
    
    # Mock retrieved documents
    mock_documents = [
        {
            "content": f"This is a sample document that relates to the query about {query}. It contains relevant information that can help answer the user's question.",
            "source": "knowledge_base/sample_doc_1.txt",
            "score": 0.95,
            "metadata": {
                "document_id": "doc_001",
                "category": "general",
                "last_updated": "2024-01-15"
            }
        },
        {
            "content": f"Another relevant document discussing aspects of {query}. This provides additional context and supporting information.",
            "source": "knowledge_base/sample_doc_2.txt",
            "score": 0.87,
            "metadata": {
                "document_id": "doc_002",
                "category": "reference",
                "last_updated": "2024-01-20"
            }
        }
    ]
    
    # Extract sources
    sources = [doc["source"] for doc in mock_documents]
    
    # Create retrieval metadata
    retrieval_metadata = {
        "query": query,
        "num_results": len(mock_documents),
        "search_method": "vector_similarity",
        "timestamp": datetime.now(),
        "max_score": max(doc["score"] for doc in mock_documents) if mock_documents else 0
    }
    
    return {
        "retrieved_docs": mock_documents,
        "sources": sources,
        "retrieval_metadata": retrieval_metadata
    }


def rerank_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Rerank retrieved documents by relevance."""
    retrieved_docs = state.get("retrieved_docs", [])
    query = state.get("query", "")
    
    if not retrieved_docs:
        return {
            "retrieved_docs": [],
            "reranking_metadata": {
                "message": "No documents to rerank",
                "timestamp": datetime.now()
            }
        }
    
    # Simple reranking by score (in practice, would use a reranking model)
    reranked_docs = sorted(
        retrieved_docs,
        key=lambda x: x.get("score", 0),
        reverse=True
    )
    
    # Limit to top k documents
    top_k = 5
    final_docs = reranked_docs[:top_k]
    
    # Add reranking metadata
    reranking_metadata = {
        "original_count": len(retrieved_docs),
        "final_count": len(final_docs),
        "reranking_method": "score_based",
        "timestamp": datetime.now()
    }
    
    return {
        "retrieved_docs": final_docs,
        "reranking_metadata": reranking_metadata
    }


# =====================================
# MEMORY MANAGEMENT NODES
# =====================================

def load_buffer_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load recent conversation from buffer memory."""
    messages = state.get("messages", [])
    session_id = state.get("session_id")
    
    # Buffer memory contains recent messages
    buffer_size = 10  # Keep last 10 messages
    buffer_memory = messages[-buffer_size:] if messages else []
    
    return {
        "buffer_memory": buffer_memory,
        "memory_metadata": {
            "buffer_size": len(buffer_memory),
            "loaded_at": datetime.now(),
            "session_id": session_id
        }
    }


def load_summary_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load conversation summary from memory."""
    session_id = state.get("session_id")
    user_id = state.get("user_id")
    
    # Placeholder for summary memory retrieval
    # In practice, this would retrieve from a memory store
    
    summary_memory = f"User {user_id} has been discussing various topics in session {session_id}."
    
    return {
        "summary_memory": summary_memory,
        "memory_metadata": {
            **state.get("memory_metadata", {}),
            "summary_loaded_at": datetime.now()
        }
    }


def update_conversation_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    """Update conversation memory with new interactions."""
    messages = state.get("messages", [])
    buffer_memory = state.get("buffer_memory", [])
    session_id = state.get("session_id")
    
    # Add new messages to buffer memory
    updated_buffer = list(buffer_memory)  # Copy existing buffer
    
    # Add new messages (avoid duplicates)
    for message in messages:
        if message not in updated_buffer:
            updated_buffer.append(message)
    
    # Keep buffer size manageable
    max_buffer_size = 20
    if len(updated_buffer) > max_buffer_size:
        updated_buffer = updated_buffer[-max_buffer_size:]
    
    # Update memory metadata
    memory_metadata = {
        **state.get("memory_metadata", {}),
        "buffer_updated_at": datetime.now(),
        "current_buffer_size": len(updated_buffer),
        "session_id": session_id
    }
    
    return {
        "buffer_memory": updated_buffer,
        "memory_metadata": memory_metadata
    }


# =====================================
# RESPONSE GENERATION NODES
# =====================================

def generate_llm_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate response using LLM."""
    if not LANGCHAIN_AVAILABLE:
        # Fallback response when LangChain is not available
        response_message = {
            "role": "assistant",
            "content": "I understand your message. However, LLM functionality is not currently available.",
            "timestamp": datetime.now()
        }
        return {"messages": [response_message]}
    
    # Get context and messages
    context = state.get("context", {})
    messages = state.get("messages", [])
    
    # Extract user input
    processed_input = context.get("processed_input", {})
    user_input = processed_input.get("content", "")
    
    # Build prompt based on available context
    conversation_history = context.get("conversation_history", [])
    user_profile = context.get("user_profile", {})
    
    # Create prompt
    prompt = build_conversation_prompt(
        current_input=user_input,
        conversation_history=conversation_history,
        user_context=user_profile
    )
    
    try:
        # Initialize LLM (using configuration)
        config = get_config()
        llm = ChatOpenAI(
            model=config.openai.model,
            temperature=config.openai.temperature,
            api_key=config.openai.api_key
        )
        
        # Generate response
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Create response message
        response_message = {
            "role": "assistant",
            "content": response.content,
            "timestamp": datetime.now(),
            "model": config.openai.model
        }
        
        return {"messages": [response_message]}
        
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        
        # Fallback response
        response_message = {
            "role": "assistant",
            "content": f"I apologize, but I encountered an error while processing your request: {str(e)}",
            "timestamp": datetime.now(),
            "error": True
        }
        
        return {"messages": [response_message]}


def generate_rag_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate response using retrieved documents (RAG)."""
    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    sources = state.get("sources", [])
    
    if not retrieved_docs:
        response_message = {
            "role": "assistant",
            "content": "I don't have enough information to answer your question based on the available documents.",
            "timestamp": datetime.now()
        }
        return {"messages": [response_message]}
    
    # Format documents into context
    context_parts = []
    for doc in retrieved_docs[:3]:  # Use top 3 documents
        content = doc.get("content", "")
        source = doc.get("source", "Unknown")
        context_parts.append(f"Source: {source}\nContent: {content}")
    
    context = "\n\n".join(context_parts)
    
    # Build RAG prompt
    prompt = build_rag_prompt(
        context=context,
        question=query,
        sources=sources[:3]
    )
    
    if LANGCHAIN_AVAILABLE:
        try:
            # Initialize LLM
            config = get_config()
            llm = ChatOpenAI(
                model=config.openai.model,
                temperature=config.openai.temperature,
                api_key=config.openai.api_key
            )
            
            # Generate response
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            content = f"Based on the retrieved documents, I can provide some information about {query}, but encountered an error in processing."
    else:
        # Fallback without LLM
        content = f"Based on the retrieved documents about {query}: {context[:200]}..."
    
    # Create response message
    response_message = {
        "role": "assistant",
        "content": content,
        "timestamp": datetime.now(),
        "sources": sources[:3],
        "retrieval_based": True
    }
    
    return {
        "messages": [response_message],
        "context": context
    }


# =====================================
# OUTPUT PROCESSING NODES
# =====================================

def format_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format the final response for output."""
    messages = state.get("messages", [])
    
    if not messages:
        return {
            "error_info": {
                "message": "No response generated",
                "timestamp": datetime.now()
            }
        }
    
    # Get the latest assistant message
    latest_response = None
    for message in reversed(messages):
        if message.get("role") == "assistant":
            latest_response = message
            break
    
    if not latest_response:
        return {
            "error_info": {
                "message": "No assistant response found",
                "timestamp": datetime.now()
            }
        }
    
    # Format the response
    formatted_response = {
        "content": latest_response.get("content", ""),
        "timestamp": latest_response.get("timestamp", datetime.now()),
        "metadata": {
            "model": latest_response.get("model"),
            "sources": latest_response.get("sources"),
            "retrieval_based": latest_response.get("retrieval_based", False),
            "error": latest_response.get("error", False)
        }
    }
    
    return {
        "formatted_response": formatted_response
    }


# =====================================
# UTILITY NODES
# =====================================

def log_interaction(state: Dict[str, Any]) -> Dict[str, Any]:
    """Log the interaction for monitoring and debugging."""
    messages = state.get("messages", [])
    session_id = state.get("session_id")
    user_id = state.get("user_id")
    
    # Create interaction log
    interaction_log = {
        "session_id": session_id,
        "user_id": user_id,
        "timestamp": datetime.now(),
        "message_count": len(messages),
        "conversation_length": sum(len(m.get("content", "")) for m in messages),
        "context_keys": list(state.get("context", {}).keys())
    }
    
    logger.info(f"Interaction logged: {interaction_log}")
    
    # Add to state for potential storage
    return {
        "interaction_log": interaction_log
    }


if __name__ == "__main__":
    # Test some nodes
    print("Testing processing nodes...")
    
    # Test input processing
    test_state = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "session_id": "test_session",
        "user_id": "test_user"
    }
    
    result = process_user_input(test_state)
    print(f"Input processing result: {result}")
    
    # Test intent extraction
    result = extract_intent({**test_state, **result})
    print(f"Intent extraction result: {result}")
    
    print("Node testing completed!")
