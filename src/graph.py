"""
LangGraph workflow orchestration module.

This module provides the main graph structures and workflow orchestration
for context-aware LLM applications using LangGraph.
"""

from typing import Dict, List, Any, Optional, Callable, TypedDict
import logging
from datetime import datetime

# Import LangGraph components when available
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback when LangGraph is not available
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"

from .states import (
    MessagesState, ContextAwareState, RetrievalState,
    MemoryEnhancedState, WorkflowState
)
from .configuration import get_config


# =====================================
# GRAPH BUILDER CLASSES
# =====================================

class BaseGraphBuilder:
    """Base class for building LangGraph workflows."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().__dict__
        self.logger = logging.getLogger(__name__)
        
        if not LANGGRAPH_AVAILABLE:
            self.logger.warning("LangGraph is not available. Some functionality will be limited.")
    
    def create_graph(self) -> Optional[StateGraph]:
        """Create the main workflow graph. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_graph method")
    
    def add_error_handling(self, graph: StateGraph) -> None:
        """Add error handling nodes to the graph."""
        if not LANGGRAPH_AVAILABLE:
            return
        
        def error_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            """Handle errors in the workflow."""
            error_info = state.get("error_info", {})
            self.logger.error(f"Workflow error: {error_info}")
            
            # Add error message to state
            error_message = {
                "role": "system",
                "content": f"An error occurred: {error_info.get('message', 'Unknown error')}",
                "timestamp": datetime.now()
            }
            
            return {
                "messages": [error_message],
                "error_info": error_info
            }
        
        graph.add_node("error_handler", error_handler)


class ContextAwareGraphBuilder(BaseGraphBuilder):
    """Builder for context-aware conversation graphs."""
    
    def create_graph(self) -> Optional[StateGraph]:
        """Create a context-aware conversation graph."""
        if not LANGGRAPH_AVAILABLE:
            self.logger.error("LangGraph is required for graph creation")
            return None
        
        # Create the graph with ContextAwareState
        graph = StateGraph(ContextAwareState)
        
        # Add nodes
        graph.add_node("process_input", self._process_input)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("update_memory", self._update_memory)
        
        # Add edges
        graph.set_entry_point("process_input")
        graph.add_edge("process_input", "retrieve_context")
        graph.add_edge("retrieve_context", "generate_response")
        graph.add_edge("generate_response", "update_memory")
        graph.add_edge("update_memory", END)
        
        # Add error handling
        self.add_error_handling(graph)
        
        return graph
    
    def _process_input(self, state: ContextAwareState) -> Dict[str, Any]:
        """Process and validate input."""
        messages = state.get("messages", [])
        if not messages:
            return {"error_info": {"message": "No input messages provided"}}
        
        # Extract the latest user message
        latest_message = messages[-1] if messages else {}
        
        # Basic input processing
        processed_context = {
            "user_input": latest_message.get("content", ""),
            "timestamp": datetime.now(),
            "session_id": state.get("session_id"),
            "user_id": state.get("user_id")
        }
        
        return {
            "context": {**state.get("context", {}), **processed_context}
        }
    
    def _retrieve_context(self, state: ContextAwareState) -> Dict[str, Any]:
        """Retrieve relevant context from memory and knowledge base."""
        context = state.get("context", {})
        user_input = context.get("user_input", "")
        
        # Placeholder for context retrieval logic
        # In a real implementation, this would:
        # 1. Search vector database for relevant documents
        # 2. Retrieve conversation history
        # 3. Get user preferences
        
        retrieved_context = {
            "relevant_docs": [],  # Would be populated by vector search
            "conversation_history": state.get("messages", [])[-5:],  # Last 5 messages
            "user_preferences": {}  # Would be retrieved from user profile
        }
        
        return {
            "context": {**context, **retrieved_context}
        }
    
    def _generate_response(self, state: ContextAwareState) -> Dict[str, Any]:
        """Generate LLM response using context."""
        try:
            import openai
        except ImportError:
            self.logger.error("OpenAI package not available")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "Error: OpenAI package not available. Please install it.",
                    "timestamp": datetime.now()
                }]
            }
        
        context = state.get("context", {})
        messages = state.get("messages", [])
        
        # Get configuration
        from .configuration import get_config
        config = get_config()
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=config.openai.api_key)
        
        # Prepare messages for OpenAI API
        api_messages = []
        
        # Add system message with context
        system_content = "You are a helpful AI assistant with access to conversation context."
        if context.get("user_preferences"):
            system_content += f" User preferences: {context['user_preferences']}"
        if context.get("relevant_docs"):
            system_content += f" Relevant information: {context['relevant_docs']}"
        
        api_messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add recent conversation messages
        for msg in messages[-5:]:  # Last 5 messages for context
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        try:
            # Generate response using OpenAI
            response = client.chat.completions.create(
                model=config.openai.model,
                messages=api_messages,
                temperature=config.openai.temperature,
                max_tokens=config.openai.max_tokens
            )
            
            response_content = response.choices[0].message.content
            
            response_message = {
                "role": "assistant",
                "content": response_content,
                "timestamp": datetime.now()
            }
            
            return {
                "messages": [response_message]
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error: {str(e)}",
                    "timestamp": datetime.now()
                }]
            }
    
    def _update_memory(self, state: ContextAwareState) -> Dict[str, Any]:
        """Update memory with new interaction."""
        messages = state.get("messages", [])
        
        # Update memory with the latest interaction
        memory_update = {
            "last_interaction": datetime.now(),
            "message_count": len(messages)
        }
        
        return {
            "memory": {**state.get("memory", {}), **memory_update}
        }


class RetrievalGraphBuilder(BaseGraphBuilder):
    """Builder for retrieval-augmented generation graphs."""
    
    def create_graph(self) -> Optional[StateGraph]:
        """Create a RAG workflow graph."""
        if not LANGGRAPH_AVAILABLE:
            self.logger.error("LangGraph is required for graph creation")
            return None
        
        # Create the graph with RetrievalState
        graph = StateGraph(RetrievalState)
        
        # Add nodes
        graph.add_node("process_query", self._process_query)
        graph.add_node("retrieve_documents", self._retrieve_documents)
        graph.add_node("rerank_documents", self._rerank_documents)
        graph.add_node("generate_answer", self._generate_answer)
        
        # Add edges
        graph.set_entry_point("process_query")
        graph.add_edge("process_query", "retrieve_documents")
        graph.add_edge("retrieve_documents", "rerank_documents")
        graph.add_edge("rerank_documents", "generate_answer")
        graph.add_edge("generate_answer", END)
        
        # Add error handling
        self.add_error_handling(graph)
        
        return graph
    
    def _process_query(self, state: RetrievalState) -> Dict[str, Any]:
        """Process and optimize the query for retrieval."""
        query = state.get("query", "")
        
        # Basic query processing
        processed_query = query.strip().lower()
        
        return {
            "query": processed_query,
            "retrieval_metadata": {
                "original_query": query,
                "processed_at": datetime.now()
            }
        }
    
    def _retrieve_documents(self, state: RetrievalState) -> Dict[str, Any]:
        """Retrieve relevant documents from vector database."""
        query = state.get("query", "")
        
        # Placeholder for document retrieval
        # In a real implementation, this would use vector similarity search
        
        retrieved_docs = [
            {
                "content": f"Sample document content related to: {query}",
                "source": "sample_document.txt",
                "score": 0.85
            }
        ]
        
        sources = [doc["source"] for doc in retrieved_docs]
        
        return {
            "retrieved_docs": retrieved_docs,
            "sources": sources
        }
    
    def _rerank_documents(self, state: RetrievalState) -> Dict[str, Any]:
        """Rerank documents based on relevance to query."""
        retrieved_docs = state.get("retrieved_docs", [])
        query = state.get("query", "")
        
        # Simple reranking by score (in practice, would use a reranking model)
        reranked_docs = sorted(retrieved_docs, key=lambda x: x.get("score", 0), reverse=True)
        
        return {
            "retrieved_docs": reranked_docs[:5]  # Top 5 documents
        }
    
    def _generate_answer(self, state: RetrievalState) -> Dict[str, Any]:
        """Generate answer based on retrieved documents using OpenAI."""
        try:
            import openai
        except ImportError:
            self.logger.error("OpenAI package not available")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "Error: OpenAI package not available. Please install it.",
                    "timestamp": datetime.now()
                }]
            }
        
        query = state.get("query", "")
        retrieved_docs = state.get("retrieved_docs", [])
        
        # Get configuration
        from .configuration import get_config
        config = get_config()
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=config.openai.api_key)
        
        # Create context from documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5]):  # Top 5 documents
            context_parts.append(f"Document {i+1}:")
            context_parts.append(f"Source: {doc.get('source', 'Unknown')}")
            context_parts.append(f"Content: {doc.get('content', '')}")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Prepare messages for OpenAI API
        system_message = """You are a helpful AI assistant that answers questions based on provided documents. 
Use the context from the retrieved documents to provide accurate and helpful answers. 
If the documents don't contain enough information to answer the question, say so clearly.
Always cite the sources when you use information from the documents."""
        
        user_message = f"""Question: {query}

Context from retrieved documents:
{context}

Please provide a comprehensive answer based on the above context."""
        
        api_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            # Generate response using OpenAI
            response = client.chat.completions.create(
                model=config.openai.model,
                messages=api_messages,
                temperature=config.openai.temperature,
                max_tokens=config.openai.max_tokens
            )
            
            response_content = response.choices[0].message.content
            
            response_message = {
                "role": "assistant",
                "content": response_content,
                "timestamp": datetime.now()
            }
            
            return {
                "messages": [response_message],
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error while generating the response: {str(e)}",
                    "timestamp": datetime.now()
                }],
                "context": context
            }


class MemoryEnhancedGraphBuilder(BaseGraphBuilder):
    """Builder for memory-enhanced conversation graphs."""
    
    def create_graph(self) -> Optional[StateGraph]:
        """Create a memory-enhanced conversation graph."""
        if not LANGGRAPH_AVAILABLE:
            self.logger.error("LangGraph is required for graph creation")
            return None
        
        # Create the graph with MemoryEnhancedState
        graph = StateGraph(MemoryEnhancedState)
        
        # Add nodes
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("process_with_memory", self._process_with_memory)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("update_memory", self._update_memory)
        
        # Add edges
        graph.set_entry_point("load_memory")
        graph.add_edge("load_memory", "process_with_memory")
        graph.add_edge("process_with_memory", "generate_response")
        graph.add_edge("generate_response", "update_memory")
        graph.add_edge("update_memory", END)
        
        # Add error handling
        self.add_error_handling(graph)
        
        return graph
    
    def _load_memory(self, state: MemoryEnhancedState) -> Dict[str, Any]:
        """Load various types of memory."""
        # Placeholder for memory loading
        # In practice, would load from different memory systems
        
        buffer_memory = state.get("buffer_memory", [])
        summary_memory = state.get("summary_memory", "")
        
        return {
            "buffer_memory": buffer_memory,
            "summary_memory": summary_memory,
            "memory_metadata": {
                "loaded_at": datetime.now(),
                "buffer_size": len(buffer_memory)
            }
        }
    
    def _process_with_memory(self, state: MemoryEnhancedState) -> Dict[str, Any]:
        """Process input considering all memory types."""
        messages = state.get("messages", [])
        buffer_memory = state.get("buffer_memory", [])
        
        # Combine current messages with buffer memory for context
        full_context = buffer_memory + messages
        
        return {
            "memory_metadata": {
                **state.get("memory_metadata", {}),
                "context_length": len(full_context),
                "processed_at": datetime.now()
            }
        }
    
    def _generate_response(self, state: MemoryEnhancedState) -> Dict[str, Any]:
        """Generate response using memory context and OpenAI."""
        try:
            import openai
        except ImportError:
            self.logger.error("OpenAI package not available")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "Error: OpenAI package not available. Please install it.",
                    "timestamp": datetime.now()
                }]
            }
        
        messages = state.get("messages", [])
        buffer_memory = state.get("buffer_memory", [])
        summary_memory = state.get("summary_memory", "")
        user_profile = state.get("user_profile", {})
        
        # Get configuration
        from .configuration import get_config
        config = get_config()
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=config.openai.api_key)
        
        # Prepare messages for OpenAI API
        api_messages = []
        
        # Add system message with memory context
        system_content = """You are a helpful AI assistant with access to conversation memory and user context. 
Use the provided memory to maintain continuity and personalize your responses."""
        
        if summary_memory:
            system_content += f"\n\nConversation Summary: {summary_memory}"
        
        if user_profile:
            system_content += f"\n\nUser Profile: {user_profile}"
        
        api_messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add buffer memory messages (recent conversation history)
        for msg in buffer_memory[-10:]:  # Last 10 messages from buffer
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current messages
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        try:
            # Generate response using OpenAI
            response = client.chat.completions.create(
                model=config.openai.model,
                messages=api_messages,
                temperature=config.openai.temperature,
                max_tokens=config.openai.max_tokens
            )
            
            response_content = response.choices[0].message.content
            
            response_message = {
                "role": "assistant",
                "content": response_content,
                "timestamp": datetime.now()
            }
            
            return {
                "messages": [response_message]
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error: {str(e)}",
                    "timestamp": datetime.now()
                }]
            }
    
    def _update_memory(self, state: MemoryEnhancedState) -> Dict[str, Any]:
        """Update all memory systems."""
        messages = state.get("messages", [])
        buffer_memory = state.get("buffer_memory", [])
        
        # Add new messages to buffer memory
        updated_buffer = buffer_memory + messages
        
        # Keep buffer size manageable
        max_buffer_size = 20
        if len(updated_buffer) > max_buffer_size:
            updated_buffer = updated_buffer[-max_buffer_size:]
        
        return {
            "buffer_memory": updated_buffer,
            "memory_metadata": {
                **state.get("memory_metadata", {}),
                "updated_at": datetime.now()
            }
        }


# =====================================
# GRAPH FACTORY
# =====================================

class GraphFactory:
    """Factory for creating different types of workflow graphs."""
    
    @staticmethod
    def create_context_aware_graph(config: Optional[Dict[str, Any]] = None) -> Optional[StateGraph]:
        """Create a context-aware conversation graph."""
        builder = ContextAwareGraphBuilder(config)
        return builder.create_graph()
    
    @staticmethod
    def create_retrieval_graph(config: Optional[Dict[str, Any]] = None) -> Optional[StateGraph]:
        """Create a retrieval-augmented generation graph."""
        builder = RetrievalGraphBuilder(config)
        return builder.create_graph()
    
    @staticmethod
    def create_memory_enhanced_graph(config: Optional[Dict[str, Any]] = None) -> Optional[StateGraph]:
        """Create a memory-enhanced conversation graph."""
        builder = MemoryEnhancedGraphBuilder(config)
        return builder.create_graph()
    
    @staticmethod
    def get_available_graphs() -> List[str]:
        """Get list of available graph types."""
        return [
            "context_aware",
            "retrieval",
            "memory_enhanced"
        ]


# =====================================
# WORKFLOW RUNNER
# =====================================

class WorkflowRunner:
    """Runner for executing LangGraph workflows."""
    
    def __init__(self, graph: StateGraph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)
    
    def run(
        self,
        initial_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run the workflow with given initial state."""
        if not LANGGRAPH_AVAILABLE:
            self.logger.error("LangGraph is required for workflow execution")
            return {"error": "LangGraph not available"}
        
        try:
            # Compile the graph if not already compiled
            compiled_graph = self.graph.compile()
            
            # Run the workflow
            result = compiled_graph.invoke(initial_state, config)
            
            self.logger.info("Workflow completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def stream(
        self,
        initial_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ):
        """Stream workflow execution results."""
        if not LANGGRAPH_AVAILABLE:
            self.logger.error("LangGraph is required for workflow execution")
            yield {"error": "LangGraph not available"}
            return
        
        try:
            # Compile the graph if not already compiled
            compiled_graph = self.graph.compile()
            
            # Stream the workflow execution
            for result in compiled_graph.stream(initial_state, config):
                yield result
                
        except Exception as e:
            self.logger.error(f"Workflow streaming failed: {e}")
            yield {
                "error": str(e),
                "error_type": type(e).__name__
            }


# =====================================
# CONVENIENCE FUNCTIONS
# =====================================

def create_simple_chat_workflow() -> Optional[WorkflowRunner]:
    """Create a simple chat workflow ready to use."""
    graph = GraphFactory.create_context_aware_graph()
    if graph:
        return WorkflowRunner(graph)
    return None


def create_rag_workflow() -> Optional[WorkflowRunner]:
    """Create a RAG workflow ready to use."""
    graph = GraphFactory.create_retrieval_graph()
    if graph:
        return WorkflowRunner(graph)
    return None


def create_memory_chat_workflow() -> Optional[WorkflowRunner]:
    """Create a memory-enhanced chat workflow ready to use."""
    graph = GraphFactory.create_memory_enhanced_graph()
    if graph:
        return WorkflowRunner(graph)
    return None


if __name__ == "__main__":
    # Test graph creation
    print("Testing graph creation...")
    
    if LANGGRAPH_AVAILABLE:
        # Test context-aware graph
        context_graph = GraphFactory.create_context_aware_graph()
        print(f"Context-aware graph created: {context_graph is not None}")
        
        # Test retrieval graph
        retrieval_graph = GraphFactory.create_retrieval_graph()
        print(f"Retrieval graph created: {retrieval_graph is not None}")
        
        # Test memory-enhanced graph
        memory_graph = GraphFactory.create_memory_enhanced_graph()
        print(f"Memory-enhanced graph created: {memory_graph is not None}")
        
        print("Available graphs:", GraphFactory.get_available_graphs())
    else:
        print("LangGraph not available - install dependencies to test graph creation")
    
    print("Graph testing completed!")
