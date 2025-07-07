from typing import Optional, Dict, Any, List

from .graph import ContextAwareGraphBuilder, RetrievalGraphBuilder, MemoryEnhancedGraphBuilder
from langgraph.graph import StateGraph, END

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
