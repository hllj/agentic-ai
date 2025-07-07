from typing import Optional

from .factory import GraphFactory
from .workflow import WorkflowRunner

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