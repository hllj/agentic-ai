from typing import Dict, Any, Optional
import logging

# Import LangGraph components when available
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback when LangGraph is not available
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"

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

