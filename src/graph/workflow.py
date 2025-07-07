from typing import Dict, Any, Optional
import logging

from langgraph.graph import StateGraph, END

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

