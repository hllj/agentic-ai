
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from langgraph.graph import StateGraph, END
from ..configuration import get_config

class BaseGraphBuilder:
    """Base class for building LangGraph workflows."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().__dict__
        self.logger = logging.getLogger(__name__)
    
    def create_graph(self) -> Optional[StateGraph]:
        """Create the main workflow graph. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_graph method")
    
    def add_error_handling(self, graph: StateGraph) -> None:
        """Add error handling nodes to the graph."""
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
