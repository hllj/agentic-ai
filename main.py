"""
Main application entry point for the Agentic AI system.

This module provides the main interface for running context-aware LLM applications
with different workflow configurations and capabilities.
"""

import sys
import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import core modules
from src.configuration import get_config, validate_config
from src.states import create_context_aware_state, create_retrieval_state
from src.tools import tool_manager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Setup basic logging configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[logging.StreamHandler(sys.stdout)]
)


# =====================================
# APPLICATION CLASS
# =====================================

class AgenticAIApp:
    """Main application class for the Agentic AI system."""
    
    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        """Initialize the application."""
        self.config = get_config()
        self.initialized = False
        
        # Apply configuration overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the application components."""
        try:
            logger.info("Initializing Agentic AI Application...")
            
            # Validate configuration
            validate_config()
            
            # Initialize components based on available dependencies
            self._check_dependencies()
            
            self.initialized = True
            logger.info("Application initialized successfully")
            
        except Exception as e:
            if logger:
                logger.error(f"Failed to initialize application: {e}")
            else:
                print(f"Failed to initialize application: {e}")
            raise
    
    def _check_dependencies(self):
        """Check which dependencies are available."""
        dependencies = {
            "langchain": False,
            "langgraph": False,
            "openai": False,
            "chroma": False
        }
        
        # Check LangChain
        try:
            import langchain
            dependencies["langchain"] = True
        except ImportError:
            pass
        
        # Check LangGraph
        try:
            import langgraph
            dependencies["langgraph"] = True
        except ImportError:
            pass
        
        # Check OpenAI
        try:
            import openai
            dependencies["openai"] = True
        except ImportError:
            pass
        
        # Check Chroma
        try:
            import chromadb
            dependencies["chroma"] = True
        except ImportError:
            pass
        
        self.dependencies = dependencies
        
        # Log available dependencies
        available = [k for k, v in dependencies.items() if v]
        missing = [k for k, v in dependencies.items() if not v]
        
        if available and logger:
            logger.info(f"Available dependencies: {', '.join(available)}")
        if missing and logger:
            logger.warning(f"Missing dependencies: {', '.join(missing)}")
    
    def create_chat_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        workflow_type: str = "context_aware"
    ) -> "ChatSession":
        """Create a new chat session."""
        if not self.initialized:
            raise RuntimeError("Application not initialized")
        
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return ChatSession(
            app=self,
            user_id=user_id,
            session_id=session_id,
            workflow_type=workflow_type
        )
    
    def get_available_workflows(self) -> List[str]:
        """Get list of available workflow types."""
        workflows = ["simple_chat"]
        
        if self.dependencies.get("langgraph"):
            workflows.extend([
                "context_aware",
                "retrieval", 
                "memory_enhanced"
            ])
        elif self.dependencies.get("openai"):
            # Fallback workflows that work without LangGraph but with OpenAI
            workflows.append("context_aware_fallback")
        
        return workflows
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and status."""
        return {
            "initialized": self.initialized,
            "environment": self.config.environment,
            "debug": self.config.debug,
            "dependencies": self.dependencies,
            "available_workflows": self.get_available_workflows(),
            "available_tools": [tool["name"] for tool in tool_manager.list_tools()],
            "config": {
                "openai_model": self.config.openai.model,
                "temperature": self.config.openai.temperature,
                "vector_db_provider": self.config.vector_db.provider,
                "memory_buffer_size": self.config.memory.buffer_size
            }
        }


# =====================================
# CHAT SESSION CLASS
# =====================================

class ChatSession:
    """Chat session for handling conversations."""
    
    def __init__(self, app: AgenticAIApp, user_id: str, session_id: str, workflow_type: str):
        """Initialize chat session."""
        self.app = app
        self.user_id = user_id
        self.session_id = session_id
        self.workflow_type = workflow_type
        
        # Session state
        self.messages: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.memory: Dict[str, Any] = {}
        
        # Initialize workflow
        self._initialize_workflow()
        
        # Initialize workflow-specific state
        self._initialize_session_state()
        
        if logger:
            logger.info(f"Created chat session {session_id} for user {user_id}")
    
    def _initialize_session_state(self) -> None:
        """Initialize session state based on workflow type."""
        # Base state for all workflows
        self.context = {
            "workflow_type": self.workflow_type,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat()
        }
        
        self.memory = {
            "conversation_count": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Workflow-specific initialization
        if self.workflow_type == "retrieval":
            self.context.update({
                "retrieved_context": "",
                "retrieved_docs": [],
                "sources": [],
                "retrieval_metadata": {}
            })
        
        elif self.workflow_type == "memory_enhanced":
            self.memory.update({
                "summary": "",
                "buffer_memory": [],
                "vector_memory_ids": [],
                "memory_metadata": {},
                "user_profile": {}
            })
        
        elif self.workflow_type == "context_aware":
            self.context.update({
                "conversation_context": {},
                "user_preferences": {}
            })
            self.memory.update({
                "context_summary": "",
                "interaction_history": []
            })
        
        if logger:
            logger.debug(f"Initialized session state for workflow: {self.workflow_type}")
    
    def _initialize_workflow(self):
        """Initialize the workflow based on type."""
        try:
            if self.workflow_type == "simple_chat":
                self.workflow = None  # Simple chat doesn't need workflow
                
            elif self.workflow_type == "context_aware":
                if self.app.dependencies.get("langgraph"):
                    from src.graph import GraphFactory
                    graph = GraphFactory.create_context_aware_graph()
                    if graph:
                        from src.graph import WorkflowRunner
                        self.workflow = WorkflowRunner(graph)
                    else:
                        self.workflow = None
                        if logger:
                            logger.warning("Failed to create context-aware graph")
                else:
                    self.workflow = None
                    if logger:
                        logger.warning("LangGraph not available, using fallback chat")
                        
            elif self.workflow_type == "retrieval":
                if self.app.dependencies.get("langgraph"):
                    from src.graph import GraphFactory
                    graph = GraphFactory.create_retrieval_graph()
                    if graph:
                        from src.graph import WorkflowRunner
                        self.workflow = WorkflowRunner(graph)
                    else:
                        self.workflow = None
                        if logger:
                            logger.warning("Failed to create retrieval graph")
                else:
                    self.workflow = None
                    if logger:
                        logger.warning("LangGraph not available for retrieval workflow")
                        
            elif self.workflow_type == "memory_enhanced":
                if self.app.dependencies.get("langgraph"):
                    from src.graph import GraphFactory
                    graph = GraphFactory.create_memory_enhanced_graph()
                    if graph:
                        from src.graph import WorkflowRunner
                        self.workflow = WorkflowRunner(graph)
                    else:
                        self.workflow = None
                        if logger:
                            logger.warning("Failed to create memory-enhanced graph")
                else:
                    self.workflow = None
                    if logger:
                        logger.warning("LangGraph not available for memory-enhanced workflow")
                        
            elif self.workflow_type == "context_aware_fallback":
                # OpenAI-only fallback without LangGraph
                self.workflow = None
                if logger:
                    logger.info("Using OpenAI fallback mode for context-aware workflow")
                    
            else:
                self.workflow = None
                if logger:
                    logger.warning(f"Unknown workflow type: {self.workflow_type}")
                
        except Exception as e:
            if logger:
                logger.error(f"Failed to initialize workflow: {e}")
            self.workflow = None
    
    def send_message(self, content: str) -> Dict[str, Any]:
        """Send a message and get response."""
        if not content.strip():
            return {
                "error": "Empty message content",
                "timestamp": datetime.now()
            }
        
        # Create user message
        user_message = {
            "role": "user",
            "content": content.strip(),
            "timestamp": datetime.now()
        }
        
        # Update conversation metadata
        self.memory["conversation_count"] = self.memory.get("conversation_count", 0) + 1
        self.memory["last_updated"] = datetime.now().isoformat()
        
        # Add to messages
        self.messages.append(user_message)
        
        print(f"Workflow: {self.workflow_type}")
        print(self.workflow)
        
        try:
            # Process with workflow or fallback
            if self.workflow:
                response = self._process_with_workflow(user_message)
            else:
                response = self._process_fallback(user_message)
            
            # Add assistant response to messages
            if "messages" in response:
                self.messages.extend(response["messages"])
            
            return response
            
        except Exception as e:
            if logger:
                logger.error(f"Error processing message: {e}")
            
            error_response = {
                "role": "assistant",
                "content": f"I apologize, but I encountered an error: {str(e)}",
                "timestamp": datetime.now(),
                "error": True
            }
            
            self.messages.append(error_response)
            return {"messages": [error_response]}
    
    def _process_with_workflow(self, user_message: Dict[str, Any]) -> Dict[str, Any]:
        """Process message using LangGraph workflow."""
        if logger:
            logger.info(f"Processing message with workflow: {self.workflow_type}")
        
        if not self.workflow:
            # Fallback if workflow is None
            return self._process_fallback(user_message)
        
        try:
            # Get appropriate state based on workflow type
            state_dict = self._get_workflow_state(user_message)
            
            # Run workflow
            result = self.workflow.run(state_dict)
            
            logger.info(f"Workflow execution result: {result}")
            
            # Update session state with proper type handling
            self._update_session_state(result)
            
            return result
        
        except Exception as e:
            if logger:
                logger.error(f"Workflow execution failed: {e}")
            return self._process_fallback(user_message)
    
    def _process_fallback(self, user_message: Dict[str, Any]) -> Dict[str, Any]:
        """Process message using fallback (no LangGraph)."""
        content = user_message["content"]
        
        # Simple response generation
        if content.lower().startswith(("hello", "hi", "hey")):
            response_content = f"Hello! How can I help you today?"
        elif "?" in content:
            response_content = f"That's an interesting question about '{content}'. I'd be happy to help, but I have limited capabilities in this mode."
        elif content.lower() in ("bye", "goodbye", "see you"):
            response_content = "Goodbye! Have a great day!"
        else:
            response_content = f"I understand you said: '{content}'. I'm a basic AI assistant ready to help."
        
        # Check if it's a tool request
        if any(tool_name in content.lower() for tool_name in ["calculate", "convert", "analyze"]):
            tool_response = self._try_tool_execution(content)
            if tool_response:
                response_content = tool_response
        
        response_message = {
            "role": "assistant",
            "content": response_content,
            "timestamp": datetime.now(),
            "fallback_mode": True
        }
        
        return {"messages": [response_message]}
    
    def _try_tool_execution(self, content: str) -> Optional[str]:
        """Try to execute a tool based on message content."""
        try:
            # Simple tool detection and execution
            if "calculate" in content.lower() or "math" in content.lower():
                # Extract mathematical expression
                numbers_and_ops = re.findall(r'[\d+\-*/().]+', content)
                if numbers_and_ops:
                    expression = ''.join(numbers_and_ops)
                    result = tool_manager.execute_tool("calculator", expression)
                    if result.success:
                        return f"Calculation result: {result.result}"
            
            elif "convert" in content.lower():
                # Simple unit conversion detection
                if "fahrenheit" in content.lower() and "celsius" in content.lower():
                    numbers = re.findall(r'\d+', content)
                    if numbers:
                        value = float(numbers[0])
                        result = tool_manager.execute_tool("unit_converter", value, "fahrenheit", "celsius")
                        if result.success:
                            return f"Conversion result: {value}°F = {result.result}°C"
            
            elif "analyze" in content.lower():
                # Text analysis
                result = tool_manager.execute_tool("text_analyzer", content)
                if result.success:
                    analysis = result.result
                    return f"Text analysis: {analysis['word_count']} words, {analysis['sentence_count']} sentences, estimated reading time: {analysis['estimated_reading_time_minutes']} minutes"
        
        except Exception as e:
            if logger:
                logger.error(f"Tool execution error: {e}")
        
        return None
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history."""
        if limit:
            return self.messages[-limit:]
        return self.messages.copy()
    
    def clear_conversation(self):
        """Clear conversation history and reset session state."""
        self.messages.clear()
        
        # Reset conversation metadata but keep session info
        session_info = {
            "workflow_type": self.workflow_type,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.context.get("created_at", datetime.now().isoformat())
        }
        
        # Re-initialize session state
        self._initialize_session_state()
        
        # Restore session info
        self.context.update(session_info)
        
        if logger:
            logger.info(f"Cleared conversation for session {self.session_id}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "workflow_type": self.workflow_type,
            "message_count": len(self.messages),
            "has_workflow": self.workflow is not None,
            "context_keys": list(self.context.keys()),
            "memory_keys": list(self.memory.keys()),
            "workflow_state": {
                "context_size": len(str(self.context)),
                "memory_size": len(str(self.memory)),
                "last_updated": self.memory.get("last_updated", "unknown")
            }
        }
    
    def _update_session_state(self, result: Dict[str, Any]) -> None:
        """Update session state with proper type handling for context and memory."""
        if "context" in result:
            context_data = result["context"]
            
            # Handle different context types based on workflow
            if isinstance(context_data, dict):
                # For context_aware and memory_enhanced workflows
                self.context.update(context_data)
            elif isinstance(context_data, str):
                # For retrieval workflows where context might be a string
                if self.workflow_type == "retrieval":
                    self.context["retrieved_context"] = context_data
                else:
                    self.context["context_string"] = context_data
            elif isinstance(context_data, list):
                # For workflows that return context as a list
                self.context["context_list"] = context_data
            else:
                # Fallback for other types
                self.context["context_data"] = context_data
                if logger:
                    logger.warning(f"Unexpected context type: {type(context_data)}")
        
        if "memory" in result:
            memory_data = result["memory"]
            
            # Handle different memory types based on workflow
            if isinstance(memory_data, dict):
                # For memory_enhanced and context_aware workflows
                self.memory.update(memory_data)
            elif isinstance(memory_data, str):
                # For workflows where memory is a summary string
                if self.workflow_type == "memory_enhanced":
                    self.memory["summary"] = memory_data
                elif self.workflow_type == "context_aware":
                    self.memory["context_summary"] = memory_data
                else:
                    self.memory["memory_string"] = memory_data
            elif isinstance(memory_data, list):
                # For workflows that return memory as a list (e.g., conversation buffer)
                self.memory["buffer"] = memory_data
            else:
                # Fallback for other types
                self.memory["memory_data"] = memory_data
                if logger:
                    logger.warning(f"Unexpected memory type: {type(memory_data)}")
        
        # Handle other workflow-specific state updates
        if "retrieved_docs" in result:
            self.context["retrieved_docs"] = result["retrieved_docs"]
        
        if "sources" in result:
            self.context["sources"] = result["sources"]
        
        if "retrieval_metadata" in result:
            self.context["retrieval_metadata"] = result["retrieval_metadata"]
        
        if "user_profile" in result:
            self.memory["user_profile"] = result["user_profile"]
        
        if "buffer_memory" in result:
            self.memory["buffer_memory"] = result["buffer_memory"]
        
        if "summary_memory" in result:
            self.memory["summary_memory"] = result["summary_memory"]
        
        if "vector_memory_ids" in result:
            self.memory["vector_memory_ids"] = result["vector_memory_ids"]
        
        if "memory_metadata" in result:
            self.memory["memory_metadata"] = result["memory_metadata"]
        
        if logger:
            logger.debug(f"Updated session state for workflow type: {self.workflow_type}")
            logger.debug(f"Context keys: {list(self.context.keys())}")
            logger.debug(f"Memory keys: {list(self.memory.keys())}")
            logger.debug(f"Memory: {self.memory}")
    
    def _get_workflow_state(self, user_message: Dict[str, Any]) -> Dict[str, Any]:
        """Get state dictionary appropriate for the workflow type."""
        base_state = {
            "messages": self.messages.copy(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": datetime.now()
        }
        
        if self.workflow_type == "retrieval":
            return {
                **base_state,
                "query": user_message["content"],
                "retrieved_docs": self.context.get("retrieved_docs", []),
                "context": self.context.get("retrieved_context", ""),
                "sources": self.context.get("sources", []),
                "retrieval_metadata": self.context.get("retrieval_metadata", {})
            }
        
        elif self.workflow_type == "memory_enhanced":
            return {
                **base_state,
                "buffer_memory": self.memory.get("buffer_memory", []),
                "summary_memory": self.memory.get("summary", ""),
                "vector_memory_ids": self.memory.get("vector_memory_ids", []),
                "memory_metadata": self.memory.get("memory_metadata", {}),
                "user_profile": self.memory.get("user_profile", {})
            }
        
        elif self.workflow_type == "context_aware":
            return {
                **base_state,
                "context": self.context.copy(),
                "memory": self.memory.copy()
            }
        
        else:
            # Default state for unknown workflow types
            return {
                **base_state,
                "context": self.context.copy(),
                "memory": self.memory.copy()
            }
       
    
# =====================================
# COMMAND LINE INTERFACE
# =====================================

def create_cli():
    """Create a simple command line interface."""
    print("🤖 Agentic AI - Context-Aware LLM Application")
    print("=" * 50)
    
    try:
        # Initialize application
        app = AgenticAIApp()
        
        # Show system info
        system_info = app.get_system_info()
        print(f"Environment: {system_info['environment']}")
        print(f"Available workflows: {', '.join(system_info['available_workflows'])}")
        print(f"Available tools: {', '.join(system_info['available_tools'])}")
        print(f"Dependencies: {', '.join([k for k, v in system_info['dependencies'].items() if v])}")
        print()
        
        # Create chat session
        user_id = input("Enter your user ID (or press Enter for 'user'): ").strip() or "user"
        
        # Select workflow
        workflows = system_info['available_workflows']
        print(f"Available workflows: {', '.join(workflows)}")
        workflow = input(f"Select workflow (default: {workflows[0]}): ").strip() or workflows[0]
        
        if workflow not in workflows:
            print(f"Unknown workflow '{workflow}', using '{workflows[0]}'")
            workflow = workflows[0]
        
        session = app.create_chat_session(user_id=user_id, workflow_type=workflow)
        
        print(f"\n💬 Chat session started! (Type 'quit' to exit, 'clear' to clear history)")
        print(f"Session ID: {session.session_id}")
        print(f"Workflow: {workflow}")
        print("-" * 50)
        
        # Chat loop
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == "clear":
                    session.clear_conversation()
                    print("Conversation cleared! 🗑️")
                    continue
                
                if user_input.lower() == "info":
                    info = session.get_session_info()
                    print(f"Session info: {info}")
                    continue
                
                if not user_input:
                    continue
                
                # Send message and get response
                response = session.send_message(user_input)
                
                # Display response
                if "messages" in response:
                    for message in response["messages"]:
                        if message.get("role") == "assistant":
                            content = message.get("content", "")
                            error = message.get("error", False)
                            fallback = message.get("fallback_mode", False)
                            
                            prefix = "🤖"
                            if error:
                                prefix = "❌"
                            elif fallback:
                                prefix = "🔄"
                            
                            print(f"{prefix} Assistant: {content}")
                elif "error" in response:
                    print(f"❌ Error: {response['error']}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return 1
    
    return 0


# =====================================
# MAIN ENTRY POINT
# =====================================

def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        return create_cli()
    else:
        # Default: show system info
        try:
            app = AgenticAIApp()
            system_info = app.get_system_info()
            
            print("🤖 Agentic AI System Information")
            print("=" * 40)
            print(f"Initialized: {system_info['initialized']}")
            print(f"Environment: {system_info['environment']}")
            print(f"Available workflows: {', '.join(system_info['available_workflows'])}")
            print(f"Available tools: {', '.join(system_info['available_tools'])}")
            print(f"Dependencies: {', '.join([k for k, v in system_info['dependencies'].items() if v])}")
            print()
            print("To start interactive chat, run: python main.py --cli")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
