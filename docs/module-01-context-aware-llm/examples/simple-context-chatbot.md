# Example 1: Simple Context-Aware Chatbot

This example demonstrates how to build a basic context-aware chatbot using our framework. The chatbot maintains conversation history and can reference previous messages.

## Code

```python
#!/usr/bin/env python3
"""
Simple Context-Aware Chatbot Example

This example shows how to:
1. Maintain conversation history
2. Use context in responses
3. Handle memory management
4. Implement basic error handling
"""

import sys
import json
from datetime import datetime
from typing import List, Dict, Any

from src.configuration import get_config
from src.states import ConversationState, Message
from src.utils import setup_logging

try:
    import openai
except ImportError:
    print("OpenAI package not available. Install with: uv pip install openai")
    sys.exit(1)


class SimpleContextChatbot:
    """A simple chatbot that maintains conversation context."""
    
    def __init__(self):
        """Initialize the chatbot with configuration and state."""
        self.config = get_config()
        self.logger = setup_logging("SimpleContextChatbot")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.config.openai.api_key)
        
        # Initialize conversation state
        self.state = ConversationState(
            conversation_id=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            messages=[],
            context={
                "user_name": None,
                "topics_discussed": [],
                "conversation_start": datetime.now().isoformat()
            },
            user_id="demo_user"
        )
        
        # Add system message
        self.add_system_message()
        
        self.logger.info(f"Chatbot initialized with conversation ID: {self.state.conversation_id}")
    
    def add_system_message(self):
        """Add initial system message to set chatbot behavior."""
        system_message = Message(
            role="system",
            content="""You are a helpful and friendly assistant. You have access to our conversation history, 
so you can reference previous messages and maintain context throughout our chat. 

Key behaviors:
- Remember what we've discussed earlier in the conversation
- Reference previous topics when relevant
- Ask clarifying questions when needed
- Be conversational and engaging
- If the user tells you their name, remember it and use it occasionally

Current conversation context will be provided with each message.""",
            timestamp=datetime.now().isoformat()
        )
        
        self.state.messages.append(system_message)
    
    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        message = Message(
            role="user",
            content=content,
            timestamp=datetime.now().isoformat()
        )
        
        self.state.messages.append(message)
        self.update_context(content)
        self.logger.info(f"User message added: {content[:50]}...")
    
    def add_assistant_message(self, content: str):
        """Add an assistant message to the conversation."""
        message = Message(
            role="assistant",
            content=content,
            timestamp=datetime.now().isoformat()
        )
        
        self.state.messages.append(message)
        self.logger.info(f"Assistant message added: {content[:50]}...")
    
    def update_context(self, user_input: str):
        """Update conversation context based on user input."""
        # Extract user name if mentioned
        if "my name is" in user_input.lower() or "i'm " in user_input.lower():
            # Simple name extraction (could be improved with NLP)
            words = user_input.lower().split()
            for i, word in enumerate(words):
                if word in ["name", "i'm", "im"] and i + 1 < len(words):
                    potential_name = words[i + 1].strip(".,!?").title()
                    if potential_name.isalpha():
                        self.state.context["user_name"] = potential_name
                        break
        
        # Track topics (simple keyword extraction)
        topics = ["weather", "food", "music", "movies", "sports", "technology", "travel"]
        for topic in topics:
            if topic in user_input.lower() and topic not in self.state.context["topics_discussed"]:
                self.state.context["topics_discussed"].append(topic)
    
    def build_context_prompt(self) -> str:
        """Build a context summary for the current conversation."""
        context_parts = []
        
        # Add conversation metadata
        context_parts.append(f"Conversation ID: {self.state.conversation_id}")
        context_parts.append(f"Started: {self.state.context['conversation_start']}")
        
        # Add user information
        if self.state.context.get("user_name"):
            context_parts.append(f"User name: {self.state.context['user_name']}")
        
        # Add topics discussed
        if self.state.context["topics_discussed"]:
            topics = ", ".join(self.state.context["topics_discussed"])
            context_parts.append(f"Topics discussed: {topics}")
        
        # Add recent message count
        context_parts.append(f"Messages in conversation: {len(self.state.messages) - 1}")  # -1 for system message
        
        return "\\n".join(context_parts)
    
    def prepare_messages_for_api(self) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API call with context."""
        api_messages = []
        
        # Convert our messages to OpenAI format
        for message in self.state.messages:
            if message.role == "system":
                # Enhance system message with current context
                enhanced_content = f"{message.content}\\n\\nCurrent conversation context:\\n{self.build_context_prompt()}"
                api_messages.append({
                    "role": "system",
                    "content": enhanced_content
                })
            else:
                api_messages.append({
                    "role": message.role,
                    "content": message.content
                })
        
        # Limit message history to avoid token limits
        max_messages = self.config.memory.max_messages
        if len(api_messages) > max_messages:
            # Keep system message and recent messages
            system_msg = api_messages[0]
            recent_messages = api_messages[-(max_messages-1):]
            api_messages = [system_msg] + recent_messages
            self.logger.info(f"Truncated messages to {len(api_messages)} for API call")
        
        return api_messages
    
    def get_response(self, user_input: str) -> str:
        """Get a response from the chatbot."""
        try:
            # Add user message
            self.add_user_message(user_input)
            
            # Prepare messages for API
            api_messages = self.prepare_messages_for_api()
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.config.openai.model,
                messages=api_messages,
                temperature=self.config.openai.temperature,
                max_tokens=self.config.openai.max_tokens
            )
            
            assistant_response = response.choices[0].message.content
            
            # Add assistant message
            self.add_assistant_message(assistant_response)
            
            return assistant_response
            
        except Exception as e:
            self.logger.error(f"Error getting response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        return {
            "conversation_id": self.state.conversation_id,
            "message_count": len(self.state.messages) - 1,  # -1 for system message
            "context": self.state.context,
            "duration": "Active session"
        }
    
    def save_conversation(self, filename: str = None):
        """Save the conversation to a JSON file."""
        if filename is None:
            filename = f"conversation_{self.state.conversation_id}.json"
        
        conversation_data = {
            "conversation_id": self.state.conversation_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in self.state.messages
            ],
            "context": self.state.context,
            "summary": self.get_conversation_summary()
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Conversation saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving conversation: {e}")
            return None


def main():
    """Main function to run the chatbot."""
    print("🤖 Simple Context-Aware Chatbot")
    print("=" * 50)
    print("This chatbot remembers our conversation!")
    print("Try mentioning your name, asking about previous topics, or having a natural conversation.")
    print("Commands: 'quit' to exit, 'summary' for conversation summary, 'save' to save conversation")
    print("=" * 50)
    
    # Initialize chatbot
    try:
        chatbot = SimpleContextChatbot()
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        return
    
    # Main conversation loop
    while True:
        try:
            user_input = input("\\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() == 'quit':
                # Offer to save conversation
                save_choice = input("\\nWould you like to save this conversation? (y/n): ").strip().lower()
                if save_choice in ['y', 'yes']:
                    filename = chatbot.save_conversation()
                    if filename:
                        print(f"💾 Conversation saved as {filename}")
                
                print("👋 Thanks for chatting! Goodbye!")
                break
            
            elif user_input.lower() == 'summary':
                summary = chatbot.get_conversation_summary()
                print("\\n📊 Conversation Summary:")
                print(f"ID: {summary['conversation_id']}")
                print(f"Messages: {summary['message_count']}")
                if summary['context']['user_name']:
                    print(f"User: {summary['context']['user_name']}")
                if summary['context']['topics_discussed']:
                    print(f"Topics: {', '.join(summary['context']['topics_discussed'])}")
                continue
            
            elif user_input.lower() == 'save':
                filename = chatbot.save_conversation()
                if filename:
                    print(f"💾 Conversation saved as {filename}")
                else:
                    print("❌ Failed to save conversation")
                continue
            
            # Get chatbot response
            response = chatbot.get_response(user_input)
            print(f"\\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\\n\\n👋 Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\\n❌ Unexpected error: {e}")
            print("The conversation will continue...")


if __name__ == "__main__":
    main()
```

## How to Run

1. Make sure your environment is set up (see Exercise 1)
2. Save this code as `examples/simple_context_chatbot.py`
3. Run: `python examples/simple_context_chatbot.py`

## What This Example Demonstrates

### Context Awareness
- **Message History**: Maintains full conversation history
- **User Information**: Remembers user's name when mentioned
- **Topic Tracking**: Keeps track of topics discussed
- **Context Integration**: Provides context summary to the LLM

### Memory Management
- **Message Limiting**: Prevents token limit issues by truncating old messages
- **State Persistence**: Can save conversations to JSON files
- **Context Updates**: Dynamically updates context based on user input

### Error Handling
- **API Errors**: Gracefully handles OpenAI API errors
- **Configuration Issues**: Validates configuration on startup
- **User Interruption**: Handles Ctrl+C gracefully

## Example Interaction

```
🤖 Simple Context-Aware Chatbot
==================================================

You: Hi there! My name is Alice and I love photography.
