"""
Prompt templates and management for LLM interactions.

This module contains reusable prompt templates for various use cases
including context-aware conversations, retrieval-augmented generation,
and memory-enhanced interactions.
"""

from typing import Dict, List, Any, Optional


# =====================================
# SYSTEM PROMPTS
# =====================================

CONTEXT_AWARE_SYSTEM_PROMPT = """You are a helpful AI assistant with access to conversation context and memory.

Your capabilities include:
- Maintaining awareness of the ongoing conversation
- Accessing relevant information from previous interactions
- Using retrieved documents to provide accurate information
- Adapting your responses based on user preferences and context

Guidelines:
1. Always consider the conversation history when formulating responses
2. Use provided context and retrieved information when relevant
3. Be transparent about your knowledge limitations
4. Maintain consistency with previous statements in the conversation
5. Ask clarifying questions when context is ambiguous

Current conversation context: {context}
User preferences: {user_preferences}
"""

RETRIEVAL_SYSTEM_PROMPT = """You are a knowledgeable assistant that uses retrieved documents to answer questions accurately.

Your approach:
1. Carefully analyze the retrieved documents
2. Synthesize information from multiple sources when available
3. Provide specific citations or references when possible
4. Acknowledge when retrieved information is insufficient
5. Distinguish between information from documents vs. your general knowledge

Retrieved documents: {retrieved_docs}
Sources: {sources}

Answer the user's question based on the retrieved information.
"""

MEMORY_ENHANCED_SYSTEM_PROMPT = """You are an AI assistant with advanced memory capabilities.

Memory systems available:
- Buffer Memory: Recent conversation messages
- Summary Memory: Condensed conversation history
- Vector Memory: Semantic search of past interactions

Instructions:
1. Use buffer memory for immediate context
2. Reference summary memory for long-term conversation themes
3. Leverage vector memory for semantically relevant past interactions
4. Update user profile based on preferences and patterns
5. Maintain conversation coherence across sessions

Buffer Memory: {buffer_memory}
Summary Memory: {summary_memory}
Relevant Past Interactions: {vector_memory}
User Profile: {user_profile}
"""

# =====================================
# TASK-SPECIFIC PROMPTS
# =====================================

DOCUMENT_QA_TEMPLATE = """Based on the following context, please answer the question. If the answer cannot be found in the context, please say "I don't have enough information to answer this question based on the provided context."

Context:
{context}

Question: {question}

Answer:"""

CONVERSATION_SUMMARY_TEMPLATE = """Progressively summarize the conversation, adding to the existing summary.

Existing summary:
{existing_summary}

New conversation to add:
{conversation_history}

Updated summary:"""

USER_PREFERENCE_EXTRACTION_TEMPLATE = """Analyze the following conversation and extract user preferences, interests, and communication style.

Conversation:
{conversation}

Extract the following in JSON format:
{{
    "preferences": {{}},
    "interests": [],
    "communication_style": "",
    "topics_of_interest": [],
    "expertise_level": ""
}}

Extracted information:"""

CONTEXT_COMPRESSION_TEMPLATE = """Given the following context and query, extract only the most relevant information that would help answer the query.

Query: {query}

Full Context:
{context}

Compressed Relevant Context:"""

# =====================================
# SIMPLE PROMPT TEMPLATE CLASS
# =====================================

class SimplePromptTemplate:
    """A simple prompt template class for when LangChain is not available."""
    
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        missing_vars = set(self.input_variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        return self.template.format(**kwargs)


# Create simple prompt template instances
DOCUMENT_QA_PROMPT = SimplePromptTemplate(
    template=DOCUMENT_QA_TEMPLATE,
    input_variables=["context", "question"]
)

CONVERSATION_SUMMARY_PROMPT = SimplePromptTemplate(
    template=CONVERSATION_SUMMARY_TEMPLATE,
    input_variables=["conversation_history", "existing_summary"]
)

USER_PREFERENCE_EXTRACTION_PROMPT = SimplePromptTemplate(
    template=USER_PREFERENCE_EXTRACTION_TEMPLATE,
    input_variables=["conversation"]
)

CONTEXT_COMPRESSION_PROMPT = SimplePromptTemplate(
    template=CONTEXT_COMPRESSION_TEMPLATE,
    input_variables=["context", "query"]
)


# =====================================
# CHAT PROMPT CREATION FUNCTIONS
# =====================================

def create_context_aware_chat_prompt() -> Dict[str, Any]:
    """Create a chat prompt configuration for context-aware conversations."""
    return {
        "system": CONTEXT_AWARE_SYSTEM_PROMPT,
        "human": "{input}",
        "variables": ["context", "user_preferences", "input"]
    }


def create_retrieval_chat_prompt() -> Dict[str, Any]:
    """Create a chat prompt configuration for retrieval-augmented generation."""
    return {
        "system": RETRIEVAL_SYSTEM_PROMPT,
        "human": "Question: {question}",
        "variables": ["retrieved_docs", "sources", "question"]
    }


def create_memory_enhanced_chat_prompt() -> Dict[str, Any]:
    """Create a chat prompt configuration for memory-enhanced conversations."""
    return {
        "system": MEMORY_ENHANCED_SYSTEM_PROMPT,
        "human": "{input}",
        "variables": ["buffer_memory", "summary_memory", "vector_memory", "user_profile", "input"]
    }


# =====================================
# DYNAMIC PROMPT BUILDERS
# =====================================

def build_rag_prompt(
    context: str,
    question: str,
    sources: Optional[List[str]] = None,
    additional_instructions: Optional[str] = None
) -> str:
    """Build a retrieval-augmented generation prompt dynamically."""
    
    prompt_parts = [
        "You are a helpful assistant that answers questions based on provided context.",
        "",
        "Context:",
        context,
        ""
    ]
    
    if sources:
        prompt_parts.extend([
            "Sources:",
            "\n".join(f"- {source}" for source in sources),
            ""
        ])
    
    if additional_instructions:
        prompt_parts.extend([
            "Additional Instructions:",
            additional_instructions,
            ""
        ])
    
    prompt_parts.extend([
        f"Question: {question}",
        "",
        "Answer:"
    ])
    
    return "\n".join(prompt_parts)


def build_conversation_prompt(
    current_input: str,
    conversation_history: List[Dict[str, Any]],
    user_context: Optional[Dict[str, Any]] = None,
    system_instructions: Optional[str] = None
) -> str:
    """Build a conversation prompt with history and context."""
    
    prompt_parts = []
    
    # Add system instructions if provided
    if system_instructions:
        prompt_parts.extend([
            "System Instructions:",
            system_instructions,
            ""
        ])
    
    # Add user context if available
    if user_context:
        prompt_parts.extend([
            "User Context:",
            str(user_context),
            ""
        ])
    
    # Add conversation history
    if conversation_history:
        prompt_parts.append("Conversation History:")
        for msg in conversation_history[-5:]:  # Last 5 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", str(msg))
            prompt_parts.append(f"{role.title()}: {content}")
        prompt_parts.append("")
    
    # Add current input
    prompt_parts.extend([
        f"Current Input: {current_input}",
        "",
        "Response:"
    ])
    
    return "\n".join(prompt_parts)


def build_memory_context_prompt(
    input_text: str,
    buffer_memory: List[Dict[str, Any]],
    summary_memory: str = "",
    vector_memory: Optional[List[Dict[str, Any]]] = None,
    user_profile: Optional[Dict[str, Any]] = None
) -> str:
    """Build a prompt that incorporates multiple memory types."""
    
    if vector_memory is None:
        vector_memory = []
    if user_profile is None:
        user_profile = {}
    
    prompt_parts = [
        "You have access to multiple types of memory to provide contextual responses.",
        ""
    ]
    
    # Buffer memory (recent conversation)
    if buffer_memory:
        prompt_parts.extend([
            "Recent Conversation:",
            "\n".join(f"- {msg.get('content', str(msg))}" for msg in buffer_memory[-3:]),
            ""
        ])
    
    # Summary memory
    if summary_memory:
        prompt_parts.extend([
            "Conversation Summary:",
            summary_memory,
            ""
        ])
    
    # Vector memory (semantically relevant)
    if vector_memory:
        prompt_parts.extend([
            "Relevant Past Interactions:",
            "\n".join(f"- {interaction.get('content', str(interaction))}" for interaction in vector_memory[:2]),
            ""
        ])
    
    # User profile
    if user_profile:
        prompt_parts.extend([
            "User Profile:",
            str(user_profile),
            ""
        ])
    
    prompt_parts.extend([
        f"Current Input: {input_text}",
        "",
        "Response:"
    ])
    
    return "\n".join(prompt_parts)


# =====================================
# PROMPT VALIDATION AND UTILITIES
# =====================================

def validate_prompt_variables(template: str, variables: Dict[str, Any]) -> bool:
    """Validate that all required variables are provided for a template."""
    import re
    
    # Find all variables in the template
    template_vars = set(re.findall(r'{(\w+)}', template))
    provided_vars = set(variables.keys())
    
    missing_vars = template_vars - provided_vars
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")
    
    return True


def format_prompt_with_fallbacks(
    template: str,
    variables: Dict[str, Any],
    fallbacks: Optional[Dict[str, str]] = None
) -> str:
    """Format a prompt template with fallback values for missing variables."""
    
    if fallbacks:
        # Apply fallbacks for missing variables
        for var, fallback in fallbacks.items():
            if var not in variables:
                variables[var] = fallback
    
    try:
        return template.format(**variables)
    except KeyError as e:
        missing_var = str(e).strip("'")
        raise ValueError(f"Required variable '{missing_var}' not provided and no fallback available")


def truncate_context(context: str, max_tokens: int = 4000) -> str:
    """Truncate context to fit within token limits (rough estimation)."""
    # Rough estimation: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    
    if len(context) <= max_chars:
        return context
    
    # Truncate and add indication
    truncated = context[:max_chars - 50]
    return truncated + "\n\n[Context truncated due to length limits]"


# =====================================
# PRESET PROMPT CONFIGURATIONS
# =====================================

PROMPT_CONFIGS = {
    "basic_chat": {
        "template": create_context_aware_chat_prompt(),
        "required_vars": ["context", "user_preferences", "input"]
    },
    "document_qa": {
        "template": DOCUMENT_QA_PROMPT,
        "required_vars": ["context", "question"]
    },
    "conversation_summary": {
        "template": CONVERSATION_SUMMARY_PROMPT,
        "required_vars": ["conversation_history", "existing_summary"]
    },
    "memory_enhanced": {
        "template": create_memory_enhanced_chat_prompt(),
        "required_vars": ["buffer_memory", "summary_memory", "vector_memory", "user_profile", "input"]
    }
}


def get_prompt_config(name: str) -> Dict[str, Any]:
    """Get a preset prompt configuration by name."""
    if name not in PROMPT_CONFIGS:
        raise ValueError(f"Unknown prompt configuration: {name}. Available: {list(PROMPT_CONFIGS.keys())}")
    
    return PROMPT_CONFIGS[name]
