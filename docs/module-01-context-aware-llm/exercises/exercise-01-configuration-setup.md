# Exercise 1: Configuration and Basic Setup

## Objective
Set up your development environment and create your first context-aware LLM application configuration.

## Prerequisites
- Python 3.11+
- OpenAI API key
- Text editor or IDE

## Part 1: Environment Setup (30 minutes)

### Step 1: Project Setup
1. Ensure you have the project cloned and virtual environment activated
2. Install dependencies: `uv pip install -e .`
3. Verify installation: `python -c "import src.configuration; print('Success!')"`

### Step 2: Configuration File
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key
3. Configure the following settings:
   ```bash
   OPENAI_API_KEY=your_key_here
   OPENAI_MODEL=gpt-3.5-turbo
   OPENAI_TEMPERATURE=0.7
   VECTOR_DB_PROVIDER=chroma
   VECTOR_DB_COLLECTION=exercise1
   MEMORY_MAX_MESSAGES=50
   ```

### Step 3: Test Your Configuration
Create a file `test_my_config.py`:

```python
#!/usr/bin/env python3
"""Test script for Exercise 1."""

from src.configuration import get_config, validate_config
import os

def main():
    print("🔧 Testing Configuration...")
    
    # Test 1: Load configuration
    try:
        config = get_config()
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return
    
    # Test 2: Validate configuration
    try:
        validate_config(config)
        print("✅ Configuration validation passed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return
    
    # Test 3: Display current settings
    print("\n📋 Current Configuration:")
    print(f"OpenAI Model: {config.openai.model}")
    print(f"Temperature: {config.openai.temperature}")
    print(f"Vector DB: {config.vector_db.provider}")
    print(f"Collection: {config.vector_db.collection_name}")
    print(f"Memory Limit: {config.memory.max_messages}")
    
    # Test 4: Check for API key (without exposing it)
    api_key = config.openai.api_key
    if api_key and len(api_key) > 10:
        print(f"✅ OpenAI API key configured (ends with: ...{api_key[-4:]})")
    else:
        print("❌ OpenAI API key not properly configured")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
```

Run the test: `python test_my_config.py`

## Part 2: Simple LLM Interaction (20 minutes)

Create `simple_chat.py`:

```python
#!/usr/bin/env python3
"""Simple chat example for Exercise 1."""

import openai
from src.configuration import get_config

def simple_chat():
    """Demonstrate basic OpenAI interaction."""
    config = get_config()
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=config.openai.api_key)
    
    print("🤖 Simple Chat (type 'quit' to exit)")
    print("=" * 40)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            # Make API call
            response = client.chat.completions.create(
                model=config.openai.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ],
                temperature=config.openai.temperature,
                max_tokens=config.openai.max_tokens
            )
            
            assistant_response = response.choices[0].message.content
            print(f"\nAssistant: {assistant_response}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    simple_chat()
```

Run the chat: `python simple_chat.py`

## Part 3: Configuration Experimentation (20 minutes)

### Experiment 1: Temperature Effects
Modify your `.env` file and test different temperature values:

1. Set `OPENAI_TEMPERATURE=0.1` (very deterministic)
2. Ask the same question multiple times: "Tell me a creative story about a robot."
3. Set `OPENAI_TEMPERATURE=1.5` (very creative)
4. Ask the same question again
5. Compare the differences in responses

### Experiment 2: Model Comparison
If you have access to different models:

1. Try `OPENAI_MODEL=gpt-3.5-turbo`
2. Try `OPENAI_MODEL=gpt-4` (if available)
3. Ask complex questions and compare response quality

### Experiment 3: Token Limits
1. Set `OPENAI_MAX_TOKENS=50`
2. Ask for a long explanation of something
3. Set `OPENAI_MAX_TOKENS=500`
4. Ask the same question and observe the difference

## Part 4: Using the Main Application (15 minutes)

### Step 1: Explore Available Commands
```bash
python main.py --help
```

### Step 2: Start a Chat Session
```bash
python main.py chat
```

### Step 3: Try Different Workflows
```bash
python main.py workflow --type context-aware
python main.py workflow --type retrieval
```

### Step 4: Use Built-in Tools
```bash
python main.py tools list
python main.py tools calculator "2 + 2 * 3"
```

## Deliverables

1. **Screenshot** of successful configuration test
2. **Screenshot** of working simple chat
3. **Documentation** of your temperature experiments:
   - What question did you ask?
   - How did responses differ between temperature 0.1 and 1.5?
   - Which temperature setting do you prefer for different use cases?

## Reflection Questions

1. **Configuration Management**: Why is it important to separate configuration from code?
2. **Environment Variables**: What are the security benefits of using environment variables for API keys?
3. **Temperature Parameter**: When would you use a low temperature vs. high temperature?
4. **Error Handling**: What happens when your API key is invalid? How could we improve error handling?

## Common Issues and Solutions

### Issue: "No module named 'src'"
**Solution**: Run `uv pip install -e .` from the project root

### Issue: "OpenAI API key is required"
**Solution**: Check that your `.env` file has `OPENAI_API_KEY=your_actual_key`

### Issue: API rate limit errors
**Solution**: Add delays between requests or reduce `max_tokens`

### Issue: Import errors with openai package
**Solution**: Ensure openai package is installed: `uv pip install openai`

## Extension Challenges

### Challenge 1: Custom Configuration Validator
Add a custom validator that checks if the OpenAI model exists:

```python
def validate_openai_model(model: str) -> bool:
    """Check if the specified model is available."""
    valid_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    return model in valid_models
```

### Challenge 2: Configuration from File
Implement loading configuration from a YAML file instead of environment variables.

### Challenge 3: Dynamic Configuration
Create a system that can reload configuration without restarting the application.

## Next Steps

After completing this exercise, you should:
- Have a working development environment
- Understand configuration management principles
- Be able to interact with OpenAI's API
- Know how to use the main application

In the next exercise, we'll build our first context-aware application with memory!
