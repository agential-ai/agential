# ReAct Agent - Simplified & Scalable

This folder contains a simplified and scalable implementation of the ReAct agent.

## Files

- **`agent.py`** - Main agent implementation with plugin-based architecture, output structures, and constants
- **`prompts.py`** - Prompt templates for different benchmarks
- **`example_usage.py`** - Example usage of the agent
- **`__init__.py`** - Exports the main ReAct agent

## Key Features

### 🚀 **Plugin-Based Architecture**
- Easy to add new benchmarks without modifying core code
- Handler classes for different benchmark types (QA, Math, Code)
- Runtime registration of new benchmarks

### 📊 **Professional Logging**
- Rich library integration for colorful output
- Real-time step-by-step progress
- Detailed metrics tracking
- Structured logging with serialization support

### 🔍 **Search Tools**
- Wikipedia integration via langchain-community
- Graceful fallback when dependencies are missing
- Search and lookup functionality for QA benchmarks

### 🎯 **Scalability**
- Add new benchmarks by creating handler classes
- No need to modify the main agent class
- Clean separation of concerns

## Adding a New Benchmark

```python
# 1. Create a handler class
class MyNewBenchmarkHandler(QAHandler):  # or MathHandler, CodeHandler
    def get_prompt(self) -> str:
        return "Your custom prompt here"
    
    def handle_observation(self, action_type, query, scratchpad):
        # Custom observation handling
        pass

# 2. Register it
ReAct.register_benchmark("my_benchmark", MyNewBenchmarkHandler)

# 3. Use it
agent = ReAct(llm, "my_benchmark")
```

## Usage

```python
from agential.agents.react import ReAct
from agential.core.llm import YourLLM

# Create agent
agent = ReAct(
    llm=YourLLM(),
    benchmark="gsm8k",
    max_steps=6,
    verbose=True
)

# Generate answer
result = agent.generate("What is 2 + 2?")
print(result.answer)
```

## Dependencies

- `rich` - For colorful logging and real-time output
- `langchain-community` - For Wikipedia search functionality

These dependencies are required for the ReAct agent to function properly. 