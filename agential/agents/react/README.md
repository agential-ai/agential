# ReAct Agent - Simplified & Scalable

This folder contains a simplified and scalable implementation of the ReAct agent.

## Files

- **`agent.py`** - Main agent implementation with plugin-based architecture, output structures, and constants
- **`handlers.py`** - Benchmark handlers and registry
- **`prompts.py`** - Prompt templates for different benchmarks
- **`example_usage.py`** - Example usage of the agent
- **`__init__.py`** - Exports the main ReAct agent

## Key Features

### 🚀 **Plugin-Based Architecture**
- Easy to add new benchmarks without modifying core code
- Handler classes for different benchmark types (QA, Math, Code)
- Runtime registration of new benchmarks
- **Flexible handler selection** - auto-detect or specify custom handlers

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

### Method 1: Simple Auto-Handler (Recommended for most cases)

```python
from agential.agents.react.agent import add_benchmark

# Add a new benchmark - handler automatically determined
add_benchmark(
    benchmark_name="my_qa_benchmark",
    prompt="Your custom prompt template here with {question}, {examples}, {max_steps}, {scratchpad}"
)

# Use immediately
agent = ReAct(llm, "my_qa_benchmark")
```

### Method 2: Full Custom Handler (For special requirements)

```python
from agential.agents.react.handlers import QAHandler

class MyCustomHandler(QAHandler):
    def get_prompt(self) -> str:
        return "Your custom prompt"
    
    def parse_action(self, action: str):
        # Custom parsing logic
        return action_type, query
    
    def handle_observation(self, action_type, query, scratchpad):
        # Custom observation handling
        return obs, answer, finished, external_info

# Register it
add_benchmark(
    benchmark_name="my_benchmark",
    prompt="",  # Prompt handled by custom handler
    handler_class=MyCustomHandler
)
```

## Handler Types

- **QAHandler**: For question-answering tasks (Search, Lookup, Finish actions)
- **MathHandler**: For mathematical reasoning (Calculate, Finish actions)  
- **CodeHandler**: For code generation (Implement, Test, Finish actions)

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
print(f"Steps taken: {result.num_steps}")
print(f"Total cost: ${result.total_cost:.4f}")
```

## Dependencies

- `rich` - For colorful logging and real-time output
- `langchain-community` - For Wikipedia search functionality

These dependencies are required for the ReAct agent to function properly. 