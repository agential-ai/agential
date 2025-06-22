"""Example usage of the ReAct agent."""

from agential.agents.react import ReAct
from agential.core.llm import LLM
from agential.agents.react.agent import add_benchmark
from agential.agents.react.handlers import QAHandler, MathHandler

# Initialize the language model
llm = LLM("gpt-3.5-turbo")

if __name__ == "__main__":
    print("=== ReAct Agent Examples ===\n")
    
    # Example 1: Math problem solving
    print("="*50)
    print("EXAMPLE 1: Math problem solving")
    print("="*50)
    
    agent = ReAct(llm, "gsm8k", verbose=True)
    result = agent.generate("If a train travels 120 miles in 2 hours, what is its speed in miles per hour?")
    
    print(f"Answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Total cost: ${result.total_cost:.4f}")
    
    # Show summary
    summary = result.summary()
    print(f"Summary: {summary}")
    
    # Example 2: Question answering with Wikipedia
    print("\n" + "="*50)
    print("EXAMPLE 2: Question answering with Wikipedia")
    print("="*50)
    
    agent = ReAct(llm, "hotpotqa", max_steps=4, verbose=True)
    result = agent.generate("What is the capital of France?")
    
    print(f"Answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}")
    
    # Example 3: List available benchmarks
    print("\n" + "="*50)
    print("EXAMPLE 3: Available benchmarks")
    print("="*50)
    
    benchmarks = ReAct.list_benchmarks()
    print("Available benchmarks:")
    for benchmark in benchmarks:
        print(f"  - {benchmark}")
    
    # Example 4: Adding a simple benchmark (auto-handler selection)
    print("\n" + "="*50)
    print("EXAMPLE 4: Adding a simple benchmark (auto-handler)")
    print("="*50)
    
    # Add a new QA benchmark - handler automatically determined
    add_benchmark(
        benchmark_name="simple_qa",
        prompt="""
Answer the following question by thinking step by step and using search tools if needed.

Question: {question}

Examples:
{examples}

You can use the following actions:
- Search[query] - Search for information
- Lookup[query] - Look up specific information
- Finish[answer] - Provide the final answer

Think step by step and use at most {max_steps} steps.

{scratchpad}
"""
    )
    
    # Use the new benchmark immediately
    agent = ReAct(llm, "simple_qa", verbose=True)
    result = agent.generate("What is the capital of France?")
    print(f"Simple QA Answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}")
    
    # Example 5: Adding a benchmark with custom handler
    print("\n" + "="*50)
    print("EXAMPLE 5: Adding a benchmark with custom handler")
    print("="*50)
    
    # Create a custom handler with special parsing
    class CustomQAHandler(QAHandler):
        def get_prompt(self) -> str:
            return """
This is a custom QA benchmark with special action format.

Question: {question}

Examples:
{examples}

Use actions in format: ACTION:query
- SEARCH:query - Search for information
- LOOKUP:query - Look up specific information  
- FINISH:answer - Provide the final answer

Think step by step and use at most {max_steps} steps.

{scratchpad}
"""
        
        def parse_action(self, action: str):
            """Custom action parsing for special format."""
            # Parse actions like "SEARCH:query" or "FINISH:answer"
            if ":" in action:
                action_type, query = action.split(":", 1)
                return action_type.strip().upper(), query.strip()
            return "", ""
    
    # Add benchmark with custom handler
    add_benchmark(
        benchmark_name="custom_qa",
        prompt="",  # Prompt is handled by custom handler
        handler_class=CustomQAHandler
    )
    
    # Use the custom benchmark
    agent = ReAct(llm, "custom_qa", verbose=True)
    result = agent.generate("What is the population of Tokyo?")
    print(f"Custom QA Answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}")
    
    # Example 6: Adding a math benchmark with custom calculation
    print("\n" + "="*50)
    print("EXAMPLE 6: Adding a math benchmark with custom calculation")
    print("="*50)
    
    class CustomMathHandler(MathHandler):
        def get_prompt(self) -> str:
            return """
Solve this math problem with custom operations.

Question: {question}

Examples:
{examples}

You can use:
- Calculate[python_code] - Execute Python code
- Finish[answer] - Provide final answer

Special operations: sqrt(x) for square root

Think step by step and use at most {max_steps} steps.

{scratchpad}
"""
        
        def handle_observation(self, action_type: str, query: str, scratchpad: str):
            """Custom observation handling for special math operations."""
            if action_type.lower() == "calculate":
                # Add custom math logic here
                try:
                    # Example: Add special handling for certain operations
                    if "sqrt" in query.lower():
                        import math
                        # Extract number from sqrt(x) format
                        import re
                        match = re.search(r'sqrt\((\d+)\)', query)
                        if match:
                            num = int(match.group(1))
                            result = math.sqrt(num)
                            return f"Result: {result}", "", False, {"custom_calc": True}
                except:
                    pass
                # Fall back to default behavior
                return super().handle_observation(action_type, query, scratchpad)
            return super().handle_observation(action_type, query, scratchpad)
    
    add_benchmark(
        benchmark_name="custom_math",
        prompt="",  # Prompt handled by custom handler
        handler_class=CustomMathHandler
    )
    
    # Use the custom math benchmark
    agent = ReAct(llm, "custom_math", verbose=True)
    result = agent.generate("What is the square root of 16?")
    print(f"Custom Math Answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}") 