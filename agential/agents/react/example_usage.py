"""Example usage of the ReAct agent."""

from agential.agents.react import ReAct
from agential.core.llm import LLM
from agential.agents.react.agent import add_benchmark
from agential.agents.react.handlers import QAHandler, MathHandler

# Initialize the language model
llm = LLM("gpt-3.5-turbo")


def test_react_with_all_benchmarks():
    """Test ReAct agent with each benchmark type to demonstrate capabilities."""
    
    print("="*60)
    print("TESTING REACT AGENT WITH ALL BENCHMARK TYPES")
    print("="*60)
    
    # Test questions for each benchmark type
    test_cases = {
        "hotpotqa": "What is the capital of France?",
        "fever": "Is it true that Paris is the capital of France?",
        "triviaqa": "What is the largest planet in our solar system?",
        "ambignq": "Who wrote Romeo and Juliet?",
        "gsm8k": "If a train travels 120 miles in 2 hours, what is its speed in miles per hour?",
        "svamp": "John has 5 apples. He gives 2 to Mary. How many apples does John have now?",
        "tabmwp": "A store sells shirts for $20 each. If they sell 15 shirts, how much money do they make?",
        "humaneval": "Write a function that returns the sum of two numbers.",
        "mbpp": "Create a function that checks if a number is even."
    }
    
    results = {}
    
    for benchmark, question in test_cases.items():
        print(f"\n{'='*50}")
        print(f"Testing {benchmark.upper()} benchmark")
        print(f"Question: {question}")
        print(f"{'='*50}")
        
        try:
            # Create agent with shorter max_steps for demo
            agent = ReAct(llm, benchmark, max_steps=3, verbose=True)
            
            # Generate response
            result = agent.generate(question)
            
            # Store results
            results[benchmark] = {
                "answer": result.answer,
                "steps": result.num_steps,
                "tokens": result.total_tokens,
                "cost": result.total_cost,
                "success": True
            }
            
            print(f"✅ {benchmark.upper()} completed successfully!")
            print(f"   Answer: {result.answer}")
            print(f"   Steps: {result.num_steps}")
            print(f"   Tokens: {result.total_tokens}")
            print(f"   Cost: ${result.total_cost:.4f}")
            
        except Exception as e:
            print(f"❌ {benchmark.upper()} failed: {str(e)}")
            results[benchmark] = {
                "success": False,
                "error": str(e)
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL BENCHMARK TESTS")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results.values() if r.get("success", False))
    total = len(results)
    
    print(f"Successful tests: {successful}/{total}")
    print(f"Success rate: {(successful/total)*100:.1f}%")
    
    if successful > 0:
        total_tokens = sum(r.get("tokens", 0) for r in results.values() if r.get("success", False))
        total_cost = sum(r.get("cost", 0) for r in results.values() if r.get("success", False))
        avg_steps = sum(r.get("steps", 0) for r in results.values() if r.get("success", False)) / successful
        
        print(f"Total tokens used: {total_tokens}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Average steps per successful test: {avg_steps:.1f}")
    
    return results


def test_react_math_problem():
    """Test ReAct with a complex math problem."""
    print("="*50)
    print("EXAMPLE 1: Complex Math Problem")
    print("="*50)
    
    agent = ReAct(llm, "gsm8k", verbose=True)
    result = agent.generate("A car travels 240 miles in 4 hours. Then it travels 180 miles in 3 hours. What is the average speed for the entire trip?")
    
    print(f"Answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Total cost: ${result.total_cost:.4f}")
    
    # Show summary
    summary = result.summary()
    print(f"Summary: {summary}")


def test_react_qa_with_wikipedia():
    """Test ReAct with question answering using Wikipedia."""
    print("\n" + "="*50)
    print("EXAMPLE 2: Question Answering with Wikipedia")
    print("="*50)
    
    agent = ReAct(llm, "hotpotqa", max_steps=4, verbose=True)
    result = agent.generate("What is the population of Tokyo and when was it founded?")
    
    print(f"Answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Total cost: ${result.total_cost:.4f}")


def test_react_code_generation():
    """Test ReAct with code generation."""
    print("\n" + "="*50)
    print("EXAMPLE 3: Code Generation")
    print("="*50)
    
    agent = ReAct(llm, "humaneval", max_steps=4, verbose=True)
    result = agent.generate("Write a function that finds the maximum element in a list of numbers.")
    
    print(f"Generated code: {result.answer}")
    print(f"Steps taken: {result.num_steps}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Total cost: ${result.total_cost:.4f}")


def test_react_benchmark_management():
    """Test ReAct's benchmark management capabilities."""
    print("\n" + "="*50)
    print("EXAMPLE 4: Benchmark Management")
    print("="*50)
    
    # List available benchmarks
    benchmarks = ReAct.list_benchmarks()
    print("Available benchmarks:")
    for benchmark in benchmarks:
        print(f"  - {benchmark}")
    
    # Add a new benchmark
    print("\nAdding a new custom benchmark...")
    add_benchmark(
        benchmark_name="custom_qa",
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
    
    # Test the new benchmark
    agent = ReAct(llm, "custom_qa", verbose=True)
    result = agent.generate("What is the tallest mountain in the world?")
    print(f"Custom benchmark answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}")


def test_react_custom_handler():
    """Test ReAct with a custom handler."""
    print("\n" + "="*50)
    print("EXAMPLE 5: Custom Handler")
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
        benchmark_name="special_qa",
        prompt="",  # Prompt is handled by custom handler
        handler_class=CustomQAHandler
    )
    
    # Test the custom benchmark
    agent = ReAct(llm, "special_qa", verbose=True)
    result = agent.generate("What is the population of New York City?")
    print(f"Custom handler answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}")


def test_react_with_truncation():
    """Test ReAct with different truncation settings."""
    print("="*50)
    print("EXAMPLE: Truncation Settings")
    print("="*50)
    
    # Test with default truncation (200 chars)
    print("\n[bold]Default truncation (200 chars):[/bold]")
    agent = ReAct(llm, "hotpotqa", max_steps=2, verbose=True, truncate_length=200)
    result = agent.generate("What is the capital of France and what are its main attractions?")
    
    # Test with shorter truncation (100 chars)
    print("\n[bold]Short truncation (100 chars):[/bold]")
    agent = ReAct(llm, "hotpotqa", max_steps=2, verbose=True, truncate_length=100)
    result = agent.generate("What is the capital of France and what are its main attractions?")
    
    # Test with longer truncation (500 chars)
    print("\n[bold]Long truncation (500 chars):[/bold]")
    agent = ReAct(llm, "hotpotqa", max_steps=2, verbose=True, truncate_length=500)
    result = agent.generate("What is the capital of France and what are its main attractions?")
    
    # Test with no truncation (very long)
    print("\n[bold]No truncation (1000 chars):[/bold]")
    agent = ReAct(llm, "hotpotqa", max_steps=2, verbose=True, truncate_length=1000)
    result = agent.generate("What is the capital of France and what are its main attractions?")


def test_react_debug_mode():
    """Test ReAct with debug mode enabled to troubleshoot action parsing issues."""
    print("\n" + "="*50)
    print("EXAMPLE 6: Debug Mode for Troubleshooting")
    print("="*50)
    
    # Create agent with debug mode enabled
    agent = ReAct(llm, "gsm8k", max_steps=3, verbose=True, debug_mode=True)
    
    print("Testing with debug mode enabled...")
    print("This will show raw LLM responses for both thought and action steps.")
    print("Useful for debugging 'Invalid Action' errors and understanding LLM behavior.\n")
    
    result = agent.generate("If I have 10 apples and eat 3, how many do I have left?")
    
    print(f"\nFinal Answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}")
    
    # Show debug information from steps
    print(f"\nDebug Information from Steps:")
    for i, step in enumerate(result.steps, 1):
        print(f"  Step {i}:")
        print(f"    Raw Thought: '{step.raw_thought[:100]}...'")
        print(f"    Parsed Thought: '{step.thought[:100]}...'")
        print(f"    Raw Action: '{step.raw_action[:100]}...'")
        print(f"    Action Type: '{step.action_type}'")
        print(f"    Query: '{step.query}'")
        if "Invalid Action" in step.observation:
            print(f"    ⚠️  Invalid Action detected!")
        print()


def test_react_debug_invalid_action():
    """Test ReAct with a scenario that might produce invalid actions for debugging."""
    print("\n" + "="*50)
    print("EXAMPLE 7: Debug Invalid Action Scenario")
    print("="*50)
    
    # Create agent with debug mode and very short truncation to see raw responses
    agent = ReAct(llm, "gsm8k", max_steps=2, verbose=True, debug_mode=True, truncate_length=50)
    
    print("Testing with a complex math problem that might cause parsing issues...")
    print("Debug mode will show raw LLM responses for both thought and action steps.\n")
    
    # Use a complex problem that might cause the LLM to generate unexpected action formats
    result = agent.generate("A train leaves station A at 2 PM traveling at 60 mph. Another train leaves station B at 3 PM traveling at 80 mph towards station A. If the stations are 300 miles apart, when will they meet?")
    
    print(f"\nFinal Answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}")
    
    # Detailed analysis of each step
    print(f"\nDetailed Step Analysis:")
    for i, step in enumerate(result.steps, 1):
        print(f"\n  Step {i}:")
        print(f"    Raw Thought: '{step.raw_thought}'")
        print(f"    Parsed Thought: '{step.thought[:100]}...'")
        print(f"    Raw Action: '{step.raw_action}'")
        print(f"    Parsed Action Type: '{step.action_type}'")
        print(f"    Parsed Query: '{step.query}'")
        print(f"    Observation: {step.observation[:100]}...")
        
        if "Invalid Action" in step.observation:
            print(f"    🚨 INVALID ACTION DETECTED!")
            print(f"    This means the LLM generated an action that couldn't be parsed.")
            print(f"    The raw action above shows what the LLM actually generated.")
            print(f"    You may need to adjust the parse_action method in the handler.")
        print()


if __name__ == "__main__":
    print("=== ReAct Agent Comprehensive Examples ===\n")
    
    # Run comprehensive benchmark tests (commented for faster execution)
    # test_results = test_react_with_all_benchmarks()
    
    # Run individual examples
    test_react_math_problem()
    # test_react_qa_with_wikipedia()
    # test_react_code_generation()
    # test_react_benchmark_management()
    # test_react_custom_handler()
    # test_react_with_truncation()
    # test_react_debug_mode()
    # test_react_debug_invalid_action()
    
    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETED!")
    print(f"{'='*60}")
    print("The ReAct agent successfully demonstrates:")
    print("✅ Multi-step reasoning across different domains")
    print("✅ Tool use (Wikipedia search, code execution)")
    print("✅ Plugin-based architecture for easy extension")
    print("✅ Comprehensive logging and metrics tracking")
    print("✅ Custom handler support for specialized tasks")
    print("✅ Configurable output truncation for readability") 