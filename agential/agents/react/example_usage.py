"""Example usage of the ReAct agent."""

from agential.agents.react import ReAct
from agential.core.llm import LLM
from agential.core.fewshots.gsm8k import GSM8K_FEWSHOT_EXAMPLES_REACT
from agential.agents.react.prompts import REACT_INSTRUCTION_GSM8K

# Initialize the language model
llm = LLM("gpt-3.5-turbo")

# Example 1: Math problem solving
def math_example():
    """Solve a math problem using ReAct."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4 eggs. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    
    agent = ReAct(
        llm=llm,
        benchmark="gsm8k",
        max_steps=6,
        verbose=True
    )
    
    result = agent.generate(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_GSM8K,
        reset=True
    )
    
    print(f"🎯 Final Answer: {result.answer}")
    print(f"📊 Total Steps: {len(result.additional_info)}")
    print(f"💰 Total Cost: ${result.total_cost:.4f}")

# Example 2: Question answering with Wikipedia
def qa_example():
    """Answer a question using Wikipedia search."""
    question = "What is the capital of France?"
    
    agent = ReAct(
        llm=llm,
        benchmark="hotpotqa",
        max_steps=4,
        verbose=True
    )
    
    result = agent.generate(
        question=question,
        examples="",
        reset=True
    )
    
    print(f"🎯 Final Answer: {result.answer}")

# Example 3: List available benchmarks
def list_benchmarks():
    """Show all available benchmarks."""
    benchmarks = ReAct.list_benchmarks()
    print("Available benchmarks:")
    for benchmark in benchmarks:
        print(f"  - {benchmark}")

if __name__ == "__main__":
    print("=== ReAct Agent Examples ===\n")
    
    print("1. Math Problem Solving:")
    math_example()
    
    print("\n2. Question Answering:")
    qa_example()
    
    print("\n3. Available Benchmarks:")
    list_benchmarks() 