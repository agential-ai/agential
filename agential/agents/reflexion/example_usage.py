"""Example usage of the Reflexion agent."""

from agential.core.llm import LLM
from agential.agents.reflexion.agent_1 import Reflexion

def run_gsm8k_example():
    """Run GSM8K math benchmark example."""
    print("=" * 60)
    print("GSM8K MATH BENCHMARK EXAMPLE")
    print("=" * 60)
    
    # Initialize the LLM
    llm = LLM(model="gpt-3.5-turbo", temperature=0)
    
    # Initialize the Reflexion agent for GSM8K
    agent = Reflexion(
        llm=llm,
        benchmark="gsm8k",
        max_steps=6,
        max_trials=3,
        max_reflections=3,
        reflect_strategy="last_attempt_and_reflexion"
    )
    
    # Example question and answer key
    question = "Janet's dogs eat 2 pounds of dog food each day. How many pounds of dog food do her dogs eat in 7 days?"
    key = "14"
    
    print(f"Question: {question}")
    print(f"Expected Answer: {key}")
    print("-" * 60)
    
    # Generate answer
    result = agent.generate(question=question, key=key)
    
    # Print results
    print(f"Generated Answer: {result['answer']}")
    print(f"Correct: {result['correct']}")
    print(f"Trials taken: {result['metrics']['trials_taken']}")
    print(f"Total time: {result['metrics']['total_time']:.2f}s")
    print(f"Total tokens: {result['metrics']['total_tokens']}")
    print(f"Total cost: ${result['metrics']['total_cost']:.4f}")
    
    # Print reflections if any
    if result['reflections']:
        print(f"\nReflections:\n{result['reflections']}")
    
    # Print trial details
    for trial in result['trials']:
        print(f"\nTrial {trial['trial']}:")
        print(f"  Answer: {trial['answer']}")
        print(f"  Correct: {trial['correct']}")
        print(f"  Steps: {len(trial['steps'])}")
        print(f"  Time: {trial['trial_time']:.2f}s")
        print(f"  Tokens: {trial['trial_tokens']}")
        print(f"  Cost: ${trial['trial_cost']:.4f}")


def run_hotpotqa_example():
    """Run HotpotQA benchmark example."""
    print("\n" + "=" * 60)
    print("HOTPOTQA BENCHMARK EXAMPLE")
    print("=" * 60)
    
    # Initialize the LLM
    llm = LLM(model="gpt-3.5-turbo", temperature=0)
    
    # Initialize the Reflexion agent for HotpotQA
    agent = Reflexion(
        llm=llm,
        benchmark="hotpotqa",
        max_steps=6,
        max_trials=2,
        max_reflections=2,
        reflect_strategy="last_attempt_and_reflexion"
    )
    
    # Example question and answer key
    question = "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    key = "1,800 to 7,000 ft"
    
    print(f"Question: {question}")
    print(f"Expected Answer: {key}")
    print("-" * 60)
    
    # Generate answer
    result = agent.generate(question=question, key=key)
    
    # Print results
    print(f"Generated Answer: {result['answer']}")
    print(f"Correct: {result['correct']}")
    print(f"Trials taken: {result['metrics']['trials_taken']}")
    print(f"Total time: {result['metrics']['total_time']:.2f}s")
    print(f"Total tokens: {result['metrics']['total_tokens']}")
    print(f"Total cost: ${result['metrics']['total_cost']:.4f}")


def run_humaneval_example():
    """Run HumanEval code benchmark example."""
    print("\n" + "=" * 60)
    print("HUMANEVAL CODE BENCHMARK EXAMPLE")
    print("=" * 60)
    
    # Initialize the LLM
    llm = LLM(model="gpt-3.5-turbo", temperature=0)
    
    # Initialize the Reflexion agent for HumanEval
    agent = Reflexion(
        llm=llm,
        benchmark="humaneval",
        max_steps=6,
        max_trials=2,
        max_reflections=2,
        reflect_strategy="last_attempt_and_reflexion"
    )
    
    # Example question and answer key
    question = "Write a function that returns the sum of two numbers."
    key = "Done"  # Code benchmarks expect "Done" for correct execution
    
    print(f"Question: {question}")
    print(f"Expected Answer: {key}")
    print("-" * 60)
    
    # Generate answer
    result = agent.generate(question=question, key=key)
    
    # Print results
    print(f"Generated Answer: {result['answer']}")
    print(f"Correct: {result['correct']}")
    print(f"Trials taken: {result['metrics']['trials_taken']}")
    print(f"Total time: {result['metrics']['total_time']:.2f}s")
    print(f"Total tokens: {result['metrics']['total_tokens']}")
    print(f"Total cost: ${result['metrics']['total_cost']:.4f}")


def show_available_benchmarks():
    """Show all available benchmarks."""
    print("\n" + "=" * 60)
    print("AVAILABLE BENCHMARKS")
    print("=" * 60)
    
    benchmarks = Reflexion.list_benchmarks()
    print("Available benchmarks:")
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"  {i}. {benchmark}")
    
    print(f"\nTotal: {len(benchmarks)} benchmarks")


def main():
    """Run comprehensive examples."""
    
    # Show available benchmarks
    # show_available_benchmarks()
    
    # Run examples for different benchmark types
    # run_gsm8k_example()
    run_hotpotqa_example()
    # run_humaneval_example()



if __name__ == "__main__":
    main()