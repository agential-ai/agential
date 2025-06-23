"""Example usage of ReflexionCoT agent."""

from agential.core.llm import LLM
from agential.agents.reflexion import ReflexionCoT


def main():
    """Run ReflexionCoT agent example."""
    # Initialize LLM
    llm = LLM(model="gpt-3.5-turbo")
    
    # Initialize ReflexionCoT agent with verbose output
    agent = ReflexionCoT(
        llm=llm,
        benchmark="gsm8k",
        max_trials=3,
        verbose=True  # Enable verbose output
    )
    
    # Example question
    question = "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"
    
    print("Running ReflexionCoT agent...")
    print(f"Question: {question}")
    print("-" * 50)
    
    # Generate answer
    result = agent.generate(question)
    
    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    print(f"Answer: {result['answer']}")
    print(f"Trials taken: {result['metrics']['trials_taken']}")
    print(f"Total time: {result['metrics']['total_time']:.2f}s")
    print(f"Total tokens: {result['metrics']['total_tokens']}")
    print(f"Total cost: ${result['metrics']['total_cost']:.4f}")
    
    # Print detailed trials
    print("\nDetailed Trials:")
    for i, trial in enumerate(result['trials'], 1):
        print(f"\nTrial {i}:")
        print(f"  Answer: {trial['answer']}")
        print(f"  Reflection: {trial['reflection']}")
        print(f"  Time: {trial['time']:.2f}s")
        print(f"  Tokens: {trial['tokens']}")
        print(f"  Cost: ${trial['cost']:.4f}")


if __name__ == "__main__":
    main() 