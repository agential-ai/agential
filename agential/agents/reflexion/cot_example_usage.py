"""Example usage of ReflexionCoT agent."""

from agential.core.llm import LLM
from agential.agents.reflexion import ReflexionCoT


def main():
    """Run ReflexionCoT agent example."""
    # Initialize LLM
    llm = LLM(model="gpt-3.5-turbo")
    
    # Initialize ReflexionCoT agent with verbose output and LLM I/O enabled
    agent = ReflexionCoT(
        llm=llm,
        benchmark="gsm8k",
        max_trials=3,
        verbose=True,  # Enable verbose output
        verbosity_level=2  # Enable LLM I/O output (level 2)
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
        print(f"  Time: {trial['total_time']:.2f}s")
        print(f"  Tokens: {trial['total_tokens']}")
        print(f"  Cost: ${trial['total_cost']:.4f}")


def demo_verbosity_levels():
    """Demo different verbosity levels."""
    print("\n" + "="*60)
    print("VERBOSITY LEVELS DEMO - ReflexionCoT")
    print("="*60)
    
    llm = LLM(model="gpt-3.5-turbo")
    question = "If you have 5 apples and give 2 to your friend, how many do you have left?"
    
    # Level 0: No verbose output
    print("\n🔇 Level 0: No verbose output")
    agent0 = ReflexionCoT(llm=llm, benchmark="gsm8k", verbose=False)
    result0 = agent0.generate(question)
    print(f"Answer: {result0['answer']}")
    
    # Level 1: Basic verbose output (trials, answers, metrics)
    print("\n🔊 Level 1: Basic verbose output")
    agent1 = ReflexionCoT(llm=llm, benchmark="gsm8k", verbose=True, verbosity_level=1)
    result1 = agent1.generate(question)
    
    # Level 2: LLM I/O output (shows actual prompts and responses)
    print("\n🔍 Level 2: LLM I/O output")
    agent2 = ReflexionCoT(llm=llm, benchmark="gsm8k", verbose=True, verbosity_level=2)
    result2 = agent2.generate(question)


if __name__ == "__main__":
    main()
    
    # Uncomment to see verbosity levels demo
    # demo_verbosity_levels() 