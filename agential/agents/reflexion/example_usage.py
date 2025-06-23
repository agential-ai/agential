"""Example usage of Reflexion agent."""

from agential.core.llm import LLM
from agential.agents.reflexion import Reflexion


def main():
    """Run Reflexion agent example."""
    # Initialize LLM
    llm = LLM(model="gpt-3.5-turbo")
    
    # Initialize Reflexion agent with verbose output and LLM I/O enabled
    agent = Reflexion(
        llm=llm,
        benchmark="gsm8k",
        max_steps=6,
        verbose=True,
        verbosity_level=2
    )
    
    # Example question
    question = "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"
    
    # print("Running Reflexion agent...")
    # print(f"Question: {question}")
    # print("-" * 50)
    
    # Generate answer
    result = agent.generate(question)
    
    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    print(f"Answer: {result['answer']}")
    print(f"Steps taken: {result['metrics']['steps_taken']}")
    print(f"Total time: {result['metrics']['total_time']:.2f}s")
    print(f"Total tokens: {result['metrics']['total_tokens']}")
    print(f"Total cost: ${result['metrics']['total_cost']:.4f}")
    
    # Print detailed steps
    # print("\nDetailed Steps:")
    # for i, step in enumerate(result['steps'], 1):
    #     print(f"\nStep {i}:")
    #     print(f"  Thought: {step['thought']}")
    #     print(f"  Action: {step['action_type']}[{step['query']}]")
    #     print(f"  Observation: {step['observation']}")
    #     if step['answer']:
    #         print(f"  Answer: {step['answer']}")



if __name__ == "__main__":
    main()