"""Example usage of the Reflexion agent."""

from agential.core.llm import LLM
from agential.agents.reflexion.agent_1 import Reflexion

def main():
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
    
    # Generate answer
    result = agent.generate(question=question, key=key)
    
    # Print results
    print(f"Question: {question}")
    print(f"Expected Answer: {key}")
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

if __name__ == "__main__":
    main()