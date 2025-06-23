"""Example usage of the new handler-based ReflexionAgent for all supported benchmarks."""

from agential.agents.reflexion.agent import ReflexionAgent
from agential.core.llm import LLM

# Example questions/tasks for each benchmark
def get_example_question(benchmark):
    examples = {
        "hotpotqa": "Who was the president of the United States in 2000?",
        "fever": "The Eiffel Tower is located in Berlin.",
        "triviaqa": "What is the capital of France?",
        "ambignq": "Who wrote the novel '1984'?",
        "gsm8k": "If you have 10 apples and eat 3, how many are left?",
        "svamp": "John has 5 more apples than Tom. Tom has 3 apples. How many apples does John have?",
        "tabmwp": "A table has 4 legs. How many legs do 3 tables have?",
        "humaneval": "Write a function that returns the sum of two numbers.",
        "mbpp": "Write a function to check if a number is even.",
    }
    return examples.get(benchmark, "What is 2+2?")

def main():
    # Initialize the language model (replace with your model as needed)
    llm = LLM("gpt-3.5-turbo")

    # Run the agent for all supported benchmarks
    for benchmark in ReflexionAgent.list_benchmarks():
        print(f"\n{'='*80}\nBenchmark: {benchmark}\n{'='*80}")
        agent = ReflexionAgent(llm, benchmark, max_steps=3, debug_mode=True, verbose=True)
        question = get_example_question(benchmark)
        print(f"\nQuestion: {question}\n{'-'*60}")
        result = agent.generate(question)
        print(f"\nFinal Answer: {result.answer}")
        print(f"Summary: {result.summary()}")
        print(f"\nStep-by-step details:")
        for i, step in enumerate(result.steps, 1):
            print(f"\nStep {i}:")
            print(f"  Thought: {step.thought}")
            print(f"  Action: {step.action_type}[{step.query}]")
            print(f"  Observation: {step.observation}")
            print(f"  Answer: {step.answer}")
            if step.reflection:
                print(f"  Reflection: {step.reflection}")
            if step.raw_thought:
                print(f"  Raw Thought: {step.raw_thought}")
            if step.raw_action:
                print(f"  Raw Action: {step.raw_action}")
            if step.external_tool_info:
                print(f"  External Tool Info: {step.external_tool_info}")

if __name__ == "__main__":
    main() 