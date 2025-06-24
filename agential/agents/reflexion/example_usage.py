from . import ReflexionQA, ReflexionMath, ReflexionCode, BENCHMARK_CONFIG


def example_usage_qa():
    """Example usage for ReflexionQA (QA benchmark)."""
    from agential.core.llm import MockLLM
    llm = MockLLM("gpt-3.5-turbo", responses=[
        "Thought: Let's search for the answer.\nAction: Search[Albert Einstein]",
        "Thought: Now let's finish.\nAction: Finish[Albert Einstein was a physicist.]"
    ])
    agent = ReflexionQA(llm, "hotpotqa", max_steps=3, max_trials=1, max_reflections=2, verbose=True, config=BENCHMARK_CONFIG["hotpotqa"])
    result = agent.generate("Who developed the theory of relativity?", key="Albert Einstein was a physicist.")
    print(result)


def example_usage_math():
    """Example usage for ReflexionMath (Math benchmark)."""
    from agential.core.llm import MockLLM
    llm = MockLLM("gpt-3.5-turbo", responses=[
        "Thought: Let's calculate.\nAction: Calculate[```python\n2 + 2\n```]",
        "Thought: Now let's finish.\nAction: Finish[4]"
    ])
    agent = ReflexionMath(llm, "gsm8k", max_steps=3, max_trials=1, max_reflections=2, verbose=True, config=BENCHMARK_CONFIG["gsm8k"])
    result = agent.generate("What is 2 + 2?", key="4")
    print(result)


def example_usage_code():
    """Example usage for ReflexionCode (Code benchmark)."""
    from agential.core.llm import MockLLM
    llm = MockLLM("gpt-3.5-turbo", responses=[
        "Thought: Let's implement the function.\nAction: Implement[```python\ndef add(a, b):\n    return a + b\n```]",
        "Thought: Now let's finish.\nAction: Finish[```python\ndef add(a, b):\n    return a + b\n```]"
    ])
    agent = ReflexionCode(llm, "humaneval", max_steps=3, max_trials=1, max_reflections=2, verbose=True, config=BENCHMARK_CONFIG["humaneval"])
    result = agent.generate("Write a function add(a, b) that returns their sum.", key="def add(a, b):\n    return a + b")
    print(result)


if __name__ == "__main__":
    print("--- QA Example ---")
    example_usage_qa()
    print("\n--- Math Example ---")
    example_usage_math()
    print("\n--- Code Example ---")
    example_usage_code() 