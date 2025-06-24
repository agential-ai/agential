from agential.agents.reflexion import Reflexion


def print_stats(result):
    metrics = result["metrics"]
    total_tokens = metrics["total_tokens"]
    total_time = metrics["total_time"]
    total_cost = metrics["total_cost"]
    trials = result["trials"]
    total_steps = sum(len(trial["steps"]) for trial in trials)
    avg_tokens = total_tokens / total_steps if total_steps else 0
    avg_time = total_time / total_steps if total_steps else 0
    avg_cost = total_cost / total_steps if total_steps else 0
    print("Answer: ", result["answer"])
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"Average tokens per step: {avg_tokens:.2f}")
    print(f"Average time per step: {avg_time:.2f} seconds")
    print(f"Average cost per step: ${avg_cost:.6f}")


def example_usage_qa():
    """Example usage for Reflexion (QA benchmark) with a real LLM. Requires LLM API key."""
    from agential.core.llm import LLM

    # NOTE: You must have your LLM API key set up for this to work.
    llm = LLM("gpt-3.5-turbo")
    agent = Reflexion(
        llm, "hotpotqa", max_steps=3, max_trials=1, max_reflections=2, verbose=True
    )
    result = agent.generate(
        "Who developed the theory of relativity?",
        key="Albert Einstein was a physicist.",
    )
    return result


def example_usage_math():
    """Example usage for Reflexion (Math benchmark) with a real LLM. Requires LLM API key."""
    from agential.core.llm import LLM

    # NOTE: You must have your LLM API key set up for this to work.
    llm = LLM("gpt-3.5-turbo")
    agent = Reflexion(
        llm, "gsm8k", max_steps=3, max_trials=1, max_reflections=2, verbose=True
    )
    result = agent.generate("What is 2 + 2?", key="4")
    return result


def example_usage_code():
    """Example usage for Reflexion (Code benchmark) with a real LLM. Requires LLM API key."""
    from agential.core.llm import LLM

    # NOTE: You must have your LLM API key set up for this to work.
    llm = LLM("gpt-3.5-turbo")
    agent = Reflexion(
        llm, "humaneval", max_steps=3, max_trials=1, max_reflections=2, verbose=True
    )
    result = agent.generate(
        "Write a function add(a, b) that returns their sum.",
        key="def add(a, b):\n    return a + b",
    )
    return result


if __name__ == "__main__":
    print("--- QA Example ---")
    qa_result = example_usage_qa()
    print("\n--- Math Example ---")
    math_result = example_usage_math()
    print("\n--- Code Example ---")
    code_result = example_usage_code()

    print_stats(qa_result)
    print_stats(math_result)
    print_stats(code_result)
