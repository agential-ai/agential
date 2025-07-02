"""
Example usage of ExpeL agents.

This script demonstrates how to use the ExpeL agent for different benchmark types and tests individual methods for output formatting.
"""

from rich.console import Console
from agential.agents.expel import ExpeL, EXPEL_BENCHMARK_CONFIG
from agential.eval.classification import EM, fuzzy_EM
from agential.utils.general import safe_execute

console = Console()


def evaluate_answer(benchmark: str, answer: str, key: str) -> bool:
    if not answer or answer.strip() == "":
        return False
    if benchmark in ["hotpotqa", "fever", "ambignq", "triviaqa"]:
        return fuzzy_EM(answer, key)
    elif benchmark in ["gsm8k", "svamp", "tabmwp"]:
        return EM(answer, key, is_numeric=True)
    elif benchmark in ["humaneval", "mbpp"]:
        return evaluate_code_answer(answer, key, benchmark)
    else:
        return fuzzy_EM(answer, key)


def evaluate_code_answer(answer: str, key: str, benchmark: str) -> bool:
    try:
        code_str = answer.replace("```python", "").replace("```", "").strip()
        _, execution_status = safe_execute(f"from typing import *\n\n{code_str}\n{key}")
        return EM(execution_status, "Done", normalize=False)
    except Exception:
        return False


def print_stats(result):
    metrics = result["metrics"]
    total_tokens = metrics["total_tokens"]
    total_time = metrics["total_time"]
    total_cost = metrics["total_cost"]
    print("Answer: ", result["answer"])
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total cost: ${total_cost:.6f}")


def calculate_benchmark_stats(benchmark_results):
    total_runs = len(benchmark_results)
    correct_runs = sum(1 for result in benchmark_results if result["correct"])
    accuracy = correct_runs / total_runs if total_runs > 0 else 0
    total_tokens = sum(
        result["metrics"]["total_tokens"] for result in benchmark_results
    )
    total_time = sum(result["metrics"]["total_time"] for result in benchmark_results)
    total_cost = sum(result["metrics"]["total_cost"] for result in benchmark_results)
    avg_tokens = total_tokens / total_runs if total_runs > 0 else 0
    avg_time = total_time / total_runs if total_runs > 0 else 0
    avg_cost = total_cost / total_runs if total_runs > 0 else 0
    return {
        "total_runs": total_runs,
        "correct_runs": correct_runs,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "total_cost": total_cost,
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "avg_cost": avg_cost,
    }


def get_benchmark_examples():
    inst = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    return {
        # QA
        "hotpotqa": ("Which book is the most popular in the world?", "The Bible"),
        "fever": (
            "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
            "REFUTES",
        ),
        "ambignq": ("When did the simpsons first air on television?", "1989"),
        "triviaqa": (
            "Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
            "Sinclair Lewis",
        ),
        # Math
        "gsm8k": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "-9867630",
        ),
        "svamp": (
            "There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups. How big is each group of bananas?",
            "145",
        ),
        "tabmwp": (
            'Read the following table regarding "Bowling Scores" and then write Python code to answer a question:\n\nName | Score\nAmanda | 117\nSam | 236\nIrma | 144\nMike | 164\n\nQuestion: Some friends went bowling and kept track of their scores. How many more points did Mike score than Irma?',
            "20",
        ),
        # Code
        "humaneval": (inst["prompt"], f"{inst['test']}\ncheck({inst['entry_point']})"),
        "mbpp": (
            "Write a python function to find the first repeated character in a given string.",
            'assert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
        ),
    }


def run_single_benchmark(benchmark: str, num_runs: int = 3):
    from agential.core.llm import LLM

    llm = LLM("gpt-4.1")
    examples = get_benchmark_examples()
    if benchmark not in examples:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    question, key = examples[benchmark]
    agent = ExpeL(llm, benchmark, verbose=True, reflexion_kwargs={"verbose": True})
    print(f"Running {benchmark.upper()} benchmark {num_runs} times...")
    print("=" * 60)
    benchmark_results = []

    # Set up additional keys for MBPP
    additional_keys = {}
    reflect_additional_keys = {}
    if benchmark == "mbpp":
        additional_keys = {"tests": key}
        reflect_additional_keys = {"tests": key}

    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...", end=" ")
        result = agent.generate(
            question,
            key=key,
            additional_keys=additional_keys,
            reflect_additional_keys=reflect_additional_keys,
        )
        result["correct"] = evaluate_answer(benchmark, result["answer"], key)
        benchmark_results.append(result)
        status = "✓" if result.get("correct", False) else "✗"
        print(f"{status} ({result['metrics']['total_time']:.2f}s)")
    stats = calculate_benchmark_stats(benchmark_results)
    print(f"\n{benchmark.upper()} BENCHMARK RESULTS")
    print("-" * 40)
    print(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
    print(f"Expected Answer: {key}")
    print(
        f"Accuracy: {stats['accuracy']:.1%} ({stats['correct_runs']}/{stats['total_runs']})"
    )
    print(f"Average Time: {stats['avg_time']:.2f} seconds")
    print(f"Average Tokens: {stats['avg_tokens']:.0f}")
    print(f"Average Cost: ${stats['avg_cost']:.6f}")
    print("\nIndividual Run Results:")
    for i, result in enumerate(benchmark_results):
        status = "✓" if result.get("correct", False) else "✗"
        answer_preview = str(result["answer"])
        print(f"  Run {i + 1:2d}: {status} | {answer_preview}")
    return {
        "benchmark": benchmark,
        "question": question,
        "key": key,
        "results": benchmark_results,
        "stats": stats,
    }


def run_all_benchmarks():
    from agential.core.llm import LLM

    llm = LLM("gpt-4.1")
    examples = get_benchmark_examples()
    all_results = []
    print("Starting ExpeL benchmark evaluation...")
    print("=" * 60)
    for benchmark in EXPEL_BENCHMARK_CONFIG:
        print(f"\n--- Running {benchmark.upper()} Benchmark ---")
        question, key = examples[benchmark]
        agent = ExpeL(llm, benchmark)
        benchmark_results = []

        # Set up additional keys for MBPP
        additional_keys = {}
        reflect_additional_keys = {}
        if benchmark == "mbpp":
            additional_keys = {"tests": key}
            reflect_additional_keys = {"tests": key}

        for run in range(3):
            print(f"  Run {run + 1}/3...", end=" ")
            result = agent.generate(
                question,
                key=key,
                additional_keys=additional_keys,
                reflect_additional_keys=reflect_additional_keys,
            )
            result["correct"] = evaluate_answer(benchmark, result["answer"], key)
            benchmark_results.append(result)
            status = "✓" if result.get("correct", False) else "✗"
            print(f"{status} ({result['metrics']['total_time']:.2f}s)")
        stats = calculate_benchmark_stats(benchmark_results)
        all_results.append(
            {
                "benchmark": benchmark,
                "question": question,
                "key": key,
                "results": benchmark_results,
                "stats": stats,
            }
        )
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EXPEL BENCHMARK RESULTS")
    print("=" * 80)
    print(
        f"\n{'Benchmark':<12} {'Accuracy':<10} {'Avg Time':<10} {'Avg Tokens':<12} {'Avg Cost':<12}"
    )
    print("-" * 60)
    for entry in all_results:
        stats = entry["stats"]
        benchmark = entry["benchmark"]
        accuracy_pct = stats["accuracy"] * 100
        print(
            f"{benchmark:<12} {accuracy_pct:>6.1f}%   {stats['avg_time']:>8.2f}s   {stats['avg_tokens']:>10.0f}   ${stats['avg_cost']:.6f}"
        )
    print("-" * 60)
    print("\nDETAILED RESULTS BY BENCHMARK")
    print("=" * 80)
    for entry in all_results:
        benchmark = entry["benchmark"]
        stats = entry["stats"]
        results = entry["results"]
        print(f"\n{benchmark.upper()} BENCHMARK")
        print("-" * 40)
        print(
            f"Question: {entry['question'][:100]}{'...' if len(entry['question']) > 100 else ''}"
        )
        print(f"Expected Answer: {entry['key']}")
        print(
            f"Accuracy: {stats['accuracy']:.1%} ({stats['correct_runs']}/{stats['total_runs']})"
        )
        print(f"Average Time: {stats['avg_time']:.2f} seconds")
        print(f"Average Tokens: {stats['avg_tokens']:.0f}")
        print(f"Average Cost: ${stats['avg_cost']:.6f}")
        print("\nIndividual Run Results:")
        for i, result in enumerate(results):
            status = "✓" if result.get("correct", False) else "✗"
            answer_preview = str(result["answer"])
            print(f"  Run {i + 1:2d}: {status} | {answer_preview}")
        print("-" * 40)
    return all_results


def test_expel_generate():
    """Test ExpeL.generate method for output formatting."""
    from agential.core.llm import LLM

    llm = LLM("gpt-4.1")
    agent = ExpeL(llm, "fever", verbose=True, reflexion_kwargs={"verbose": True})
    out = agent.generate(
        "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.", key="REFUTES"
    )
    out_1 = agent.generate(
        "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.", key="REFUTES"
    )
    return out, out_1


def test_expel_memory_access():
    """Test ExpeL memory access methods."""
    from agential.core.llm import LLM

    llm = LLM("gpt-4.1")
    agent = ExpeL(llm, "fever")
    print("\nTesting ExpeL memory access...")

    # Access the underlying agent's memory
    if hasattr(agent._agent, "experience_memory"):
        print("Experience Memory:", agent._agent.experience_memory.show_memories())
    if hasattr(agent._agent, "insight_memory"):
        print("Insight Memory:", agent._agent.insight_memory.show_memories())

    return {
        "experience_memory": getattr(agent._agent, "experience_memory", None),
        "insight_memory": getattr(agent._agent, "insight_memory", None),
    }


def test_expel_config():
    """Test ExpeL configuration access."""
    from agential.core.llm import LLM

    llm = LLM("gpt-4.1")
    agent = ExpeL(llm, "fever")
    return {
        "config": agent.config,
        "benchmark": agent.benchmark,
        "llm": agent.llm,
    }


def test_expel_methods():
    """Test individual methods of ExpeL for output formatting."""
    print("=" * 60)
    print("TESTING EXPEL METHODS")
    print("=" * 60)

    # Test each method individually
    generate_result = test_expel_generate()
    memory_result = test_expel_memory_access()
    config_result = test_expel_config()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

    return {
        "generate": generate_result,
        "memory_access": memory_result,
        "config": config_result,
    }


if __name__ == "__main__":
    # Example: Run just one benchmark
    # run_single_benchmark("mbpp", num_runs=2)

    # Or run all benchmarks
    run_all_benchmarks()

    # Test ExpeL methods for output formatting
    # out, out_1 = test_expel_generate()
