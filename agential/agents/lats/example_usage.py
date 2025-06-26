"""
Example usage of LATS agents.

This script demonstrates how to use the LATS agents for different benchmark types.
"""

from rich.console import Console

from agential.agents.lats import LATS, BENCHMARK_CONFIG
from agential.eval.classification import EM, fuzzy_EM
from agential.utils.general import safe_execute

console = Console()


def evaluate_answer(benchmark: str, answer: str, key: str) -> bool:
    """Evaluate if the answer is correct based on the benchmark type."""
    if not answer or answer.strip() == "":
        return False

    # QA benchmarks - use fuzzy matching
    if benchmark in ["hotpotqa", "fever", "ambignq", "triviaqa"]:
        return fuzzy_EM(answer, key)

    # Math benchmarks - use exact numeric matching
    elif benchmark in ["gsm8k", "svamp", "tabmwp"]:
        return EM(answer, key, is_numeric=True)

    # Code benchmarks - execute and test
    elif benchmark in ["humaneval", "mbpp"]:
        return evaluate_code_answer(answer, key, benchmark)

    else:
        # Default to fuzzy matching
        return fuzzy_EM(answer, key)


def evaluate_code_answer(answer: str, key: str, benchmark: str) -> bool:
    """Evaluate code answers by executing them."""
    try:
        # Extract code from answer (remove markdown if present)
        code_str = answer.replace("```python", "").replace("```", "").strip()

        if benchmark == "humaneval":
            # For HumanEval, the key contains the test cases
            _, execution_status = safe_execute(
                f"from typing import *\n\n{code_str}\n{key}"
            )
            return EM(execution_status, "Done", normalize=False)
        elif benchmark == "mbpp":
            # For MBPP, the key contains the test cases
            _, execution_status = safe_execute(
                f"from typing import *\n\n{code_str}\n{key}"
            )
            return EM(execution_status, "Done", normalize=False)
        else:
            return False
    except Exception:
        return False


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


def calculate_benchmark_stats(benchmark_results):
    """Calculate comprehensive statistics for a benchmark's results."""
    total_runs = len(benchmark_results)
    correct_runs = sum(1 for result in benchmark_results if result["correct"])
    accuracy = correct_runs / total_runs if total_runs > 0 else 0

    # Aggregate metrics
    total_tokens = sum(
        result["metrics"]["total_tokens"] for result in benchmark_results
    )
    total_time = sum(result["metrics"]["total_time"] for result in benchmark_results)
    total_cost = sum(result["metrics"]["total_cost"] for result in benchmark_results)

    # Calculate averages
    avg_tokens = total_tokens / total_runs if total_runs > 0 else 0
    avg_time = total_time / total_runs if total_runs > 0 else 0
    avg_cost = total_cost / total_runs if total_runs > 0 else 0

    # Calculate total steps across all runs
    total_steps = sum(len(result["steps"]) for result in benchmark_results)
    avg_steps = total_steps / total_runs if total_runs > 0 else 0

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
        "total_steps": total_steps,
        "avg_steps": avg_steps,
    }


# Example questions/keys for each benchmark
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
        "hotpotqa": (
            "Which book is the most popular in the world?",
            "The Bible",
        ),
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


def run_single_benchmark(benchmark: str, num_runs: int = 5):
    """Run a single benchmark multiple times and return results."""
    from agential.core.llm import LLM

    llm = LLM("gpt-4.1")
    examples = get_benchmark_examples()

    if benchmark not in examples:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    question, key = examples[benchmark]
    agent = LATS(llm, benchmark, max_iterations=10, verbose=True)

    print(f"Running {benchmark.upper()} benchmark {num_runs} times...")
    print("=" * 60)

    # Run the benchmark multiple times
    benchmark_results = []
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...", end=" ")

        if benchmark == "mbpp":
            # MBPP expects tests as additional_keys and reflect_additional_keys
            result = agent.generate(
                question,
                key=key,
                additional_keys={"tests": key},
            )
        else:
            result = agent.generate(question, key=key)

        # Add correctness evaluation
        result["correct"] = evaluate_answer(benchmark, result["answer"], key)

        benchmark_results.append(result)
        status = "✓" if result.get("correct", False) else "✗"
        print(f"{status} ({result['metrics']['total_time']:.2f}s)")

    # Calculate stats for this benchmark
    stats = calculate_benchmark_stats(benchmark_results)

    # Print results
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
    print(f"Average Steps: {stats['avg_steps']:.1f}")

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

    print("Starting LATS benchmark evaluation...")
    print("=" * 60)

    for benchmark in BENCHMARK_CONFIG:
        print(f"\n--- Running {benchmark.upper()} Benchmark ---")
        question, key = examples[benchmark]
        agent = LATS(llm, benchmark, max_iterations=10, verbose=False)

        # Run the same benchmark 5 times
        benchmark_results = []
        for run in range(5):
            print(f"  Run {run + 1}/5...", end=" ")

            if benchmark == "mbpp":
                # MBPP expects tests as additional_keys and reflect_additional_keys
                result = agent.generate(
                    question,
                    key=key,
                    additional_keys={"tests": key},
                )
            else:
                result = agent.generate(question, key=key)

            # Add correctness evaluation
            result["correct"] = evaluate_answer(benchmark, result["answer"], key)

            benchmark_results.append(result)
            status = "✓" if result.get("correct", False) else "✗"
            print(f"{status} ({result['metrics']['total_time']:.2f}s)")

        # Calculate stats for this benchmark
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

    # Print comprehensive results
    print("\n" + "=" * 80)
    print("COMPREHENSIVE LATS BENCHMARK RESULTS")
    print("=" * 80)

    # Summary table
    print(
        f"\n{'Benchmark':<12} {'Accuracy':<10} {'Avg Time':<10} {'Avg Tokens':<12} {'Avg Cost':<12} {'Avg Steps':<10}"
    )
    print("-" * 80)

    total_accuracy = 0
    total_benchmarks = len(all_results)

    for entry in all_results:
        stats = entry["stats"]
        benchmark = entry["benchmark"]
        accuracy_pct = stats["accuracy"] * 100
        total_accuracy += accuracy_pct

        print(
            f"{benchmark:<12} {accuracy_pct:>6.1f}%   {stats['avg_time']:>8.2f}s   {stats['avg_tokens']:>10.0f}   ${stats['avg_cost']:.6f}   {stats['avg_steps']:>8.1f}"
        )

    overall_accuracy = total_accuracy / total_benchmarks if total_benchmarks > 0 else 0
    print("-" * 80)
    print(f"{'OVERALL':<12} {overall_accuracy:>6.1f}%")

    # Detailed results for each benchmark
    print("\n" + "=" * 80)
    print("DETAILED RESULTS BY BENCHMARK")
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
        print(f"Average Steps: {stats['avg_steps']:.1f}")

        print("\nIndividual Run Results:")
        for i, result in enumerate(results):
            status = "✓" if result.get("correct", False) else "✗"
            answer_preview = str(result["answer"])
            print(f"  Run {i + 1:2d}: {status} | {answer_preview}")

        print("-" * 40)

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    total_runs = sum(entry["stats"]["total_runs"] for entry in all_results)
    total_correct = sum(entry["stats"]["correct_runs"] for entry in all_results)
    total_tokens = sum(entry["stats"]["total_tokens"] for entry in all_results)
    total_time = sum(entry["stats"]["total_time"] for entry in all_results)
    total_cost = sum(entry["stats"]["total_cost"] for entry in all_results)

    print(f"Total Benchmarks: {total_benchmarks}")
    print(f"Total Runs: {total_runs}")
    print(f"Total Correct: {total_correct}")
    print(f"Overall Accuracy: {total_correct / total_runs:.1%}")
    print(f"Total Tokens Used: {total_tokens:,}")
    print(f"Total Time: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)")
    print(f"Total Cost: ${total_cost:.6f}")


def demonstrate_lats_agents():
    """Demonstrate the different LATS agents with simple examples."""
    from agential.core.llm import LLM

    llm = LLM("gpt-4.1")

    print("LATS Agent Demonstrations")
    print("=" * 50)

    # QA Agent demonstration
    print("\n1. LATS QA Agent (HotpotQA)")
    print("-" * 30)
    qa_agent = LATS(llm, "hotpotqa", max_iterations=5, verbose=True)
    qa_result = qa_agent.generate(
        "Which book is the most popular in the world?", key="The Bible"
    )
    print(f"Answer: {qa_result['answer']}")
    print(f"Correct: {evaluate_answer('hotpotqa', qa_result['answer'], 'The Bible')}")
    print(f"Steps: {len(qa_result['steps'])}")
    print(f"Time: {qa_result['metrics']['total_time']:.2f}s")

    # Math Agent demonstration
    print("\n2. LATS Math Agent (GSM8K)")
    print("-" * 30)
    math_agent = LATS(llm, "gsm8k", max_iterations=5, verbose=True)
    math_result = math_agent.generate("What is 15 + 27?", key="42")
    print(f"Answer: {math_result['answer']}")
    print(f"Correct: {evaluate_answer('gsm8k', math_result['answer'], '42')}")
    print(f"Steps: {len(math_result['steps'])}")
    print(f"Time: {math_result['metrics']['total_time']:.2f}s")

    # Code Agent demonstration
    print("\n3. LATS Code Agent (HumanEval)")
    print("-" * 30)
    code_agent = LATS(llm, "humaneval", max_iterations=5, verbose=True)
    code_result = code_agent.generate(
        "def add(a, b):\n    return a + b",
        key="assert add(1, 2) == 3\nassert add(-1, 1) == 0",
    )
    print(f"Answer: {code_result['answer']}")
    print(
        f"Correct: {evaluate_answer('humaneval', code_result['answer'], 'assert add(1, 2) == 3\nassert add(-1, 1) == 0')}"
    )
    print(f"Steps: {len(code_result['steps'])}")
    print(f"Time: {code_result['metrics']['total_time']:.2f}s")


if __name__ == "__main__":
    # Demonstrate the LATS agents
    demonstrate_lats_agents()

    # Example: Run just one benchmark
    # run_single_benchmark("gsm8k", num_runs=3)

    # Or run all benchmarks
    # run_all_benchmarks()
