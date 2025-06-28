from agential.agents.reflexion import Reflexion, BENCHMARK_CONFIG


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
    total_trials = sum(
        result["metrics"]["trials_taken"] for result in benchmark_results
    )

    # Calculate averages
    avg_tokens = total_tokens / total_runs if total_runs > 0 else 0
    avg_time = total_time / total_runs if total_runs > 0 else 0
    avg_cost = total_cost / total_runs if total_runs > 0 else 0
    avg_trials = total_trials / total_runs if total_runs > 0 else 0

    # Calculate total steps across all runs
    total_steps = sum(
        sum(len(trial["steps"]) for trial in result["trials"])
        for result in benchmark_results
    )
    avg_steps = total_steps / total_runs if total_runs > 0 else 0

    return {
        "total_runs": total_runs,
        "correct_runs": correct_runs,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "total_cost": total_cost,
        "total_trials": total_trials,
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "avg_cost": avg_cost,
        "avg_trials": avg_trials,
        "total_steps": total_steps,
        "avg_steps": avg_steps,
    }


# Example questions/keys for each benchmark (from the reflexion notebook)
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
            "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?\"\n",
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


def test_single_benchmark(benchmark_name: str, num_runs: int = 1, max_steps: int = 6, max_trials: int = 3, max_reflections: int = 2, verbose: bool = True):
    """
    Test a single benchmark with configurable parameters.
    
    Args:
        benchmark_name: Name of the benchmark to test (e.g., "hotpotqa", "gsm8k", "humaneval")
        num_runs: Number of times to run the benchmark
        max_steps: Maximum number of steps per trial
        max_trials: Maximum number of trials per run
        max_reflections: Maximum number of reflections
        verbose: Whether to show detailed output
    """
    from agential.core.llm import LLM

    if benchmark_name not in BENCHMARK_CONFIG:
        print(f"Error: Unknown benchmark '{benchmark_name}'")
        print(f"Available benchmarks: {list(BENCHMARK_CONFIG.keys())}")
        return

    llm = LLM("gpt-4.1")
    examples = get_benchmark_examples()
    
    if benchmark_name not in examples:
        print(f"Error: No example found for benchmark '{benchmark_name}'")
        return

    question, key = examples[benchmark_name]
    
    print(f"Testing {benchmark_name.upper()} Benchmark")
    print("=" * 60)
    print(f"Question: {question}")
    print(f"Expected Answer: {key}")
    print(f"Parameters: {num_runs} runs, {max_steps} max steps, {max_trials} max trials, {max_reflections} max reflections")
    print("-" * 60)

    agent = Reflexion(
        llm, benchmark_name, 
        max_steps=max_steps, 
        max_trials=max_trials, 
        max_reflections=max_reflections, 
        verbose=verbose
    )

    benchmark_results = []
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...", end=" ")
        if benchmark_name == "mbpp":
            # MBPP expects tests as additional_keys and reflect_additional_keys
            result = agent.generate(
                question,
                key=key,
                additional_keys={"tests": key},
                reflect_additional_keys={"tests": key},
            )
        else:
            result = agent.generate(question, key=key)

        benchmark_results.append(result)
        status = "✓" if result["correct"] else "✗"
        print(f"{status} ({result['metrics']['total_time']:.2f}s)")
        
        if verbose:
            print(f"  Answer: {result['answer']}")
            print(f"  Steps taken: {len(result['steps'])}")
            print(f"  Trials taken: {result['metrics']['trials_taken']}")
            print(f"  Total tokens: {result['metrics']['total_tokens']}")
            print(f"  Total cost: ${result['metrics']['total_cost']:.6f}")
            if result.get('reflections'):
                print(f"  Reflections: {len(result['reflections'])}")
            print()


    # Calculate stats for this benchmark
    stats = calculate_benchmark_stats(benchmark_results)

    # Print results
    print("=" * 60)
    print(f"{benchmark_name.upper()} RESULTS")
    print("=" * 60)
    print(f"Accuracy: {stats['accuracy']:.1%} ({stats['correct_runs']}/{stats['total_runs']})")
    print(f"Average Time: {stats['avg_time']:.2f} seconds")
    print(f"Average Tokens: {stats['avg_tokens']:.0f}")
    print(f"Average Cost: ${stats['avg_cost']:.6f}")
    print(f"Average Steps: {stats['avg_steps']:.1f}")
    print(f"Average Trials: {stats['avg_trials']:.1f}")

    return {
        "benchmark": benchmark_name,
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

    print("Starting benchmark evaluation...")
    print("=" * 60)

    for benchmark in BENCHMARK_CONFIG:
        if benchmark != "hotpotqa":
            continue
        print(f"\n--- Running {benchmark.upper()} Benchmark ---")
        question, key = examples[benchmark]
        agent = Reflexion(
            llm, benchmark, max_steps=3, max_trials=1, max_reflections=2, verbose=True
        )

        # Run the same benchmark 10 times
        benchmark_results = []
        for run in range(1):
            print(f"  Run {run + 1}/10...", end=" ")
            try:
                if benchmark == "mbpp":
                    # MBPP expects tests as additional_keys and reflect_additional_keys
                    result = agent.generate(
                        question,
                        key=key,
                        additional_keys={"tests": key},
                        reflect_additional_keys={"tests": key},
                    )
                else:
                    result = agent.generate(question, key=key)

                benchmark_results.append(result)
                status = "✓" if result["correct"] else "✗"
                print(f"{status} ({result['metrics']['total_time']:.2f}s)")

            except Exception as e:
                print(f"ERROR: {str(e)}")
                # Add a failed result for consistency
                benchmark_results.append(
                    {
                        "answer": "ERROR",
                        "correct": False,
                        "metrics": {
                            "total_tokens": 0,
                            "total_time": 0,
                            "total_cost": 0,
                            "trials_taken": 0,
                        },
                    }
                )

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
    print("COMPREHENSIVE BENCHMARK RESULTS")
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
            f"{benchmark:<12} {accuracy_pct:>6.1f}%   {stats['avg_time']:>8.2f}s   {stats['avg_tokens']:>10.0f}   ${stats['avg_cost']:>10.6f}   {stats['avg_steps']:>8.1f}"
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
        print(f"Average Trials: {stats['avg_trials']:.1f}")

        print("\nIndividual Run Results:")
        for i, result in enumerate(results):
            status = "✓" if result["correct"] else "✗"
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


if __name__ == "__main__":
    # Example usage of the new single benchmark test function
    print("Example: Testing a single benchmark")
    print("=" * 50)
    
    # Test hotpotqa with default parameters
    test_single_benchmark("mbpp", num_runs=1, max_steps=3, max_trials=2, verbose=True)
    
    # Test gsm8k with custom parameters
    # test_single_benchmark("gsm8k", num_runs=3, max_steps=4, max_trials=2, verbose=False)
    
    # Test humaneval with minimal parameters
    # test_single_benchmark("humaneval", num_runs=1, max_steps=3, max_trials=1, verbose=True)
    
    # Uncomment one of the above lines to test a specific benchmark
    # Or run the full benchmark suite:
    # run_all_benchmarks()
