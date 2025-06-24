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


def run_all_benchmarks():
    from agential.core.llm import LLM

    llm = LLM("gpt-4.1")
    examples = get_benchmark_examples()
    all_results = []
    for benchmark in BENCHMARK_CONFIG:
        if benchmark != "triviaqa":
            continue
        print(f"\n--- {benchmark.upper()} Example ---")
        question, key = examples[benchmark]
        agent = Reflexion(
            llm, benchmark, max_steps=3, max_trials=1, max_reflections=2, verbose=True
        )

        # Run the same benchmark 10 times
        benchmark_results = []
        for run in range(10):
            print(f"Run {run + 1}/10...")
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

        all_results.append(
            {
                "benchmark": benchmark,
                "question": question,
                "key": key,
                "results": benchmark_results,
            }
        )

    print("\n=== All Results ===")
    for entry in all_results:
        print(f"Benchmark: {entry['benchmark']}")
        print("Answers from all 10 runs:")
        for i, result in enumerate(entry["results"]):
            print(f"  Run {i + 1}: {result['answer']}")
        print("-" * 40)


if __name__ == "__main__":
    run_all_benchmarks()
