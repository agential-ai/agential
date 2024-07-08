"""Few-sshot factory class."""

from agential.manager.fewshot_mapping import BENCHMARK_FEWSHOTS


class FewShotFactory:
    """A factory class for retrieving few-shot examples for a given benchmark and few-shot type."""

    @staticmethod
    def get_benchmark_fewshots(benchmark: str, fewshot_type: str) -> str:
        """Retrieve few-shot examples for a given benchmark and few-shot type.

        Available Benchmarks:
            - hotpotqa: Supports "cot", "direct", "react"
            - fever: Supports "cot", "direct", "react"
            - triviaqa: Supports "cot", "direct", "react"
            - ambignq: Supports "cot", "direct", "react"
            - gsm8k: Supports "pot", "cot", "react"
            - svamp: Supports "pot", "cot", "react"
            - tabmwp: Supports "pot", "cot", "react"
            - humaneval: Supports "pot", "cot", "react"
            - mbpp: Supports "pot", "cot", "react"

        Available Few-Shot Types:
            - "cot"
            - "direct"
            - "react"
            - "pot"

        Args:
            benchmark (str): The benchmark name.
            fewshot_type (str): The type of few-shot examples. It should be one of the predefined types in the FewShotType class.

        Returns:
            str: The few-shot examples corresponding to the given benchmark and type.
            If the benchmark or few-shot type is not found, returns a detailed error message.
        """
        if benchmark not in BENCHMARK_FEWSHOTS:
            raise ValueError(f"Benchmark '{benchmark}' not found.")

        examples = BENCHMARK_FEWSHOTS[benchmark].get(fewshot_type)
        if examples is None:
            raise ValueError(
                f"Few-shot type '{fewshot_type}' not found for benchmark '{benchmark}'."
            )

        return examples
