"""ExpeL Agent.

Original Paper: https://arxiv.org/pdf/2308.10144.pdf
Paper Repository: https://github.com/LeapLabTHU/ExpeL
"""

from typing import Any, Dict, Optional

from agential.agents.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.agents.expel.output import ExpeLOutput
from agential.agents.expel.prompts import (
    AMBIGNQ_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    EXPEL_REFLEXION_REACT_INSTRUCTION_AMBIGNQ,
    EXPEL_REFLEXION_REACT_INSTRUCTION_FEVER,
    EXPEL_REFLEXION_REACT_INSTRUCTION_GSM8K,
    EXPEL_REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    EXPEL_REFLEXION_REACT_INSTRUCTION_HUMANEVAL,
    EXPEL_REFLEXION_REACT_INSTRUCTION_MBPP,
    EXPEL_REFLEXION_REACT_INSTRUCTION_SVAMP,
    EXPEL_REFLEXION_REACT_INSTRUCTION_TABMWP,
    EXPEL_REFLEXION_REACT_INSTRUCTION_TRIVIAQA,
    EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_AMBIGNQ,
    EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_FEVER,
    EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
    EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_HUMANEVAL,
    EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
    EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_SVAMP,
    EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_TABMWP,
    EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_TRIVIAQA,
    FEVER_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    HUMANEVAL_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    SVAMP_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    TABMWP_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
)
from agential.agents.expel.strategies.base import ExpeLBaseStrategy
from agential.agents.expel.strategies.code import (
    ExpeLHEvalStrategy,
    ExpeLMBPPStrategy,
)
from agential.agents.expel.strategies.math import (
    ExpeLGSM8KStrategy,
    ExpeLSVAMPStrategy,
    ExpeLTabMWPStrategy,
)
from agential.agents.expel.strategies.qa import (
    ExpeLAmbigNQStrategy,
    ExpeLFEVERStrategy,
    ExpeLHotQAStrategy,
    ExpeLTriviaQAStrategy,
)
from agential.agents.reflexion.agent import ReflexionReAct
from agential.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.core.base.agents.agent import BaseAgent
from agential.llm.llm import BaseLLM

EXPEL_BENCHMARK_FEWSHOTS = {
    Benchmarks.HOTPOTQA: [FewShotType.REACT],
    Benchmarks.FEVER: [FewShotType.REACT],
    Benchmarks.TRIVIAQA: [FewShotType.REACT],
    Benchmarks.AMBIGNQ: [FewShotType.REACT],
    Benchmarks.GSM8K: [FewShotType.REACT],
    Benchmarks.SVAMP: [FewShotType.REACT],
    Benchmarks.TABMWP: [FewShotType.REACT],
    Benchmarks.HUMANEVAL: [FewShotType.REACT],
    Benchmarks.MBPP: [FewShotType.REACT],
}

EXPEL_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_FEVER,
        "reflect_prompt": EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_TRIVIAQA,
        "reflect_prompt": EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_AMBIGNQ,
        "reflect_prompt": EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_GSM8K,
        "reflect_prompt": EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_SVAMP,
        "reflect_prompt": EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_TABMWP,
        "reflect_prompt": EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_HUMANEVAL,
        "reflect_prompt": EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_MBPP,
        "reflect_prompt": EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
    },
}

EXPEL_FEWSHOTS = {
    Benchmarks.HOTPOTQA: {
        "reflect_examples": HOTPOTQA_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.TRIVIAQA: {
        "reflect_examples": TRIVIAQA_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.AMBIGNQ: {
        "reflect_examples": AMBIGNQ_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.FEVER: {
        "reflect_examples": FEVER_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.GSM8K: {
        "reflect_examples": GSM8K_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.SVAMP: {
        "reflect_examples": SVAMP_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.TABMWP: {
        "reflect_examples": TABMWP_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.HUMANEVAL: {
        "reflect_examples": HUMANEVAL_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.MBPP: {
        "reflect_examples": MBPP_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    },
}


EXPEL_STRATEGIES = {
    Benchmarks.HOTPOTQA: ExpeLHotQAStrategy,
    Benchmarks.FEVER: ExpeLFEVERStrategy,
    Benchmarks.TRIVIAQA: ExpeLTriviaQAStrategy,
    Benchmarks.AMBIGNQ: ExpeLAmbigNQStrategy,
    Benchmarks.GSM8K: ExpeLGSM8KStrategy,
    Benchmarks.SVAMP: ExpeLSVAMPStrategy,
    Benchmarks.TABMWP: ExpeLTabMWPStrategy,
    Benchmarks.HUMANEVAL: ExpeLHEvalStrategy,
    Benchmarks.MBPP: ExpeLMBPPStrategy,
}


class ExpeL(BaseAgent):
    """Implements ExpeL, a reflective, experiential learning agent.

    Attributes:
        llm (BaseLLM): Primary language model for general tasks.
        benchmark (str): The benchmark name.
        reflexion_react_strategy_kwargs (Dict[str, Any]): Configuration options for the ReflexionReAct agent.
            Defaults max_steps=7 and max_trials=3 for the ReflexionReAct.
        reflexion_react_agent (Optional[ReflexionReAct]): The ReflexionReAct agent. Optional.
        experience_memory (Optional[ExpeLExperienceMemory]): Memory module for storing experiences.
        insight_memory (Optional[ExpeLInsightMemory]): Memory module for storing insights derived from experiences.
        success_batch_size (int): Batch size for processing success experiences in generating insights.
        testing (bool, optional): Whether to run in testing mode. Defaults to False.

    Methods:
        generate(question, key): Generates a response based on a given question and key, potentially extracting insights and applying self-reflection in the process.
        reset(): Resets the agent's state for a new problem-solving session, clearing memory modules and the ReAct agent's state.
        gather_experience(questions, keys): Collects experiences from interactions, storing them for future reference and insight extraction.
        extract_insights(experiences): Analyzes stored experiences to extract and store insights for improving future interactions.
        update_insights(operations): Updates the stored insights based on the analysis of new experiences.
        retrieve(): Retrieves the current state of the agent's memories, including both experiences and insights.
    """

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        reflexion_react_agent: Optional[ReflexionReAct] = None,
        experience_memory: Optional[ExpeLExperienceMemory] = None,
        insight_memory: Optional[ExpeLInsightMemory] = None,
        reflexion_react_strategy_kwargs: Dict[str, Any] = {
            "max_steps": 7,
            "max_trials": 3,
        },
        testing: bool = False,
        **strategy_kwargs: Any,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, benchmark=benchmark, testing=testing)

        reflexion_react_agent = reflexion_react_agent or ReflexionReAct(
            llm=llm,
            benchmark=benchmark,
            testing=testing,
            **reflexion_react_strategy_kwargs,
        )

        self.strategy = ExpeL.get_strategy(
            benchmark=self.benchmark,
            llm=self.llm,
            reflexion_react_agent=reflexion_react_agent,
            experience_memory=experience_memory,
            insight_memory=insight_memory,
            testing=self.testing,
            **strategy_kwargs,
        )

    @staticmethod
    def get_fewshots(
        benchmark: str, fewshot_type: str, **kwargs: Any
    ) -> Dict[str, str]:
        """Retrieve few-shot examples based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            fewshot_type (str): The benchmark few-shot type.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of few-shot examples.
        """
        if benchmark not in EXPEL_FEWSHOTS:
            raise ValueError(f"Benchmark '{benchmark}' few-shots not found for ExpeL.")

        if fewshot_type not in EXPEL_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for ExpeL."
            )

        benchmark_fewshots = BENCHMARK_FEWSHOTS[benchmark][fewshot_type]

        return {"examples": benchmark_fewshots, **EXPEL_FEWSHOTS[benchmark]}

    @staticmethod
    def get_prompts(benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve the prompt instruction based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: The prompt instructions.
        """
        if benchmark not in EXPEL_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for ExpeL.")

        return EXPEL_PROMPTS[benchmark]

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> ExpeLBaseStrategy:
        """Returns an instance of the appropriate ExpeL strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            ExpeLBaseStrategy: An instance of the appropriate ExpeL strategy.
        """
        if benchmark not in EXPEL_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent ExpeL")

        strategy = EXPEL_STRATEGIES[benchmark]
        return strategy(**kwargs)

    def generate(
        self,
        question: str,
        key: str,
        examples: str = "",
        prompt: str = "",
        reflect_examples: str = "",
        reflect_prompt: str = "",
        reflect_strategy: str = "reflexion",
        additional_keys: Dict[str, str] = {},
        reflect_additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        use_dynamic_examples: bool = True,
        extract_insights: bool = True,
        patience: int = 3,
        k_docs: int = 24,
        num_fewshots: int = 6,
        max_fewshot_tokens: int = 1500,
        reranker_strategy: Optional[str] = None,
        reset: bool = False,
    ) -> ExpeLOutput:
        """Collects and stores experiences from interactions based on specified questions and strategies.

        This method invokes the ReflexionReAct agent to process a set of questions with corresponding keys,
        using the provided strategy, prompts, and examples. It captures the trajectories of the agent's reasoning
        and reflection process, storing them for future analysis and insight extraction.

        Parameters:
            questions (List[str]): A list of questions for the agent to process.
            keys (List[str]): Corresponding keys to the questions, used for internal tracking and analysis.
            examples (str): Examples to provide context or guidance for the ReflexionReAct agent. Defaults to "".
            prompt (str): The initial prompt or instruction to guide the ReflexionReAct agent's process. Defaults to "".
            reflect_examples (str): Examples specifically for the reflection phase of processing. Defaults to "".
            reflect_prompt (str): The prompt or instruction guiding the reflection process. Defaults to "".
            reflect_strategy (Optional[str]): The strategy to use for processing questions. Defaults to "reflexion".
            additional_keys (Dict[str, str]): The additional keys. Defaults to {}.
            reflect_additional_keys (Dict[str, str]): Additional keys for the reflection phase. Defaults to {}.
            fewshot_type (str): The type of fewshot to use. Defaults to "".
            use_dynamic_examples (bool): A boolean specifying whether or not to use dynamic examples from ExpeL's memory. Defaults to True.
            extract_insights (bool): Whether to extract insights from the experiences. Defaults to True.
            patience (int): The number of times to retry the agent's process if it fails. Defaults to 3.
            k_docs (int): The number of documents to retrieve for the fewshot. Defaults to 24.
            num_fewshots (int): The number of examples to use for the fewshot. Defaults to 6.
            max_fewshot_tokens (int): The maximum number of tokens to use for the fewshot. Defaults to 1500.
            reranker_strategy (Optional[str]): The strategy to use for re-ranking the retrieved. Defaults to None.
            reset (bool): Whether to reset the agent's state for a new problem-solving session. Defaults to False.

        Returns:
            ExpeLOutput: The output of the ExpeL agent.
        """
        if not prompt or not reflect_prompt or not examples or not reflect_examples:
            if not fewshot_type:
                fewshot_type = EXPEL_BENCHMARK_FEWSHOTS[self.benchmark][0]  # type: ignore
            fewshots = ExpeL.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type
            )
            prompts = ExpeL.get_prompts(benchmark=self.benchmark)
            examples = fewshots["examples"]
            prompt = prompts["prompt"]
            reflect_examples = fewshots["reflect_examples"]
            reflect_prompt = prompts["reflect_prompt"]

        out = self.strategy.generate(
            question=question,
            key=key,
            examples=examples,
            prompt=prompt,
            reflect_examples=reflect_examples,
            reflect_prompt=reflect_prompt,
            reflect_strategy=reflect_strategy,
            additional_keys=additional_keys,
            reflect_additional_keys=reflect_additional_keys,
            use_dynamic_examples=use_dynamic_examples,
            extract_insights=extract_insights,
            patience=patience,
            k_docs=k_docs,
            num_fewshots=num_fewshots,
            max_fewshot_tokens=max_fewshot_tokens,
            reranker_strategy=reranker_strategy,
            reset=reset,
        )

        return out
