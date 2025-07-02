"""Self-Refine agents for iterative self-refinement."""

from agential.agents.self_refine.qa_agent import SelfRefineQA
from agential.agents.self_refine.math_agent import SelfRefineMath
from agential.agents.self_refine.code_agent import SelfRefineCode
from agential.agents.base import BaseMethod
from agential.agents.self_refine.prompts import (
    SELF_REFINE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_INSTRUCTION_FEVER,
    SELF_REFINE_INSTRUCTION_TRIVIAQA,
    SELF_REFINE_INSTRUCTION_AMBIGNQ,
    SELF_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_INSTRUCTION_SVAMP,
    SELF_REFINE_INSTRUCTION_TABMWP,
    SELF_REFINE_INSTRUCTION_HUMANEVAL,
    SELF_REFINE_INSTRUCTION_MBPP,
    HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
    FEVER_CRITIQUE_FEWSHOT_EXAMPLES,
    TRIVIAQA_CRITIQUE_FEWSHOT_EXAMPLES,
    AMBIGNQ_CRITIQUE_FEWSHOT_EXAMPLES,
    GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
    SVAMP_CRITIQUE_FEWSHOT_EXAMPLES,
    TABMWP_CRITIQUE_FEWSHOT_EXAMPLES,
    HUMANEVAL_CRITIQUE_FEWSHOT_EXAMPLES,
    MBPP_CRITIQUE_FEWSHOT_EXAMPLES,
    HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
    FEVER_REFINE_FEWSHOT_EXAMPLES,
    TRIVIAQA_REFINE_FEWSHOT_EXAMPLES,
    AMBIGNQ_REFINE_FEWSHOT_EXAMPLES,
    GSM8K_REFINE_FEWSHOT_EXAMPLES,
    SVAMP_REFINE_FEWSHOT_EXAMPLES,
    TABMWP_REFINE_FEWSHOT_EXAMPLES,
    HUMANEVAL_REFINE_FEWSHOT_EXAMPLES,
    MBPP_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_CRITIQUE_INSTRUCTION_FEVER,
    SELF_REFINE_CRITIQUE_INSTRUCTION_TRIVIAQA,
    SELF_REFINE_CRITIQUE_INSTRUCTION_AMBIGNQ,
    SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
    SELF_REFINE_CRITIQUE_INSTRUCTION_SVAMP,
    SELF_REFINE_CRITIQUE_INSTRUCTION_TABMWP,
    SELF_REFINE_CRITIQUE_INSTRUCTION_HUMANEVAL,
    SELF_REFINE_CRITIQUE_INSTRUCTION_MBPP,
    SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_REFINE_INSTRUCTION_FEVER,
    SELF_REFINE_REFINE_INSTRUCTION_TRIVIAQA,
    SELF_REFINE_REFINE_INSTRUCTION_AMBIGNQ,
    SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_REFINE_INSTRUCTION_SVAMP,
    SELF_REFINE_REFINE_INSTRUCTION_TABMWP,
    SELF_REFINE_REFINE_INSTRUCTION_HUMANEVAL,
    SELF_REFINE_REFINE_INSTRUCTION_MBPP,
    # Initial fewshot examples
    HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
    FEVER_FEWSHOT_EXAMPLES_DIRECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
    GSM8K_FEWSHOT_EXAMPLES_POT,
    SVAMP_FEWSHOT_EXAMPLES_POT,
    TABMWP_FEWSHOT_EXAMPLES_POT,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
    MBPP_FEWSHOT_EXAMPLES_POT,
)


SELF_REFINE_BENCHMARK_CONFIG = {
    # QA
    "hotpotqa": {
        "prompt": SELF_REFINE_INSTRUCTION_HOTPOTQA,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
        "examples": HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
        "critique_examples": HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
        "agent": SelfRefineQA,
    },
    "fever": {
        "prompt": SELF_REFINE_INSTRUCTION_FEVER,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_FEVER,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_FEVER,
        "examples": FEVER_FEWSHOT_EXAMPLES_DIRECT,
        "critique_examples": FEVER_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": FEVER_REFINE_FEWSHOT_EXAMPLES,
        "agent": SelfRefineQA,
    },
    "triviaqa": {
        "prompt": SELF_REFINE_INSTRUCTION_TRIVIAQA,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_TRIVIAQA,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_TRIVIAQA,
        "examples": TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
        "critique_examples": TRIVIAQA_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": TRIVIAQA_REFINE_FEWSHOT_EXAMPLES,
        "agent": SelfRefineQA,
    },
    "ambignq": {
        "prompt": SELF_REFINE_INSTRUCTION_AMBIGNQ,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_AMBIGNQ,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_AMBIGNQ,
        "examples": AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
        "critique_examples": AMBIGNQ_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": AMBIGNQ_REFINE_FEWSHOT_EXAMPLES,
        "agent": SelfRefineQA,
    },
    # Math
    "gsm8k": {
        "prompt": SELF_REFINE_INSTRUCTION_GSM8K,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
        "examples": GSM8K_FEWSHOT_EXAMPLES_POT,
        "critique_examples": GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": GSM8K_REFINE_FEWSHOT_EXAMPLES,
        "agent": SelfRefineMath,
    },
    "svamp": {
        "prompt": SELF_REFINE_INSTRUCTION_SVAMP,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_SVAMP,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_SVAMP,
        "examples": SVAMP_FEWSHOT_EXAMPLES_POT,
        "critique_examples": SVAMP_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": SVAMP_REFINE_FEWSHOT_EXAMPLES,
        "agent": SelfRefineMath,
    },
    "tabmwp": {
        "prompt": SELF_REFINE_INSTRUCTION_TABMWP,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_TABMWP,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_TABMWP,
        "examples": TABMWP_FEWSHOT_EXAMPLES_POT,
        "critique_examples": TABMWP_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": TABMWP_REFINE_FEWSHOT_EXAMPLES,
        "agent": SelfRefineMath,
    },
    # Code
    "humaneval": {
        "prompt": SELF_REFINE_INSTRUCTION_HUMANEVAL,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_HUMANEVAL,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_HUMANEVAL,
        "examples": HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        "critique_examples": HUMANEVAL_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": HUMANEVAL_REFINE_FEWSHOT_EXAMPLES,
        "agent": SelfRefineCode,
    },
    "mbpp": {
        "prompt": SELF_REFINE_INSTRUCTION_MBPP,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_MBPP,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_MBPP,
        "examples": MBPP_FEWSHOT_EXAMPLES_POT,
        "critique_examples": MBPP_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": MBPP_REFINE_FEWSHOT_EXAMPLES,
        "agent": SelfRefineCode,
    },
}


class SelfRefine(BaseMethod):
    _agent: object

    def __new__(cls, llm, benchmark, *args, **kwargs):
        try:
            config = SELF_REFINE_BENCHMARK_CONFIG[benchmark]
            agent_cls = config["agent"]
        except KeyError:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        agent = agent_cls(llm, benchmark, *args, **kwargs, config=config)
        instance = super().__new__(cls)
        instance._agent = agent
        instance.llm = agent.llm
        instance.benchmark = agent.benchmark
        instance.verbose = getattr(agent, "verbose", False)
        instance.config = getattr(agent, "config", {})
        return instance

    def generate(self, question: str, **kwargs):
        return self._agent.generate(question, **kwargs)  # type: ignore


__all__ = ["SelfRefine"]
