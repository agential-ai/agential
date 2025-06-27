"""Language Agent Tree Search (LATS) Agent."""

from agential.agents.lats.qa_agent import LATSQA
from agential.agents.lats.math_agent import LATSMath
from agential.agents.lats.code_agent import LATSCode
from agential.agents.lats.prompts import (
    LATS_INSTRUCTION_HOTPOTQA,
    LATS_INSTRUCTION_FEVER,
    LATS_INSTRUCTION_TRIVIAQA,
    LATS_INSTRUCTION_AMBIGNQ,
    LATS_INSTRUCTION_GSM8K,
    LATS_INSTRUCTION_SVAMP,
    LATS_INSTRUCTION_TABMWP,
    LATS_INSTRUCTION_HUMANEVAL,
    LATS_INSTRUCTION_MBPP,
    LATS_REFLECT_INSTRUCTION_HOTPOTQA,
    LATS_REFLECT_INSTRUCTION_FEVER,
    LATS_REFLECT_INSTRUCTION_TRIVIAQA,
    LATS_REFLECT_INSTRUCTION_AMBIGNQ,
    LATS_REFLECT_INSTRUCTION_GSM8K,
    LATS_REFLECT_INSTRUCTION_SVAMP,
    LATS_REFLECT_INSTRUCTION_TABMWP,
    LATS_REFLECT_INSTRUCTION_HUMANEVAL,
    LATS_REFLECT_INSTRUCTION_MBPP,
    LATS_VALUE_INSTRUCTION_HOTPOTQA,
    LATS_VALUE_INSTRUCTION_FEVER,
    LATS_VALUE_INSTRUCTION_TRIVIAQA,
    LATS_VALUE_INSTRUCTION_AMBIGNQ,
    LATS_VALUE_INSTRUCTION_GSM8K,
    LATS_VALUE_INSTRUCTION_SVAMP,
    LATS_VALUE_INSTRUCTION_TABMWP,
    LATS_VALUE_INSTRUCTION_HUMANEVAL,
    LATS_VALUE_INSTRUCTION_MBPP,
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    FEVER_FEWSHOT_EXAMPLES_LATS_REFLECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_LATS_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
    SVAMP_FEWSHOT_EXAMPLES_LATS_REFLECT,
    TABMWP_FEWSHOT_EXAMPLES_LATS_REFLECT,
    HUMANEVAL_FEWSHOT_EXAMPLES_LATS_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_LATS_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    FEVER_FEWSHOT_EXAMPLES_LATS_VALUE,
    TRIVIAQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    AMBIGNQ_FEWSHOT_EXAMPLES_LATS_VALUE,
    GSM8K_FEWSHOT_EXAMPLES_LATS_VALUE,
    SVAMP_FEWSHOT_EXAMPLES_LATS_VALUE,
    TABMWP_FEWSHOT_EXAMPLES_LATS_VALUE,
    HUMANEVAL_FEWSHOT_EXAMPLES_LATS_VALUE,
    MBPP_FEWSHOT_EXAMPLES_LATS_VALUE,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
    FEVER_FEWSHOT_EXAMPLES_REACT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
    GSM8K_FEWSHOT_EXAMPLES_REACT,
    SVAMP_FEWSHOT_EXAMPLES_REACT,
    TABMWP_FEWSHOT_EXAMPLES_REACT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
    MBPP_FEWSHOT_EXAMPLES_REACT,
)
from agential.agents.base import BaseAgent

BENCHMARK_CONFIG = {
    # QA
    "hotpotqa": {
        "prompt": LATS_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_HOTPOTQA,
        "value_prompt": LATS_VALUE_INSTRUCTION_HOTPOTQA,
        "fewshot": HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "reflect_fewshot": HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_fewshot": HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
        "agent": LATSQA,
    },
    "fever": {
        "prompt": LATS_INSTRUCTION_FEVER,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_FEVER,
        "value_prompt": LATS_VALUE_INSTRUCTION_FEVER,
        "fewshot": FEVER_FEWSHOT_EXAMPLES_REACT,
        "reflect_fewshot": FEVER_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_fewshot": FEVER_FEWSHOT_EXAMPLES_LATS_VALUE,
        "agent": LATSQA,
    },
    "triviaqa": {
        "prompt": LATS_INSTRUCTION_TRIVIAQA,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_TRIVIAQA,
        "value_prompt": LATS_VALUE_INSTRUCTION_TRIVIAQA,
        "fewshot": TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
        "reflect_fewshot": TRIVIAQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_fewshot": TRIVIAQA_FEWSHOT_EXAMPLES_LATS_VALUE,
        "agent": LATSQA,
    },
    "ambignq": {
        "prompt": LATS_INSTRUCTION_AMBIGNQ,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_AMBIGNQ,
        "value_prompt": LATS_VALUE_INSTRUCTION_AMBIGNQ,
        "fewshot": AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
        "reflect_fewshot": AMBIGNQ_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_fewshot": AMBIGNQ_FEWSHOT_EXAMPLES_LATS_VALUE,
        "agent": LATSQA,
    },
    # Math
    "gsm8k": {
        "prompt": LATS_INSTRUCTION_GSM8K,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_GSM8K,
        "value_prompt": LATS_VALUE_INSTRUCTION_GSM8K,
        "fewshot": GSM8K_FEWSHOT_EXAMPLES_REACT,
        "reflect_fewshot": GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_fewshot": GSM8K_FEWSHOT_EXAMPLES_LATS_VALUE,
        "agent": LATSMath,
    },
    "svamp": {
        "prompt": LATS_INSTRUCTION_SVAMP,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_SVAMP,
        "value_prompt": LATS_VALUE_INSTRUCTION_SVAMP,
        "fewshot": SVAMP_FEWSHOT_EXAMPLES_REACT,
        "reflect_fewshot": SVAMP_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_fewshot": SVAMP_FEWSHOT_EXAMPLES_LATS_VALUE,
        "agent": LATSMath,
    },
    "tabmwp": {
        "prompt": LATS_INSTRUCTION_TABMWP,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_TABMWP,
        "value_prompt": LATS_VALUE_INSTRUCTION_TABMWP,
        "fewshot": TABMWP_FEWSHOT_EXAMPLES_REACT,
        "reflect_fewshot": TABMWP_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_fewshot": TABMWP_FEWSHOT_EXAMPLES_LATS_VALUE,
        "agent": LATSMath,
    },
    # Code
    "humaneval": {
        "prompt": LATS_INSTRUCTION_HUMANEVAL,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_HUMANEVAL,
        "value_prompt": LATS_VALUE_INSTRUCTION_HUMANEVAL,
        "fewshot": HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
        "reflect_fewshot": HUMANEVAL_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_fewshot": HUMANEVAL_FEWSHOT_EXAMPLES_LATS_VALUE,
        "agent": LATSCode,
    },
    "mbpp": {
        "prompt": LATS_INSTRUCTION_MBPP,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_MBPP,
        "value_prompt": LATS_VALUE_INSTRUCTION_MBPP,
        "fewshot": MBPP_FEWSHOT_EXAMPLES_REACT,
        "reflect_fewshot": MBPP_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_fewshot": MBPP_FEWSHOT_EXAMPLES_LATS_VALUE,
        "agent": LATSCode,
    },
}


class LATS(BaseAgent):
    _agent: BaseAgent  # type: ignore

    def __new__(cls, llm, benchmark, *args, **kwargs):
        try:
            config = BENCHMARK_CONFIG[benchmark]
            agent_cls = config["agent"]
        except KeyError:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        # Create the agent instance
        agent = agent_cls(llm, benchmark, *args, **kwargs, config=config)
        # Create a LATS instance and store the agent
        instance = super().__new__(cls)
        instance._agent = agent
        # Copy attributes for BaseAgent compliance
        instance.llm = agent.llm
        instance.benchmark = agent.benchmark
        instance.verbose = getattr(agent, "verbose", False)
        instance.config = getattr(agent, "config", {})
        return instance

    def generate(self, question: str, **kwargs):
        return self._agent.generate(question, **kwargs)


__all__ = ["LATS"]
