"""ReAct agents for reasoning and acting."""

from agential.agents.react.qa_agent import ReActQA
from agential.agents.react.math_agent import ReActMath
from agential.agents.react.code_agent import ReActCode
from agential.agents.react.prompts import (
    REACT_INSTRUCTION_HOTPOTQA,
    REACT_INSTRUCTION_FEVER,
    REACT_INSTRUCTION_TRIVIAQA,
    REACT_INSTRUCTION_AMBIGNQ,
    REACT_INSTRUCTION_GSM8K,
    REACT_INSTRUCTION_SVAMP,
    REACT_INSTRUCTION_TABMWP,
    REACT_INSTRUCTION_HUMANEVAL,
    REACT_INSTRUCTION_MBPP,
    AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
    FEVER_FEWSHOT_EXAMPLES_REACT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
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
        "prompt": REACT_INSTRUCTION_HOTPOTQA,
        "fewshot": HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "agent": ReActQA,
    },
    "fever": {
        "prompt": REACT_INSTRUCTION_FEVER,
        "fewshot": FEVER_FEWSHOT_EXAMPLES_REACT,
        "agent": ReActQA,
    },
    "triviaqa": {
        "prompt": REACT_INSTRUCTION_TRIVIAQA,
        "fewshot": TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
        "agent": ReActQA,
    },
    "ambignq": {
        "prompt": REACT_INSTRUCTION_AMBIGNQ,
        "fewshot": AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
        "agent": ReActQA,
    },
    # Math
    "gsm8k": {
        "prompt": REACT_INSTRUCTION_GSM8K,
        "fewshot": GSM8K_FEWSHOT_EXAMPLES_REACT,
        "agent": ReActMath,
    },
    "svamp": {
        "prompt": REACT_INSTRUCTION_SVAMP,
        "fewshot": SVAMP_FEWSHOT_EXAMPLES_REACT,
        "agent": ReActMath,
    },
    "tabmwp": {
        "prompt": REACT_INSTRUCTION_TABMWP,
        "fewshot": TABMWP_FEWSHOT_EXAMPLES_REACT,
        "agent": ReActMath,
    },
    # Code
    "humaneval": {
        "prompt": REACT_INSTRUCTION_HUMANEVAL,
        "fewshot": HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
        "agent": ReActCode,
    },
    "mbpp": {
        "prompt": REACT_INSTRUCTION_MBPP,
        "fewshot": MBPP_FEWSHOT_EXAMPLES_REACT,
        "agent": ReActCode,
    },
}


class ReAct(BaseAgent):
    _agent: BaseAgent  # type: ignore

    def __new__(cls, llm, benchmark, *args, **kwargs):
        try:
            config = BENCHMARK_CONFIG[benchmark]
            agent_cls = config["agent"]
        except KeyError:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        # Create the agent instance
        agent = agent_cls(llm, benchmark, *args, **kwargs, config=config)
        # Create a ReAct instance and store the agent
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


__all__ = ["ReAct", "ReActQA", "ReActMath", "ReActCode"]
