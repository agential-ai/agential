"""CRITIC Agent.

Original Paper: https://arxiv.org/pdf/2305.11738
Paper Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
"""

from agential.methods.base import BaseMethod
from agential.methods.critic.code_agent import CriticCode
from agential.methods.critic.math_agent import CriticMath
from agential.methods.critic.qa_agent import CriticQA
from agential.methods.critic.prompts import (
    AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
    AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
    CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
    CRITIC_CRITIQUE_INSTRUCTION_FEVER,
    CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_INSTRUCTION_MBPP,
    CRITIC_CRITIQUE_INSTRUCTION_SVAMP,
    CRITIC_CRITIQUE_INSTRUCTION_TABMWP,
    CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_SVAMP,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_TABMWP,
    CRITIC_INSTRUCTION_AMBIGNQ,
    CRITIC_INSTRUCTION_FEVER,
    CRITIC_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_TRIVIAQA,
    CRITIC_POT_INSTRUCTION_GSM8K,
    CRITIC_POT_INSTRUCTION_HUMANEVAL,
    CRITIC_POT_INSTRUCTION_MBPP,
    CRITIC_POT_INSTRUCTION_SVAMP,
    CRITIC_POT_INSTRUCTION_TABMWP,
    FEVER_FEWSHOT_EXAMPLES_CRITIC,
    FEVER_FEWSHOT_EXAMPLES_DIRECT,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    GSM8K_FEWSHOT_EXAMPLES_POT,
    HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
    HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
    MBPP_FEWSHOT_EXAMPLES_CRITIC,
    MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    MBPP_FEWSHOT_EXAMPLES_POT,
    SVAMP_FEWSHOT_EXAMPLES_CRITIC,
    SVAMP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    SVAMP_FEWSHOT_EXAMPLES_POT,
    TABMWP_FEWSHOT_EXAMPLES_CRITIC,
    TABMWP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    TABMWP_FEWSHOT_EXAMPLES_POT,
    TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
    TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
)


CRITIC_BENCHMARK_CONFIG = {
    "hotpotqa": {
        "prompt": CRITIC_INSTRUCTION_HOTPOTQA,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
        "examples": HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
        "critique_examples": HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
        "agent": CriticQA,
    },
    "fever": {
        "prompt": CRITIC_INSTRUCTION_FEVER,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_FEVER,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_FEVER,
        "examples": FEVER_FEWSHOT_EXAMPLES_DIRECT,
        "critique_examples": FEVER_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": FEVER_FEWSHOT_EXAMPLES_CRITIC,
        "agent": CriticQA,
    },
    "triviaqa": {
        "prompt": CRITIC_INSTRUCTION_TRIVIAQA,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
        "examples": TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
        "critique_examples": TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
        "agent": CriticQA,
    },
    "ambignq": {
        "prompt": CRITIC_INSTRUCTION_AMBIGNQ,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
        "examples": AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
        "critique_examples": AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
        "agent": CriticQA,
    },
    "gsm8k": {
        "prompt": CRITIC_POT_INSTRUCTION_GSM8K,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
        "examples": GSM8K_FEWSHOT_EXAMPLES_POT,
        "critique_examples": GSM8K_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        "agent": CriticMath,
    },
    "svamp": {
        "prompt": CRITIC_POT_INSTRUCTION_SVAMP,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_SVAMP,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_SVAMP,
        "examples": SVAMP_FEWSHOT_EXAMPLES_POT,
        "critique_examples": SVAMP_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": SVAMP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        "agent": CriticMath,
    },
    "tabmwp": {
        "prompt": CRITIC_POT_INSTRUCTION_TABMWP,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_TABMWP,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_TABMWP,
        "examples": TABMWP_FEWSHOT_EXAMPLES_POT,
        "critique_examples": TABMWP_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": TABMWP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        "agent": CriticMath,
    },
    "humaneval": {
        "prompt": CRITIC_POT_INSTRUCTION_HUMANEVAL,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
        "examples": HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        "critique_examples": HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        "agent": CriticCode,
    },
    "mbpp": {
        "prompt": CRITIC_POT_INSTRUCTION_MBPP,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_MBPP,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
        "examples": MBPP_FEWSHOT_EXAMPLES_POT,
        "critique_examples": MBPP_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        "agent": CriticCode,
    },
}


class Critic(BaseMethod):
    _agent: BaseMethod  # type: ignore

    def __new__(cls, llm, benchmark, *args, **kwargs):
        try:
            config = CRITIC_BENCHMARK_CONFIG[benchmark]
            agent_cls = config["agent"]
        except KeyError:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        # Pass the entire config to the agent constructor (matching React pattern)
        agent = agent_cls(llm, benchmark, *args, **kwargs, config=config)
        instance = super().__new__(cls)
        instance._agent = agent
        instance.llm = agent.llm
        instance.benchmark = agent.benchmark
        instance.verbose = getattr(agent, "verbose", False)
        instance.config = getattr(agent, "config", {})
        return instance

    def generate(self, question: str, **kwargs):
        return self._agent.generate(question, **kwargs)


__all__ = ["Critic"]
