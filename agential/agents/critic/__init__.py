"""CRITIC Agent."""

from agential.agents.critic.qa_agent import CriticQA
from agential.agents.critic.prompts import (
    CRITIC_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_FEVER,
    CRITIC_INSTRUCTION_TRIVIAQA,
    CRITIC_INSTRUCTION_AMBIGNQ,
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_CRITIQUE_INSTRUCTION_FEVER,
    CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
    CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
    HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
    FEVER_FEWSHOT_EXAMPLES_CRITIC,
    TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
    AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
)
from agential.agents.base import BaseAgent

CRITIC_BENCHMARK_CONFIG = {
    "hotpotqa": {
        "prompt": CRITIC_INSTRUCTION_HOTPOTQA,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
        "examples": HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
        "agent": CriticQA,
    },
    "fever": {
        "prompt": CRITIC_INSTRUCTION_FEVER,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_FEVER,
        "examples": FEVER_FEWSHOT_EXAMPLES_CRITIC,
        "agent": CriticQA,
    },
    "triviaqa": {
        "prompt": CRITIC_INSTRUCTION_TRIVIAQA,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
        "examples": TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
        "agent": CriticQA,
    },
    "ambignq": {
        "prompt": CRITIC_INSTRUCTION_AMBIGNQ,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
        "examples": AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
        "agent": CriticQA,
    },
}

class Critic(BaseAgent):
    _agent: BaseAgent  # type: ignore

    def __new__(cls, llm, benchmark, *args, **kwargs):
        try:
            config = CRITIC_BENCHMARK_CONFIG[benchmark]
            agent_cls = config["agent"]
        except KeyError:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        agent = agent_cls(llm, benchmark, *args, **kwargs)
        instance = super().__new__(cls)
        instance._agent = agent
        instance.llm = agent.llm
        instance.benchmark = agent.benchmark
        instance.verbose = getattr(agent, "verbose", False)
        instance.config = getattr(agent, "config", {})
        return instance

    def generate(self, question: str, **kwargs):
        return self._agent.generate(question, **kwargs)

__all__ = ["CriticQA", "CRITIC_BENCHMARK_CONFIG", "Critic"]
