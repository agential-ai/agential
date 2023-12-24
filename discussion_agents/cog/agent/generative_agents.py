"""Generative Agents module implementation adapted from LangChain.

This implementation includes functions for performing the operations
in the Generative Agents paper without the graphic interface.

Original Paper: https://arxiv.org/abs/2304.03442
LangChain: https://github.com/langchain-ai/langchain
LangChain Generative Agents:
https://github.com/langchain-ai/langchain/tree/master/libs/experimental/langchain_experimental/generative_agents
LangChain Generative Agents Doc Page:
https://python.langchain.com/docs/use_cases/more/agents/agent_simulations/characters
"""
from typing import List, Optional

from langchain_core.language_models import LLM

from pydantic.v1 import root_validator

from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.modules.reflect.base import BaseReflector
from discussion_agents.cog.modules.score.base import BaseScorer
from discussion_agents.cog.modules.memory.generative_agents import GenerativeAgentMemory
from discussion_agents.cog.modules.reflect.generative_agents import GenerativeAgentReflector
from discussion_agents.cog.modules.score.generative_agents import GenerativeAgentScorer


class GenerativeAgent(BaseAgent):
    llm: LLM
    memory: GenerativeAgentMemory
    reflector: Optional[BaseReflector] = None
    scorer: Optional[BaseScorer] = None
    importance_weight: float = 0.15
    reflection_threshold: Optional[int] = 8

    # A bit petty, but it allows us to define the class attributes in a specific order.
    @root_validator
    def set_default_components(cls, values):
        if 'reflector' not in values or values['reflector'] is None:
            values['reflector'] = GenerativeAgentReflector(
                llm=values['llm'], retriever=values['memory'].retriever
            )
        if 'scorer' not in values or values['scorer'] is None:
            values['scorer'] = GenerativeAgentScorer(
                llm=values['llm'], importance_weight=values['importance_weight']
            )
        return values

    # Internal variables.
    is_reflecting: bool = False  #: :meta private:
    aggregate_importance: float = 0.0  #: :meta private:

    def reflect(self, last_k: int = 50) -> List[str]:
        pass
