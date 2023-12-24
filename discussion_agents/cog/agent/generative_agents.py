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
from typing import Optional, List
from langchain_core.language_models import LLM

from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.modules.memory.generative_agents import GenerativeAgentMemory


class GenerativeAgent(BaseAgent):
    llm: LLM
    memory: GenerativeAgentMemory
    
    reflection_threshold: Optional[int] = 8

    # Internal variables.
    reflecting: bool = False  #: :meta private:
    aggregate_importance: float = 0.0  #: :meta private:

    def reflect(self, last_k: int = 50) -> List[str]:
        pass