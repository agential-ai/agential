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
from typing import List, Optional, Tuple
from datetime import datetime
from itertools import chain

from langchain_core.language_models import LLM

from pydantic.v1 import root_validator

from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.modules.reflect.base import BaseReflector
from discussion_agents.cog.modules.score.base import BaseScorer
from discussion_agents.cog.modules.memory.generative_agents import GenerativeAgentMemory
from discussion_agents.cog.modules.reflect.generative_agents import GenerativeAgentReflector
from discussion_agents.cog.modules.score.generative_agents import GenerativeAgentScorer
from discussion_agents.cog.functional.generative_agents import (
    get_insights_on_topics,
    get_topics_of_reflection,
    reflect,
    score_memories_importance,
)
from discussion_agents.utils.format import (
    format_memories_detail,
)
from discussion_agents.utils.fetch import fetch_memories

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

    def get_topics_of_reflection(self, last_k: int = 50) -> List[str]:
        """Generate high-level reflection topics based on recent observations.

        This method leverages the `get_topics_of_reflection` function to generate a list of
        high-level reflection topics based on recent observations from the agent's retriever
        memory. The method considers the last 'last_k' observations and formats them for
        reflection.

        Args:
            last_k (int, optional): The number of recent observations to consider. Default is 50.

        Returns:
            List[str]: A list of high-level reflection topics.

        Example:
            memory = GenerativeAgentMemory(...)
            reflection_topics = memory.get_topics_of_reflection(last_k=100)
        """
        observations = self.memory.load_memories(last_k=last_k)["most_recent_memories"]
        observations = "\n".join([format_memories_detail(o) for o in observations])
        return get_topics_of_reflection(observations=observations, llm=self.llm)

    def get_insights_on_topic(
        self,
        topics: List[str],
        now: Optional[datetime] = None,
    ) -> List[List[str]]:
        """Generate insights on specified topics based on related memories.

        This method acts as a wrapper for generating insights on specified topics. It leverages
        the `discussion_agents.reflecting.generative_agents.get_insights_on_topic` function.

        Args:
            topics (List[str]): A list of topics for which insights are to be generated.
            now (Optional[datetime], optional): The current date and time for temporal context. Default is None.

        Returns:
            List[List[str]]: Lists of high-level insights corresponding to each specified topic.

        Example:
            selected_topics = ["Artificial Intelligence trends", "Recent travel experiences"]
            memory = GenerativeAgentMemory(...)
            generated_insights = memory.get_insights_on_topic(selected_topics, now=datetime.now())
        """
        related_memories = []
        for topic in topics:
            fetched_memories = self.memory.load_memories(queries=topic, now=now)["relevant_memories"]
            topic_related_memories = "\n".join(
                [
                    format_memories_detail(memories=memory, prefix=f"{i+1}. ")
                    for i, memory in enumerate(fetched_memories)
                ]
            )
            related_memories.append(topic_related_memories)

        new_insights = get_insights_on_topics(
            topics=topics, related_memories=related_memories, llm=self.llm
        )

        return new_insights
    
    def reflect(
        self, last_k: int = 50, now: Optional[datetime] = None
    ) -> Tuple[List[str], List[str]]:
        """Pause for reflection and enrich memory with insights.

        This method acts as a wrapper for the reflection process, leveraging the
        `discussion_agents.reflecting.generative_agents.reflect` functionality. It aims to
        generate insights on recent observations, add these insights to the agent's memory,
        and return the generated insights.

        Args:
            last_k (int, optional): The number of recent observations to consider. Default is 50.
            now (Optional[datetime], optional): The current date and time for temporal context. Default is None.

        Returns:
            Tuple[List[str], List[str]]: A list of topics and insights generated during the reflection process.

        Example:
            memory = GenerativeAgentMemory(...)
            topics, reflection_insights = memory.pause_to_reflect(last_k=100, now=datetime.now())
        """
        observations = self.memory.load_memories(last_k=last_k)["most_recent_memories"]
        observations = "\n".join([format_memories_detail(o) for o in observations])

        topics, topics_insights = reflect(
            observations=observations, llm=self.llm, retriever=self.retriever, now=now
        )
        insights = list(chain(*topics_insights))

        self.add_memories(memory_contents=insights, now=now)
        return topics, insights