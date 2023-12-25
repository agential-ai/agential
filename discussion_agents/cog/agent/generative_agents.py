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
from datetime import datetime
from itertools import chain
from typing import List, Optional, Union

from langchain_core.language_models import LLM
from pydantic.v1 import root_validator

from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.functional.generative_agents import (
    get_insights_on_topics,
    get_topics_of_reflection,
)
from discussion_agents.cog.modules.memory.generative_agents import GenerativeAgentMemory
from discussion_agents.cog.modules.reflect.base import BaseReflector
from discussion_agents.cog.modules.reflect.generative_agents import (
    GenerativeAgentReflector,
)
from discussion_agents.cog.modules.score.base import BaseScorer
from discussion_agents.cog.modules.score.generative_agents import GenerativeAgentScorer
from discussion_agents.utils.format import (
    format_memories_detail,
)


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
        if "reflector" not in values or values["reflector"] is None:
            values["reflector"] = GenerativeAgentReflector(
                llm=values["llm"], retriever=values["memory"].retriever
            )
        if "scorer" not in values or values["scorer"] is None:
            values["scorer"] = GenerativeAgentScorer(llm=values["llm"])
        return values

    # Internal variables.
    is_reflecting: bool = False  #: :meta private:
    aggregate_importance: float = 0.0  #: :meta private:

    def get_topics_of_reflection(self, last_k: int = 50) -> List[str]:
        """Generate high-level reflection topics based on recent observations.

        This method leverages the `get_topics_of_reflection` function to generate a list of
        high-level reflection topics based on recent observations from the agent's memory.
        The method considers the last 'last_k' observations and formats them for
        reflection. This method exposes the `get_topics_of_reflection` method from
        the functional counterpart of Generative Agents.

        Args:
            last_k (int, optional): The number of recent observations to consider. Default is 50.

        Returns:
            List[str]: A list of high-level reflection topics.
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

        This method exposes the `get_insights_on_topic` method from
        the functional counterpart of Generative Agents.

        Args:
            topics (List[str]): A list of topics for which insights are to be generated.
            now (Optional[datetime], optional): The current date and time for temporal context. Default is None.

        Returns:
            List[List[str]]: Lists of high-level insights corresponding to each specified topic.
        """
        related_memories = []
        for topic in topics:
            fetched_memories = self.memory.load_memories(queries=topic, now=now)[
                "relevant_memories"
            ]
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
    ) -> List[List[str]]:
        """Pause for reflection and enrich memory with insights.

        This method acts as a wrapper for the reflector's reflect method. It aims to
        find topics to reflect on, generate insights on recent observations,
        add these insights to the agent's memory, and return the generated insights.

        Args:
            last_k (int, optional): The number of recent observations to consider. Default is 50.
            now (Optional[datetime], optional): The current date and time for temporal context. Default is None.

        Returns:
            List[List[str]]: A list of insights generated during the reflection process,
                one list of insights for every topic.
        """
        observations = self.memory.load_memories(last_k=last_k)["most_recent_memories"]
        observations = "\n".join([format_memories_detail(o) for o in observations])

        topics_insights = self.reflector.reflect(observations=observations, now=now)
        insights = list(chain(*topics_insights))

        self.add_memories(memory_contents=insights, now=now)
        return insights

    def add_memories(
        self,
        memory_contents: Union[str, List[str]],
        now: Optional[datetime] = None,
        importance_weight: float = 0.15,
        last_k: int = 50,
    ) -> None:
        """Add observations/memories to the agent's memory.

        This method allows adding new observations or memories to the agent's memory store.
        It calculates the importance scores for the added memories, updates the aggregate
        importance, and adds them to the memory store using the memory retriever. This
        method wraps around the memory attribute's load_memory method.

        Args:
            memory_contents (Union[str, List[str]]): The memory contents to be added.
                It can be a single string or a list of strings representing observations
                or memories.
            now (Optional[datetime], optional): Current time context for memory addition.
                Defaults to None.
            importance_weight (float, optional): Weight for importance scores. Default is 0.15.
            last_k (int, optional): The number of recent observations to consider. Default is 50.

        Note:
            - Importance scores are calculated for the added memories and contribute to
            the agent's aggregate importance.
            - If the aggregate importance surpasses a reflection threshold, the agent
            enters a reflection phase to add synthesized memories.
        """
        if isinstance(memory_contents, str):
            memory_contents = [memory_contents]

        importance_scores = []
        for memory_content in memory_contents:
            fetched_memories = self.memory.load_memories(queries=memory_content)[
                "relevant_memories"
            ]
            relevant_memories: str = "\n".join(
                [mem.page_content for mem in fetched_memories]
            )
            relevant_memories = "N/A" if not relevant_memories else relevant_memories
            importance_score = self.scorer.score(
                memory_contents=memory_content,
                relevant_memories=relevant_memories,
                importance_weight=importance_weight,
            )
            importance_scores.append(importance_score[0])
        self.aggregate_importance += max(importance_scores)

        assert len(importance_scores) == len(
            memory_contents
        ), "The length of the generated list of importance_scores does not match the length of memory_contents."

        self.memory.add_memories(
            memory_contents=memory_contents,
            importance_scores=importance_scores,
            now=now,
        )

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            _ = self.reflect(last_k=last_k, now=now)
            self.aggregate_importance = 0.0
            self.reflecting = False

    def score(
        self,
        memory_contents: Union[str, List[str]],
        relevant_memories: Union[str, List[str]],
        importance_weight: float = 0.15,  # Less important than relevance and recency.
    ) -> List[float]:
        """Calculate importance scores for agent memories w.r.t. relevant_memories.

        This method serves as a wrapper around the scorer's score method for calculating importance scores
        for agent memories. It scores methods based on their importance to the relevant_memories.

        Args:
            memory_contents (Union[str, List[str]): The memory contents to be scored.
            relevant_memories (Union[str, List[str]): Relevant memories for context.
            importance_weight (float, optional): Weight for importance scores. Default is 0.15.

        Returns:
            List[float]: A list of importance scores for each memory content.
        """
        return self.scorer.score(
            memory_contents=memory_contents,
            relevant_memories=relevant_memories,
            importance_weight=importance_weight,
        )
