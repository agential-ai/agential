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
from typing import List, Optional, Union, Dict, Any

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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

    @root_validator(pre=False)
    def set_reflector_and_scorer(cls, values):
        llm = values.get('llm')
        memory = values.get('memory')
        if llm is not None:
            values['reflector'] = GenerativeAgentReflector(llm=llm, retriever=memory.retriever)
            values['scorer'] = GenerativeAgentScorer(llm=llm)
        return values

    # Personal state.
    name: str = "Vincent"
    age: int = 20
    traits: str = "Enjoys working on this library"
    status: str = ""
    lifestyle: str = ""

    # Internal variables.
    summary: str = ""  #: :meta private:
    last_refreshed: datetime = datetime.now()  # : :meta private:
    summary_refresh_seconds: int = 3600  #: :meta private:
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

        self.add_memories(memory_contents=list(chain(*topics_insights)), now=now)
        return topics_insights

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

    def get_entity_from_observation(self, observation: str) -> str:
        """Extract the observed entity from a given observation text.

        Args:
            observation (str): The observation text from which to extract the entity.

        Returns:
            str: The extracted entity name.

        This method uses a prompt to extract and identify the entity mentioned in the
        observation text.
        """
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}\n"
            + "Entity="
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(observation=observation).strip()

        return result

    def get_entity_action(self, observation: str, entity_name: str) -> str:
        """Determine the action performed by the specified entity in an observation.

        Args:
            observation (str): The observation text containing the entity's action.
            entity_name (str): The name of the entity whose action to determine.

        Returns:
            str: The action performed by the specified entity.

        This method uses a prompt to identify and describe the action performed by the
        specified entity in the given observation.
        """
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}\n"
            + "The {entity} is"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(entity=entity_name, observation=observation).strip()

        return result

    def summarize_related_memories(self, observation: str) -> str:
        """Generate a summary of memories most relevant to an observation.

        Args:
            observation (str): The observation for which to summarize related memories.

        Returns:
            str: A summary of the most relevant memories in the context of the observation.

        This method generates a summary by posing questions about the relationship between
        the character represented by this `GenerativeAgent` and the entity mentioned in
        the observation. It then incorporates relevant context from memory to provide a
        coherent summary.
        """
        prompt = PromptTemplate.from_template(
            "{q1}?\n"
            + "Context from memory:\n"
            + "{relevant_memories}\n"
            + "Relevant context:\n"
        )
        entity_name = self.get_entity_from_observation(observation)
        entity_action = self.get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {self.name} and {entity_name}"
        q2 = f"{entity_name} is {entity_action}"

        chain = LLMChain(llm=self.llm, prompt=prompt)
        relevant_memories = self.memory.load_memories(queries=[q1, q2])["relevant_memories"]
        relevant_memories = "\n".join([mem.page_content for mem in relevant_memories])
        result = chain.run(q1=q1, relevant_memories=relevant_memories).strip()

        return result

    def get_summary(
        self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        """Return a descriptive summary of the agent.

        Args:
            force_refresh (bool, optional): If True, force a refresh of the summary
                even if it was recently computed. Defaults to False.
            now (datetime, optional): The current datetime to use for refreshing
                the summary. Defaults to None, which uses the current system time.

        Returns:
            str: A descriptive summary of the agent.

        This method returns a descriptive summary of the agent, including the agent's
        name, age (if available), innate traits, and a summary of the agent's core
        characteristics based on relevant memories. The summary is refreshed
        periodically, but it can be forced to refresh by setting `force_refresh` to True.
        """
        current_time = datetime.now() if now is None else now
        since_refresh = (current_time - self.last_refreshed).seconds
        if (
            not self.summary
            or since_refresh >= self.summary_refresh_seconds
            or force_refresh
        ):
            prompt = PromptTemplate.from_template(
                "How would you summarize {name}'s core characteristics given the"
                + " following statements:\n"
                + "{relevant_memories}"
                + "Do not embellish."
                + "\n\nSummary: "
            )
            chain = LLMChain(llm=self.llm, prompt=prompt)

            # The agent seeks to think about their core characteristics.
            relevant_memories = self.memory.load_memories(queries=[f"{self.name}'s core characteristics"])["relevant_memories"]
            relevant_memories = "\n".join([mem.page_content for mem in relevant_memories])
            self.summary = chain.run(
                name=self.name, relevant_memories=relevant_memories
            ).strip()

            self.last_refreshed = current_time

        summary = (
            f"Name: {self.name}\n"
            + f"Age: {self.age}\n"
            + f"Innate traits: {self.traits}\n"
            + f"Status: {self.status}\n"
            + f"Lifestyle: {self.lifestyle}\n"
            + f"{self.summary}\n"
        )

        return summary

    def _generate_reaction(
        self, observation: str, suffix: str, now: Optional[datetime] = None, max_tokens_limit: Optional[int] = 1200
    ) -> str:
        """A helper method that generates a reaction or response to a given observation or dialogue act.

        Args:
            observation (str): The observation or dialogue act to react to.
            suffix (str): The suffix to append to the generated reaction; a call-to-action.
            now (Optional[datetime], optional): The timestamp for the current time.
                Defaults to None.
            max_tokens_limit (Optional[int], optional): Max number of tokens to be fed into
                the llm for it to generate a reaction. Defaults to 1200.

        Returns:
            str: The generated reaction or response.

        This helper method generates a reaction or response by providing contextual information
        about the agent, including a summary, current time, relevant memories, and
        recent observations. It then adds the provided `suffix` to create a complete
        reaction or response.
        """
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}\n"
            + "It is {current_time}.\n"
            + "{agent_name}'s lifestyle: {lifestyle}\n"
            + "Summary of relevant context from {agent_name}'s memory:\n"
            + "{relevant_memories}\n"
            + "Most recent observations: {most_recent_memories_limit}\n"
            + "Observation: {observation}\n\n"
            + "{suffix}"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)

        agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            agent_name=self.name,
            lifestyle=self.lifestyle,
            relevant_memories=relevant_memories_str,
            observation=observation,
            suffix=suffix,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        most_recent_memories_limit = self.memory.load_memories(consumed_tokens=consumed_tokens, max_tokens_limit=max_tokens_limit, llm=self.llm)["most_recent_memories_limit"]
        most_recent_memories_limit = "\n".join([mem.page_content for mem in most_recent_memories_limit])
        kwargs["most_recent_memories_limit"] = most_recent_memories_limit
        result = chain.run(**kwargs).strip()

        return result