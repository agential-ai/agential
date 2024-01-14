"""Generative Agents module implementation adapted from LangChain.

This implementation includes functions for performing the operations
in the Generative Agents paper without the graphic interface.

Original Paper: https://arxiv.org/abs/2304.03442
Paper Repository: https://github.com/joonspk-research/generative_agents
LangChain: https://github.com/langchain-ai/langchain
LangChain Generative Agents:
https://github.com/langchain-ai/langchain/tree/master/libs/experimental/langchain_experimental/generative_agents
LangChain Generative Agents Doc Page:
https://python.langchain.com/docs/use_cases/more/agents/agent_simulations/characters
"""
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.functional.generative_agents import (
    get_insights_on_topics,
    get_topics_of_reflection,
    _create_default_time_weighted_retriever
)
from discussion_agents.cog.modules.memory.generative_agents import GenerativeAgentMemory
from discussion_agents.cog.modules.reflect.generative_agents import (
    GenerativeAgentReflector,
)
from discussion_agents.cog.modules.score.generative_agents import GenerativeAgentScorer
from discussion_agents.cog.persona.base import BasePersona
from discussion_agents.cog.persona.generative_agents import GenerativeAgentPersona
from discussion_agents.utils.format import (
    format_memories_detail,
)
from discussion_agents.utils.parse import remove_name


class GenerativeAgent(BaseAgent):
    """GenerateAgent class complete with memory, reflecting, and scoring capabilities.

    This class extends BaseAgent and implements `reflect`, `score`, `retrieve`, and `generate`.

    Attributes:
        llm (LLM): An instance of a language model used for processing and generating content.
        memory (Optional[GenerativeAgentMemory]): A memory management component responsible for handling
            storage and retrieval of memories. Automatically set if not provided.
        reflector (Optional[GenerativeAgentReflector]): A component for reflecting on observations and memories.
            Automatically set based on the LLM and memory if not provided.
        scorer (Optional[GenerativeAgentScorer]): A component for scoring and evaluating memories or observations.
            Automatically set based on the LLM if not provided.
        persona (Optional[BasePersona]): A persona component representing agent's characteristics.
            Automatically set based on provided personal attributes below if not provided.
        importance_weight (float): A weight factor used in scoring calculations.
        reflection_threshold (Optional[int]): A threshold value used in reflection decisions.

        name (str): The name of the agent's persona. Defaults to "Klaus Mueller".
        age (int): The age of the agent's persona. Defaults to 20.
        traits (str): The traits of the agent's persona, described as a string. Defaults to "kind, inquisitive, passionate".
        status (str): The current status of the agent's persona. Defaults to a predefined status about writing a research paper.
        lifestyle (str): The lifestyle description of the agent's persona, including typical daily routines.

    The `set_args` root validator is used to automatically set the `reflector`, `scorer`, and `persona` components
    if they are not explicitly provided. It ensures that these components are appropriately initialized based on the LLM,
    memory, and persona attributes.
    """

    def __init__(
        self,
        llm: Any,
        memory: Optional[GenerativeAgentMemory] = None,
        reflector: Optional[GenerativeAgentReflector] = None,
        scorer: Optional[GenerativeAgentScorer] = None,
        persona: Optional[BasePersona] = None,
        importance_weight: float = 0.15,
        reflection_threshold: Optional[int] = 8,
        name: str = "Klaus Mueller",
        age: int = 20,
        traits: str = "kind, inquisitive, passionate",
        status: str = "Klaus Mueller is writing a research paper on the effects of gentrification in low-income communities.",
        lifestyle: str = "Klaus Mueller goes to bed around 11pm, awakes up around 7am, eats dinner around 5pm."
    ) -> None:
        """Initialize agent."""
        self.llm = llm

        self.memory = memory
        if not self.memory:
            self.memory = GenerativeAgentMemory(
                retriever=_create_default_time_weighted_retriever()
            )

        self.reflector = reflector
        if self.llm and self.memory and not self.reflector:
            self.reflector = GenerativeAgentReflector(
                llm=self.llm, retriever=self.memory.retriever
            )

        self.scorer = scorer
        if self.llm and not self.scorer:
            self.scorer = GenerativeAgentScorer(llm=self.llm)

        self.persona = persona
        self.name = name
        self.age = age
        self.traits = traits
        self.status = status
        self.lifestyle = lifestyle
        if not self.persona:
            self.persona = GenerativeAgentPersona(
                name=self.name,
                age=self.age,
                traits=self.traits,
                status=self.status,
                lifestyle=self.lifestyle,
            )

        self.importance_weight = importance_weight
        self.reflection_threshold = reflection_threshold

        # Internal variables.
        self.__summary: str = ""  #: :meta private:
        self.__last_refreshed: datetime = datetime.now()  # : :meta private:
        self.__summary_refresh_seconds: int = 3600  #: :meta private:
        self.__is_reflecting: bool = False  #: :meta private:
        self.__aggregate_importance: float = 0.0  #: :meta private:

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

        if self.reflector:
            topics_insights = self.reflector.reflect(observations=observations, now=now)
        else:
            raise ValueError("`reflector` was incorrectly defined.")
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
            if self.scorer:
                importance_score = self.scorer.score(
                    memory_contents=memory_content,
                    relevant_memories=relevant_memories,
                    importance_weight=importance_weight,
                )
            else:
                raise ValueError("`scorer` was incorrectly defined.")
            importance_scores.append(importance_score[0])
        self.__aggregate_importance += max(importance_scores)

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
            and self.__aggregate_importance > self.reflection_threshold
            and not self.__is_reflecting
        ):
            self.__is_reflecting = True
            _ = self.reflect(last_k=last_k, now=now)
            self.__aggregate_importance = 0.0
            self.__is_reflecting = False

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
        if self.scorer:
            return self.scorer.score(
                memory_contents=memory_contents,
                relevant_memories=relevant_memories,
                importance_weight=importance_weight,
            )
        else:
            raise ValueError("`scorer` was incorrectly defined.")

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
        relevant_memories = self.memory.load_memories(queries=[q1, q2])[
            "relevant_memories"
        ]
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
        since_refresh = (current_time - self.__last_refreshed).seconds
        if (
            not self.__summary
            or since_refresh >= self.__summary_refresh_seconds
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
            relevant_memories = self.memory.load_memories(
                queries=[f"{self.name}'s core characteristics"]
            )["relevant_memories"]
            relevant_memories = "\n".join(
                [mem.page_content for mem in relevant_memories]
            )
            self.__summary = chain.run(
                name=self.name, relevant_memories=relevant_memories
            ).strip()

            self.__last_refreshed = current_time

        summary = (
            f"Name: {self.name}\n"
            + f"Age: {self.age}\n"
            + f"Innate traits: {self.traits}\n"
            + f"Status: {self.status}\n"
            + f"Lifestyle: {self.lifestyle}\n"
            + f"{self.__summary}\n"
        )

        return summary

    def _generate_reaction(
        self,
        observation: str,
        suffix: str,
        now: Optional[datetime] = None,
        max_tokens_limit: Optional[int] = 1200,
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
            prompt.format(most_recent_memories_limit="", **kwargs)
        )
        most_recent_memories_limit = self.memory.load_memories(
            consumed_tokens=consumed_tokens,
            max_tokens_limit=max_tokens_limit,
            llm=self.llm,
        )["most_recent_memories_limit"]
        most_recent_memories_limit = "\n".join(
            [mem.page_content for mem in most_recent_memories_limit]
        )
        kwargs["most_recent_memories_limit"] = most_recent_memories_limit
        result = chain.run(**kwargs).strip()

        return result

    def generate_reaction(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """Generate a reaction to a given observation.

        Args:
            observation (str): The observation or input to react to.
            now (Optional[datetime], optional): The current timestamp for the reaction.
                Defaults to None.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean value indicating whether the
            agent should respond (True) or not (False), and the generated reaction or
            response text.

        This method generates a reaction or response to a given observation. It uses a
        template that instructs the agent on how to react to the observation. The
        agent can choose to react, say something, or do nothing based on the
        instructions in the template.
        """
        call_to_action_template = (
            "Should {agent_name} react to the observation, and if so,"
            + " what would be an appropriate reaction? Respond in one line."
            + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
            + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
            + "\nEither do nothing, react, or say something but not both.\n\n"
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        result = f"{self.name} observed {observation} and reacted by {result}"
        self.add_memories(memory_contents=result, now=now)

        if "REACT:" in result:
            reaction = remove_name(text=result.split("REACT:")[-1], name=self.name)
            return False, f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = remove_name(text=result.split("SAY:")[-1], name=self.name)
            return True, f"{self.name} said {said_value}"
        else:
            return False, result

    def generate_dialogue_response(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """Generate a dialogue response to a given observation.

        Args:
            observation (str): The observation or input to respond to.
            now (Optional[datetime], optional): The current timestamp for the response.
                Defaults to None.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean value indicating whether the
            agent should continue the dialogue (True) or end it (False), and the
            generated dialogue response text.

        This method generates a dialogue response to a given observation. It uses a
        template that instructs the agent on how to respond to the observation, either
        by ending the conversation or continuing it. The agent can choose to say
        something in response or end the conversation based on the instructions in the
        template.
        """
        call_to_action_template = (
            "What would {agent_name} say? To end the conversation, write:"
            ' GOODBYE: "what to say". Otherwise to continue the conversation,'
            ' write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        if "GOODBYE:" in result:
            farewell = remove_name(text=result.split("GOODBYE:")[-1], name=self.name)
            farewell = f"{self.name} observed {observation} and said {farewell}"
            self.add_memories(memory_contents=farewell, now=now)
            return False, farewell
        if "SAY:" in result:
            response_text = remove_name(text=result.split("SAY:")[-1], name=self.name)
            response_text = (
                f"{self.name} observed {observation} and said {response_text}"
            )
            self.add_memories(memory_contents=response_text, now=now)
            return True, response_text
        else:
            return False, result

    def show_memories(self, memories_key: str = "memory_stream") -> Dict[str, Any]:
        """Retrieves all stored memories and returns them in a dictionary.

        This method wraps around and exposes the memory module's `show_memories` method.

        Args:
            memories_key (str, optional): The key under which the memories are stored. Defaults to "memory_stream".

        Returns:
            Dict[str, Any]: A dictionary containing the stored memories, keyed by `memories_key`.
        """
        return self.memory.show_memories(memories_key=memories_key)

    def retrieve(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        last_k: Optional[int] = None,
        consumed_tokens: Optional[int] = None,
        max_tokens_limit: Optional[int] = None,
        llm: Optional[Any] = None,
        now: Optional[datetime] = None,
        queries_key: str = "relevant_memories",
        most_recent_key: str = "most_recent_memories",
        consumed_tokens_key: str = "most_recent_memories_limit",
    ) -> Dict[str, Any]:
        """Wraps around the memory's `load_memories` method.

        If `load_memories` uses `consumed_tokens` and `max_tokens_limit`,
        llm will default to `GenerativeAgent` llm if not specified.
        """
        return self.memory.load_memories(
            queries=queries,
            last_k=last_k,
            consumed_tokens=consumed_tokens,
            max_tokens_limit=max_tokens_limit,
            llm=llm if llm else self.llm,
            now=now,
            queries_key=queries_key,
            most_recent_key=most_recent_key,
            consumed_tokens_key=consumed_tokens_key,
        )

    def generate(
        self, observation: str, is_react: bool, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """Wrapper around `generate_reaction` and `generate_dialogue_response`."""
        if is_react:
            return self.generate_reaction(observation=observation, now=now)
        return self.generate_dialogue_response(observation=observation, now=now)
