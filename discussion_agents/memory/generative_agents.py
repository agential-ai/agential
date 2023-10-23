"""Generative Agent Memory implementation from LangChain.

Note: The following classes are versions of LangChain's Generative Agent
implementations with my improvements.

Original Paper: https://arxiv.org/abs/2304.03442
LangChain: https://github.com/langchain-ai/langchain
LangChain Generative Agents:
https://github.com/langchain-ai/langchain/tree/master/libs/experimental/langchain_experimental/generative_agents
LangChain Generative Agents Doc Page:
https://python.langchain.com/docs/use_cases/more/agents/agent_simulations/characters
"""
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseMemory, Document

from discussion_agents.core.base import BaseCore
from discussion_agents.memory.base import BaseMemoryInterface
from discussion_agents.reflecting.generative_agents import (
    get_insights_on_topics,
    get_topics_of_reflection,
    reflect,
)
from discussion_agents.scoring.generative_agents import score_memories_importance
from discussion_agents.utils.fetch import fetch_memories
from discussion_agents.utils.format import (
    format_memories_detail,
    format_memories_simple,
)


class GenerativeAgentMemory(BaseMemory, BaseMemoryInterface):
    """Memory for the generative agent.

    This class represents the memory system used by the generative agent. It stores
    and manages memories, interacts with a large language model (LLM), and provides methods
    for reflection, and scores importance.

    Attributes:
        core (BaseCore): The agent core for this class. Note, GenerativeAgentMemory
            requires a TimeWeightedVectorStoreRetriever instance to be specified
            as the core's retriever.
        reflection_threshold (float, optional): When the aggregate importance of recent
            memories exceeds this threshold, the agent triggers a reflection process.
            Defaults to None.
        max_tokens_limit (int): The maximum token limit for text generation. Defaults
            to 1200 tokens.

    Keys for Loading Memory Variables:
        These attribute names are used as keys for loading memory variables.

        - queries_key (str): The key for loading queries from inputs.
        - most_recent_memories_token_key (str): The key for loading the token limit for
          most recent memories.
        - add_memory_key (str): The key for loading the memory to be added.
        - relevant_memories_key (str): The key for loading relevant memories.
        - relevant_memories_simple_key (str): The key for loading simplified relevant
          memories.
        - most_recent_memories_key (str): The key for loading most recent memories.
        - now_key (str): The key for loading the current timestamp.
    """

    core: BaseCore  # Must Use retriever=TimeWeightedVectorStoreRetriever!
    reflection_threshold: Optional[float] = 8
    aggregate_importance: float = 0.0  # : :meta private:
    max_tokens_limit: int = 1200  # : :meta private:

    # Keys for loading memory variables.

    # Input keys.
    queries_key: str = "queries"
    most_recent_memories_token_key: str = "recent_memories_token"
    add_memory_key: str = "add_memory"

    # Output keys.
    relevant_memories_key: str = "relevant_memories"
    relevant_memories_simple_key: str = "relevant_memories_simple"
    most_recent_memories_key: str = "most_recent_memories"
    now_key: str = "now"

    # Internal reflecting flag.
    reflecting: bool = False

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
        if not isinstance(self.core.get_retriever(), TimeWeightedVectorStoreRetriever):
            raise TypeError(
                "The core's 'retriever' attribute must be an instance of TimeWeightedVectorStoreRetriever."
            )
        observations = self.core.get_retriever().memory_stream[-last_k:]
        observations = "\n".join([format_memories_detail(o) for o in observations])
        return get_topics_of_reflection(observations=observations, core=self.core)

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
            topic_related_memories = fetch_memories(
                observation=topic, memory_retriever=self.core.get_retriever(), now=now
            )
            topic_related_memories = "\n".join(
                [
                    format_memories_detail(memories=memory, prefix=f"{i+1}. ")
                    for i, memory in enumerate(topic_related_memories)
                ]
            )
            related_memories.append(topic_related_memories)

        new_insights = get_insights_on_topics(
            topics=topics, related_memories=related_memories, core=self.core
        )

        return new_insights

    def pause_to_reflect(
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
        if not isinstance(self.core.get_retriever(), TimeWeightedVectorStoreRetriever):
            raise TypeError(
                "The core's 'retriever' attribute must be an instance of TimeWeightedVectorStoreRetriever."
            )
        observations = self.core.get_retriever().memory_stream[-last_k:]
        observations = "\n".join([format_memories_detail(o) for o in observations])

        topics, insights = reflect(observations=observations, core=self.core, now=now)
        insights = list(chain(*insights))

        self.add_memories(memory_contents=insights, now=now)
        return topics, insights

    def score_memories_importance(
        self,
        memory_contents: Union[str, List[str]],
        relevant_memories: Union[str, List[str]],
        importance_weight: float = 0.15,  # Less important than relevance and recency.
    ) -> List[float]:
        """Calculate importance scores for agent memories w.r.t. relevant_memories.

        This method serves as a wrapper for calculating importance scores for agent memories. It
        leverages the `discussion_agents.scoring.generative_agents.score_memories_importance` function.
        Refer to the aforementioned method for more information.

        Args:
            memory_contents (Union[str, List[str]): The memory contents to be scored.
            relevant_memories (Union[str, List[str]): Relevant memories for context.
            importance_weight (float, optional): Weight for importance scores. Default is 0.15.

        Returns:
            List[float]: A list of importance scores for each memory content.

        Example:
            memories = ["Visited the museum.", "Had a meaningful conversation."]
            relevance_context = ["History buffs meeting.", "Art gallery visit."]
            memory = GenerativeAgentMemory(...)
            importance_scores = memory.score_memories_importance(memories, relevance_context, importance_weight=0.2)
        """
        return score_memories_importance(
            memory_contents=memory_contents,
            relevant_memories=relevant_memories,
            core=self.core,
            importance_weight=importance_weight,
        )

    def add_memories(
        self,
        memory_contents: Union[str, List[str]],
        now: Optional[datetime] = None,
        importance_weight: float = 0.15,
        last_k: int = 50,
    ) -> List[str]:
        """Add observations/memories to the agent's memory.

        This method allows adding new observations or memories to the agent's memory store.
        It calculates the importance scores for the added memories, updates the aggregate
        importance, and adds them to the memory store using the memory retriever.

        Args:
            memory_contents (Union[str, List[str]]): The memory contents to be added.
                It can be a single string or a list of strings representing observations
                or memories.
            now (Optional[datetime], optional): Current time context for memory addition.
                Defaults to None.
            importance_weight (float, optional): Weight for importance scores. Default is 0.15.
            last_k (int, optional): The number of recent observations to consider. Default is 50.

        Returns:
            List[str]: A list of string IDs indicating the results of the memory addition.

        Example usage:
            memory = GenerativeAgentMemory(...)
            memories_to_add = ["Visited the museum.", "Learned about space exploration."]
            ids = memory.add_memories(memories_to_add)
            # 'ids' contains information about the result of the memory addition.

        Note:
            - Importance scores are calculated for the added memories and contribute to
            the agent's aggregate importance.
            - The method handles both single string and list input for memory addition.
            - If the aggregate importance surpasses a reflection threshold, the agent
            enters a reflection phase to add synthesized memories.
        """
        if type(memory_contents) is str:
            memory_contents = [memory_contents]

        importance_scores = []
        for memory_content in memory_contents:
            relevant_memories = fetch_memories(
                observation=memory_content,
                memory_retriever=self.core.get_retriever(),
            )
            relevant_memories = "\n".join(
                [mem.page_content for mem in relevant_memories]
            )
            relevant_memories = "N/A" if not relevant_memories else relevant_memories
            importance_score = self.score_memories_importance(
                memory_contents=memory_content,
                relevant_memories=relevant_memories,
                importance_weight=importance_weight,
            )
            importance_scores.append(importance_score[0])
        self.aggregate_importance += max(importance_scores)

        assert len(importance_scores) == len(memory_contents)

        documents = []
        for i in range(len(memory_contents)):
            documents.append(
                Document(
                    page_content=memory_contents[i],
                    metadata={"importance": importance_scores[i]},
                )
            )

        result = self.core.get_retriever().add_documents(documents, current_time=now)

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            _, _ = self.pause_to_reflect(last_k=last_k, now=now)
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result

    def get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Retrieve memories from the agent's memory until a token consumption limit is reached.

        This method retrieves documents from the agent's memory until the total number of tokens
        consumed by the retrieved documents reaches or exceeds the specified token limit.

        Args:
            consumed_tokens (int): The current number of tokens consumed.

        Returns:
            str: A formatted string containing the retrieved documents, separated by semicolons.

        Example:
            consumed_tokens = 1000
            memory = GenerativeAgentMemory(...)
            retrieved_docs = memory.get_memories_until_limit(consumed_tokens)
        """
        if not isinstance(self.core.get_retriever(), TimeWeightedVectorStoreRetriever):
            raise TypeError(
                "The core's 'retriever' attribute must be an instance of TimeWeightedVectorStoreRetriever."
            )
        result = []
        for doc in self.core.get_retriever().memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.core.get_llm().get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc)
        return format_memories_simple(result)

    @property
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""
        return []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        r"""Load and return key-value pairs based on the provided input.

        This function takes a dictionary of inputs and retrieves relevant memories
        or memory-related information based on the input keys. It returns a dictionary
        of key-value pairs containing formatted memory data.

        Args:
            inputs (Dict[str, Any]): A dictionary of input data with specific keys.

        Returns:
            Dict[str, str]: A dictionary of key-value pairs, where keys represent memory-related
            information and values contain formatted memory data.

        Example usage:
            inputs = {
                'queries': ['query1', 'query2'],
                'now': '2023-09-16T12:00:00',
            }
            memory = GenerativeAgentMemory(...)
            memory_variables = memory.load_memory_variables(inputs)
            # Resulting memory_variables may contain:
            # {
            #     'relevant_memories': 'Formatted memories with timestamps and prefix',
            #     'relevant_memories_simple': 'Formatted memories separated by \';\''
            # }

        Note:
            - This function supports multiple input scenarios based on the presence of specific keys
            in the 'inputs' dictionary:
                - If 'queries' key is present, relevant memories are fetched and formatted.
                - If 'most_recent_memories_token' key is present, recent memories are retrieved.
                - If none of the supported keys are present, an empty dictionary is returned.
        """
        queries = inputs.get(self.queries_key)
        now = inputs.get(self.now_key)
        if queries is not None:
            relevant_memories = [
                mem
                for query in queries
                for mem in fetch_memories(
                    observation=query,
                    memory_retriever=self.core.get_retriever(),
                    now=now,
                )
            ]
            return {
                self.relevant_memories_key: format_memories_detail(
                    memories=relevant_memories, prefix="- "
                ),
                self.relevant_memories_simple_key: format_memories_simple(
                    relevant_memories
                ),
            }

        most_recent_memories_token = inputs.get(self.most_recent_memories_token_key)
        if most_recent_memories_token is not None:
            return {
                self.most_recent_memories_key: self.get_memories_until_limit(
                    most_recent_memories_token
                )
            }
        return {}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context of this model run to memory.

        This function is responsible for saving the context of the current model run to memory.
        It takes a dictionary of inputs and a dictionary of outputs, and it can save relevant
        information from the outputs to memory for future reference.

        Args:
            inputs (Dict[str, Any]): A dictionary of input data used for the current model run.
            outputs (Dict[str, Any]): A dictionary of output data generated by the current model run.

        Returns:
            None: This function does not return a value.

        Example usage:
            inputs = {
                'user_query': 'Tell me about topic X.',
            }
            outputs = {
                'response': 'Here is information about topic X.',
                'additional_memory': 'New memory to be saved.',
            }
            memory = GenerativeAgentMemory(...)
            memory.save_context(inputs, outputs)
            # The 'additional_memory' from 'outputs' will be saved to memory for future use.
        """
        mem = outputs.get(self.add_memory_key)
        now = outputs.get(self.now_key)
        if mem:
            self.add_memories(mem, now=now)

    def clear(self) -> None:
        """Clear method."""
        pass
