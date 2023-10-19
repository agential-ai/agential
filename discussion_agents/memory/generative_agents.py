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
from typing import Any, Dict, List, Optional, Union

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseMemory, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.utils import mock_now

from discussion_agents.memory.base import BaseMemoryInterface
from discussion_agents.reflecting.generative_agents import (
    get_insights_on_topic,
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
        llm (BaseLanguageModel): The core language model used for text generation.
        memory_retriever (TimeWeightedVectorStoreRetriever):
            The retriever responsible for fetching related memories.
        reflection_threshold (float, optional): When the aggregate importance of recent
            memories exceeds this threshold, the agent triggers a reflection process.
            Defaults to None.
        importance_weight (float): A weight to assign to memory importance when
            calculating aggregate importance. Defaults to 0.15.
        aggregate_importance (float): A running sum of the importance scores of recent
            memories. Triggers reflection when it reaches the reflection threshold.
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

    llm: BaseLanguageModel
    memory_retriever: TimeWeightedVectorStoreRetriever
    reflection_threshold: Optional[float] = None
    # A weight of 0.15 makes this less important than it
    # would be otherwise, relative to salience and time
    importance_weight: float = 0.15
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
        """Exposing get_topics_of_reflection.

        Wrapper for `discussion_agents.reflecting.generative_agents.get_topics_of_reflection`.
        """
        return get_topics_of_reflection(
            llm=self.llm,
            memory_retriever=self.memory_retriever,
            last_k=last_k,
        )

    def get_insights_on_topic(
        self,
        topics: Union[str, List[str]],
        now: Optional[datetime] = None,
    ) -> List[List[str]]:
        """Exposing get_insights_on_topic.

        Wrapper for `discussion_agents.reflecting.generative_agents.get_insights_on_topic`.
        """
        return get_insights_on_topic(
            llm=self.llm,
            memory_retriever=self.memory_retriever,
            topics=topics,
            now=now,
        )

    def pause_to_reflect(
        self, last_k: int = 50, now: Optional[datetime] = None
    ) -> List[str]:
        """Wrapper for Generative Agents reflection.

        Wrapper for `discussion_agents.reflecting.generative_agents.reflect`.
        Adds reflection insights to memory.
        """
        results = reflect(
            llm=self.llm,
            memory_retriever=self.memory_retriever,
            last_k=last_k,
            now=now,
        )
        self.add_memories(results, now=now)
        return results

    def score_memories_importance(
        self,
        memory_contents: Union[str, List[str]],
        relevant_memories: Union[str, List[str]],
    ) -> List[float]:
        """Wrapper for Generative Agents scoring memory importance.

        Wrapper for `discussion_agents.scoring.generative_agents.score_memories_importance`.
        """
        return score_memories_importance(
            memory_contents=memory_contents,
            relevant_memories=relevant_memories,
            llm=self.llm,
            importance_weight=self.importance_weight,
        )

    def add_memories(
        self, memory_contents: Union[str, List[str]], now: Optional[datetime] = None
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
            - If the aggregate importance surpasses a reflection threshold, the agent may
            enter a reflection phase to add synthesized memories.
        """
        if type(memory_contents) is str:
            memory_contents = [memory_contents]

        relevant_memories = fetch_memories(
            memory_retriever=self.memory_retriever,
            observation="\n".join(memory_contents),
        )
        relevant_memories = [mem.page_content for mem in relevant_memories]
        importance_scores = self.score_memories_importance(
            memory_contents, relevant_memories=relevant_memories
        )
        self.aggregate_importance += max(importance_scores)

        documents = []
        for i in range(len(memory_contents)):
            documents.append(
                Document(
                    page_content=memory_contents[i],
                    metadata={"importance": importance_scores[i]},
                )
            )

        result = self.memory_retriever.add_documents(documents, current_time=now)

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result

    def get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Get documents from memory until max_tokens_limit reached.

        Args:
            consumed_tokens (int): The current number of tokens consumed.

        Returns:
            str: A formatted string representing the retrieved documents;
                semi-colon delineated.
        """
        result = []
        for doc in self.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
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
                for mem in fetch_memories(self.memory_retriever, query, now=now)
            ]
            return {
                self.relevant_memories_key: format_memories_detail(
                    relevant_memories, prefix="- "
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
