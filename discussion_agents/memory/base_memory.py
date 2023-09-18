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
import re

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseMemory, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.utils import mock_now


class GenerativeAgentMemory(BaseMemory):
    """Memory for the generative agent.

    This class represents the memory system used by the generative agent. It stores
    and manages memories, interacts with a large language model (LLM), and provides methods
    for reflection, and scores importance.

    Attributes:
        llm (BaseLanguageModel): The core language model used for text generation.
        memory_retriever (TimeWeightedVectorStoreRetriever):
            The retriever responsible for fetching related memories.
        verbose (bool, optional): A flag to enable verbose mode for debugging and
            logging. Defaults to False.
        reflection_threshold (float, optional): When the aggregate importance of recent
            memories exceeds this threshold, the agent triggers a reflection process.
            Defaults to None.
        current_plan (List[str]): The current plan of the agent.
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
    verbose: bool = False
    reflection_threshold: Optional[float] = None
    current_plan: List[str] = []
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

    reflecting: bool = False

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        """Create a Large Language Model (LLM) chain for text generation.

        This method creates a Large Language Model (LLM) chain for text generation
        using the specified prompt template. It allows you to chain together multiple
        prompts and interactions with the LLM for more complex text generation tasks.

        Args:
            prompt (PromptTemplate): The prompt template to be used for text generation.

        Returns:
            LLMChain: An instance of the LangChain LLM chain configured with the provided
                prompt template.

        Example usage:
            memory = GenerativeAgentMemory(...)
            prompt_template = PromptTemplate.from_template("Generate a creative story.")
            llm_chain = memory.chain(prompt_template)
            generated_text = llm_chain.run()
            # 'generated_text' contains the text generated using the specified prompt.
        """
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        r"""Parse a newline-separated string into a list of strings.

        This static method takes a string that contains multiple lines separated by
        newline characters and parses it into a list of strings. It removes any empty
        lines and also removes any leading numbers followed by a period (commonly used
        in numbered lists).

        Args:
            text (str): The input string containing newline-separated lines.

        Returns:
            List[str]: A list of strings parsed from the input text.

        Example usage:
            input_text = "1. Item 1\n2. Item 2\n3. Item 3\n\n4. Item 4"
            parsed_list = GenerativeAgentMemory._parse_list(input_text)
            # 'parsed_list' contains ["Item 1", "Item 2", "Item 3", "Item 4"]

        Note:
            - This method is useful for parsing structured text into a list of items.
            - It removes leading numbers and periods often used in numbered lists.
        """
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]  # remove empty lines
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def get_topics_of_reflection(self, last_k: int = 50) -> List[str]:
        """Return the 3 most salient high-level questions about recent observations.

        This method analyzes recent observations stored in the agent's memory to identify
        the three most salient high-level questions that can be answered based on these
        observations. It follows these steps:
        1. Retrieves the last 'last_k' observations from the agent's memory.
        2. Formats these observations into a single string.
        3. Uses a predefined prompt to request the most salient high-level questions.
        4. Parses and returns the identified questions as a list of strings.

        Args:
            last_k (int, optional): The number of most recent observations to consider.
                Defaults to 50.

        Returns:
            List[str]: A list of the three most salient high-level questions based on
                recent observations.

        Example usage:
            memory = GenerativeAgentMemory(...)
            salient_questions = memory.get_topics_of_reflection()
            # 'salient_questions' contains the three most salient questions for reflection.

        Note:
            - The method uses a predefined prompt to facilitate question identification.
            - It analyzes recent observations in the agent's memory.
            - The number of observations considered can be adjusted using the 'last_k' parameter.
        """
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            "Given only the information above, what are the 3 most salient "
            "high-level questions we can answer about the subjects in the statements?\n"
            "Provide each question on a new line."
        )
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join(
            [self.format_memories_detail(o) for o in observations]
        )
        result = self.chain(prompt).run(observations=observation_str)
        return self._parse_list(result)

    def get_insights_on_topic(
        self, topics: Union[str, List[str]], now: Optional[datetime] = None
    ) -> List[List[str]]:
        """Generate insights on a topic of reflection based on pertinent memories.

        This method generates 'insights' on a given topic of reflection by analyzing
        pertinent memories associated with the topic. It follows these steps:
        1. Retrieves related memories based on the specified topic.
        2. Formats these memories into statements.
        3. Uses a predefined prompt to request high-level novel insights related to the topic.
        4. Parses and returns the generated insights as lists.

        Args:
            topics (Union[str, List[str]]): The topic or topics for which insights are
                to be generated. It can be a single string or a list of strings.
            now (Optional[datetime], optional): Current time context for memory retrieval.
                Defaults to None.

        Returns:
            List[List[str]]: A list of lists, where each inner list contains insights
                generated for a specific topic.

        Example usage:
            memory = GenerativeAgentMemory(...)
            topics_to_generate_insights = ["History of space exploration", "Artificial intelligence"]
            insights = memory.get_insights_on_topic(topics_to_generate_insights)
            # 'insights' contains lists of insights generated for the specified topics.

        Note:
            - Insights are generated based on memories related to the given topic.
            - The method uses a predefined prompt template to facilitate insight generation.
            - It supports both single topic string and multiple topics in a list.
        """
        prompt = PromptTemplate.from_template(
            "Statements relevant to: '{topic}'\n"
            "---\n"
            "{related_statements}\n"
            "---\n"
            "What 5 high-level novel insights can you infer from the above statements "
            "that are relevant for answering the following question?\n"
            "Do not include any insights that are not relevant to the question.\n"
            "Do not repeat any insights that have already been made.\n\n"
            "Question: {topic}\n\n"
            "(example format: insight (because of 1, 5, 3))\n"
        )

        if type(topics) is str:
            topics = [topics]

        results = []
        for topic in topics:
            related_memories = self.fetch_memories(topic, now=now)
            related_statements = "\n".join(
                [
                    self.format_memories_detail(memory, prefix=f"{i+1}. ")
                    for i, memory in enumerate(related_memories)
                ]
            )
            result = self.chain(prompt).run(
                topic=topic, related_statements=related_statements
            )
            results.append(self._parse_list(result))

        # TODO: Parse the connections between memories and insights
        return results

    def pause_to_reflect(self, now: Optional[datetime] = None) -> List[str]:
        """Pause and reflect on recent observations to generate insights.

        This method initiates a pause in the Generative Agent's operation to reflect
        on recent observations and generate 'insights.' It follows these steps:
        1. Retrieves topics for reflection.
        2. For each topic, gathers insights using the `get_insights_on_topic` method.
        3. Adds these insights to the agent's memory for future reference.
        4. Returns a list of the generated insights.

        Args:
            now (Optional[datetime], optional): Current time context for reflection.
                Defaults to None.

        Returns:
            List[str]: A list of generated insights as strings.

        Example usage:
            memory = GenerativeAgentMemory(...)
            new_insights = memory.pause_to_reflect()
            # 'new_insights' contains insights generated during reflection.

        Note:
            - Reflection is a process of generating insights based on recent observations.
            - The method relies on the `get_topics_of_reflection` and `get_insights_on_topic`
            methods for topic retrieval and insight generation.
            - Reflection may be initiated when the agent's aggregate importance surpasses
            a specified threshold.
        """
        new_insights = []
        topics = self.get_topics_of_reflection()
        for topic in topics:
            insights = self.get_insights_on_topic(topic, now=now)[0]
            for insight in insights:
                self.add_memories(insight, now=now)
            new_insights.extend(insights)
        return new_insights

    def score_memories_importance(
        self, memory_contents: Union[str, List[str]]
    ) -> List[float]:
        """Score the absolute importance of the given memory contents.

        This method calculates the absolute importance scores for the provided memory contents.
        It uses a scale from 1 to 10, where 1 represents purely mundane memories, and 10
        represents extremely poignant memories.

        Args:
            memory_contents (Union[str, List[str]]): The memory contents to be scored.
                It can be a single string or a list of strings representing memories.

        Returns:
            List[float]: A list of float values representing the calculated importance scores.

        Example usage:
            memory = GenerativeAgentMemory(...)
            memories_to_score = ["Visited the museum.", "Had a meaningful conversation."]
            importance_scores = memory.score_memories_importance(memories_to_score)
            # 'importance_scores' contains the calculated importance scores for the memories.

        Note:
            - The method converts the ratings to float values and applies an importance weight.
            - The importance weight can be configured to influence the final scores.
        """
        if type(memory_contents) is list:
            memory_contents = "; ".join(memory_contents)

        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Always answer with only a list of numbers."
            + " If just given one memory still respond in a list."
            + " Memories are separated by semi colons (;)"
            + "\Memories: {memory_contents}"
            + "\nRating: "
        )
        scores = self.chain(prompt).run(memory_contents=memory_contents).strip()

        # Split into list of strings and convert to floats
        scores_list = [
            float(x) / 10 * self.importance_weight for x in scores.split(";")
        ]

        return scores_list

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
        importance_scores = self.score_memories_importance(memory_contents)
        self.aggregate_importance += max(importance_scores)

        if type(memory_contents) is str:
            memory_contents = [memory_contents]

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

    def fetch_memories(
        self, observation: str, now: Optional[datetime] = None
    ) -> List[Document]:
        """Fetch related memories based on an observation.

        Args:
            observation (str): The query or observation used for memory retrieval.
            now (Optional[datetime], optional): Current time context for retrieval. Defaults to None.

        Returns:
            List[Document]: A list of related memories.

        Example:
            memory = GenerativeAgentMemory(...)
            observation = "Tell me about the history of space exploration."
            memories = memory.fetch_memories(observation)
            # 'memories' contains related memories based on the observation.

        Note:
            - This method uses the associated Memory Retriever to fetch relevant memories.
            - 'now' allows setting a specific time context for retrieval.
        """
        if now is not None:
            with mock_now(now):
                return self.memory_retriever.get_relevant_documents(observation)
        else:
            return self.memory_retriever.get_relevant_documents(observation)

    def format_memories_detail(
        self, relevant_memories: Union[Document, List[Document]], prefix: str = ""
    ) -> str:
        """Formats memories with created_at time and an optional prefix.

        Args:
            relevant_memories (Union[Document, List[Document]]): The memories to be formatted.
                It can be a single Document or a list of Document objects.
            prefix (str, optional): A prefix to be added before each formatted memory.
                Defaults to an empty string.

        Returns:
            str: A string containing the formatted memories with timestamps and prefix;
                newline-character delineated.
        """
        if isinstance(relevant_memories, Document):
            relevant_memories = [relevant_memories]

        content = []
        for mem in relevant_memories:
            if isinstance(mem, Document):
                created_time = mem.metadata["created_at"].strftime(
                    "%B %d, %Y, %I:%M %p"
                )
                content.append(f"{prefix}[{created_time}] {mem.page_content.strip()}")
        return "\n".join([f"{mem}" for mem in content])

    def format_memories_simple(
        self, relevant_memories: Union[Document, List[Document]]
    ) -> str:
        r"""Formats memories delineated by \';\'.

        Args:
            relevant_memories (Union[Document, List[Document]]): The memories to be formatted.
                It can be a single Document or a list of Documents.

        Returns:
            str: A string containing the formatted memories separated by \';\'.
        """
        if isinstance(relevant_memories, Document):
            relevant_memories = [relevant_memories]
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

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
        return self.format_memories_simple(result)

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
                mem for query in queries for mem in self.fetch_memories(query, now=now)
            ]
            return {
                self.relevant_memories_key: self.format_memories_detail(
                    relevant_memories, prefix="- "
                ),
                self.relevant_memories_simple_key: self.format_memories_simple(
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

        Note:
            - This function can be customized to save specific information from the 'outputs' dictionary
            to memory based on the needs of your application.
            - This function does not have any use for the 'inputs' dictionary yet.
            - The 'TODO' comment suggests that the 'save memory key' needs to be fixed, and you may
            need to replace it with the appropriate key used for saving memories to memory storage.
        """
        # TODO: fix the save memory key
        mem = outputs.get(self.add_memory_key)
        now = outputs.get(self.now_key)
        if mem:
            self.add_memories(mem, now=now)

    def clear(self) -> None:
        """Clear the memory contents of the Generative Agent.

        This method clears the memory contents of the Generative Agent instance.
        It performs the following actions:
        - set aggregate_importance = 0
        - current_plan = []
        - set reflecting = False
        - set retriever's memory_stream = []
        - clear retriever's vectorstore

        As of now, there is no universal method to clear a vector store, so vector store clearing
        may be specific to the implementation used. Ideally, vector store implementations should
        provide a clear() method for consistency.

        Note:
        - Clearing the memory contents is essential for resetting the Generative Agent's state
        and starting with a clean slate.
        - Be aware that clearing the memory will remove all stored information, which cannot
        be undone.
        -  The best way, currently, is to just re-instantiate a GenerativeAgentMemory.

        Example usage:
            memory = GenerativeAgentMemory(...)
            memory.clear()  # Clears the agent's memory contents.
        """
        # TODO
        pass
