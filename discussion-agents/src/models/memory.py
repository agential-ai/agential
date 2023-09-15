import os

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseMemory, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.utils import mock_now

logger = logging.getLogger(__name__)


class GenerativeAgentMemory(BaseMemory):
    """Memory for the generative agent."""

    llm: BaseLanguageModel
    """The core language model."""
    memory_retriever: TimeWeightedVectorStoreRetriever
    """The retriever to fetch related memories."""
    verbose: bool = False
    reflection_threshold: Optional[float] = None
    """When aggregate_importance exceeds reflection_threshold, stop to reflect."""
    current_plan: List[str] = []
    """The current plan of the agent."""
    # A weight of 0.15 makes this less important than it
    # would be otherwise, relative to salience and time
    importance_weight: float = 0.15
    """How much weight to assign the memory importance."""
    aggregate_importance: float = 0.0  # : :meta private:
    """Track the sum of the 'importance' of recent memories.

    Triggers reflection when it reaches reflection_threshold."""

    max_tokens_limit: int = 1200  # : :meta private:
    # input keys
    queries_key: str = "queries"
    most_recent_memories_token_key: str = "recent_memories_token"
    add_memory_key: str = "add_memory"
    # output keys
    relevant_memories_key: str = "relevant_memories"
    relevant_memories_simple_key: str = "relevant_memories_simple"
    most_recent_memories_key: str = "most_recent_memories"
    now_key: str = "now"
    reflecting: bool = False

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]  # remove empty lines
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def get_topics_of_reflection(self, last_k: int = 50) -> List[str]:
        """Return the 3 most salient high-level questions about recent observations."""
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
        """Generate 'insights' on a topic of reflection, based on pertinent memories."""
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

        if type(topics) is str: topics = [topics]

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
        """Reflect on recent observations and generate 'insights'."""
        if self.verbose:
            logger.info("Character is reflecting")
        new_insights = []
        topics = self.get_topics_of_reflection()
        for topic in topics:
            insights = self.get_insights_on_topic(topic, now=now)[0]
            for insight in insights:
                self.add_memory(insight, now=now)
            new_insights.extend(insights)
        return new_insights

    def score_memories_importance(self, memory_contents: Union[str, List[str]]) -> List[float]:
        """Score the absolute importance of the given memory."""

        if type(memory_contents) is list: memory_contents = "; ".join(memory_contents)

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

        if self.verbose:
            logger.info(f"Importance scores: {scores}")

        # Split into list of strings and convert to floats
        scores_list = [float(x) / 10 * self.importance_weight for x in scores.split(";")]

        return scores_list

    def add_memories(
        self, memory_contents: Union[str, List[str]], now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observations or memories to the agent's memory."""
        importance_scores = self.score_memories_importance(memory_contents)
        self.aggregate_importance += max(importance_scores)
            
        if type(memory_contents) is str: memory_contents = [memory_contents]

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
        """Fetch related memories."""
        if now is not None:
            with mock_now(now):
                return self.memory_retriever.get_relevant_documents(observation)
        else:
            return self.memory_retriever.get_relevant_documents(observation)

    def format_memories_detail(
        self, relevant_memories: Union[Document, List[Document]], prefix: str = ""
    ) -> str:
        if type(relevant_memories) is Document: relevant_memories = [relevant_memories]

        content = []
        for mem in relevant_memories:
            created_time = mem.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
            content.append(f"{prefix}[{created_time}] {mem.page_content.strip()}")
        return "\n".join([f"{mem}" for mem in content])

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

    def get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
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
        """Return key-value pairs given the text input to the chain."""
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
        """Save the context of this model run to memory."""
        # TODO: fix the save memory key
        mem = outputs.get(self.add_memory_key)
        now = outputs.get(self.now_key)
        if mem:
            self.add_memory(mem, now=now)

    def clear(self) -> None:
        """Clear memory contents."""
        # To clear memory:
        # - set aggregate_importance = 0
        # - current_plan = []
        # - set reflecting = False
        # - set retriever's memory_stream = []
        # - clear retriever's vectorstore

        # As of now, there is no universal method to clear a vector store.
        # To my knowledge, to clear the memory (vectorstore-agnostic), 
        # we need to delete the memory and re-instantiate the instance.
        # I can do this if a create_vectorstore() method was provided,
        # but this is too hacky. It'd be better if LangChain had all 
        # vectorstores implement a clear() method.

        # The best way, currently, is to just re-instantiate a
        # GenerativeAgentMemory.