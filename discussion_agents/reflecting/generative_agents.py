"""Generative Agents reflection."""

from typing import List, Union, Optional

from datetime import datetime

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.utils import mock_now

from discussion_agents.utils.format import format_memories_detail
from discussion_agents.utils.parse import parse_list

def get_topics_of_reflection(
    llm: BaseLanguageModel,
    memory_retriever: TimeWeightedVectorStoreRetriever, 
    verbose: bool = False,
    last_k: int = 50
) -> List[str]:
    """Return the 3 most salient high-level questions about recent observations.

    This method analyzes recent observations stored in the agent's memory to identify
    the three most salient high-level questions that can be answered based on these
    observations. It follows these steps:
    1. Retrieves the last 'last_k' observations from the agent's memory.
    2. Formats these observations into a single string.
    3. Uses a predefined prompt to request the most salient high-level questions.
    4. Parses and returns the identified questions as a list of strings.
    """
    prompt = PromptTemplate.from_template(
        "{observations}\n\n"
        + "Given only the information above, what are the 3 most salient "
        + "high-level questions we can answer about the subjects in the statements?\n"
        + "Provide each question on a new line."
    )
    observations = memory_retriever.memory_stream[-last_k:]
    observation_str = "\n".join(
        [format_memories_detail(o) for o in observations]
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    result = chain(prompt).run(observations=observation_str)
    return parse_list(result)

def fetch_memories(
    memory_retriever: TimeWeightedVectorStoreRetriever, 
    observation: str, 
    now: Optional[datetime] = None
) -> List[Document]:
    """Fetch related memories based on an observation.
    """
    if now is not None:
        with mock_now(now):
            return memory_retriever.get_relevant_documents(observation)
    else:
        return memory_retriever.get_relevant_documents(observation)

def get_insights_on_topic(
    llm: BaseLanguageModel, 
    memory_retriever: TimeWeightedVectorStoreRetriever,
    topics: Union[str, List[str]], 
    now: Optional[datetime] = None,
    verbose: bool = False
) -> List[List[str]]:
    """Generate insights on a topic of reflection based on pertinent memories.

    This method generates 'insights' on a given topic of reflection by analyzing
    pertinent memories associated with the topic. It follows these steps:
    1. Retrieves related memories based on the specified topic.
    2. Formats these memories into statements.
    3. Uses a predefined prompt to request high-level novel insights related to the topic.
    4. Parses and returns the generated insights as lists.
    """
    prompt = PromptTemplate.from_template(
        "Statements relevant to: '{topic}'\n"
        + "---\n"
        + "{related_statements}\n"
        + "---\n"
        + "What 5 high-level novel insights can you infer from the above statements "
        + "that are relevant for answering the following question?\n"
        + "Do not include any insights that are not relevant to the question.\n"
        + "Do not repeat any insights that have already been made.\n\n"
        + "Question: {topic}\n\n"
        + "(example format: insight (because of 1, 5, 3))\n"
    )

    if type(topics) is str:
        topics = [topics]

    chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    results = []
    for topic in topics:
        related_memories = fetch_memories(memory_retriever, topic, now=now)
        related_statements = "\n".join(
            [
                format_memories_detail(memory, prefix=f"{i+1}. ")
                for i, memory in enumerate(related_memories)
            ]
        )
        result = chain(prompt).run(
            topic=topic, related_statements=related_statements
        )
        results.append(parse_list(result))

    return results

def reflect(
    llm: BaseLanguageModel,
    memory_retriever: TimeWeightedVectorStoreRetriever, 
    last_k: int = 50,
    verbose: bool = False,
    now: Optional[datetime] = None
) -> List[str]:
    """Pause and reflect on recent observations to generate insights.

    This method initiates a pause in the Generative Agent's operation to reflect
    on recent observations and generate 'insights.' It follows these steps:
    1. Retrieves topics for reflection.
    2. For each topic, gathers insights using the `get_insights_on_topic` method.
    3. Adds these insights to the agent's memory for future reference.
    4. Returns a list of the generated insights.
    """
    new_insights = []
    topics = get_topics_of_reflection(
        llm, 
        memory_retriever, 
        verbose=verbose, 
        last_k=last_k
    )
    for topic in topics:
        insights = get_insights_on_topic(
            llm,
            memory_retriever,
            topic, 
            now=now, 
            verbose=verbose
        )[0]
        new_insights.extend(insights)
    return new_insights