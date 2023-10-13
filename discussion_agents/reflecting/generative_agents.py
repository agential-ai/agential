"""Generative Agents methods related to reflection."""

from datetime import datetime
from typing import List, Optional, Union

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseRetriever
from langchain.schema.language_model import BaseLanguageModel

from discussion_agents.utils.fetch import fetch_memories
from discussion_agents.utils.format import format_memories_detail
from discussion_agents.utils.parse import parse_list


def get_topics_of_reflection(
    llm: BaseLanguageModel,
    memory_retriever: TimeWeightedVectorStoreRetriever,
    last_k: int = 50,
) -> List[str]:
    """Identify the three most salient high-level questions based on recent observations.

    Args:
        llm (BaseLanguageModel): Language model for question generation.
        memory_retriever (TimeWeightedVectorStoreRetriever): Memory retriever.
        last_k (int, optional): Number of recent observations to consider. Default is 50.

    Returns:
        List[str]: List of the three most salient high-level questions.

    Note:
        This method retrieves recent observations from memory and uses them to
        generate insightful questions about the observed subjects.

    Example:
        questions = get_topics_of_reflection(llm_model, memory_retriever)
    """
    assert isinstance(
        memory_retriever, TimeWeightedVectorStoreRetriever
    ), f"memory_retriever must be of type {TimeWeightedVectorStoreRetriever}."

    prompt = PromptTemplate.from_template(
        "{observations}\n\n"
        + "Given only the information above, what are the 3 most salient "
        + "high-level questions we can answer about the subjects in the statements?\n"
        + "Provide each question on a new line."
    )
    observations = memory_retriever.memory_stream[-last_k:]
    observation_str = "\n".join([format_memories_detail(o) for o in observations])
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(observations=observation_str)
    return parse_list(result)


def get_insights_on_topic(
    llm: BaseLanguageModel,
    memory_retriever: BaseRetriever,
    topics: Union[str, List[str]],
    now: Optional[datetime] = None,
) -> List[List[str]]:
    """Generate high-level insights on a given topic based on pertinent memories.

    Args:
        llm (BaseLanguageModel): Language model for insight generation.
        memory_retriever (TimeWeightedVectorStoreRetriever): Memory retriever.
        topics (Union[str, List[str]]): Topic or list of topics for insight generation.
        now (Optional[datetime], optional): Current date and time for temporal context. Default is None.

    Returns:
        List[List[str]]: Lists of high-level novel insights for each specified topic.

    Note:
        This method extracts pertinent memories related to the given topics and
        generates unique insights based on the gathered information.

    Example:
        topics = ["Favorite books", "Hiking experiences"]
        insights = get_insights_on_topic(llm_model, memory_retriever, topics)
    """
    prompt = PromptTemplate.from_template(
        "Statements relevant to: '{topic}'\n"
        + "---\n"
        + "{related_statements}\n"
        + "---\n"
        + "What 5 high-level novel insights can you infer from the above statements "
        + "that are relevant to answering the following question?\n"
        + "Do not include any insights that are not relevant to the question.\n"
        + "Do not repeat any insights that have already been made.\n\n"
        + "Question: {topic}\n\n"
    )

    if type(topics) is str:
        topics = [topics]

    chain = LLMChain(llm=llm, prompt=prompt)
    results = []
    for topic in topics:
        related_memories = fetch_memories(memory_retriever, topic, now=now)
        related_statements = "\n".join(
            [
                format_memories_detail(memory, prefix=f"{i+1}. ")
                for i, memory in enumerate(related_memories)
            ]
        )
        result = chain.run(topic=topic, related_statements=related_statements)
        results.append(parse_list(result))

    return results


def reflect(
    llm: BaseLanguageModel,
    memory_retriever: TimeWeightedVectorStoreRetriever,
    last_k: int = 50,
    now: Optional[datetime] = None,
) -> List[str]:
    """Pause to reflect on recent observations and generate insights.

    Args:
        llm (BaseLanguageModel): Language model for insight generation.
        memory_retriever (TimeWeightedVectorStoreRetriever): Memory retriever.
        last_k (int, optional): Number of recent observations to consider. Default is 50.
        now (Optional[datetime], optional): Current date and time for temporal context. Default is None.

    Returns:
        List[str]: List of generated insights based on recent observations.

    Note:
        This method retrieves topics for reflection, gathers insights for each topic,
        and adds these insights to the agent's memory for future reference.

    Example:
        insights = reflect(llm_model, memory_retriever, last_k=100)
    """
    assert isinstance(
        memory_retriever, TimeWeightedVectorStoreRetriever
    ), f"memory_retriever must be of type {TimeWeightedVectorStoreRetriever}."

    new_insights = []
    topics = get_topics_of_reflection(llm, memory_retriever, last_k=last_k)
    for topic in topics:
        insights = get_insights_on_topic(llm, memory_retriever, topic, now=now)[0]
        new_insights.extend(insights)
    return new_insights
