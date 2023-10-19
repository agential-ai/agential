"""Generative Agents methods related to reflection."""

from typing import List, Union

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from discussion_agents.core.base import BaseCore

from discussion_agents.utils.parse import parse_list


def get_topics_of_reflection(
    observations: str,
    core: BaseCore,
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
    prompt = PromptTemplate.from_template(
        "{observations}\n\n"
        + "Given only the information above, what are the 3 most salient "
        + "high-level questions we can answer about the subjects in the statements?\n"
        + "Provide each question on a new line."
    )
    chain = LLMChain(llm=core.llm, prompt=prompt)
    result = chain.run(observations=observations)
    return parse_list(result)


def get_insights_on_topic(
    related_memories: Union[str, List[str]],
    topics: Union[str, List[str]],
    core: BaseCore,
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
        + "{related_memory}\n"
        + "---\n"
        + "What 5 high-level novel insights can you infer from the above statements "
        + "that are relevant to answering the following question?\n"
        + "Do not include any insights that are not relevant to the question.\n"
        + "Do not repeat any insights that have already been made.\n\n"
        + "Question: {topic}\n\n"
    )

    assert type(topics) is type(related_memories)
    if type(topics) is list and type(related_memories) is list:
        assert len(topics) == len(related_memories)

    if type(topics) is str and type(related_memories) is str:
        topics = [topics]
        related_memories = [related_memories]

    chain = LLMChain(llm=core.llm, prompt=prompt)

    results = []
    for topic, related_memory in zip(topics, related_memories):
        result = chain.run(topic=topic, related_memory=related_memory)
        results.append(parse_list(result))

    return results


def reflect(
    observations: str,
    related_memories: str,
    core: BaseCore,
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
    topics = get_topics_of_reflection(observations, core)
    new_insights = get_insights_on_topic(related_memories, topics, core)
    return new_insights
