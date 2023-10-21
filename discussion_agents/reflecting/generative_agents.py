"""Generative Agents methods related to reflection."""

from typing import List, Union

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from discussion_agents.core.base import BaseCore
from discussion_agents.utils.parse import parse_list


def get_topics_of_reflection(
    observations: Union[str, List[str]],
    core: BaseCore,
) -> List[str]:
    """Generate three insightful high-level questions based on recent observation(s).

    Args:
        observations (Union[str, List[str]]): Recent observations to derive questions from; can be multiple but must be str.
        core (BaseCore): The agent's core component.

    Returns:
        List[str]: A list of the three most salient high-level questions.

    Note:
        This function takes recent observations as input and utilizes the specified core
        component to generate meaningful and high-level questions related to the observed
        subjects.

    Example:
        recent_observations = "Attended a scientific lecture on AI advancements."
        core = BaseCore(llm=llm)
        generated_questions = get_topics_of_reflection(recent_observations, core)
    """
    if type(observations) is list:
        observations = "\n".join(observations)

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
    topics: List[str],
    related_memories: List[str],
    core: BaseCore,
) -> List[List[str]]:
    """Generate high-level insights on specified topics using relevant memories.

    Args:
        topics (List[str]): A list of topics for which insights are to be generated.
        related_memories (List[str]): Memories relevant to the specified topics.
        core (BaseCore): The core component used for insight generation.

    Returns:
        List[List[str]]: Lists of high-level, unique insights corresponding to each topic.

    Note:
        This function leverages pertinent memories associated with the specified topics
        and employs the provided core component to create novel, high-level insights
        based on the gathered information.

    Example:
        selected_topics = ["Artificial Intelligence trends", "Recent travel experiences"]
        relevant_memories = ["Attended an AI conference.", "Traveled to Japan last month."]
        generated_insights = get_insights_on_topic(selected_topics, relevant_memories, insight_core)
    """   

    # str list
    # list str
    # str str
    # list list

    assert type(topics) == type(related_memories) == list
    assert len(topics) == len(related_memories)

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

    chain = LLMChain(llm=core.llm, prompt=prompt)

    results = []
    for topic, related_memory in zip(topics, related_memories):
        result = chain.run(topic=topic, related_memory=related_memory)
        results.append(parse_list(result))

    return results


def reflect(
    observations: str,
    related_memories: List[str],
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
    topics = get_topics_of_reflection(observations=observations, core=core)
    new_insights = get_insights_on_topic(
        topics=topics, related_memories=related_memories, core=core
    )
    return new_insights
