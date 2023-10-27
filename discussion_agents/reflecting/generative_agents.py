"""Generative Agents methods related to reflection."""

from datetime import datetime
from typing import List, Optional, Tuple, Union

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

from discussion_agents.core.base import BaseCore
from discussion_agents.utils.fetch import fetch_memories
from discussion_agents.utils.format import format_memories_detail
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
    chain = core.chain(prompt=prompt)
    result = parse_list(chain.run(observations=observations))
    return result


def get_insights_on_topics(
    topics: Union[str, List[str]],
    related_memories: Union[str, List[str]],
    core: BaseCore,
) -> List[List[str]]:
    """Generate high-level insights on specified topics using relevant memories.

    Args:
        topics (Union[str, List[str]]): A list of topics (or a str for 1 topic) for which insights are to be generated.
        related_memories (Union[str, List[str]]): Memories relevant to the specified topic(s);
            if topics and related_memories are both str/list, then they correspond 1-to-1;
            if topics is str and related_memories is list, then the topic will use all related_memories;
            if topics is list and related_memories is str, then related_memories is broadcasted to all topics.
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
        core = BaseCore(llm=llm)
        generated_insights = get_insights_on_topics(selected_topics, relevant_memories, core)
    """
    if isinstance(topics, str) and isinstance(related_memories, str):
        topics, related_memories = [topics], [related_memories]
    elif isinstance(topics, str) and isinstance(related_memories, list):
        topics = [topics]
        related_memories = ["\n".join(related_memories)]
    elif isinstance(topics, list) and isinstance(related_memories, str):
        related_memories = [related_memories] * len(topics)

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

    chain = core.chain(prompt=prompt)

    results = []
    for topic, related_memory in zip(topics, related_memories):
        result = chain.run(topic=topic, related_memory=related_memory)
        results.append(parse_list(result))

    return results


def reflect(
    observations: Union[str, List[str]], core: BaseCore, now: Optional[datetime] = None
) -> Tuple[List[str], List[List[str]]]:
    """Generate insights on recent observations through reflection.

    This function generates a list of topics w.r.t observations and extracts
    salient insights from each of these topics using related memories (to these topics)
    as context. Related memories are specific to each topic/observation.

    Args:
        observations (Union[str, List[str]]): Observations to derive reflections from.
        core (BaseCore): The agent's core component; needs a retriever.
        now (Optional[datetime]): current datetime or one specified.

    Returns:
        Tuple[List[str], List[List[str]]]: A list of generated topics from the
            observation and a list of lists of insights, a list of insights for
            every topic.

    Example:
        observations = "Attended a tech conference on AI advancements."
        core = BaseCore(llm=llm, retriever=retriever)
        topics, insights = reflect(observations, core, now=datetime.now())
    """
    topics = get_topics_of_reflection(observations=observations, core=core)

    related_memories = []
    for topic in topics:
        topic_related_memories = fetch_memories(
            observation=topic, memory_retriever=core.get_retriever(), now=now
        )
        topic_related_memories_joined = "\n".join(
            [
                format_memories_detail(memories=memory, prefix=f"{i+1}. ")
                for i, memory in enumerate(topic_related_memories)
            ]
        )
        related_memories.append(topic_related_memories_joined)

    insights = get_insights_on_topics(
        topics=topics, related_memories=related_memories, core=core
    )
    return topics, insights
