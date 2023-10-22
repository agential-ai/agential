"""Memory-scoring methods for Generative Agents."""
import re

from typing import List, Union

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from discussion_agents.core.base import BaseCore


def score_memories_importance(
    memory_contents: Union[str, List[str]],
    relevant_memories: Union[str, List[str]],
    core: BaseCore,
    importance_weight: float = 0.15,
) -> List[float]:
    """Calculate absolute importance scores for given memory contents.

    Args:
        memory_contents (Union[str, List[str]): Memories or memory contents to score.
        relevant_memories (Union[str, List[str]): Relevant memories to all memory_contents;
            if memory_contents and relevant_memories are both str/list, then they correspond 1-to-1;
            if memory_contents is str and relevant_memories is list, then the topic will use all relevant_memories;
            if memory_contents is list and relevant_memories is str, then relevant_memories is broadcasted to all memory_contents.
        core (BaseCore): The agent core component.
        importance_weight (float, optional): Weight for importance scores. Default is 0.15.

    Returns:
        List[float]: List of importance scores (1.0 to 10.0 scale normalized and weighted).

    Example:
        memories = ["Visited the museum."]
        relevant_memories = ["Enjoyed the outdoors at the museum.", "Took lots of pictures at the museum."]
        core = ...
        importance_scores = score_memories_importance(memories, relevant_memories, core)
        # [0.9]
    """
    if isinstance(memory_contents, str) and isinstance(relevant_memories, str):
        memory_contents, relevant_memories = [memory_contents], [relevant_memories]
    elif isinstance(memory_contents, str) and isinstance(relevant_memories, list):
        memory_contents = [memory_contents]
        relevant_memories = ["\n".join(relevant_memories)]
    elif isinstance(memory_contents, list) and isinstance(relevant_memories, str):
        relevant_memories = [relevant_memories] * len(memory_contents)

    assert len(memory_contents) == len(relevant_memories)

    relevant_memories = "\n".join(relevant_memories)

    prompt = PromptTemplate.from_template(
        "On the scale of 1 to 10, where 1 is purely mundane "
        + "and 10 is extremely poignant "
        + ", rate the likely poignancy of the "
        + "following piece of memory with respect to these following relevant memories:\n"
        + "{relevant_memories}\n\n"
        + "Provide only a single rating.\n"
        + "\Memory: {memory_content}\n"
        + "Rating: "
    )
    chain = LLMChain(llm=core.llm, prompt=prompt)

    scores = []
    for i, (memory_content, relevant_memory) in enumerate(
        zip(memory_contents, relevant_memories)
    ):
        score = chain.run(
            relevant_memories=relevant_memory, memory_content=memory_content
        ).strip()
        score = re.findall(r"\d+", score)
        score = (
            [0] if not score else score
        )  # Default value if the model cannot assign a value.
        assert (
            len(score) == 1
        ), f"Found multiple scores in LLM output for memory_content at index {i}: {score}."
        scores.append(score[0])

    assert len(scores) == len(
        memory_contents
    ), f"The llm generated {len(scores)} scores for {len(memory_contents)} memories."

    # Split into list of strings and convert to floats.
    scores = [float(x) / 10 * importance_weight for x in scores]

    return scores
