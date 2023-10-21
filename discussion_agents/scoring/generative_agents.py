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
        memory_contents (Union[str, List[str]]): Memories to score.
        llm (BaseLanguageModel): Language model for scoring.
        importance_weight (float, optional): Weight for importance scores. Default is 0.15.

    Returns:
        List[float]: List of importance scores (1.0 to 10.0 scale).

    Example:
        memories = ["Visited the museum.", "Had a meaningful conversation."]
        importance_scores = score_memories_importance(memories, llm)
    """
    if type(memory_contents) is str:
        memory_contents = [memory_contents]
    if type(relevant_memories) is str:
        relevant_memories = [relevant_memories]

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
    for i, memory_content in enumerate(memory_contents):
        score = chain.run(
            relevant_memories=relevant_memories, memory_content=memory_content
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

    # Split into list of strings and convert to floats
    scores = [float(x) / 10 * importance_weight for x in scores]

    return scores
