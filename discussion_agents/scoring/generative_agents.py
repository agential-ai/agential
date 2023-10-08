from typing import List, Union

import re

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel


def score_memories_importance(
    memory_contents: Union[str, List[str]],
    llm: BaseLanguageModel,
    verbose: bool = False,
    importance_weight: float = 0.15,
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
    chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

    scores = chain.run(memory_contents=memory_contents).strip()
    scores = re.findall(r'\d+', scores)  # In place of scores.split(";").

    # Split into list of strings and convert to floats
    scores_list = [float(x) / 10 * importance_weight for x in scores]

    return scores_list
