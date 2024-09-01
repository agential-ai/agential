"""CoT functional module."""

from typing import Any, Dict, List, Optional

from agential.llm.llm import BaseLLM, Response
from agential.prompting.cot.output import CoTStepOutput


def _build_prompt(
    question: str,
    examples: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Builds a prompt for questioning the llm using a template.

    Parameters:
        question (str): The question to be answered by the llm.
        examples (str): Contextual examples related to the question.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt ready for use with the language model.
    """
    prompt = prompt.format(question=question, examples=examples, **additional_keys)
    return prompt


def _prompt_llm(
    llm: BaseLLM,
    question: str,
    examples: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
    temperature: Optional[float] = None,
) -> Response:
    """Prompts the llm to answer a question using the language model.

    Parameters:
        llm (BaseLLM): The language model to use for generating the answer.
        question (str): The question to be answered.
        examples (str): Contextual examples relevant to the question.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
        temperature (Optional[float]): The temperature to use for generating the answer. Defaults to None.

    Returns:
        Response: The answer from the language model, with no leading or trailing whitespace.
    """
    prompt = _build_prompt(
        question=question,
        examples=examples,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    print("<PROMPT LLM==============================================================>")
    print(prompt)
    print("<PROMPT LLM==============================================================>")
    out = llm(prompt, temperature=temperature)
    print("<OUT LLM==============================================================>")
    print(repr(out.output_text))
    print("<OUT LLM==============================================================>")
    return out


def accumulate_metrics(steps: List[List[CoTStepOutput]]) -> Dict[str, Any]:
    """Accumulate total metrics from a list of CoTStepOutput objects.

    This function calculates and aggregates various metrics across all steps in the input list.
    It sums up token counts, costs, and time measurements for both thought and action components.

    Args:
        steps (List[List[CoTStepOutput]]): A list of a list of CoTStepOutput objects representing individual steps.

    Returns:
        Dict[str, Any]: A dictionary containing the following accumulated metrics:
            - total_prompt_tokens (int): Total number of prompt tokens used.
            - total_completion_tokens (int): Total number of completion tokens generated.
            - total_tokens (int): Total number of tokens (prompt + completion).
            - total_prompt_cost (float): Total cost associated with prompts.
            - total_completion_cost (float): Total cost associated with completions.
            - total_cost (float): Total overall cost (prompt + completion).
            - total_prompt_time (float): Total time spent on prompts.
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    total_cost = 0.0
    total_prompt_time = 0.0

    for step in steps:
        for warming_step in step:
            total_prompt_tokens += (
                warming_step.thought_response.prompt_tokens
                + warming_step.answer_response.prompt_tokens
            )
            total_completion_tokens += (
                warming_step.thought_response.completion_tokens
                + warming_step.answer_response.completion_tokens
            )
            total_tokens += (
                warming_step.thought_response.prompt_tokens
                + warming_step.answer_response.prompt_tokens
                + warming_step.thought_response.completion_tokens
                + warming_step.answer_response.completion_tokens
            )
            total_prompt_cost += (
                warming_step.thought_response.prompt_cost
                + warming_step.answer_response.prompt_cost
            )
            total_completion_cost += (
                warming_step.thought_response.completion_cost
                + warming_step.answer_response.completion_cost
            )
            total_cost += (
                warming_step.thought_response.prompt_cost
                + warming_step.answer_response.prompt_cost
                + warming_step.thought_response.completion_cost
                + warming_step.answer_response.completion_cost
            )
            total_prompt_time += (
                warming_step.thought_response.prompt_time
                + warming_step.answer_response.prompt_time
            )

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_prompt_cost": total_prompt_cost,
        "total_completion_cost": total_completion_cost,
        "total_cost": total_cost,
        "total_prompt_time": total_prompt_time,
    }
