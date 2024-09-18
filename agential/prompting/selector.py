"""Prompting methods selector."""

from typing import Any

from agential.prompting.base.prompting import BasePrompting
from agential.prompting.cot.prompting import CoT
from agential.prompting.standard.prompting import Standard

PROMPTING_METHODS = {"standard": Standard, "cot": CoT}


def select_prompting_method(method: str, init_kwargs: Any) -> BasePrompting:
    """Select the prompting method.

    Args:
        method (str): The name of the prompting method.
        init_kwargs (Any): Initialization keyword arguments for the prompting method.

    Returns:
        BasePrompting: An instance of the selected prompting method.
    """
    if method not in PROMPTING_METHODS:
        raise ValueError(f"Invalid prompting method: {method}")

    prompting_class = PROMPTING_METHODS[method]
    return prompting_class(**init_kwargs)
