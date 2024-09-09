"""Helper function to select a method from a list of methods."""


from typing import Any, Union
from agential.agents.base.agent import BaseAgent
from agential.agents.react.agent import ReAct
from agential.core.llm import BaseLLM
from agential.prompting.base.prompting import BasePrompting
from agential.prompting.cot.prompting import CoT
from agential.prompting.standard.prompting import Standard

prompting_methods = [
    "standard",
    "cot",
]

agent_methods = [
    "react",
]

def select_prompting_method(method_name: str, llm: BaseLLM, benchmark: str, **kwargs: Any) -> BasePrompting:
    """Select a method from a list of prompting methods.
    
    Args:
        method_name (str): The method name.
        llm (BaseLLM): The LLM.
        benchmark (str):  The benchmark name.
        **kwargs (Any): The additional kwargs.

    Returns:
        BasePrompting: The prompting method. 
    """
    if method_name == "standard":
        return Standard(llm=llm, benchmark=benchmark, **kwargs)
    elif method_name == "cot":
        return CoT(llm=llm, benchmark=benchmark, **kwargs)
    else:
        raise ValueError(f"Method {method_name} not found.")
    

def select_agent_method(method_name: str, llm: BaseLLM, benchmark: str, **kwargs: Any) -> BaseAgent:
    """Select a method from a list of agent methods.

    Args:
        method_name (str): The method name.
        llm (BaseLLM): The LLM.
        benchmark (str):  The benchmark name.
        **kwargs (Any): The additional kwargs.

    Returns:
        BaseAgent: The agent method.
    """
    if method_name == "react":
        return ReAct(llm=llm, benchmark=benchmark, **kwargs)
    else:
        raise ValueError(f"Method {method_name} not found.")
    

def select_method(method_name: str, llm: BaseLLM, benchmark: str, **kwargs: Any) -> Union[BasePrompting, BaseAgent]:
    """Select a method from a list of methods.

    Args:
        method_name (str): The method name.
        llm (BaseLLM): The LLM.
        benchmark (str):  The benchmark name.
        **kwargs (Any): The additional kwargs.

    Returns:
        Union[BasePrompting, BaseAgent]: The method.
    """
    if method_name in prompting_methods:
        return select_prompting_method(method_name=method_name, llm=llm, benchmark=benchmark, **kwargs)
    elif method_name in agent_methods:
        return select_agent_method(method_name=method_name, llm=llm, benchmark=benchmark, **kwargs)
    else:
        raise ValueError(f"Method {method_name} not found.")