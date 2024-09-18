"""Agent methods selector."""

from typing import Any

from agential.agents.base.agent import BaseAgent
from agential.agents.critic.agent import Critic
from agential.agents.expel.agent import ExpeL
from agential.agents.lats.agent import LATS
from agential.agents.react.agent import ReAct
from agential.agents.reflexion.agent import ReflexionCoT, ReflexionReAct
from agential.agents.self_refine.agent import SelfRefine

AGENT_METHODS = {
    "critic": Critic,
    "expel": ExpeL,
    "lats": LATS,
    "react": ReAct,
    "reflexion_cot": ReflexionCoT,
    "reflexion_react": ReflexionReAct,
    "self_refine": SelfRefine,
}


def select_agent_method(method: str, **init_kwargs: Any) -> BaseAgent:
    """Select the agent method.

    Args:
        method (str): The name of the agent method.
        **init_kwargs (Any): Initialization keyword arguments for the agent method.

    Returns:
        BaseAgent: An instance of the selected agent method.
    """
    if method not in AGENT_METHODS:
        raise ValueError(f"Invalid agent method: {method}")

    agent_class = AGENT_METHODS[method]
    return agent_class(**init_kwargs)
