"""ExpeL training components."""

from typing import List
from discussion_agents.cog.agent.reflexion import ReflexionReActAgent


# Q1: Should this experience be a part of the ExpeLAgent class?
# Q2: Should this Experience Gathering Section be a function or a class?

# llm = FakeListChatModel(responses=["1"])
# agent = ReflexionReActAgent(
#     self_reflect_llm=llm,
#     action_llm=llm,
# )
def gather_trajectories(reflexion_react_agent: ReflexionReActAgent, questions: List[str], keys: List[str], strategy: str = None):
    for (question, key) in zip(questions, keys):
        out = reflexion_react_agent.generate(question=question, key=key, strategy=strategy)
        