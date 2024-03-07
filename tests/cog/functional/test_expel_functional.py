"""Unit tests for ExpeL functional module."""

from discussion_agents.cog.agent.reflexion import ReflexionReActAgent
from discussion_agents.cog.functional.expel import (
    gather_experience,
)

def test_gather_experience(reflexion_react_agent: ReflexionReActAgent) -> None:
    """Test gather_experience."""
    questions = [""]
    keys = [""]
    experiences = gather_experience(
        reflexion_react_agent,
        questions,
        keys,
    )
    gt_experiences = {
        'idxs': [0], 
        'questions': [''], 
        'keys': [''], 
        'trajectories': [[]], 
        'reflections': [[]]
    }
    assert experiences == gt_experiences