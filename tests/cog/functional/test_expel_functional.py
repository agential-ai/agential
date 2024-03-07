"""Unit tests for ExpeL functional module."""

import joblib

from discussion_agents.cog.agent.reflexion import ReflexionReActAgent
from discussion_agents.cog.functional.expel import (
    gather_experience,
    categorize_experiences
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

def test_categorize_experiences(expel_15_compare_fake_path: str) -> None:
    """Test categorize_experiences."""
    experiences = joblib.load(expel_15_compare_fake_path)
    categories = categorize_experiences(experiences)
    print(repr(categories))
    gt_categories = {
        'compare': [10, 11, 12, 13, 14], 
        'success': [1, 3, 6, 7, 8], 
        'fail': [0, 2, 4, 5, 9]
    }
    assert categories == gt_categories