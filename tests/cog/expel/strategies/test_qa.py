"""Unit tests for ExpeL QA strategies."""

import joblib
from langchain_community.chat_models.fake import FakeListChatModel
from agential.cog.expel.strategies.qa import ExpeLQAStrategy
from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.cog.reflexion.agent import ReflexionReActAgent
def test_generate():
    pass

def test_get_dynamic_examples():
    pass

def test_gather_experience():
    pass


def test_extract_insights(expel_experiences_10_fake_path: str) -> None:
    """Test extract_insights."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    selected_indices = [3]
    selected_dict = {
        key: [value[i] for i in selected_indices] for key, value in experiences.items()
    }
    selected_dict["idxs"] = list(range(len(selected_indices)))

    gt_insights = [
        {
            "insight": "Always try multiple variations of search terms when looking for specific information.",
            "score": 2,
        },
        {
            "insight": "If unable to find relevant information through initial searches, consider looking for official announcements or press releases from the company.",
            "score": 2,
        },
    ]
    responses = [
        "ADD 11: Always try multiple variations of search terms when looking for specific information.\nADD 12: If unable to find relevant information through initial searches, consider looking for official announcements or press releases from the company.\nREMOVE 3: Always use the exact search term provided in the question, do not try variations.\nEDIT 7: Make sure to exhaust all possible search options before concluding that the information is unavailable.",
    ]
    llm = FakeListChatModel(responses=responses)
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLQAStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
    
    strategy.extract_insights(selected_dict)
    assert strategy.insight_memory.insights == gt_insights

def test_update_insights() -> None:
    """Test update_insights."""
    insights = [
        {"insight": "Test 1", "score": 1},
        {"insight": "Test 2", "score": 2},
        {"insight": "Test 3", "score": 3},
    ]
    memory = ExpeLInsightMemory(insights, max_num_insights=3)
    llm = FakeListChatModel(responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLQAStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent, insight_memory=memory)

    # Valid remove.
    gt_insights = [{"insight": "Test 2", "score": 2}, {"insight": "Test 3", "score": 3}]
    strategy.update_insights(
        [
            ("REMOVE 0", "Test 1"),
        ]
    )
    assert strategy.insight_memory.insights == gt_insights

    # Invalid remove.
    strategy.update_insights(
        [
            ("REMOVE 0", "Test askdasf"),
        ]
    )
    assert strategy.insight_memory.insights == gt_insights

    # Valid agree.
    gt_insights = [{"insight": "Test 2", "score": 3}, {"insight": "Test 3", "score": 3}]
    strategy.update_insights([("AGREE 0", "Test 2")])
    assert strategy.insight_memory.insights == gt_insights

    # Invalid agree.
    strategy.update_insights([("AGREE 0", "Test asdjafh")])
    assert strategy.insight_memory.insights == gt_insights

    # Edit.
    gt_insights = [{"insight": "Test 2", "score": 3}, {"insight": "Test 4", "score": 4}]
    strategy.update_insights([("EDIT 1", "Test 4")])
    assert strategy.insight_memory.insights == gt_insights

    # Add.
    gt_insights = [
        {"insight": "Test 2", "score": 3},
        {"insight": "Test 4", "score": 4},
        {"insight": "Another insight", "score": 2},
    ]
    strategy.update_insights([("ADD", "Another insight")])
    assert strategy.insight_memory.insights == gt_insights


def test_reset() -> None:
    """Test reset."""
    llm = FakeListChatModel(responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLQAStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
    
    strategy.reflexion_react_agent.strategy._scratchpad = "cat"
    strategy.experience_memory.experiences = "dog"
    strategy.insight_memory.insights = ["turtle"]
    strategy.reset()
    assert strategy.reflexion_react_agent.strategy._scratchpad == ""
    assert strategy.experience_memory.experiences == {
        "idxs": [],
        "questions": [],
        "keys": [],
        "trajectories": [],
        "reflections": [],
    }
    assert strategy.insight_memory.insights == []

    # Test only_reflexion=True.
    llm = FakeListChatModel(responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLQAStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
    
    strategy.reflexion_react_agent.strategy._scratchpad = "cat"
    strategy.experience_memory.experiences = "dog"
    strategy.insight_memory.insights = ["turtle"]
    strategy.reset(only_reflexion=True)
    assert strategy.reflexion_react_agent.strategy._scratchpad == ""
    assert strategy.experience_memory.experiences == "dog"
    assert strategy.insight_memory.insights == ["turtle"]