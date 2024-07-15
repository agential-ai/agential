"""Unit tests for ExpeL QA strategies."""
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

def test_extract_insights():
    pass

def test_update_insights() -> None:
    """Tests update_insights."""
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


def test_reset():
    pass