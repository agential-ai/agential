"""Unit tests for ExpeL memory module."""

import joblib
import pytest

from langchain_core.embeddings import Embeddings
from tiktoken.core import Encoding

from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)

def test_expel_experience_memory_init(expel_experiences_10_fake_path: str) -> None:
    """Test ExpeLExperienceMemory initialization."""
    experiences = joblib.load(expel_experiences_10_fake_path)

    # Test empty initialization.
    memory = ExpeLExperienceMemory()
    assert memory.experiences == []
    assert memory.strategy == "task"
    assert isinstance(memory.embedder, Embeddings)
    assert isinstance(memory.encoder, Encoding)

    assert not memory.success_traj_docs
    assert memory.vectorstore is None

    # Test with experiences parameter.
    memory = ExpeLExperienceMemory(experiences)
    assert memory.experiences == experiences
    assert memory.strategy == "task"
    assert isinstance(memory.embedder, Embeddings)
    assert isinstance(memory.encoder, Encoding)
    assert len(memory.success_traj_docs) == 23
    assert memory.vectorstore

    success_traj_doc_types = [
        "task",
        "action",
        "action",
        "action",
        "action",
        "thought",
        "thought",
        "thought",
        "thought",
        "step",
    ]

    for type_, doc in zip(success_traj_doc_types, memory.success_traj_docs):
        assert type_ == doc.metadata["type"]


def test_expel_experience_memory_len(expel_experiences_10_fake_path: str) -> None:
    """Test ExpeLExperienceMemory len method."""
    memory = ExpeLExperienceMemory()
    assert len(memory) == 0

    experiences = joblib.load(expel_experiences_10_fake_path)
    memory = ExpeLExperienceMemory(experiences)
    assert len(memory) == 5


def test_expel_experience_memory_clear(expel_experiences_10_fake_path: str) -> None:
    """Test ExpeLExperienceMemory clear method."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    memory = ExpeLExperienceMemory(experiences)
    assert memory.experiences
    assert memory.success_traj_docs
    assert memory.vectorstore
    memory.clear()
    assert memory.experiences == []
    assert not memory.success_traj_docs
    assert not memory.vectorstore


def test_expel_experience_memory_add_memories(
    expel_experiences_10_fake_path: str,
) -> None:
    """Test ExpeLExperienceMemory add_memories method."""
    experiences = joblib.load(expel_experiences_10_fake_path)

    # Successful trajectory.
    success_questions = [experiences[3]["questions"][3]]
    success_keys = [experiences["keys"][3]]
    success_trajectories = [experiences["trajectories"][3]]
    success_reflections = [[]]

    # Failed trajectories (multiple).
    fail_questions = [
        experiences["questions"][0],
        experiences["questions"][1],
    ]
    fail_keys = [
        experiences["keys"][0],
        experiences["keys"][1],
    ]
    fail_trajectories = [experiences["trajectories"][0], experiences["trajectories"][1]]
    fail_reflections = [experiences["reflections"][0], experiences["reflections"][1]]

    # Test with empty memory (with and without reflection).
    memory = ExpeLExperienceMemory()
    memory.add_memories(
        success_questions, success_keys, success_trajectories, success_reflections
    )
    assert memory.experiences["idxs"] == [0]
    assert memory.experiences["questions"][0] == success_questions[0]
    assert memory.experiences["keys"][0] == success_keys[0]
    assert memory.experiences["trajectories"][0] == success_trajectories[0]
    assert memory.experiences["reflections"][0] == success_reflections[0]
    assert len(memory.success_traj_docs) == 10
    assert memory.success_traj_docs[0].metadata["task_idx"] == 0
    assert memory.vectorstore

    memory.add_memories(
        success_questions,
        success_keys,
        success_trajectories,
    )
    assert memory.experiences["idxs"] == [0, 1]
    assert memory.experiences["questions"][1] == success_questions[0]
    assert memory.experiences["keys"][1] == success_keys[0]
    assert memory.experiences["trajectories"][1] == success_trajectories[0]
    assert memory.experiences["reflections"][1] == success_reflections[0]
    assert len(memory.success_traj_docs) == 20
    assert memory.success_traj_docs[0].metadata["task_idx"] == 0
    assert memory.success_traj_docs[-1].metadata["task_idx"] == 1
    assert memory.vectorstore

    # Test with non-empty memory (with reflection).
    memory.add_memories(
        success_questions, success_keys, success_trajectories, success_reflections
    )
    assert memory.experiences["idxs"] == [0, 1, 2]
    assert memory.experiences["questions"][2] == success_questions[0]
    assert memory.experiences["keys"][2] == success_keys[0]
    assert memory.experiences["trajectories"][2] == success_trajectories[0]
    assert memory.experiences["reflections"][2] == success_reflections[0]
    assert len(memory.success_traj_docs) == 30
    assert memory.success_traj_docs[0].metadata["task_idx"] == 0
    assert memory.success_traj_docs[20].metadata["task_idx"] == 2
    assert memory.success_traj_docs[-1].metadata["task_idx"] == 2
    assert memory.vectorstore

    # Test with adding only failed trajectories.
    memory.add_memories(fail_questions, fail_keys, fail_trajectories, fail_reflections)
    assert memory.experiences["idxs"] == [0, 1, 2, 3, 4]
    assert memory.experiences["questions"][3] == fail_questions[0]
    assert memory.experiences["questions"][4] == fail_questions[1]
    assert memory.experiences["keys"][3] == fail_keys[0]
    assert memory.experiences["keys"][4] == fail_keys[1]
    assert memory.experiences["trajectories"][3] == fail_trajectories[0]
    assert memory.experiences["trajectories"][4] == fail_trajectories[1]
    assert memory.experiences["reflections"][3] == fail_reflections[0]
    assert memory.experiences["reflections"][4] == fail_reflections[1]
    assert len(memory.success_traj_docs) == 43
    assert memory.success_traj_docs[0].metadata["task_idx"] == 0
    assert memory.success_traj_docs[20].metadata["task_idx"] == 2
    assert memory.vectorstore

    # Test with a mix of failed and successful trajectories.
    memory.add_memories(
        success_questions + fail_questions,
        success_keys + fail_keys,
        success_trajectories + fail_trajectories,
        success_reflections + fail_reflections,
    )
    assert memory.experiences["idxs"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert memory.experiences["questions"][5] == success_questions[0]
    assert memory.experiences["questions"][6] == fail_questions[0]
    assert memory.experiences["questions"][7] == fail_questions[1]
    assert memory.experiences["keys"][5] == success_keys[0]
    assert memory.experiences["keys"][6] == fail_keys[0]
    assert memory.experiences["keys"][7] == fail_keys[1]
    assert memory.experiences["trajectories"][5] == success_trajectories[0]
    assert memory.experiences["trajectories"][6] == fail_trajectories[0]
    assert memory.experiences["trajectories"][7] == fail_trajectories[1]
    assert memory.experiences["reflections"][5] == success_reflections[0]
    assert memory.experiences["reflections"][6] == fail_reflections[0]
    assert memory.experiences["reflections"][7] == fail_reflections[1]
    assert len(memory.success_traj_docs) == 66
    assert memory.success_traj_docs[0].metadata["task_idx"] == 0
    assert memory.success_traj_docs[20].metadata["task_idx"] == 2
    assert memory.vectorstore


def test_expel_experience_memory__fewshot_doc_token_count(
    expel_experiences_10_fake_path: str,
) -> None:
    """Test ExpeLExperienceMemory _fewshot_doc_token_count method."""
    experiences = joblib.load(expel_experiences_10_fake_path)

    # Testing with just experiences (1 success, a dupe).
    memory = ExpeLExperienceMemory(experiences)
    gt_token_counts = [
        554,
        554,
        554,
        554,
        554,
        554,
        554,
        554,
        554,
        554,
        554,
        554,
        554,
        971,
        971,
        971,
        971,
        971,
        971,
        971,
        971,
        971,
        971,
    ]
    for doc, gt_token_count in zip(memory.success_traj_docs, gt_token_counts):
        token_count = memory._fewshot_doc_token_count(doc)
        assert token_count == gt_token_count


def test_expel_experience_memory_load_memories(
    expel_experiences_10_fake_path: str,
) -> None:
    """Test ExpeLExperienceMemory load_memories method."""
    experiences = joblib.load(expel_experiences_10_fake_path)

    queries = {
        "task": 'The creator of "Wallace and Gromit" also created what animation comedy that matched animated zoo animals with a soundtrack of people talking about their homes? ',
        "thought": "Thought: I should try a different approach. Let me search for press releases, industry news sources, or announcements specifically related to the name change and new acronym for VIVA Media AG in 2004. By focusing on more specialized sources, I may be able to find the accurate information needed to answer the question correctly. ",
        "other": "Some other query.",
    }

    empty_thought_queries = {
        "task": 'The creator of "Wallace and Gromit" also created what animation comedy that matched animated zoo animals with a soundtrack of people talking about their homes? ',
        "thought": "",
    }

    # Test when memory is empty.
    memory = ExpeLExperienceMemory()
    memory_dict = memory.load_memories(query=queries["task"])
    assert list(memory_dict.keys()) == ["fewshots"]
    assert not memory_dict["fewshots"]

    # Test non-empty memory with different query types like "task" and "thought".
    # Other query types limited to keys of queries.
    memory = ExpeLExperienceMemory(experiences)
    memory_dict = memory.load_memories(query=queries["task"])
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 2

    memory_dict = memory.load_memories(query=queries["thought"])
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 2

    memory_dict = memory.load_memories(query=queries["other"])
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 2

    # Test with every reranking strategy + error.
    with pytest.raises(NotImplementedError):
        memory_dict = memory.load_memories(
            query=queries["task"], reranker_strategy="invalid input"
        )

    # First case.
    memory_dict = memory.load_memories(
        empty_thought_queries["task"], reranker_strategy="thought"
    )
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 2

    # Length case.
    memory_dict = memory.load_memories(
        query=queries["task"], reranker_strategy="length"
    )
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 2

    # Thought case.
    memory_dict = memory.load_memories(
        query=queries["task"], reranker_strategy="thought"
    )
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 2

    # Task case.
    memory_dict = memory.load_memories(query=queries["task"], reranker_strategy="task")
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 2

    # Test with varying max_fewshot_tokens.
    memory_dict = memory.load_memories(query=queries["task"], max_fewshot_tokens=0)
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 0

    # Test with varying num_fewshots.
    memory_dict = memory.load_memories(query=queries["task"], num_fewshots=3)
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 2

    memory_dict = memory.load_memories(query=queries["task"], num_fewshots=2)
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 2

    memory_dict = memory.load_memories(query=queries["task"], num_fewshots=1)
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 1

    memory_dict = memory.load_memories(query=queries["task"], num_fewshots=0)
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 0

    # Test with varying k_docs.
    memory_dict = memory.load_memories(query=queries["task"], k_docs=0)
    assert list(memory_dict.keys()) == ["fewshots"]
    assert isinstance(memory_dict["fewshots"], list)
    assert len(memory_dict["fewshots"]) == 0


def test_expel_experience_memory_show_memories(
    expel_experiences_10_fake_path: str,
) -> None:
    """Test ExpeLExperienceMemory show_memories method."""
    experiences = joblib.load(expel_experiences_10_fake_path)

    # Test with empty memory.
    memory = ExpeLExperienceMemory()
    memory_dict = memory.show_memories()
    assert list(memory_dict.keys()) == [
        "experiences",
        "success_traj_docs",
        "vectorstore",
    ]
    assert memory_dict["experiences"] == {
        "idxs": [],
        "questions": [],
        "keys": [],
        "trajectories": [],
        "reflections": [],
    }
    assert not memory_dict["success_traj_docs"]
    assert not memory_dict["vectorstore"]

    # Test with non-empty memory.
    memory = ExpeLExperienceMemory(experiences)
    memory_dict = memory.show_memories()
    assert list(memory_dict.keys()) == [
        "experiences",
        "success_traj_docs",
        "vectorstore",
    ]
    assert memory.experiences == memory_dict["experiences"]
    assert len(memory_dict["success_traj_docs"]) == 23
    assert memory_dict["vectorstore"]


def test_expel_insight_memory_init() -> None:
    """Test ExpeLInsightMemory initialization."""
    # Test with empty memory.
    memory = ExpeLInsightMemory()
    assert not memory.insights
    assert memory.max_num_insights == 20
    assert memory.leeway == 5

    # Test initialization with insights and max_num_insights.
    insights = [{"insight": "Insight", "score": 1}]
    max_num_insights = 5
    leeway = 1
    memory = ExpeLInsightMemory(
        insights=insights, max_num_insights=max_num_insights, leeway=1
    )
    assert memory.insights == insights
    assert memory.max_num_insights == max_num_insights
    assert memory.leeway == leeway


def test_expel_insight_memory_len() -> None:
    """Test ExpeLInsightMemory len method."""
    memory = ExpeLInsightMemory()
    assert len(memory) == 0

    insights = [{"insight": "Insight", "score": 1}]
    memory = ExpeLInsightMemory(insights)
    assert len(memory) == 1


def test_expel_insight_memory_clear() -> None:
    """Test ExpeLInsightMemory clear method."""
    insights = [{"insight": "Insight", "score": 1}]
    max_num_insights = 5
    memory = ExpeLInsightMemory(insights=insights, max_num_insights=max_num_insights)
    memory.clear()
    assert memory.insights == []


def test_expel_insight_memory_add_memories() -> None:
    """Test ExpeLInsightMemory add_memories method."""
    memory = ExpeLInsightMemory(max_num_insights=3)
    insights_to_add = [
        {"insight": "Test 1", "score": 1},
        {"insight": "Test 2", "score": 2},
    ]
    memory.add_memories(insights_to_add)
    assert len(memory.insights) == 2
    assert memory.insights == insights_to_add

    # Test exceeding max_num_insights.
    memory.add_memories(
        [{"insight": "Test 3", "score": 3}, {"insight": "Test 4", "score": 4}]
    )
    assert len(memory.insights) == 4  # Should not add beyond max_num_insights


def test_expel_insight_memory_delete_memories() -> None:
    """Test ExpeLInsightMemory delete_memories method."""
    memory = ExpeLInsightMemory(max_num_insights=3, leeway=0)
    insights_to_add = [
        {"insight": "Test 1", "score": 1},
        {"insight": "Test 2", "score": 2},
    ]
    memory.add_memories(insights_to_add)

    # Test deleting the first item.
    memory.delete_memories(0)
    assert len(memory.insights) == 1 and memory.insights[0]["insight"] == "Test 2"

    # Test deleting the last item (re-add and then delete the last).
    memory.add_memories([{"insight": "Test 3", "score": 3}])
    memory.delete_memories(1)  # Now index 1 should be "Test 3"
    assert len(memory.insights) == 2 and memory.insights[0]["insight"] == "Test 2"

    # Test deleting from an empty memory (setup a new instance to ensure it's empty).
    empty_memory = ExpeLInsightMemory(max_num_insights=3)
    empty_memory.delete_memories(0)
    assert len(empty_memory.insights) == 0

    # Test deleting the only item in memory.
    single_item_memory = ExpeLInsightMemory(
        max_num_insights=3, insights=[{"insight": "Only Item", "score": 1}]
    )
    single_item_memory.delete_memories(0)
    assert (
        len(single_item_memory.insights) == 0
    ), "Memory should be empty after deleting the only item."

    # Test attempting to delete with an invalid index does not affect memory.
    initial_length = len(memory.insights)
    memory.delete_memories(5)  # Index out of range.
    assert len(memory.insights) == initial_length

    # Test deleting memory when exceeding leeway.
    memory = ExpeLInsightMemory(max_num_insights=2, leeway=1)
    insights_to_add = [
        {"insight": "Insight 1", "score": 1},
        {"insight": "Insight 2", "score": 2},
        {"insight": "Insight 3", "score": 3},
    ]

    # Add insights to memory, exceeding the max_num_insights but within leeway.
    memory.add_memories(insights_to_add)

    # Assert all insights are still in memory since we haven't exceeded max_num_insights + leeway.
    assert len(memory.insights) == 3

    # Add another insight to exceed the leeway limit.
    memory.add_memories([{"insight": "Insight 4", "score": 4}])

    # Attempt to delete an insight now that we've exceeded the leeway
    memory.delete_memories(0)  # Try deleting the first insight.

    # Check the insight was deleted.
    assert len(memory.insights) == 3

    # Further verify that the correct insights remain.
    assert memory.insights[0]["insight"] == "Insight 2"
    assert memory.insights[1]["insight"] == "Insight 3"


def test_expel_insight_memory_update_memories() -> None:
    """Test ExpeLInsightMemory update_memories method."""
    memory = ExpeLInsightMemory(max_num_insights=3)
    insights_to_add = [{"insight": "Test 1", "score": 1}]
    memory.add_memories(insights_to_add)
    memory.update_memories(0, "EDIT", "Updated Test 1")
    assert memory.insights[0]["insight"] == "Updated Test 1"
    assert memory.insights[0]["score"] == 2

    memory.update_memories(0, "AGREE")
    assert memory.insights[0]["score"] == 3


def test_expel_insight_memory_load_memories() -> None:
    """Test ExpeLInsightMemory load_memories method."""
    # Test empty.
    memory = ExpeLInsightMemory(max_num_insights=3)
    loaded_memories = memory.load_memories()
    assert loaded_memories == {"insights": []}

    # Test non-empty.
    insights = [{"insight": "Test 1", "score": 1}]
    memory = ExpeLInsightMemory(insights, max_num_insights=3)
    loaded_memories = memory.load_memories()
    assert loaded_memories == {"insights": insights}


def test_expel_insight_memory_show_memories() -> None:
    """Test ExpeLInsightMemory show_memories method."""
    # Test empty.
    memory = ExpeLInsightMemory(max_num_insights=3)
    loaded_memories = memory.show_memories()
    assert loaded_memories == {"insights": []}

    # Test non-empty.
    insights = [{"insight": "Test 1", "score": 1}]
    memory = ExpeLInsightMemory(insights, max_num_insights=3)
    loaded_memories = memory.show_memories()
    assert loaded_memories == {"insights": insights}
