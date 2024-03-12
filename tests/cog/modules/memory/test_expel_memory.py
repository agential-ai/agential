"""Unit tests for ExpeL memory module."""

import joblib

from discussion_agents.cog.modules.memory.expel import ExpeLExperienceMemory


def test_expel_experience_memory_init(expel_experiences_10_fake_path: str) -> None:
    """Test ExpeLExperienceMemory initialization."""
    # Test empty.
    memory = ExpeLExperienceMemory()
    assert isinstance(memory.experiences, dict)
    for k, v in memory.experiences.items():
        assert isinstance(k, str)
        assert isinstance(v, list)
        assert len(v) == 0
    assert list(memory.experiences.keys()) == [
        "idxs",
        "questions",
        "keys",
        "trajectories",
        "reflections",
    ]

    # Test non-empty.
    experiences = joblib.load(expel_experiences_10_fake_path)
    _ = experiences.pop("idxs")
    memory = ExpeLExperienceMemory(**experiences)
    assert list(memory.experiences.keys()) == [
        "idxs",
        "questions",
        "keys",
        "trajectories",
        "reflections",
    ]
    for k, v in memory.experiences.items():
        assert len(v) == 10
    assert memory.experiences["idxs"] == list(range(10))

    # Test with no reflection.
    experiences = joblib.load(expel_experiences_10_fake_path)
    _ = experiences.pop("idxs")
    _ = experiences.pop("reflections")
    memory = ExpeLExperienceMemory(**experiences)
    assert list(memory.experiences.keys()) == [
        "idxs",
        "questions",
        "keys",
        "trajectories",
        "reflections",
    ]
    for k, v in memory.experiences.items():
        assert len(v) == 10
    assert memory.experiences["idxs"] == list(range(10))
    for reflection in memory.experiences["reflections"]:
        assert not reflection


def test_expel_experience_memory_clear(expel_experiences_10_fake_path: str) -> None:
    """Test ExpeLExperienceMemory clear method."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    _ = experiences.pop("idxs")
    memory = ExpeLExperienceMemory(**experiences)
    memory.clear()

    for k, v in memory.experiences.items():
        assert isinstance(k, str)
        assert isinstance(v, list)
        assert len(v) == 0
    assert list(memory.experiences.keys()) == [
        "idxs",
        "questions",
        "keys",
        "trajectories",
        "reflections",
    ]


def test_expel_experience_memory_show_memories() -> None:
    """Test ExpeLExperienceMemory show_memories method."""
    memory = ExpeLExperienceMemory()
    memories = memory.show_memories()
    for k, v in memories.items():
        assert isinstance(k, str)
        assert isinstance(v, list)
        assert len(v) == 0
    assert list(memory.experiences.keys()) == [
        "idxs",
        "questions",
        "keys",
        "trajectories",
        "reflections",
    ]


def test_expel_experience_memory_show_memories() -> None:
    pass
