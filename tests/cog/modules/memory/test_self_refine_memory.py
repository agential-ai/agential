"""Unit tests for Self-Refine memory module."""

from agential.cog.modules.memory.self_refine import SelfRefineMemory


def test_clear() -> None:
    """Test clear function."""
    memory = SelfRefineMemory(solution=["solution1"], feedback=["feedback1"])
    memory.clear()
    assert memory.solution == []
    assert memory.feedback == []


def test_add_memories() -> None:
    """Test add_memories function."""
    memory = SelfRefineMemory()
    memory.add_memories("solution1", "feedback1")
    assert memory.solution == ["solution1"]
    assert memory.feedback == ["feedback1"]


def test_load_memories() -> None:
    """Test load_memories function."""
    memory = SelfRefineMemory()
    memories = memory.load_memories()
    assert memories["solution"] == []
    assert memories["feedback"] == []


def test_show_memories() -> None:
    """Test show_memories function."""
    memory = SelfRefineMemory(solution=["solution1"], feedback=["feedback1"])
    displayed_memories = memory.show_memories()
    assert displayed_memories["solution"] == ["solution1"]
    assert displayed_memories["feedback"] == ["feedback1"]
