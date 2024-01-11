"""Unit tests for Reflexion memory module."""

from discussion_agents.cog.modules.memory.reflexion import ReflexionMemory


def test_clear() -> None:
    """Test clear function."""
    memory = ReflexionMemory()
    assert memory.scratchpad == ""
    memory.scratchpad = "Text"
    assert memory.scratchpad
    memory.clear()
    assert memory.scratchpad == ""


def test_add_memories() -> None:
    """Test add_memories function."""
    memory = ReflexionMemory()
    assert memory.scratchpad == ""
    memory.add_memories(observation="Some text")
    assert memory.scratchpad == "Some text"


def test_load_memories() -> None:
    """Test load_memories function."""
    memory = ReflexionMemory()
    out = memory.load_memories()
    assert "scratchpad" in out
    assert out["scratchpad"] == ""
    memory.scratchpad = "Text"
    out = memory.load_memories()
    assert "scratchpad" in out
    assert out["scratchpad"] == "Text"


def test_show_memories() -> None:
    """Test show_memories function."""
    memory = ReflexionMemory()
    out = memory.show_memories()
    assert "scratchpad" in out
    assert out["scratchpad"] == ""
    memory.scratchpad = "Text"
    out = memory.show_memories()
    assert "scratchpad" in out
    assert out["scratchpad"] == "Text"
