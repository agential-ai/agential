"""Unit tests for ReAct memory module."""

from agential.cog.modules.memory.react import ReActMemory


def test_clear() -> None:
    """Test clear function."""
    memory = ReActMemory()
    assert memory.scratchpad == ""
    memory.scratchpad = "Text"
    assert memory.scratchpad
    memory.clear()
    assert memory.scratchpad == ""


def test_add_memories() -> None:
    """Test add_memories function."""
    memory = ReActMemory()
    assert memory.scratchpad == ""
    memory.add_memories(observation="Some text")
    assert memory.scratchpad == "Some text"


def test_load_memories() -> None:
    """Test load_memories function."""
    memory = ReActMemory()
    out = memory.load_memories()
    assert "scratchpad" in out
    assert out["scratchpad"] == ""
    memory.scratchpad = "Text"
    out = memory.load_memories()
    assert "scratchpad" in out
    assert out["scratchpad"] == "Text"


def test_show_memories() -> None:
    """Test show_memories function."""
    memory = ReActMemory()
    out = memory.show_memories()
    assert "scratchpad" in out
    assert out["scratchpad"] == ""
    memory.scratchpad = "Text"
    out = memory.show_memories()
    assert "scratchpad" in out
    assert out["scratchpad"] == "Text"
