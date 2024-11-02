"""Tests for the CLINMemory class."""

from agential.agents.clin.memory import CLINMemory


def test_memory_initialization() -> None:
    """Tests memory initialization with default values."""
    memory = CLINMemory()
    assert memory.k == 10
    assert memory.memories == {}
    assert memory.meta_summaries == {}
    assert memory.history == []


def test_memory_clear() -> None:
    """Tests memory clear method."""
    memory = CLINMemory()
    memory.memories["sample_question"] = [{"summaries": "sample_summary"}]
    memory.meta_summaries["sample_question"] = ["sample_meta_summary"]
    memory.history.append("sample_question")
    memory.clear()
    assert memory.memories == {}
    assert memory.meta_summaries == {}
    assert memory.history == []


def test_add_memories() -> None:
    """Tests adding memories to the CLINMemory."""
    memory = CLINMemory()
    memory.add_memories(
        question="What is AI?",
        summaries="AI is artificial intelligence.",
        eval_report="Good response.",
        is_correct=True,
    )
    assert "What is AI?" in memory.memories
    assert len(memory.memories["What is AI?"]) == 1
    assert (
        memory.memories["What is AI?"][0]["summaries"]
        == "AI is artificial intelligence."
    )
    assert (
        "EVALUATION REPORT: Good response."
        in memory.memories["What is AI?"][0]["trial"]
    )
    assert memory.memories["What is AI?"][0]["is_correct"] == True


def test_add_meta_summaries() -> None:
    """Tests adding meta-summaries to the CLINMemory."""
    memory = CLINMemory()
    memory.add_meta_summaries(
        question="What is AI?", meta_summaries="Meta-summary for AI."
    )
    assert "What is AI?" in memory.meta_summaries
    assert memory.meta_summaries["What is AI?"][-1] == "Meta-summary for AI."
    assert memory.history[-1] == "What is AI?"


def test_load_memories_no_existing_question() -> None:
    """Tests loading memories for a non-existing question."""
    memory = CLINMemory()
    result = memory.load_memories("Non-existing question")
    assert result == {"previous_trials": "", "latest_summaries": ""}


def test_load_memories_with_existing_question() -> None:
    """Tests loading memories for an existing question."""
    memory = CLINMemory()
    memory.add_memories(
        question="What is AI?",
        summaries="AI is artificial intelligence.",
        eval_report="Good response.",
        is_correct=True,
    )
    result = memory.load_memories("What is AI?")
    assert "previous_trials" in result
    assert "latest_summaries" in result
    assert "AI is artificial intelligence." in result["latest_summaries"]
    assert "EVALUATION REPORT: Good response." in result["previous_trials"]


def test_load_meta_summaries_empty() -> None:
    """Tests loading meta-summaries when none are available."""
    memory = CLINMemory()
    result = memory.load_meta_summaries()
    assert result == {"meta_summaries": ""}


def test_load_meta_summaries_with_data() -> None:
    """Tests loading meta-summaries with data present."""
    memory = CLINMemory(k=2)
    memory.add_meta_summaries("What is AI?", "Meta-summary for AI.")
    memory.add_meta_summaries("What is ML?", "Meta-summary for ML.")
    result = memory.load_meta_summaries()
    assert "Meta-summary for AI." in result["meta_summaries"]
    assert "Meta-summary for ML." in result["meta_summaries"]


def test_show_memories() -> None:
    """Tests showing all memories."""
    memory = CLINMemory()
    memory.add_memories(
        question="What is AI?",
        summaries="AI is artificial intelligence.",
        eval_report="Good response.",
        is_correct=True,
    )
    result = memory.show_memories()
    assert "memories" in result
    assert "What is AI?" in result["memories"]
    assert (
        result["memories"]["What is AI?"][0]["summaries"]
        == "AI is artificial intelligence."
    )
