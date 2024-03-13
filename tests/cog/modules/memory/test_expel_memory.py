"""Unit tests for ExpeL memory module."""

import re
import joblib

from langchain_core.embeddings import Embeddings
from tiktoken.core import Encoding
from discussion_agents.cog.modules.memory.expel import ExpeLExperienceMemory
from discussion_agents.cog.prompts.react import REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES

fewshot_questions = re.findall(r'Question: (.+?)\n', REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES)
fewshot_keys = re.findall(r'Action \d+: Finish\[(.+?)\]', REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES)
blocks = re.split(r'(?=Question: )', REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES)[1:]  # Split and ignore the first empty result

fewshot_examples = []
for block in blocks:
    # Extract all thoughts, actions, and observations within each block
    thoughts = re.findall(r'(Thought \d+: .+?)\n', block)
    actions = re.findall(r'(Action \d+: .+?)\n', block)
    observations = re.findall(r'(Observation \d+: .+)', block)
    
    # Combine them into tuples and add to the examples list
    fewshot_examples.append(list(zip(thoughts, actions, observations)))


def test_expel_experience_memory_init(expel_experiences_10_fake_path: str) -> None:
    """Test ExpeLExperienceMemory initialization."""
    experiences = joblib.load(expel_experiences_10_fake_path)

    # Test empty initialization.
    memory = ExpeLExperienceMemory()
    assert memory.experiences == {'idxs': [], 'questions': [], 'keys': [], 'trajectories': [], 'reflections': []}
    assert not memory.fewshot_questions
    assert not memory.fewshot_keys
    assert not memory.fewshot_examples
    assert memory.strategy == "task"
    assert memory.reranker_strategy is None
    assert isinstance(memory.embedder, Embeddings)
    assert memory.k_docs == 24
    assert isinstance(memory.encoder, Encoding)
    assert memory.max_fewshot_tokens == 500
    assert memory.num_fewshots == 6
    assert not memory.success_traj_docs
    assert memory.vectorstore is None

    # Test with experiences parameter.
    memory = ExpeLExperienceMemory(experiences)
    assert memory.experiences == experiences
    assert not memory.fewshot_questions
    assert not memory.fewshot_keys
    assert not memory.fewshot_examples
    assert memory.strategy == "task"
    assert memory.reranker_strategy is None
    assert isinstance(memory.embedder, Embeddings)
    assert memory.k_docs == 24
    assert isinstance(memory.encoder, Encoding)
    assert memory.max_fewshot_tokens == 500
    assert memory.num_fewshots == 6
    assert len(memory.success_traj_docs) == 38
    assert memory.vectorstore
    
    success_traj_doc_types = [
        "task",
        "action",
        "action",
        "action",
        "action",
        "action",
        "action",
        "thought",
        "thought",
        "thought",
        "thought",
        "thought",
        "thought",
        "step",
        "step",
        "step",
        "step",
        "step",
        "step",
    ] * 2

    for type_, doc in zip(success_traj_doc_types, memory.success_traj_docs):
        assert type_ == doc.metadata['type']

    # Test with no experiences and fewshot examples.
    memory = ExpeLExperienceMemory(
        fewshot_questions=fewshot_questions,
        fewshot_keys=fewshot_keys,
        fewshot_examples=fewshot_examples
    )
    assert list(memory.experiences.keys()) == ['idxs', 'questions', 'keys', 'trajectories', 'reflections']
    for v in memory.experiences.values():
        assert len(v) == 6
    assert memory.fewshot_questions
    assert memory.fewshot_keys
    assert memory.fewshot_examples
    assert memory.strategy == "task"
    assert memory.reranker_strategy is None
    assert isinstance(memory.embedder, Embeddings)
    assert memory.k_docs == 24
    assert isinstance(memory.encoder, Encoding)
    assert memory.max_fewshot_tokens == 500
    assert memory.num_fewshots == 6
    assert len(memory.success_traj_docs) == 48
    assert memory.vectorstore

    # Test with experiences and fewshot examples.
    memory = ExpeLExperienceMemory(
        experiences=experiences,
        fewshot_questions=fewshot_questions,
        fewshot_keys=fewshot_keys,
        fewshot_examples=fewshot_examples
    )
    assert list(memory.experiences.keys()) == ['idxs', 'questions', 'keys', 'trajectories', 'reflections']
    for v in memory.experiences.values():
        assert len(v) == 16
    assert memory.fewshot_questions
    assert memory.fewshot_keys
    assert memory.fewshot_examples
    assert memory.strategy == "task"
    assert memory.reranker_strategy is None
    assert isinstance(memory.embedder, Embeddings)
    assert memory.k_docs == 24
    assert isinstance(memory.encoder, Encoding)
    assert memory.max_fewshot_tokens == 500
    assert memory.num_fewshots == 6
    assert len(memory.success_traj_docs) == 86
    assert memory.vectorstore


def test_expel_experience_memory_clear(expel_experiences_10_fake_path: str) -> None:
    """Test ExpeLExperienceMemory clear method."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    memory = ExpeLExperienceMemory(experiences)
    assert memory.experiences
    assert memory.success_traj_docs
    assert memory.vectorstore
    memory.clear()
    for v in memory.experiences.values():
        assert not v
    assert not memory.success_traj_docs
    assert not memory.vectorstore


def test_expel_experience_memory_add_memories(expel_experiences_10_fake_path: str) -> None:
    """Test ExpeLExperienceMemory add_memories method."""
    pass


def test_expel_experience_memory__fewshot_doc_token_count(expel_experiences_10_fake_path: str) -> None:
    """Test ExpeLExperienceMemory _fewshot_doc_token_count method."""
    experiences = joblib.load(expel_experiences_10_fake_path)

    # Testing with just experiences (1 success, a dupe).
    memory = ExpeLExperienceMemory(experiences)
    for doc in memory.success_traj_docs:
        token_count = memory._fewshot_doc_token_count(doc)
        assert token_count == 1245

    # Testing with fewshots only.
    gt_token_counts = [273] * 13 + [149] * 7 + [156] * 7 + [163] * 7 + [134] * 7 + [154] * 7
    memory = ExpeLExperienceMemory(
        fewshot_questions=fewshot_questions,
        fewshot_keys=fewshot_keys,
        fewshot_examples=fewshot_examples
    )
    for gt_token_count, doc in zip(gt_token_counts, memory.success_traj_docs):
        token_count = memory._fewshot_doc_token_count(doc)
        assert gt_token_count == token_count


def test_expel_experience_memory_load_memories(expel_experiences_10_fake_path: str) -> None:
    """Test ExpeLExperienceMemory load_memories method."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    memory = ExpeLExperienceMemory(experiences)

    # Test when memory is empty.

    # Test with every query type.

    # Test with every reranking strategy + error.

    # Test with varying max_fewshot_tokens.
    
    # Test with varying num_fewshots.


def test_expel_experience_memory_show_memories(expel_experiences_10_fake_path: str) -> None:
    """Test ExpeLExperienceMemory show_memories method."""
    experiences = joblib.load(expel_experiences_10_fake_path)

    # Test with empty memory.
    memory = ExpeLExperienceMemory()
    memory_dict = memory.show_memories()
    assert list(memory_dict.keys()) == ["experiences", "success_traj_docs", "vectorstore"]
    assert memory_dict['experiences'] == {'idxs': [], 'questions': [], 'keys': [], 'trajectories': [], 'reflections': []}
    assert not memory_dict['success_traj_docs']
    assert not memory_dict['vectorstore']

    # Test with non-empty memory.
    memory = ExpeLExperienceMemory(experiences)
    memory_dict = memory.show_memories()
    assert list(memory_dict.keys()) == ["experiences", "success_traj_docs", "vectorstore"]
    assert memory.experiences == memory_dict['experiences']
    assert len(memory_dict['success_traj_docs']) == 38
    assert memory_dict['vectorstore']
