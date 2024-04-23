"""Unit tests for Generative Agents memory module."""

from datetime import datetime

import faiss
import pytest

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.fake import FakeListLLM
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from agential.cog.modules.memory.generative_agents import GenerativeAgentMemory

test_date = datetime(year=2022, month=11, day=14, hour=3, minute=14)


def test_clear(time_weighted_retriever: TimeWeightedVectorStoreRetriever) -> None:
    """Test clear method."""
    mem = GenerativeAgentMemory(retriever=time_weighted_retriever)
    mem.retriever.memory_stream = ["Populating memory stream..."]

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
    index = faiss.IndexFlatL2(768)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, otherScoreKeys=["importance"], k=5
    )

    mem.clear(retriever=retriever)
    assert mem.retriever.memory_stream == []


def test_add_memories(
    time_weighted_retriever: TimeWeightedVectorStoreRetriever,
) -> None:
    """Test add_memories."""
    mem = GenerativeAgentMemory(retriever=time_weighted_retriever)
    mem.add_memories(
        memory_contents="An observation.", importance_scores=0.1, now=test_date
    )
    assert len(mem.retriever.memory_stream) == 1
    assert mem.retriever.memory_stream[0].page_content == "An observation."
    assert mem.retriever.memory_stream[0].metadata["last_accessed_at"] == test_date
    assert mem.retriever.memory_stream[0].metadata["created_at"] == test_date
    assert mem.retriever.memory_stream[0].metadata["importance"] == 0.1

    obs = ["Another observation.", "Yet another observation."]
    scores = [0.3, 0.4]

    mem.add_memories(memory_contents=obs, importance_scores=scores, now=test_date)

    for i, m in enumerate(mem.retriever.memory_stream[1:]):
        assert m.page_content == obs[i]
        assert m.metadata["last_accessed_at"] == test_date
        assert m.metadata["created_at"] == test_date
        assert m.metadata["importance"] == scores[i]

    memory_contents = ["memory1", "memory2"]
    importance_scores = [0.5]

    with pytest.raises(ValueError) as excinfo:
        mem.add_memories(memory_contents, importance_scores)
    assert (
        str(excinfo.value)
        == "The length of memory_contents must match the length of importance_scores."
    )


def test_load_memories(
    time_weighted_retriever: TimeWeightedVectorStoreRetriever,
) -> None:
    """Test load_memories."""
    mem = GenerativeAgentMemory(retriever=time_weighted_retriever)
    obs = [f"An observation {i}" for i in range(51)]
    scores = [0.1] * 51

    mem.add_memories(obs, scores)
    out = mem.load_memories(queries="An observation")
    assert isinstance(out["relevant_memories"], list)

    out = mem.load_memories(last_k=1)
    assert isinstance(out["most_recent_memories"], list)
    assert len(out["most_recent_memories"]) == 1
    assert out["most_recent_memories"][0].page_content == "An observation 50"

    with pytest.raises(ValueError) as excinfo:
        mem.load_memories(consumed_tokens=1)
    assert (
        str(excinfo.value)
        == "max_tokens_limit and llm must be defined if consumed_tokens is defined."
    )

    with pytest.raises(ValueError) as excinfo:
        mem.load_memories(consumed_tokens=1, max_tokens_limit=1)
    assert (
        str(excinfo.value)
        == "max_tokens_limit and llm must be defined if consumed_tokens is defined."
    )

    with pytest.raises(ValueError) as excinfo:
        mem.load_memories(consumed_tokens=1, llm=FakeListLLM(responses=[""]))
    assert (
        str(excinfo.value)
        == "max_tokens_limit and llm must be defined if consumed_tokens is defined."
    )

    out = mem.load_memories(
        consumed_tokens=1, max_tokens_limit=5, llm=FakeListLLM(responses=[""])
    )
    assert isinstance(out["most_recent_memories_limit"], list)
    assert len(out["most_recent_memories_limit"]) == 1
    assert out["most_recent_memories_limit"][0].page_content == "An observation 50"
    assert out["most_recent_memories_limit"][0].metadata["importance"] == 0.1


def test_show_memories(
    time_weighted_retriever: TimeWeightedVectorStoreRetriever,
) -> None:
    """Test show_memories."""
    mem = GenerativeAgentMemory(retriever=time_weighted_retriever)
    out = mem.show_memories()
    assert out["memory_stream"] == []
