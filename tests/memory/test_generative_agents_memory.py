"""Unit tests for GenerativeAgentMemory."""
import warnings

from datetime import datetime

import pytest

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

from discussion_agents.core.base import BaseCore
from discussion_agents.memory.generative_agents import GenerativeAgentMemory
from tests.fixtures.retriever import memory_retriever

warnings.filterwarnings("ignore")

import os

import dotenv

dotenv.load_dotenv(".env")
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=openai_api_key, max_tokens=3000)

embedding_size = (
    768  # Embedding dimension for all-mpnet-base-v2. FAISS needs the same count.
)
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

observations = [
    "Tommie wakes up to the sound of a noisy construction site outside his window.",
    "Tommie gets out of bed and heads to the kitchen to make himself some coffee.",
    "Tommie realizes he forgot to buy coffee filters and starts rummaging through his moving boxes to find some.",
    "Tommie finally finds the filters and makes himself a cup of coffee.",
    "The coffee tastes bitter, and Tommie regrets not buying a better brand.",
    "Tommie checks his email and sees that he has no job offers yet.",
]

test_date = datetime(year=2022, month=11, day=14, hour=3, minute=14)

core = BaseCore(llm=llm, retriever=memory_retriever())


@pytest.mark.cost
def test_score_memories_importance() -> None:
    """Tests score_memories_importance in GenerativeAgentMemory."""
    # Test instantiation.
    memory = GenerativeAgentMemory(
        core=core,
        reflection_threshold=8,
    )
    # Test score_memories_importance.
    scores = memory.score_memories_importance(
        memory_contents=observations[0], relevant_memories=observations[1:]
    )
    assert len(scores) == 1
    for score in scores:
        assert type(score) is float


@pytest.mark.cost
def test_add_memories() -> None:
    """Tests add_memories."""
    # Test instantiation.
    memory = GenerativeAgentMemory(
        core=core,
        reflection_threshold=8,
    )

    # Test add_memories.
    hashes = memory.add_memories(observations)
    assert type(hashes) is list
    assert type(memory.core.retriever.memory_stream) is list
    assert len(memory.core.retriever.memory_stream) == 6


@pytest.mark.cost
@pytest.mark.slow
def test_pause_to_reflect() -> None:
    """Tests pause_to_reflect."""
    # Test instantiation.
    memory = GenerativeAgentMemory(
        core=core,
        reflection_threshold=8,
    )

    # Test pause_to_reflect.
    topics, insights = memory.pause_to_reflect(last_k=50, now=None)
    assert type(insights) is list
    assert type(topics) is list


def test_get_memories_until_limit() -> None:
    """Tests get_memories_until_limit."""
    # Test instantiation.
    memory = GenerativeAgentMemory(
        core=core,
        reflection_threshold=8,
    )

    # Test get_memories_until_limit.
    docs = []
    for i in range(1, 6):
        docs.append(
            Document(
                page_content="Some text." * i,
                metadata={
                    "created_at": test_date,
                    "importance": 0.15,
                    "buffer_idx": i - 1,
                },
            )
        )
    memory.core.retriever.memory_stream.extend(docs)

    mem_str = memory.get_memories_until_limit(consumed_tokens=10)
    assert type(mem_str) is str


def test_memory_variables() -> None:
    """Tests memory_variables property."""
    # Test instantiation.
    memory = GenerativeAgentMemory(
        core=core,
        reflection_threshold=8,
    )

    # Test memory_variables.
    assert type(memory.memory_variables) is list
    assert not memory.memory_variables


def test_load_memory_variables_empty() -> None:
    """Tests load_memory_variables when input is empty."""
    # Test instantiation.
    memory = GenerativeAgentMemory(
        core=core,
        reflection_threshold=8,
    )

    # Test load_memory_variables.
    empty = memory.load_memory_variables({})
    assert empty == {}


@pytest.mark.cost
def test_load_memory_variables_query() -> None:
    """Tests load_memory_variables when query is supplied in input."""
    # Test instantiation.
    memory = GenerativeAgentMemory(
        core=core,
        reflection_threshold=8,
    )

    mem_detail_simple = memory.load_memory_variables(
        inputs={memory.queries_key: ["Some text."]}
    )
    assert type(mem_detail_simple) is dict
    assert type(mem_detail_simple[memory.relevant_memories_key]) is str
    assert type(mem_detail_simple[memory.relevant_memories_simple_key]) is str


def test_load_memory_variables_relevant() -> None:
    """Tests load_memory_variables when most_recent_memories_token_key is supplied in input."""
    # Test instantiation.
    memory = GenerativeAgentMemory(
        core=core,
        reflection_threshold=8,
    )

    relevant_mem = memory.load_memory_variables(
        inputs={memory.most_recent_memories_token_key: 0}
    )
    assert type(relevant_mem[memory.most_recent_memories_key]) is str


@pytest.mark.cost
def test_save_context() -> None:
    """Tests save_context."""
    # Test instantiation.
    memory = GenerativeAgentMemory(
        core=core,
        reflection_threshold=8,
    )

    # Test save_context.
    memory.save_context(
        inputs={},
        outputs={memory.add_memory_key: "Some memory.", memory.now_key: test_date},
    )


@pytest.mark.cost
def test_clear() -> None:
    """Tests clear."""
    # Test instantiation.
    memory = GenerativeAgentMemory(
        core=core,
        reflection_threshold=8,
    )

    # Test clear.
    _ = memory.clear()
