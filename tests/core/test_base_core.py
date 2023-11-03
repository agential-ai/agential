"""Unit tests for base core."""

import os

import dotenv
import pytest

from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.memory import BaseMemory
from langchain.schema.retriever import BaseRetriever

from discussion_agents.core.base import BaseCore

dotenv.load_dotenv(".env")
huggingface_hub_api_key = os.getenv("HUGGINGFACE_HUB_API_KEY")

llm = HuggingFaceHub(
    repo_id="gpt2",
    huggingfacehub_api_token=huggingface_hub_api_key,
)

embedding_size = (
    768  # Embedding dimension for all-mpnet-base-v2. FAISS needs the same count.
)
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}


def test_chain_method(memory_retriever: BaseRetriever) -> None:
    """Test the chain method of BaseCore."""
    core = BaseCore(llm=llm, llm_kwargs={}, retriever=memory_retriever)

    prompt = PromptTemplate.from_template(
        "Explain the importance of eating a proper meal."
    )

    # Test chain.
    chain = core.chain(prompt=prompt)
    out = chain.run({})
    assert type(out) is str


def test_get_llm_and_set_llm(memory_retriever: BaseRetriever) -> None:
    """Test getters and setters for the language model in BaseCore."""
    core = BaseCore(llm=llm, llm_kwargs={}, retriever=memory_retriever)

    assert isinstance(core.get_llm(), BaseLanguageModel)

    core.llm = None

    with pytest.raises(TypeError):
        _ = core.get_llm()

    assert (core.set_llm(llm)) is None

    with pytest.raises(TypeError):
        core.set_llm(None)


def test_get_llm_kwargs_and_set_llm_kwargs(memory_retriever: BaseRetriever) -> None:
    """Test getters and setters for the language model kwargs in BaseCore."""
    core = BaseCore(llm=llm, llm_kwargs={}, retriever=memory_retriever)

    assert isinstance(core.get_llm_kwargs(), dict)

    core.llm_kwargs = None

    with pytest.raises(TypeError):
        _ = core.get_llm_kwargs()

    assert (core.set_llm_kwargs({})) is None

    with pytest.raises(TypeError):
        core.set_llm_kwargs([])


def test_get_retriever_and_set_retriever(memory_retriever: BaseRetriever) -> None:
    """Test getters and setters for the retriever in BaseCore."""
    core = BaseCore(llm=llm, llm_kwargs={}, retriever=memory_retriever)

    assert isinstance(core.get_retriever(), BaseRetriever)

    core.retriever = None

    with pytest.raises(TypeError):
        _ = core.get_retriever()

    assert (core.set_retriever(memory_retriever)) is None

    with pytest.raises(TypeError):
        core.set_retriever("invalid input")


def test_memory_related_methods(memory_retriever: BaseRetriever) -> None:
    """Test memory-related methods of BaseCore."""
    core = BaseCore(
        llm=llm,
        llm_kwargs={},
        retriever=memory_retriever,
        memory=ConversationBufferMemory(),
    )

    assert isinstance(core.get_memory(), BaseMemory)

    core.memory = None
    with pytest.raises(TypeError):
        _ = core.get_memory()

    with pytest.raises(TypeError):
        core.set_memory(None)

    assert (core.set_memory(ConversationBufferMemory())) is None
