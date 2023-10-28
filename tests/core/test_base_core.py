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


def test_base_core(memory_retriever: BaseRetriever) -> None:
    """Test BaseCore & chain method."""
    core = BaseCore(llm=llm, llm_kwargs={}, retriever=memory_retriever)

    prompt = PromptTemplate.from_template(
        "Explain the importance of eating a proper meal."
    )

    # Test chain.
    chain = core.chain(prompt=prompt)
    out = chain.run({})
    assert type(out) is str

    # Test getters and setters.
    assert isinstance(core.get_llm(), BaseLanguageModel)
    assert isinstance(core.get_llm_kwargs(), dict)
    assert isinstance(core.get_retriever(), BaseRetriever)

    core.llm = None
    core.llm_kwargs = None
    core.retriever = None

    with pytest.raises(TypeError):
        _ = core.get_llm()

    with pytest.raises(TypeError):
        _ = core.get_llm_kwargs()

    with pytest.raises(TypeError):
        _ = core.get_retriever()

    core = BaseCore(llm=llm, llm_kwargs={}, retriever=memory_retriever)

    assert (core.set_llm(llm)) is None
    assert (core.set_llm_kwargs({})) is None
    assert (core.set_retriever(memory_retriever)) is None

    with pytest.raises(TypeError):
        core.set_llm(None)

    with pytest.raises(TypeError):
        core.set_llm_kwargs([])

    with pytest.raises(TypeError):
        core.set_retriever("invalid input")

    # Test memory.
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
