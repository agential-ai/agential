"""Fixtures for creating agents."""

import pytest
import yaml

from langchain.llms.fake import FakeListLLM
from langchain_community.chat_models.fake import FakeListChatModel

from discussion_agents.cog.agent.generative_agents import GenerativeAgent
from discussion_agents.cog.agent.react import ReActAgent
from discussion_agents.cog.agent.reflexion import ReflexionCoTAgent, ReflexionReActAgent
from pathlib import Path

import faiss

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS




@pytest.fixture
def generative_agent() -> GenerativeAgent:
    """Creates a GenerativeAgent."""
    agent = GenerativeAgent(llm=FakeListLLM(responses=["1"]))
    return agent


@pytest.fixture
def react_agent() -> ReActAgent:
    """Creates a ReActAgent."""
    agent = ReActAgent(llm=FakeListLLM(responses=["1"]))
    return agent


@pytest.fixture
def reflexion_cot_agent() -> ReflexionCoTAgent:
    """Creates a ReflexionCoTAgent."""
    agent = ReflexionCoTAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]),
        action_llm=FakeListChatModel(responses=["1"]),
    )
    return agent


@pytest.fixture
def reflexion_react_agent() -> ReflexionReActAgent:
    """Creates a ReflexionReActAgent."""
    agent = ReflexionReActAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]),
        action_llm=FakeListChatModel(responses=["1"]),
    )
    return agent


@pytest.fixture
def data_dir(pytestconfig):
    """Dir path to asset."""
    return Path(pytestconfig.rootdir) / "tests/assets"

@pytest.fixture
def alfworld_file(data_dir):
    """Dir path to Alfworld environement file."""
    return Path(data_dir) / "base_config.yaml"

@pytest.fixture
def alfworld_env(alfworld_file):
    """Prepare for env init for Alfworld."""
    with open(alfworld_file) as reader:
        config = yaml.safe_load(reader) 
    return config

"""Fixtures for creating retrievers."""


embedding_size = (
    768  # Embedding dimension for all-mpnet-base-v2. FAISS needs the same count.
)
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}


@pytest.fixture
def time_weighted_retriever() -> TimeWeightedVectorStoreRetriever:
    """Creates a TimeWeightedVectorStoreRetriever."""
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, otherScoreKeys=["importance"], k=5
    )
    return retriever


