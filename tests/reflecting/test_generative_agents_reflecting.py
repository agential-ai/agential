"""Unit Tests for Generative Agents reflecting modules."""
import os

from datetime import datetime

import dotenv
import faiss

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from discussion_agents.reflecting.generative_agents import (
    get_insights_on_topic,
    get_topics_of_reflection,
    reflect,
)

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

test_date = datetime(year=2022, month=11, day=14, hour=3, minute=14)


def create_memory_retriever():
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


def test_get_topics_of_reflection():
    """Tests get_topics_of_reflection."""
    topics = get_topics_of_reflection(
        llm=llm, memory_retriever=create_memory_retriever(), verbose=False, last_k=10
    )
    assert type(topics) is list


def test_get_insights_on_topic():
    """Tests get_insights_on_topic."""
    insights = get_insights_on_topic(
        llm=llm,
        memory_retriever=create_memory_retriever(),
        topics="Some topic.",
        now=test_date,
        verbose=False,
    )
    assert type(insights) is list
    assert type(insights[0]) is list


def test_reflect():
    """Tests reflect."""
    insights = reflect(
        llm=llm,
        memory_retriever=create_memory_retriever(),
        last_k=10,
        verbose=False,
        now=test_date,
    )

    assert type(insights) is list
