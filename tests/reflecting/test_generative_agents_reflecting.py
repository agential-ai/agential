"""Unit Tests for Generative Agents reflecting modules."""
import os

from datetime import datetime

import dotenv
import pytest

from langchain.llms.huggingface_hub import HuggingFaceHub

from discussion_agents.core.base import BaseCore
from discussion_agents.reflecting.generative_agents import (
    get_insights_on_topics,
    get_topics_of_reflection,
    reflect,
)
from tests.fixtures.retriever import memory_retriever

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

core = BaseCore(llm=llm, retriever=memory_retriever())


@pytest.mark.slow
def test_get_topics_of_reflection() -> None:
    """Tests get_topics_of_reflection."""
    # Test observations string.
    observations = "This is an observation."
    topics = get_topics_of_reflection(observations=observations, core=core)
    assert type(topics) is list

    # Test observations list.
    observations = ["This is an observation."]
    topics = get_topics_of_reflection(observations=observations, core=core)
    assert type(topics) is list


def test_get_insights_on_topics() -> None:
    """Tests get_insights_on_topics."""
    # Test topics list and related_memories list.
    insights = get_insights_on_topics(
        topics=["Some topic."],
        related_memories=["This is another topic."],
        core=core,
    )
    assert type(insights) is list
    assert type(insights[0]) is list

    # Test topics str and related_memories str.
    insights = get_insights_on_topics(
        topics="Some topic.",
        related_memories="This is another topic.",
        core=core,
    )
    assert type(insights) is list
    assert type(insights[0]) is list

    # Test topics str and related_memories list.
    insights = get_insights_on_topics(
        topics="Some topic.",
        related_memories=["This is another topic."],
        core=core,
    )
    assert type(insights) is list
    assert type(insights[0]) is list

    # Test topics list and related_memories str.
    insights = get_insights_on_topics(
        topics=["Some topic."],
        related_memories="This is another topic.",
        core=core,
    )
    assert type(insights) is list
    assert type(insights[0]) is list


def test_reflect() -> None:
    """Tests reflect."""
    observations = "This is an observation."
    topics, insights = reflect(observations=observations, core=core, now=test_date)

    assert type(topics) is list
    assert type(insights) is list
    assert type(insights[0]) is list
