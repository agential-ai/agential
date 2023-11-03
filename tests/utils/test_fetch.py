"""Unit tests for utility fetch memory functions."""
import os

from datetime import datetime

import dotenv

from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.schema import BaseRetriever

from discussion_agents.utils.fetch import fetch_memories

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


def test_fetch_memories(memory_retriever: BaseRetriever) -> None:
    """Test fetch_memories."""
    observation = "Some observation."

    memories = fetch_memories(
        observation=observation,
        memory_retriever=memory_retriever,
        now=test_date,
    )
    assert type(memories) is list
