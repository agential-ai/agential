"""Unit tests for BaseCoreWithMemory."""

import os

import dotenv

from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

from discussion_agents.core.memory import BaseCoreWithMemory

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


def test_base_core_with_memory(memory_retriever: BaseRetriever) -> None:
    """Test BaseCore & chain method."""
    template = """You are a chatbot having a conversation with a human.

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")

    core = BaseCoreWithMemory(
        llm=llm, llm_kwargs={}, retriever=memory_retriever, memory=memory
    )

    chain = core.chain(prompt=prompt)
    out = chain.predict(human_input="Tell me a joke!")
    assert type(out) is str
