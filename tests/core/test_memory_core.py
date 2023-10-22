"""Unit tests for BaseCoreWithMemory."""

import os

import dotenv
import faiss

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

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


def test_base_core_with_memory():
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
        llm=llm, llm_kwargs={}, retriever=create_memory_retriever(), memory=memory
    )

    chain = core.chain(prompt=prompt)
    out = chain.predict(human_input="Tell me a joke!")
    assert type(out) is str
