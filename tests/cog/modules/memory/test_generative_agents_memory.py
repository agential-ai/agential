"""Unit tests for Generative Agents memory module."""

import faiss

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from langchain.retrievers import TimeWeightedVectorStoreRetriever

from discussion_agents.cog.modules.memory.generative_agents import GenerativeAgentMemory

def test_clear(memory_retriever: TimeWeightedVectorStoreRetriever) -> None:
    mem = GenerativeAgentMemory(retriever=memory_retriever)
    mem.retriever.memory_stream = ["Populating memory stream..."]

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2", 
        model_kwargs={"device": "cpu"}, 
        encode_kwargs={"normalize_embeddings": False}
    )
    index = faiss.IndexFlatL2(768)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, otherScoreKeys=["importance"], k=5
    )

    mem.clear(retriever=retriever)
    assert mem.retriever.memory_stream == []