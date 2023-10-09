from datetime import datetime
import asyncio

import faiss

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.schema import Document

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

from langchain.chat_models import ChatOpenAI
from langchain.retrievers import TimeWeightedVectorStoreRetriever

from discussion_agents.memory.generative_agents import GenerativeAgentMemory

import warnings

warnings.filterwarnings("ignore")

import dotenv
import os

dotenv.load_dotenv(".env")
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=openai_api_key, max_tokens=3000)

embedding_size = 768  # Embedding dimension for all-mpnet-base-v2. FAISS needs the same count.
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

observations = [
    "Tommie wakes up to the sound of a noisy construction site outside his window.",
    "Tommie gets out of bed and heads to the kitchen to make himself some coffee.",
    "Tommie realizes he forgot to buy coffee filters and starts rummaging through his moving boxes to find some.",
    "Tommie finally finds the filters and makes himself a cup of coffee.",
    "The coffee tastes bitter, and Tommie regrets not buying a better brand.",
    "Tommie checks his email and sees that he has no job offers yet."
]

test_date = datetime(year=2022, month=11, day=14, hour=3, minute=14)

def create_memory_retriever():
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore,
        otherScoreKeys=["importance"],
        k=5
    )
    return retriever

def test_generative_agents_memory():
    # Test instantiation.
    memory = GenerativeAgentMemory(
        llm=llm,
        memory_retriever=create_memory_retriever(),
        verbose=False,
        reflection_threshold=8,
    )   
    # # Test score_memories_importance.
    # scores = memory.score_memories_importance(memory_contents=observations, verbose=False)
    # assert len(scores) == len(observations)
    # for score in scores: assert type(score) is float

    # # Test add_memories.
    # hashes = memory.add_memories(observations)
    # assert type(hashes) is list
    # assert type(memory.memory_retriever.memory_stream) is list
    # assert len(memory.memory_retriever.memory_stream) == 6

    # # Test pause_to_reflect.
    # insights = memory.pause_to_reflect(
    #     last_k=50, 
    #     verbose=False,
    #     now=None
    # )
    # assert type(insights) is list

    # Test get_memories_until_limit.
    docs = []
    for i in range(1, 6):
        docs.append(
            Document(
                page_content="Some text." * i,
                metadata={
                    "created_at": test_date,
                    "importance": 0.15
                }
            )
        )
    memory.memory_retriever.memory_stream.extend(docs)

    mem_str = memory.get_memories_until_limit(consumed_tokens=10)
    assert type(mem_str) is str

    # Test memory_variables.
    assert type(memory.memory_variables) is list
    assert not memory.memory_variables

    # Test load_memory_variables.
    

    # Test save_context.

    # Test clear.