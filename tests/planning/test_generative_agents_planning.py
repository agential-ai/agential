import faiss

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from discussion_agents.memory.generative_agents import GenerativeAgentMemory
from discussion_agents.planning.generative_agents import (
    generate_broad_plan,
    update_status,
    update_broad_plan,
    generate_refined_plan,
)

import dotenv
import os

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

broad_plan = [
    'Choose a sturdy and durable material for the table, such as solid wood or metal',
    'Ensure that the table is properly constructed and assembled with strong joints and connections',
    'Verify that the table has a stable and level surface to prevent wobbling or tipping over',
    'Consider the weight capacity of the table to ensure it can support the intended load without any issues',
    'Regularly inspect and maintain the table, checking for any signs of wear, damage, or instability',
    'Avoid placing excessive weight or applying excessive force on the table to prevent potential damage',
    'Place the table on a suitable surface, such as a level floor, to provide additional stability',
    'Keep the table away from direct sunlight, extreme temperatures, or excessive moisture to prevent warping or deterioration',
    'Choose a table design that suits the intended purpose and environment, considering factors like size, shape, and functionality',
    'Consider purchasing a table from a reputable and trustworthy manufacturer or retailer to ensure quality and reliability'
]

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

def test_generate_broad_plan():
    instruction = "Describe what makes a table reliable."
    lifestyle = "lazy, likes to sleep late"
    name = "Bob"

    memory = GenerativeAgentMemory(
        llm=llm,
        memory_retriever=create_memory_retriever(),
        verbose=False,
        reflection_threshold=8,
    )

    broad_plan = generate_broad_plan(
        instruction=instruction,
        lifestyle=lifestyle,
        name=name,
        llm=llm,
        llm_kwargs={},
        memory=memory
    )
    assert type(broad_plan) is list
    for p in broad_plan:
        assert type(p) is str

def test_update_status():
    memory = GenerativeAgentMemory(
        llm=llm,
        memory_retriever=create_memory_retriever(),
        verbose=False,
        reflection_threshold=8,
    )

    status = update_status(
        previous_steps=broad_plan[:1],
        plan_step=broad_plan[1],
        name="Bob",
        status="Sturdy and durable tables are reliable.",
        llm=llm,
        llm_kwargs={},
        memory=memory,
    )
    assert type(status) is str

def test_update_broad_plan():
    memory = GenerativeAgentMemory(
        llm=llm,
        memory_retriever=create_memory_retriever(),
        verbose=False,
        reflection_threshold=8,
    )

    updated_broad_plan = update_broad_plan(
        instruction="Describe what makes a table reliable.", 
        name="Bob",
        plan=broad_plan,
        llm=llm,
        llm_kwargs={},
        memory=memory,
    )
    assert type(updated_broad_plan) is str

def test_generate_refined_plan():
    memory = GenerativeAgentMemory(
        llm=llm,
        memory_retriever=create_memory_retriever(),
        verbose=False,
        reflection_threshold=8,
    )

    refined_plan = generate_refined_plan(
        instruction="Describe what makes a table reliable.",
        plan=broad_plan,
        name="Bob",
        llm=llm,
        llm_kwargs={},
        memory=memory
    )
    assert type(refined_plan) is list

    