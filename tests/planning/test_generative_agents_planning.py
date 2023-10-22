import os

import dotenv
import faiss

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from discussion_agents.core.base import BaseCore
from discussion_agents.planning.generative_agents import (
    generate_broad_plan,
    generate_refined_plan,
    update_status,
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

broad_plan = [
    "Choose a sturdy and durable material for the table, such as solid wood or metal",
    "Ensure that the table is properly constructed and assembled with strong joints and connections",
    "Verify that the table has a stable and level surface to prevent wobbling or tipping over",
    "Consider the weight capacity of the table to ensure it can support the intended load without any issues",
    "Regularly inspect and maintain the table, checking for any signs of wear, damage, or instability",
    "Avoid placing excessive weight or applying excessive force on the table to prevent potential damage",
    "Place the table on a suitable surface, such as a level floor, to provide additional stability",
    "Keep the table away from direct sunlight, extreme temperatures, or excessive moisture to prevent warping or deterioration",
    "Choose a table design that suits the intended purpose and environment, considering factors like size, shape, and functionality",
    "Consider purchasing a table from a reputable and trustworthy manufacturer or retailer to ensure quality and reliability",
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

core = BaseCore(
    llm=llm,
    retriever=create_memory_retriever()
)

instruction = "Describe what makes a table reliable."
name = "Bob"
age = 25
traits = "talkative, enthusiastic"
lifestyle = "lazy, likes to sleep late"
status = ""
_summary = "Bob is a socialite with lots of friends and a heart for enthusiasm and extrovertedness."

summary = (
    f"Name: {name}\n"
    + f"Age: {age}\n"
    + f"Innate traits: {traits}\n"
    + f"Status: {status}\n"
    + f"Lifestyle: {lifestyle}\n"
    + f"{_summary}\n"
)

def test_generate_broad_plan():
    broad_plan = generate_broad_plan(
        instruction=instruction,
        summary=summary,
        core=core
    )
    assert type(broad_plan) is list
    for p in broad_plan:
        assert type(p) is str


def test_update_status():
    new_status = update_status(
        instruction=instruction,
        previous_steps=broad_plan[:1],
        plan_step=broad_plan[1],
        summary=summary,
        status=status,
        core=core
    )
    assert type(new_status) is str

def test_generate_refined_plan():
    refined_steps = generate_refined_plan(
        instruction=instruction,
        previous_steps=broad_plan[:1],
        plan_step=broad_plan[1],
        summary=summary,
        core=core
    )
    assert type(refined_steps) is list
