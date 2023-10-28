"""Unit tests for generative agent planning."""
import os

import dotenv
import pytest

from langchain.chat_models import ChatOpenAI
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.schema import BaseRetriever

from discussion_agents.core.base import BaseCore
from discussion_agents.planning.generative_agents import (
    generate_broad_plan,
    generate_refined_plan_step,
    update_status,
)
from tests.fixtures.retriever import memory_retriever

dotenv.load_dotenv(".env")
huggingface_hub_api_key = os.getenv("HUGGINGFACE_HUB_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

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

core = BaseCore(llm=llm, retriever=memory_retriever())


def test_generate_broad_plan() -> None:
    """Test generate_broad_plan."""
    broad_plan = generate_broad_plan(
        instruction=instruction, summary=summary, core=core
    )
    assert type(broad_plan) is list
    for p in broad_plan:
        assert type(p) is str


@pytest.mark.slow
def test_update_status() -> None:
    """Test update_status."""
    new_status = update_status(
        instruction=instruction,
        previous_steps=broad_plan[:1],
        plan_step=broad_plan[1],
        summary=summary,
        status=status,
        core=core,
    )
    assert type(new_status) is str


@pytest.mark.slow
def test_generate_refined_plan_step() -> None:
    """Test generate_refined_plan_step."""
    refined_steps = generate_refined_plan_step(
        instruction=instruction,
        previous_steps=broad_plan[:1],
        plan_step=broad_plan[1],
        summary=summary,
        core=core,
    )
    assert type(refined_steps) is list


@pytest.mark.cost
def test_generate_refined_plan_step_no_substep(memory_retriever: BaseRetriever) -> None:
    """Test generate_refined_plan_step where no substeps are required."""
    LLM = ChatOpenAI(openai_api_key=openai_api_key, max_tokens=1500)

    refined_steps = generate_refined_plan_step(
        instruction="Something",
        previous_steps=["Something"],
        plan_step="Something",
        summary=summary,
        core=BaseCore(llm=LLM, retriever=memory_retriever),
        k=1,
    )
    assert isinstance(refined_steps, list)
    assert len(refined_steps) == 1
    assert refined_steps[0] == "<NO_SUBSTEPS_REQUIRED>"

    refined_steps = generate_refined_plan_step(
        instruction="Something",
        previous_steps=["Something"],
        plan_step="Something",
        summary=summary,
        core=BaseCore(llm=LLM, retriever=memory_retriever),
        k=2,
    )
    assert isinstance(refined_steps, list)
    assert len(refined_steps) == 1
    assert refined_steps[0] == "<NO_SUBSTEPS_REQUIRED>"
