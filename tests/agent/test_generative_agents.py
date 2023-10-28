"""Unit tests for the GenerativeAgent class."""

from langchain.chat_models import ChatOpenAI

from discussion_agents.core.base import BaseCore
from discussion_agents.core.memory import BaseCoreWithMemory
from discussion_agents.memory.generative_agents import GenerativeAgentMemory
from discussion_agents.agent.generative_agents import GenerativeAgent

from tests.fixtures.retriever import memory_retriever

import os

import dotenv

dotenv.load_dotenv(".env")
openai_api_key = os.getenv("OPENAI_API_KEY")

LLM = ChatOpenAI(openai_api_key=openai_api_key, max_tokens=3000)

characteristics = {
    "name": "Bob",
    "age": 25,
    "traits": "enthusiastic, energetic, motivated",
    "lifestyle": "very early sleeper"
}

memory_core = BaseCore(
    llm=LLM,
    retriever=memory_retriever()
)

memory = GenerativeAgentMemory(
    core=memory_core
)

agent_core = BaseCoreWithMemory(
    llm=LLM,
    memory=memory
)

def test_instantiation() -> None:
    """Test the GenerativeAgent class and its methods."""

    # Instantiation.
    agent = GenerativeAgent(
        **characteristics,
        core=agent_core
    )

def test_add_memory() -> None:
    """Tests add_memories."""
    # Add memory.

def test_compute_agent_summary() -> None:
    """Tests compute_agent_summary"""
    pass

def test_get_summary() -> None:
    """Tests get_summary."""
    pass

def test_get_full_header() -> None:
    """Tests get_full_header."""
    pass

def test_get_entity_from_observation() -> None:
    """Tests get_entity_from_observation."""
    pass

def test_get_entity_action() -> None:
    """Tests get_entity_action."""
    pass

def test_summarize_related_memories() -> None:
    """Tests summarize_related_memories."""
    pass

def test__generate_reaction() -> None:
    """Tests _generate_reaction."""
    pass

def test_generate_dialogue_response() -> None:
    """Tests generate_dialogue_response."""
    pass

def test_generate_reaction() -> None:
    """Test generate_reaction."""
    pass

def test_generate_broad_plan() -> None:
    """Tests generate_broad_plan."""
    pass

def test_update_status() -> None:
    """Tests update_status."""
    pass

def test_generate_refined_plan_step() -> None:
    """Tests generate_refined_plan_step."""
    pass

def test_clear_plan() -> None:
    """Tests clear_plan."""
    pass