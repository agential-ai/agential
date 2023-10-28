"""Unit tests for the GenerativeAgent class."""

import os

import dotenv

from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.schema import BaseRetriever

from discussion_agents.agent.generative_agents import GenerativeAgent
from discussion_agents.core.base import BaseCore
from discussion_agents.core.memory import BaseCoreWithMemory
from discussion_agents.memory.generative_agents import GenerativeAgentMemory

dotenv.load_dotenv(".env")
huggingface_hub_api_key = os.getenv("HUGGINGFACE_HUB_API_KEY")

LLM = HuggingFaceHub(
    repo_id="gpt2",
    huggingfacehub_api_token=huggingface_hub_api_key,
)

characteristics = {
    "name": "Bob",
    "age": 25,
    "traits": "enthusiastic, energetic, motivated",
    "lifestyle": "very early sleeper",
}


def test_generative_agent_instantiation(memory_retriever: BaseRetriever) -> None:
    """Test the GenerativeAgent class instantiation."""
    memory_core = BaseCore(llm=LLM, retriever=memory_retriever)

    memory = GenerativeAgentMemory(core=memory_core)

    agent_core = BaseCoreWithMemory(llm=LLM, memory=memory)

    _ = GenerativeAgent(**characteristics, core=agent_core)


def test_generative_agent_compute_agent_summary(
    memory_retriever: BaseRetriever,
) -> None:
    """Tests compute_agent_summary."""
    memory_core = BaseCore(llm=LLM, retriever=memory_retriever)

    memory = GenerativeAgentMemory(core=memory_core)

    agent_core = BaseCoreWithMemory(llm=LLM, memory=memory)

    agent = GenerativeAgent(**characteristics, core=agent_core)
    summary = agent.compute_agent_summary()
    assert isinstance(summary, str)


def test_generative_agent_get_summary(memory_retriever: BaseRetriever) -> None:
    """Test the get_summary method."""
    memory_core = BaseCore(llm=LLM, retriever=memory_retriever)

    memory = GenerativeAgentMemory(core=memory_core)

    agent_core = BaseCoreWithMemory(llm=LLM, memory=memory)

    agent = GenerativeAgent(**characteristics, core=agent_core)
    summary = agent.get_summary()
    assert isinstance(summary, str)


def test_generative_agent_get_full_header(memory_retriever: BaseRetriever) -> None:
    """Test the get_full_header method."""
    memory_core = BaseCore(llm=LLM, retriever=memory_retriever)

    memory = GenerativeAgentMemory(core=memory_core)

    agent_core = BaseCoreWithMemory(llm=LLM, memory=memory)

    agent = GenerativeAgent(**characteristics, core=agent_core)
    header = agent.get_full_header()
    assert isinstance(header, str)


def test_generative_agent_get_entity_from_observation(
    memory_retriever: BaseRetriever,
) -> None:
    """Test the get_entity_from_observation method."""
    memory_core = BaseCore(llm=LLM, retriever=memory_retriever)

    memory = GenerativeAgentMemory(core=memory_core)

    agent_core = BaseCoreWithMemory(llm=LLM, memory=memory)

    agent = GenerativeAgent(**characteristics, core=agent_core)
    observation = "A cat is chasing a mouse."
    entity = agent.get_entity_from_observation(observation)
    assert isinstance(entity, str)


def test_generative_agent_get_entity_action(memory_retriever: BaseRetriever) -> None:
    """Test the get_entity_action method."""
    memory_core = BaseCore(llm=LLM, retriever=memory_retriever)

    memory = GenerativeAgentMemory(core=memory_core)

    agent_core = BaseCoreWithMemory(llm=LLM, memory=memory)

    agent = GenerativeAgent(**characteristics, core=agent_core)
    observation = "A dog is barking loudly."
    entity = "dog"
    action = agent.get_entity_action(observation, entity)
    assert isinstance(action, str)


def test_generative_agent_summarize_related_memories(
    memory_retriever: BaseRetriever,
) -> None:
    """Test the summarize_related_memories method."""
    memory_core = BaseCore(llm=LLM, retriever=memory_retriever)

    memory = GenerativeAgentMemory(core=memory_core)

    agent_core = BaseCoreWithMemory(llm=LLM, memory=memory)

    agent = GenerativeAgent(**characteristics, core=agent_core)
    observation = "Alice met a friendly cat in the park."
    summary = agent.summarize_related_memories(observation)
    assert isinstance(summary, str)


def test_generative_agent__generate_reaction(memory_retriever: BaseRetriever) -> None:
    """Test the _generate_reaction method."""
    memory_core = BaseCore(llm=LLM, retriever=memory_retriever)

    memory = GenerativeAgentMemory(core=memory_core)

    agent_core = BaseCoreWithMemory(llm=LLM, memory=memory)

    agent = GenerativeAgent(**characteristics, core=agent_core)
    observation = "Alice: Hello, how are you?"
    suffix = "GenerativeAgent: Hello, Alice! I'm doing well, thank you."
    reaction = agent._generate_reaction(observation, suffix)
    assert isinstance(reaction, str)


def test_generative_agent__generate_reaction(memory_retriever: BaseRetriever) -> None:
    """Test the _generate_reaction method."""
    memory_core = BaseCore(llm=LLM, retriever=memory_retriever)

    memory = GenerativeAgentMemory(core=memory_core)

    agent_core = BaseCoreWithMemory(llm=LLM, memory=memory)

    agent = GenerativeAgent(**characteristics, core=agent_core)
    observation = "Alice: Hello, how are you?"
    suffix = "GenerativeAgent: Hello, Alice! I'm doing well, thank you."
    reaction = agent._generate_reaction(observation, suffix)
    assert isinstance(reaction, str)


def test_generative_agent_generate_dialogue_response(
    memory_retriever: BaseRetriever,
) -> None:
    """Test the generate_dialogue_response method."""
    memory_core = BaseCore(llm=LLM, retriever=memory_retriever)

    memory = GenerativeAgentMemory(core=memory_core)

    agent_core = BaseCoreWithMemory(llm=LLM, memory=memory)

    agent = GenerativeAgent(**characteristics, core=agent_core)
    observation = "User: Hello, how are you?"
    should_continue, response = agent.generate_dialogue_response(observation)
    assert isinstance(should_continue, bool)
    assert isinstance(response, str)


def test_generative_agent_generate_broad_plan(memory_retriever: BaseRetriever) -> None:
    """Test the generate_broad_plan method."""
    memory_core = BaseCore(llm=LLM, retriever=memory_retriever)

    memory = GenerativeAgentMemory(core=memory_core)

    agent_core = BaseCoreWithMemory(llm=LLM, memory=memory)

    agent = GenerativeAgent(**characteristics, core=agent_core)
    instruction = "Plan a weekend getaway."
    broad_plan = agent.generate_broad_plan(instruction)
    assert isinstance(broad_plan, list)


def test_generative_agent_update_status(memory_retriever: BaseRetriever) -> None:
    """Test the update_status method."""
    memory_core = BaseCore(llm=LLM, retriever=memory_retriever)

    memory = GenerativeAgentMemory(core=memory_core)

    agent_core = BaseCoreWithMemory(llm=LLM, memory=memory)

    agent = GenerativeAgent(**characteristics, core=agent_core)
    instruction = "Prepare for a hiking trip."
    previous_steps = ["1) Pack hiking gear.", "2) Plan the route."]
    plan_step = "3) Check the weather forecast."
    updated_status = agent.update_status(instruction, previous_steps, plan_step)
    assert isinstance(updated_status, str)


def test_generative_agent_generate_refined_plan_step(
    memory_retriever: BaseRetriever,
) -> None:
    """Tests generate_refined_plan_step."""
    pass


def test_generative_agent_clear_plan(memory_retriever: BaseRetriever) -> None:
    """Tests clear_plan."""
    pass
