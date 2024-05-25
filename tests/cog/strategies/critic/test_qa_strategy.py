"""Unit tests for CRITIC QA strategies."""

import pytest
from unittest.mock import MagicMock

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from agential.cog.strategies.critic.qa_strategy import (
    CriticQAStrategy, 
    CritAmbigNQStrategy,
    CritFEVERStrategy,
    CritHotQAStrategy,
    CritTriviaQAStrategy
)

# Mock objects for testing
mock_search = MagicMock(spec=GoogleSerperAPIWrapper)

@pytest.fixture
def critic_qa_strategy():
    return CriticQAStrategy(llm=llm, search=mock_search)

def test_generate(critic_qa_strategy):
    llm.generate = MagicMock(return_value="Generated answer")
    question = "What is the capital of France?"
    examples = "Example question-answer pairs"
    prompt = "Prompt template"
    additional_keys = {}

    result = critic_qa_strategy.generate(question, examples, prompt, additional_keys)
    
    assert result == "Generated answer"
    llm.generate.assert_called_once()

def test_generate_critique(critic_qa_strategy):
    llm.generate = MagicMock(return_value="Generated critique")
    idx = 0
    question = "What is the capital of France?"
    examples = "Example question-answer pairs"
    answer = "The capital of France is Berlin."
    critique = ""
    prompt = "Prompt template"
    additional_keys = {}
    use_tool = False
    max_interactions = 5

    result, external_tool_info = critic_qa_strategy.generate_critique(
        idx, question, examples, answer, critique, prompt, additional_keys, use_tool, max_interactions
    )
    
    assert result == "Generated critique"
    assert external_tool_info == {}
    llm.generate.assert_called_once()

def test_create_output_dict(critic_qa_strategy):
    answer = "The capital of France is Paris."
    critique = "The answer is correct."
    external_tool_info = {"search_query": "capital of France", "search_result": "Paris"}

    result = critic_qa_strategy.create_output_dict(answer, critique, external_tool_info)
    
    assert result["answer"] == "The capital of France is Paris."
    assert result["critique"] == "The answer is correct."
    assert result["search_query"] == "capital of France"
    assert result["search_result"] == "Paris"

def test_update_answer_based_on_critique(critic_qa_strategy):
    question = "What is the capital of France?"
    examples = "Example question-answer pairs"
    answer = "The capital of France is Berlin."
    critique = "The answer is incorrect. The correct answer is Paris."
    prompt = "Prompt template"
    additional_keys = {}
    external_tool_info = {}

    result = critic_qa_strategy.update_answer_based_on_critique(
        question, examples, answer, critique, prompt, additional_keys, external_tool_info
    )
    
    assert result == answer

def test_halting_condition(critic_qa_strategy):
    critique = "The answer is correct."

    critic_qa_strategy._halt = True
    result = critic_qa_strategy.halting_condition(critique)
    
    assert result is True

def test_reset(critic_qa_strategy):
    critic_qa_strategy._query_history = ["query1"]
    critic_qa_strategy._evidence_history = {"evidence1"}
    critic_qa_strategy._halt = True

    critic_qa_strategy.reset()

    assert critic_qa_strategy._query_history == []
    assert critic_qa_strategy._evidence_history == set()
    assert critic_qa_strategy._halt is False

def test_handle_search_query(critic_qa_strategy) -> None:
    """Test CriticQAStrategy handle_search_query."""
    idx = 0
    question = "What is the capital of France?"
    search_query = "capital of France"
    use_tool = True
    max_interactions = 5
    kwargs = {"evidence_length": 100, "num_results": 3}

    mock_search.results = MagicMock(return_value=[{"title": "Paris", "snippet": "The capital of France is Paris."}])

    search_result, context = critic_qa_strategy.handle_search_query(
        idx, question, search_query, use_tool, max_interactions, **kwargs
    )
    
    assert search_result == {"title": "Paris", "snippet": "The capital of France is Paris."}
    assert "> Evidence: [Paris] The capital of France is Paris." in context


def test_instantiate_strategies() -> None:
    """Test instantiate all QA strategies."""
    llm = FakeListChatModel(responses=[])
    hotqa_strategy = CritHotQAStrategy(llm=llm)
    triviaqa_strategy = CritTriviaQAStrategy(llm=llm)
    ambignq_strategy = CritAmbigNQStrategy(llm=llm)
    fever_strategy = CritFEVERStrategy(llm=llm)

    assert isinstance(hotqa_strategy, CritHotQAStrategy)
    assert isinstance(triviaqa_strategy, CritTriviaQAStrategy)
    assert isinstance(ambignq_strategy, CritAmbigNQStrategy)
    assert isinstance(fever_strategy, CritFEVERStrategy)
