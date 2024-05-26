"""Unit tests for CRITIC code strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.prompts.critic import (
    CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
    CRITIC_POT_INSTRUCTION_GSM8K,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    GSM8K_FEWSHOT_EXAMPLES_POT,
)
from agential.cog.strategies.critic.code_strategy import (
    CriticCodeStrategy,
    CritMBPPCodeStrategy,
    CritHEvalCodeStrategy
)


def test_init() -> None:
    """Test CriticCodeStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = CriticCodeStrategy(llm=llm)
    assert strategy.llm == llm
    assert not strategy._halt

    
def test_generate() -> None:
    """Tests CriticCodeStrategy generate."""

def test_generate_critique() -> None:
    """Tests CriticCodeStrategy generate_critique."""

def test_create_output_dict() -> None:
    """Tests CriticCodeStrategy create_output_dict."""

def test_update_answer_based_on_critique() -> None:
    """Tests CriticCodeStrategy update_answer_based_on_critique."""

def test_halting_condition() -> None:
    """Tests CriticCodeStrategy halting_condition."""

def test_reset() -> None:
    """Tests CriticCodeStrategy reset."""

def test_instantiate_strategies() -> None:
    """Test instantiate all code strategies."""