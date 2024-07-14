"""Unit tests for Self-Refine code strategies."""

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.fewshots.humaneval import HUMANEVAL_FEWSHOT_EXAMPLES_POT
from agential.cog.self_refine.prompts import (
    HUMANEVAL_CRITIQUE_FEWSHOT_EXAMPLES,
    HUMANEVAL_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_HUMANEVAL,
    SELF_REFINE_INSTRUCTION_HUMANEVAL,
    SELF_REFINE_REFINE_INSTRUCTION_HUMANEVAL,
)
from agential.cog.self_refine.strategies.code import (
    SelfRefineHEvalStrategy,
    SelfRefineMBPPStrategy,
)   


def test_init() -> None:
    """Test SelfRefineCodeStrategy initialization."""

def test_generate() -> None:
    """Tests SelfRefineCodeStrategy generate."""

def test_generate_critique() -> None:
    """Tests SelfRefineCodeStrategy generate_critique."""

def test_create_output_dict() -> None:
    """Tests SelfRefineCodeStrategy create_output_dict."""

def test_update_answer_based_on_critique() -> None:
    """Tests SelfRefineCodeStrategy update_answer_based_on_critique."""

def test_halting_condition() -> None:
    """Tests SelfRefineCodeStrategy halting_condition."""

def test_reset() -> None:
    """Tests SelfRefineCodeStrategy reset."""
    
def test_instantiate_strategies() -> None:
    """Test instantiate all Code strategies."""