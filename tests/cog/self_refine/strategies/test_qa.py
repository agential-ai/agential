"""Unit tests for Self-Refine QA strategies."""
from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.cog.self_refine.prompts import (
    HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
    HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
)
from agential.cog.self_refine.strategies.qa import (
    SelfRefineHotQAStrategy,
    SelfRefineQAStrategy,
)


def test_init() -> None:
    """Test SelfRefineMathStrategy initialization."""
def test_generate() -> None:
    """Tests SelfRefineMathStrategy generate."""
def test_generate_critique() -> None:
    """Tests SelfRefineMathStrategy generate_critique."""
def test_create_output_dict() -> None:
    """Tests SelfRefineMathStrategy create_output_dict."""
def test_update_answer_based_on_critique() -> None:
    """Tests SelfRefineMathStrategy update_answer_based_on_critique."""
def test_halting_condition() -> None:
    """Tests SelfRefineMathStrategy halting_condition."""
def test_reset() -> None:
    """Tests SelfRefineMathStrategy reset."""

def test_instantiate_strategies() -> None:
    """Test instantiate all Math strategies."""