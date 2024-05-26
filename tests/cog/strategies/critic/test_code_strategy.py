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
    """Test CriticQAStrategy initialization."""