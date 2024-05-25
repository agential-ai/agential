"""Unit tests for CRITIC math strategies."""

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.strategies.critic.math_strategy import (
    CritGSM8KStrategy,
    CritSVAMPStrategy,
    CritTabMWPStrategy,
    MathStrategy,
)


def test_generate() -> None:
    """Tests MathStrategy generate."""
    llm = FakeListChatModel(responses=["Generated answer\n```python\n42\n```"])
    strategy = MathStrategy(llm=llm)
    question = "What is 6 multiplied by 7?"
    examples = "Example question-answer pairs"
    prompt = "Prompt template"
    additional_keys = {}

    result = strategy.generate(question, examples, prompt, additional_keys)

    assert result == "42"


def test_generate_critique() -> None:
    """Tests MathStrategy generate_critique."""
    llm = FakeListChatModel(responses=["Generated critique"])
    strategy = MathStrategy(llm=llm)
    idx = 0
    question = "What is 6 multiplied by 7?"
    examples = "Example question-answer pairs"
    answer = "42"
    critique = ""
    prompt = "Prompt template"
    additional_keys = {}
    use_tool = False
    max_interactions = 5

    result, external_tool_info = strategy.generate_critique(
        idx,
        question,
        examples,
        answer,
        critique,
        prompt,
        additional_keys,
        use_tool,
        max_interactions,
    )

    assert result == "Generated critique"
    assert external_tool_info == {}

    # Test with tool.
    llm = FakeListChatModel(
        responses=["The answer is incorrect. Here's the correct code: 6 * 7 = 42"]
    )
    strategy = MathStrategy(llm=llm)
    idx = 0
    question = "What is 6 multiplied by 7?"
    examples = "Example question-answer pairs"
    answer = "40"
    critique = ""
    prompt = "Prompt template"
    additional_keys = {}
    use_tool = True
    max_interactions = 5

    result, external_tool_info = strategy.generate_critique(
        idx,
        question,
        examples,
        answer,
        critique,
        prompt,
        additional_keys,
        use_tool,
        max_interactions,
    )

    assert "The answer is incorrect." in result
    assert external_tool_info["execution_status"] == "Done"
    assert external_tool_info["code_answer"] == ""
