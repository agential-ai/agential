"""Unit tests for Self-Refine."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.self_refine import SelfRefineAgent
from agential.cog.prompts.agents.self_refine import (
    GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
    GSM8K_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
    SELF_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
)
from agential.cog.prompts.benchmark.gsm8k import GSM8K_FEWSHOT_EXAMPLES_POT
from agential.cog.strategies.self_refine.base import SelfRefineBaseStrategy


def test_init() -> None:
    """Test initialization."""
    agent = SelfRefineAgent(llm=FakeListChatModel(responses=[]), mode={"math": "gsm8k"})
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.strategy, SelfRefineBaseStrategy)
    assert agent.mode == {"math": "gsm8k"}


def test_reset() -> None:
    """Test reset."""
    agent = SelfRefineAgent(llm=FakeListChatModel(responses=[]), mode={"math": "gsm8k"})
    agent.strategy._halt = True
    agent.reset()
    assert not agent.strategy._halt


def test_generate() -> None:
    """Test generate."""
    question = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"

    gt_out = [
        {
            "code": "blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts",
            "critique": "The error in the code is in the calculation of the white fiber needed for the robe. Since the robe takes half as much white fiber as blue fiber, the calculation for white fiber should be `white_fiber = blue_fiber / 2`, not `white_fiber = blue_fiber * 2`. This error affects the total number of bolts calculation as well. The correct calculation should be `total_bolts = blue_fiber + white_fiber`.",
        },
        {
            "code": "blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts",
            "critique": "The error in the code is in the calculation of the white fiber needed for the robe. The white fiber needed is not half of the blue fiber, but rather half of the blue fiber bolts. Therefore, the calculation for white fiber should be white_fiber = blue_fiber / 2, not white_fiber = blue_fiber / 2.",
        },
        {
            "code": "blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts",
            "critique": "The error in the code is that it incorrectly calculates the amount of white fiber needed for the robe. The question states that the robe takes half as much white fiber as blue fiber, so the calculation for white fiber should be `white_fiber = blue_fiber / 2` instead of `white_fiber = blue_fiber * 2`.",
        },
    ]
    responses = [
        "blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts",
        "The error in the code is in the calculation of the white fiber needed for the robe. Since the robe takes half as much white fiber as blue fiber, the calculation for white fiber should be `white_fiber = blue_fiber / 2`, not `white_fiber = blue_fiber * 2`. This error affects the total number of bolts calculation as well. The correct calculation should be `total_bolts = blue_fiber + white_fiber`.",
        "```python\nblue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts\n```",
        "The error in the code is in the calculation of the white fiber needed for the robe. The white fiber needed is not half of the blue fiber, but rather half of the blue fiber bolts. Therefore, the calculation for white fiber should be white_fiber = blue_fiber / 2, not white_fiber = blue_fiber / 2. ",
        "```python\nblue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts\n```",
        "The error in the code is that it incorrectly calculates the amount of white fiber needed for the robe. The question states that the robe takes half as much white fiber as blue fiber, so the calculation for white fiber should be `white_fiber = blue_fiber / 2` instead of `white_fiber = blue_fiber * 2`.",
    ]
    agent = SelfRefineAgent(
        llm=FakeListChatModel(responses=responses), mode={"math": "gsm8k"}
    )

    out = agent.generate(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_POT,
        prompt=SELF_REFINE_INSTRUCTION_GSM8K,
        critique_examples=GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
        critique_prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
        refine_examples=GSM8K_REFINE_FEWSHOT_EXAMPLES,
        refine_prompt=SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
        additional_keys={},
        critique_additional_keys={},
        refine_additional_keys={},
        max_interactions=3,
        reset=True,
    )

    assert len(gt_out) == len(out)
    for gt_i, out_i in zip(gt_out, out):
        assert gt_i["code"] == out_i["code"]
        assert gt_i["critique"] == out_i["critique"]
