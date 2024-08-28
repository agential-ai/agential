"""Unit tests for Self-Refine."""

import pytest

from agential.cog.constants import Benchmarks, FewShotType
from agential.cog.fewshots.gsm8k import GSM8K_FEWSHOT_EXAMPLES_POT
from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.cog.self_refine.agent import SelfRefineAgent
from agential.cog.self_refine.output import SelfRefineOutput, SelfRefineStepOutput
from agential.cog.self_refine.prompts import (
    GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
    GSM8K_REFINE_FEWSHOT_EXAMPLES,
    HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
    HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
    SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
)
from agential.cog.self_refine.strategies.base import SelfRefineBaseStrategy
from agential.cog.self_refine.strategies.math import (
    SelfRefineGSM8KStrategy,
    SelfRefineSVAMPStrategy,
    SelfRefineTabMWPStrategy,
)
from agential.cog.self_refine.strategies.qa import (
    SelfRefineAmbigNQStrategy,
    SelfRefineFEVERStrategy,
    SelfRefineHotQAStrategy,
    SelfRefineTriviaQAStrategy,
)
from agential.llm.llm import BaseLLM, MockLLM, Response


def test_init() -> None:
    """Test initialization."""
    agent = SelfRefineAgent(
        llm=MockLLM("gpt-3.5-turbo", responses=[]), benchmark="gsm8k"
    )
    assert isinstance(agent.llm, BaseLLM)
    assert isinstance(agent.strategy, SelfRefineBaseStrategy)
    assert agent.benchmark == "gsm8k"


def test_self_refine_factory_get_strategy() -> None:
    """Tests SelfRefineAgent get_strategy method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    # QA benchmarks.
    assert isinstance(
        SelfRefineAgent.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        SelfRefineHotQAStrategy,
    )
    assert isinstance(
        SelfRefineAgent.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        SelfRefineTriviaQAStrategy,
    )
    assert isinstance(
        SelfRefineAgent.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        SelfRefineAmbigNQStrategy,
    )
    assert isinstance(
        SelfRefineAgent.get_strategy(Benchmarks.FEVER, llm=llm),
        SelfRefineFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        SelfRefineAgent.get_strategy(Benchmarks.GSM8K, llm=llm),
        SelfRefineGSM8KStrategy,
    )
    assert isinstance(
        SelfRefineAgent.get_strategy(Benchmarks.SVAMP, llm=llm),
        SelfRefineSVAMPStrategy,
    )
    assert isinstance(
        SelfRefineAgent.get_strategy(Benchmarks.TABMWP, llm=llm),
        SelfRefineTabMWPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent Self-Refine"
    ):
        SelfRefineAgent.get_strategy("unknown", llm=llm)


def test_self_refine_factory_get_fewshots() -> None:
    """Tests SelfRefineAgent get_fewshots method."""
    # Test with valid fewshot type.
    fewshots = SelfRefineAgent.get_fewshots(Benchmarks.HOTPOTQA, FewShotType.COT)
    assert isinstance(fewshots, dict)
    assert "examples" in fewshots
    assert "critique_examples" in fewshots
    assert "refine_examples" in fewshots
    assert fewshots == {
        "examples": HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        "critique_examples": HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
    }

    # Test with invalid benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for Self-Refine."
    ):
        SelfRefineAgent.get_fewshots("unknown", FewShotType.COT)

    # Test with invalid fewshot type.
    with pytest.raises(
        ValueError,
        match="Benchmark 'hotpotqa' few-shot type not supported for Self-Refine.",
    ):
        SelfRefineAgent.get_fewshots(Benchmarks.HOTPOTQA, "invalid_type")


def test_self_refine_factory_get_prompts() -> None:
    """Tests SelfRefineAgent get_prompts method."""
    # Test with valid benchmark.
    prompts = SelfRefineAgent.get_prompts(Benchmarks.HOTPOTQA)
    assert isinstance(prompts, dict)
    assert "prompt" in prompts
    assert "critique_prompt" in prompts
    assert "refine_prompt" in prompts
    assert prompts == {
        "prompt": SELF_REFINE_INSTRUCTION_HOTPOTQA,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
    }

    # Test with invalid benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for Self-Refine."
    ):
        SelfRefineAgent.get_prompts("unknown")


def test_generate() -> None:
    """Test generate."""
    question = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"

    gt_out = SelfRefineOutput(answer='blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts', total_prompt_tokens=40, total_completion_tokens=80, total_tokens=120, total_prompt_cost=6e-05, total_completion_cost=0.00015999999999999999, total_cost=0.00021999999999999998, total_prompt_time=2.0, total_time=0.5, additional_info=[SelfRefineStepOutput(answer='blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts', critique='The error in the code is in the calculation of the white fiber needed for the robe. Since the robe takes half as much white fiber as blue fiber, the calculation for white fiber should be `white_fiber = blue_fiber / 2`, not `white_fiber = blue_fiber * 2`. This error affects the total number of bolts calculation as well. The correct calculation should be `total_bolts = blue_fiber + white_fiber`.', answer_response=Response(input_text='', output_text='blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5), critique_response=Response(input_text='', output_text='The error in the code is in the calculation of the white fiber needed for the robe. Since the robe takes half as much white fiber as blue fiber, the calculation for white fiber should be `white_fiber = blue_fiber / 2`, not `white_fiber = blue_fiber * 2`. This error affects the total number of bolts calculation as well. The correct calculation should be `total_bolts = blue_fiber + white_fiber`.', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)), SelfRefineStepOutput(answer='blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts', critique='The error in the code is in the calculation of the white fiber needed for the robe. The white fiber needed is not half of the blue fiber, but rather half of the blue fiber bolts. Therefore, the calculation for white fiber should be white_fiber = blue_fiber / 2, not white_fiber = blue_fiber / 2.', answer_response=Response(input_text='', output_text='```python\nblue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts\n```', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5), critique_response=Response(input_text='', output_text='The error in the code is in the calculation of the white fiber needed for the robe. The white fiber needed is not half of the blue fiber, but rather half of the blue fiber bolts. Therefore, the calculation for white fiber should be white_fiber = blue_fiber / 2, not white_fiber = blue_fiber / 2. ', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5))])  
    responses = [
        "blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts",
        "The error in the code is in the calculation of the white fiber needed for the robe. Since the robe takes half as much white fiber as blue fiber, the calculation for white fiber should be `white_fiber = blue_fiber / 2`, not `white_fiber = blue_fiber * 2`. This error affects the total number of bolts calculation as well. The correct calculation should be `total_bolts = blue_fiber + white_fiber`.",
        "```python\nblue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts\n```",
        "The error in the code is in the calculation of the white fiber needed for the robe. The white fiber needed is not half of the blue fiber, but rather half of the blue fiber bolts. Therefore, the calculation for white fiber should be white_fiber = blue_fiber / 2, not white_fiber = blue_fiber / 2. ",
        "```python\nblue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts\n```",
        "The error in the code is that it incorrectly calculates the amount of white fiber needed for the robe. The question states that the robe takes half as much white fiber as blue fiber, so the calculation for white fiber should be `white_fiber = blue_fiber / 2` instead of `white_fiber = blue_fiber * 2`.",
    ]
    agent = SelfRefineAgent(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), benchmark="gsm8k", testing=True
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
    assert out == gt_out

    # Test auto-select prompts and few-shots.
    agent = SelfRefineAgent(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), benchmark="gsm8k", testing=True
    )
    out = agent.generate(
        question=question,
        additional_keys={},
        critique_additional_keys={},
        refine_additional_keys={},
        max_interactions=3,
        reset=True,
    )
    assert out == gt_out

    # Test auto-select prompts and few-shots with fewshot_type.
    gt_out = SelfRefineOutput(answer='blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts', total_prompt_tokens=40, total_completion_tokens=80, total_tokens=120, total_prompt_cost=6e-05, total_completion_cost=0.00015999999999999999, total_cost=0.00021999999999999998, total_prompt_time=2.0, total_time=0.5, additional_info=[SelfRefineStepOutput(answer='blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts', critique='The error in the code is in the calculation of the white fiber needed for the robe. Since the robe takes half as much white fiber as blue fiber, the calculation for white fiber should be `white_fiber = blue_fiber / 2`, not `white_fiber = blue_fiber * 2`. This error affects the total number of bolts calculation as well. The correct calculation should be `total_bolts = blue_fiber + white_fiber`.', answer_response=Response(input_text='', output_text='blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5), critique_response=Response(input_text='', output_text='The error in the code is in the calculation of the white fiber needed for the robe. Since the robe takes half as much white fiber as blue fiber, the calculation for white fiber should be `white_fiber = blue_fiber / 2`, not `white_fiber = blue_fiber * 2`. This error affects the total number of bolts calculation as well. The correct calculation should be `total_bolts = blue_fiber + white_fiber`.', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)), SelfRefineStepOutput(answer='blue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts', critique='The error in the code is in the calculation of the white fiber needed for the robe. The white fiber needed is not half of the blue fiber, but rather half of the blue fiber bolts. Therefore, the calculation for white fiber should be white_fiber = blue_fiber / 2, not white_fiber = blue_fiber / 2.', answer_response=Response(input_text='', output_text='```python\nblue_fiber = 2\nwhite_fiber = blue_fiber / 2\ntotal_bolts = blue_fiber + white_fiber\nanswer = total_bolts\n```', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5), critique_response=Response(input_text='', output_text='The error in the code is in the calculation of the white fiber needed for the robe. The white fiber needed is not half of the blue fiber, but rather half of the blue fiber bolts. Therefore, the calculation for white fiber should be white_fiber = blue_fiber / 2, not white_fiber = blue_fiber / 2. ', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5))])  
    agent = SelfRefineAgent(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), benchmark="gsm8k", testing=True
    )
    out = agent.generate(
        question=question,
        additional_keys={},
        critique_additional_keys={},
        refine_additional_keys={},
        fewshot_type="pot",
        max_interactions=3,
        reset=True,
    )
    assert out == gt_out

    # Test auto-select prompts and few-shots with incorrect fewshot_type.
    agent = SelfRefineAgent(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), benchmark="gsm8k", testing=True
    )
    with pytest.raises(
        ValueError,
        match="Benchmark 'gsm8k' few-shot type not supported for Self-Refine.",
    ):
        out = agent.generate(
            question=question,
            additional_keys={},
            critique_additional_keys={},
            refine_additional_keys={},
            fewshot_type="cot",
            max_interactions=3,
            reset=True,
        )
    assert out == gt_out