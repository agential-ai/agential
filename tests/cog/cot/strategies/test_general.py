"""Unit tests for CoT general strategy."""

from agential.cog.cot.output import CoTOutput, CoTStepOutput
from agential.cog.cot.prompts import COT_INSTRUCTION_HOTPOTQA
from agential.cog.cot.strategies.general import CoTGeneralStrategy
from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.llm.llm import BaseLLM, MockLLM, Response


def test_init() -> None:
    """Tests the initialization of the CoT general strategy."""
    strategy = CoTGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    assert isinstance(strategy.llm, BaseLLM)


def test_generate() -> None:
    """Tests the generate method."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    gt_out = CoTOutput(answer='Badr Hari', total_prompt_tokens=20, total_completion_tokens=40, total_tokens=60, total_prompt_cost=3e-05, total_completion_cost=7.999999999999999e-05, total_cost=0.00010999999999999999, total_prompt_time=1.0, total_time=0.5, additional_info=CoTStepOutput(thought="Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.", answer='Badr Hari', thought_response=Response(input_text='', output_text="Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.\nAction: Finish[Badr Hari]", prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5), answer_response=Response(input_text='', output_text='Finish[Badr Hari]', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)))
    responses = [
        "Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.\nAction: Finish[Badr Hari]",
        'Finish[Badr Hari]',
    ]
    strategy = CoTGeneralStrategy(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), testing=True
    )
    out = strategy.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT, 
        prompt=COT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert out == gt_out

def test_reset() -> None:
    """Tests the reset method."""
    strategy = CoTGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    strategy.reset()
