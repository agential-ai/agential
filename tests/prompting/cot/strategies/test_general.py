"""Unit tests for CoT general strategy."""

from agential.core.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.llm.llm import BaseLLM, MockLLM, Response
from agential.prompting.cot.output import CoTOutput, CoTStepOutput
from agential.prompting.cot.prompts import COT_INSTRUCTION_HOTPOTQA
from agential.prompting.cot.strategies.general import CoTGeneralStrategy


def test_init() -> None:
    """Tests the initialization of the CoT general strategy."""
    strategy = CoTGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    assert isinstance(strategy.llm, BaseLLM)


def test_generate() -> None:
    """Tests the generate method."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    gt_out = CoTOutput(
        answer=[["Badr Hari", "Badr Hari", "Badr Hari", "Badr Hari"]],
        total_prompt_tokens=80,
        total_completion_tokens=160,
        total_tokens=240,
        total_prompt_cost=0.00012,
        total_completion_cost=0.00031999999999999997,
        total_cost=0.00043999999999999996,
        total_prompt_time=4.0,
        total_time=0.5,
        additional_info=[
            [
                CoTStepOutput(
                    thought="Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.",
                    answer="Badr Hari",
                    thought_response=Response(
                        input_text="",
                        output_text="Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.\nAction: Finish[Badr Hari]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    answer_response=Response(
                        input_text="",
                        output_text="Finish[Badr Hari]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                CoTStepOutput(
                    thought="Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.",
                    answer="Badr Hari",
                    thought_response=Response(
                        input_text="",
                        output_text="Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.\nAction: Finish[Badr Hari]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    answer_response=Response(
                        input_text="",
                        output_text="Finish[Badr Hari]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                CoTStepOutput(
                    thought="Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.",
                    answer="Badr Hari",
                    thought_response=Response(
                        input_text="",
                        output_text="Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.\nAction: Finish[Badr Hari]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    answer_response=Response(
                        input_text="",
                        output_text="Finish[Badr Hari]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                CoTStepOutput(
                    thought="Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.",
                    answer="Badr Hari",
                    thought_response=Response(
                        input_text="",
                        output_text="Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.\nAction: Finish[Badr Hari]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    answer_response=Response(
                        input_text="",
                        output_text="Finish[Badr Hari]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
            ]
        ],
    )
    responses = [
        "Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.\nAction: Finish[Badr Hari]",
        "Finish[Badr Hari]",
    ]
    strategy = CoTGeneralStrategy(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), testing=True
    )
    out = strategy.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        prompt=COT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        num_retries=1,
        warming=[None, 0.123, None, 0.2],
    )

    assert out == gt_out

    # Test num_retries=2.
    gt_out = CoTOutput(
        answer=[
            ["Paris", "Paris", "Paris", "Paris"],
            ["Paris", "Paris", "Paris", "Paris"],
        ],
        total_prompt_tokens=160,
        total_completion_tokens=320,
        total_tokens=480,
        total_prompt_cost=0.00024,
        total_completion_cost=0.0006399999999999999,
        total_cost=0.0008799999999999998,
        total_prompt_time=8.0,
        total_time=0.5,
        additional_info=[
            [
                CoTStepOutput(
                    thought="Let's think step by step. The capital of France is Paris. So, the answer is Paris.",
                    answer="Paris",
                    thought_response=Response(
                        input_text="",
                        output_text="Let's think step by step. The capital of France is Paris. So, the answer is Paris.\nAction: Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    answer_response=Response(
                        input_text="",
                        output_text="Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                CoTStepOutput(
                    thought="Let's think step by step. The capital of France is Paris, so the answer is Paris.",
                    answer="Paris",
                    thought_response=Response(
                        input_text="",
                        output_text="Let's think step by step. The capital of France is Paris, so the answer is Paris.\nAction: Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    answer_response=Response(
                        input_text="",
                        output_text="Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                CoTStepOutput(
                    thought="Let's think step by step. The capital of France is Paris. So, the answer is Paris.",
                    answer="Paris",
                    thought_response=Response(
                        input_text="",
                        output_text="Let's think step by step. The capital of France is Paris. So, the answer is Paris.\nAction: Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    answer_response=Response(
                        input_text="",
                        output_text="Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                CoTStepOutput(
                    thought="Let's think step by step. The capital of France is Paris, so the answer is Paris.",
                    answer="Paris",
                    thought_response=Response(
                        input_text="",
                        output_text="Let's think step by step. The capital of France is Paris, so the answer is Paris.\nAction: Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    answer_response=Response(
                        input_text="",
                        output_text="Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
            ],
            [
                CoTStepOutput(
                    thought="Let's think step by step. The capital of France is Paris. So, the answer is Paris.",
                    answer="Paris",
                    thought_response=Response(
                        input_text="",
                        output_text="Let's think step by step. The capital of France is Paris. So, the answer is Paris.\nAction: Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    answer_response=Response(
                        input_text="",
                        output_text="Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                CoTStepOutput(
                    thought="Let's think step by step. The capital of France is Paris, so the answer is Paris.",
                    answer="Paris",
                    thought_response=Response(
                        input_text="",
                        output_text="Let's think step by step. The capital of France is Paris, so the answer is Paris.\nAction: Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    answer_response=Response(
                        input_text="",
                        output_text="Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                CoTStepOutput(
                    thought="Let's think step by step. The capital of France is Paris. So, the answer is Paris.",
                    answer="Paris",
                    thought_response=Response(
                        input_text="",
                        output_text="Let's think step by step. The capital of France is Paris. So, the answer is Paris.\nAction: Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    answer_response=Response(
                        input_text="",
                        output_text="Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                CoTStepOutput(
                    thought="Let's think step by step. The capital of France is Paris, so the answer is Paris.",
                    answer="Paris",
                    thought_response=Response(
                        input_text="",
                        output_text="Let's think step by step. The capital of France is Paris, so the answer is Paris.\nAction: Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    answer_response=Response(
                        input_text="",
                        output_text="Finish[Paris]",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
            ],
        ],
    )
    responses = [
        "Let's think step by step. The capital of France is Paris. So, the answer is Paris.\nAction: Finish[Paris]",
        "Finish[Paris]",
        "Let's think step by step. The capital of France is Paris, so the answer is Paris.\nAction: Finish[Paris]",
        "Finish[Paris]",
    ]
    strategy = CoTGeneralStrategy(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), testing=True
    )
    out = strategy.generate(
        question="What is the capital of France?",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        prompt=COT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        num_retries=2,
        warming=[None, 0.123, None, 0.2],
    )
    assert out == gt_out


def test_reset() -> None:
    """Tests the reset method."""
    strategy = CoTGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    strategy.reset()
