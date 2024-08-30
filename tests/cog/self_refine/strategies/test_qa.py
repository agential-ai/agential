"""Unit tests for Self-Refine QA strategies."""

from agential.core.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.agent.self_refine.output import SelfRefineOutput, SelfRefineStepOutput
from agential.agent.self_refine.prompts import (
    HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
    HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
)
from agential.agent.self_refine.strategies.qa import (
    SelfRefineAmbigNQStrategy,
    SelfRefineFEVERStrategy,
    SelfRefineHotQAStrategy,
    SelfRefineQAStrategy,
    SelfRefineTriviaQAStrategy,
)
from agential.llm.llm import MockLLM, Response


def test_init() -> None:
    """Test SelfRefineQAStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineQAStrategy(llm=llm, patience=3)
    assert strategy.llm == llm
    assert strategy.patience == 3
    assert strategy.testing == False
    assert strategy._prev_answer == ""
    assert strategy.patience_counter == 0


def test_generate() -> None:
    """Test SelfRefineQAStrategy generate."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'
    gt_out = SelfRefineOutput(
        answer="Badr Hari",
        total_prompt_tokens=40,
        total_completion_tokens=80,
        total_tokens=120,
        total_prompt_cost=6e-05,
        total_completion_cost=0.00015999999999999999,
        total_cost=0.00021999999999999998,
        total_prompt_time=2.0,
        total_time=0.5,
        additional_info=[
            SelfRefineStepOutput(
                answer="Badr Hari",
                critique="The proposed answer \"Badr Hari\" fits the characteristics described in the question, so it seems plausible.\n\n2. Truthfulness:\n\nLet's search for information about Badr Hari's career and controversies:\n\n> Search Query: Badr Hari kickboxing controversy\n> Evidence: Badr Hari is a Moroccan-Dutch kickboxer who was once considered one of the best in the world. However, he has been involved in several controversies related to unsportsmanlike conduct in the sport and criminal activities outside the ring.\n\nThe evidence confirms that Badr Hari fits the description provided in the question, making the proposed answer correct and truthful.",
                answer_response=Response(
                    input_text="",
                    output_text="Badr Hari",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                critique_response=Response(
                    input_text="",
                    output_text="The proposed answer \"Badr Hari\" fits the characteristics described in the question, so it seems plausible.\n\n2. Truthfulness:\n\nLet's search for information about Badr Hari's career and controversies:\n\n> Search Query: Badr Hari kickboxing controversy\n> Evidence: Badr Hari is a Moroccan-Dutch kickboxer who was once considered one of the best in the world. However, he has been involved in several controversies related to unsportsmanlike conduct in the sport and criminal activities outside the ring.\n\nThe evidence confirms that Badr Hari fits the description provided in the question, making the proposed answer correct and truthful.",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
            ),
            SelfRefineStepOutput(
                answer="Badr Hari",
                critique="The proposed answer, Badr Hari, fits the description provided in the question. It is plausible that he was once considered the best kickboxer in the world and has been involved in controversies related to unsportsmanlike conduct and violence outside of the ring.\n\n2. Truthfulness:\n\nLet's search for information to verify the accuracy of the answer:\n\n> Search Query: Best kickboxer controversies violence outside ring\n> Evidence: [Badr Hari - Wikipedia] Badr Hari is a Dutch-Moroccan kickboxer who was once considered the best in the world. He has been involved in various controversies related to unsportsmanlike conduct and crimes of violence outside the ring.\n\nThe evidence confirms that Badr Hari fits the description provided in the question, making the proposed answer accurate and truthful.\n\nOverall, the proposed answer, Badr Hari, correctly aligns with the information provided in the question regarding his kickboxing career and controversies.",
                answer_response=Response(
                    input_text="",
                    output_text="Badr Hari",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                critique_response=Response(
                    input_text="",
                    output_text="The proposed answer, Badr Hari, fits the description provided in the question. It is plausible that he was once considered the best kickboxer in the world and has been involved in controversies related to unsportsmanlike conduct and violence outside of the ring.\n\n2. Truthfulness:\n\nLet's search for information to verify the accuracy of the answer:\n\n> Search Query: Best kickboxer controversies violence outside ring\n> Evidence: [Badr Hari - Wikipedia] Badr Hari is a Dutch-Moroccan kickboxer who was once considered the best in the world. He has been involved in various controversies related to unsportsmanlike conduct and crimes of violence outside the ring.\n\nThe evidence confirms that Badr Hari fits the description provided in the question, making the proposed answer accurate and truthful.\n\nOverall, the proposed answer, Badr Hari, correctly aligns with the information provided in the question regarding his kickboxing career and controversies.",
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
    )
    responses = [
        "Badr Hari",
        "The proposed answer \"Badr Hari\" fits the characteristics described in the question, so it seems plausible.\n\n2. Truthfulness:\n\nLet's search for information about Badr Hari's career and controversies:\n\n> Search Query: Badr Hari kickboxing controversy\n> Evidence: Badr Hari is a Moroccan-Dutch kickboxer who was once considered one of the best in the world. However, he has been involved in several controversies related to unsportsmanlike conduct in the sport and criminal activities outside the ring.\n\nThe evidence confirms that Badr Hari fits the description provided in the question, making the proposed answer correct and truthful.",
        "Badr Hari",
        "The proposed answer, Badr Hari, fits the description provided in the question. It is plausible that he was once considered the best kickboxer in the world and has been involved in controversies related to unsportsmanlike conduct and violence outside of the ring.\n\n2. Truthfulness:\n\nLet's search for information to verify the accuracy of the answer:\n\n> Search Query: Best kickboxer controversies violence outside ring\n> Evidence: [Badr Hari - Wikipedia] Badr Hari is a Dutch-Moroccan kickboxer who was once considered the best in the world. He has been involved in various controversies related to unsportsmanlike conduct and crimes of violence outside the ring.\n\nThe evidence confirms that Badr Hari fits the description provided in the question, making the proposed answer accurate and truthful.\n\nOverall, the proposed answer, Badr Hari, correctly aligns with the information provided in the question regarding his kickboxing career and controversies.",
    ]

    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = SelfRefineQAStrategy(llm=llm, testing=True)

    out = strategy.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,  # HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT, HOTPOTQA_FEWSHOT_EXAMPLES_REACT
        prompt=SELF_REFINE_INSTRUCTION_HOTPOTQA,
        critique_examples=HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
        critique_prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
        refine_examples=HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
        refine_prompt=SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        critique_additional_keys={},
        refine_additional_keys={},
        max_interactions=3,
        reset=True,
    )
    assert out == gt_out


def test_generate_answer() -> None:
    """Tests SelfRefineQAStrategy generate_answer."""
    llm = MockLLM("gpt-3.5-turbo", responses=["Badr Hari"])
    strategy = SelfRefineQAStrategy(llm=llm)
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    answer, out = strategy.generate_answer(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        prompt=SELF_REFINE_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert answer == "Badr Hari"
    assert out == Response(
        input_text="",
        output_text="Badr Hari",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_generate_critique() -> None:
    """Tests SelfRefineQAStrategy generate_critique."""
    gt_critique = "1"
    responses = ["1"]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = SelfRefineQAStrategy(llm=llm)
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'
    answer = "Mike Tyson"

    critique, finished, out = strategy.generate_critique(
        question=question,
        examples=HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
        answer=answer,
        prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert critique == gt_critique
    assert strategy._prev_answer == answer
    assert strategy.patience_counter == 0
    assert finished == False
    assert out == Response(
        input_text="",
        output_text="1",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )

    # Test early stopping.
    gt_critique = "1"
    answer = "Mike Tyson"
    responses = ["1"]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = SelfRefineQAStrategy(llm=llm, patience=1)
    strategy._prev_answer = "Mike Tyson"
    critique, finished, out = strategy.generate_critique(
        question=question,
        examples=HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
        answer=answer,
        prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert critique == gt_critique
    assert strategy.patience_counter == 1
    assert strategy._prev_answer == "Mike Tyson"
    assert finished
    assert out == Response(
        input_text="",
        output_text="1",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_update_answer_based_on_critique() -> None:
    """Tests SelfRefineQAStrategy update_answer_based_on_critique."""
    responses = ["1"]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = SelfRefineQAStrategy(llm=llm)
    question = "Sample question"
    answer = "Mike Tyson"
    critique = "Critique: Your solution is incorrect."

    new_answer, out = strategy.update_answer_based_on_critique(
        question=question,
        examples=HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
        answer=answer,
        critique=critique,
        prompt=SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert new_answer == "1"
    assert out == Response(
        input_text="",
        output_text="1",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_halting_condition() -> None:
    """Tests SelfRefineQAStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineQAStrategy(llm=llm, patience=2)

    # Initially, halting condition should be False.
    assert strategy.halting_condition(False) == False

    # Simulate the halting condition being met.
    assert strategy.halting_condition(True)


def test_reset() -> None:
    """Tests SelfRefineQAStrategy reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineQAStrategy(llm=llm, patience=2)

    strategy._prev_answer = "result = 42"
    strategy.patience_counter = 1
    strategy.reset()
    assert strategy._prev_answer == ""
    assert strategy.patience_counter == 0


def test_instantiate_strategies() -> None:
    """Test instantiate all QA strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    assert isinstance(SelfRefineHotQAStrategy(llm=llm), SelfRefineHotQAStrategy)
    assert isinstance(SelfRefineTriviaQAStrategy(llm=llm), SelfRefineTriviaQAStrategy)
    assert isinstance(SelfRefineAmbigNQStrategy(llm=llm), SelfRefineAmbigNQStrategy)
    assert isinstance(SelfRefineFEVERStrategy(llm=llm), SelfRefineFEVERStrategy)
