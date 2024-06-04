"""Unit tests for Reflexion QA strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.modules.reflect.reflexion import ReflexionCoTReflector
from agential.cog.prompts.agent.reflexion import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    REFLEXION_COT_INSTRUCTION_HOTPOTQA,
    REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.prompts.benchmark.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT_REACT,
)
from agential.cog.strategies.reflexion.qa import (
    ReflexionCoTQAStrategy,
    parse_qa_action,
)


def test_parse_qa_action() -> None:
    """Test the parse_qa_action function."""
    assert parse_qa_action("QA[question]") == ("QA", "question")
    assert parse_qa_action("QA[]") == ("", "")
    assert parse_qa_action("QA") == ("", "")


def test_reflexion_cot_init() -> None:
    """Test ReflexionCoTQAStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate() -> None:
    """Tests ReflexionCoTQAStrategy generate."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    gt_scratchpad = '\nThought: The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.'
    gt_out = 'The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.'
    responses = [
        'The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.',
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTQAStrategy(llm=llm)
    out = strategy.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT_REACT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert out == gt_out
    assert strategy._scratchpad == gt_scratchpad
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTQAStrategy generate_action."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    responses = ["Finish[Verwaltung von Internet Video und Audio]"]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTQAStrategy(llm=llm)
    action_type, query = strategy.generate_action(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT_REACT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert action_type == "Finish"
    assert query == "Verwaltung von Internet Video und Audio"
    assert strategy._finished == False
    assert strategy._answer == ""
    assert (
        strategy._scratchpad
        == "\nAction: Finish[Verwaltung von Internet Video und Audio]"
    )


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTQAStrategy generate_observation."""
    # Case 1: action_type is "Finish" and answer is correct.
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Finish", query="correct_answer", key="correct_answer"
    )
    assert is_correct == True
    assert obs == "Answer is CORRECT"
    assert "Observation: Answer is CORRECT" in strategy._scratchpad

    # Case 2: action_type is "Finish" and answer is incorrect.
    strategy = ReflexionCoTQAStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Finish", query="incorrect_answer", key="correct_answer"
    )
    assert is_correct == False
    assert obs == "Answer is INCORRECT"
    assert "Observation: Answer is INCORRECT" in strategy._scratchpad

    # Case 3: action_type is not "Finish".
    strategy = ReflexionCoTQAStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Calculate", query="some_query", key="correct_answer"
    )
    assert is_correct == False
    assert obs == "Invalid action type, please try again."
    assert "Observation: Invalid action type, please try again." in strategy._scratchpad


def test_reflexion_cot_create_output_dict() -> None:
    """Tests ReflexionCoTQAStrategy create_output_dict."""
    strategy = ReflexionCoTQAStrategy(llm=FakeListChatModel(responses=[]))

    # Setting a dummy answer for testing.
    strategy._answer = "correct_answer"

    # Test case 1: Correct answer.
    output = strategy.create_output_dict(
        thought="This is a thought.",
        action_type="Finish",
        query="correct_answer",
        obs="Observation: Answer is CORRECT",
        key="correct_answer",
    )
    expected_output = {
        "thought": "This is a thought.",
        "action_type": "Finish",
        "query": "correct_answer",
        "obs": "Observation: Answer is CORRECT",
        "answer": "correct_answer",
        "is_correct": True,
    }
    assert output == expected_output

    # Test case 2: Incorrect answer.
    strategy._answer = "incorrect_answer"
    output = strategy.create_output_dict(
        thought="This is a thought.",
        action_type="Finish",
        query="incorrect_answer",
        obs="Observation: Answer is INCORRECT",
        key="correct_answer",
    )
    expected_output = {
        "thought": "This is a thought.",
        "action_type": "Finish",
        "query": "incorrect_answer",
        "obs": "Observation: Answer is INCORRECT",
        "answer": "incorrect_answer",
        "is_correct": False,
    }
    assert output == expected_output

    # Test case 3: Invalid action type.
    strategy._answer = "some_answer"
    output = strategy.create_output_dict(
        thought="This is another thought.",
        action_type="Calculate",
        query="some_query",
        obs="Observation: Invalid action type, please try again.",
        key="correct_answer",
    )
    expected_output = {
        "thought": "This is another thought.",
        "action_type": "Calculate",
        "query": "some_query",
        "obs": "Observation: Invalid action type, please try again.",
        "answer": "some_answer",
        "is_correct": False,
    }
    assert output == expected_output


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTQAStrategy halting_condition."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm, max_trials=3)

    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer") == False

    strategy._answer = "correct_answer"
    assert strategy.halting_condition(2, "correct_answer") == False

    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(2, "correct_answer") == True


def test_reflexion_cot_reset() -> None:
    """Tests ReflexionCoTQAStrategy reset."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm, max_trials=3)

    # Set some initial states
    strategy._scratchpad = "Initial scratchpad content"
    strategy._finished = True
    strategy._answer = "Some answer"

    # Test case 1: Reset everything
    strategy.reset()
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""

    # Set some initial states
    strategy._scratchpad = "Initial scratchpad content"
    strategy._finished = True
    strategy._answer = "Some answer"

    # Test case 2: Reset only scratchpad
    strategy.reset(only_scratchpad=True)
    assert strategy._scratchpad == ""
    assert strategy._finished == True
    assert strategy._answer == "Some answer"


def test_reflexion_cot_reflect() -> None:
    """Tests ReflexionCoTQAStrategy reflect."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm, max_trials=3)

    gt_out = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: VIVA Media AG changed it's name in 2004. What does their new acronym stand for?\n\n(END PREVIOUS TRIAL)\n"
    out = strategy.reflect(
        reflection_strategy="last_attempt",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        prompt=REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert out == gt_out


def test_reflexion_cot_should_reflect() -> None:
    """Tests ReflexionCoTQAStrategy should_reflect."""
    answer = {"key1": True, "key2": False}
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm, answer)

    assert not strategy.should_reflect(0, "strategy1", "key1")
    assert not strategy.should_reflect(1, "strategy1", "key1")
    assert strategy.should_reflect(1, "strategy1", "key2")
    assert not strategy.should_reflect(1, "", "key2")
