"""Unit tests for Reflexion QA strategies."""

from agential.cog.fewshots.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.reflexion.output import ReflexionCoTOutput, ReflexionCoTStepOutput
from agential.cog.reflexion.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_COT_INSTRUCTION_HOTPOTQA,
    REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.reflexion.reflect import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.reflexion.strategies.qa import (
    ReflexionCoTAmbigNQStrategy,
    ReflexionCoTFEVERStrategy,
    ReflexionCoTHotQAStrategy,
    ReflexionCoTQAStrategy,
    ReflexionCoTTriviaQAStrategy,
    ReflexionReActAmbigNQStrategy,
    ReflexionReActFEVERStrategy,
    ReflexionReActHotQAStrategy,
    ReflexionReActQAStrategy,
    ReflexionReActTriviaQAStrategy,
)
from agential.llm.llm import BaseLLM, MockLLM
from agential.utils.metrics import PromptMetrics


def test_reflexion_cot_init() -> None:
    """Test ReflexionCoTQAStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 3


def test_reflexion_cot_generate() -> None:
    """Tests ReflexionCoTQAStrategy generate."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    key = "Gesellschaft mit beschränkter Haftung"

    gt_out = ReflexionCoTOutput(
        answer="Visionary International Visual Art",
        total_prompt_tokens=80,
        total_completion_tokens=160,
        total_tokens=240,
        total_prompt_cost=0.00012000000000000002,
        total_completion_cost=0.00031999999999999997,
        total_cost=0.00043999999999999996,
        total_prompt_time=4.0,
        total_time=0.5,
        additional_info=[
            ReflexionCoTStepOutput(
                thought="I'm not sure what VIVA Media AG changed their name to, but I can reason that the new acronym must represent their new name.Thought 1: The new acronym is related to their new name, so let's find out what VIVA Media AG changed their name to in 2004.Answer: Finish[I'm sorry, I can't provide the answer as the information is not available in the provided context.]",
                action_type="Finish",
                observation="Answer is INCORRECT",
                answer="I'm sorry, I can't provide the answer as the information is not available in the provided context.",
                is_correct=False,
                reflections=[],
                thought_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                reflection_metrics=None,
            ),
            ReflexionCoTStepOutput(
                thought='Let\'s think step by step. VIVA Media AG was a German music television channel. In 2004, they changed their name to VIVACOM. Based on industry trends and common practices, they might have chosen a name that reflects their focus on media and communication.Thought 2: By considering their core business and the importance of communication in the media industry, the new acronym might stand for something like "VIVA Communication" or "VIVA Media Communication."Finish[VIVA Communication]',
                action_type="Finish",
                observation="Answer is INCORRECT",
                answer="VIVA Communication",
                is_correct=False,
                reflections=[
                    "My reasoning for not providing an answer to the question about the new acronym for VIVA Media AG in 2004 failed because I did not attempt to deduce or estimate the new name based on the given information. In the future, I should try to make an educated guess or use logical deductions to provide a more informed response. To improve, I will try to infer potential names based on common practices in company name changes or industry trends to offer a more insightful answer."
                ],
                thought_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                reflection_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
            ),
            ReflexionCoTStepOutput(
                thought="Let's approach this systematically. Given that VIVA Media AG was a media company, the new name could potentially reflect their focus on media or communication. Considering the previous failed attempts, I should broaden my thinking to include various possibilities beyond just one interpretation. By combining industry knowledge and common naming practices, I can make an educated guess at what the new acronym might stand for.",
                action_type="Finish",
                observation="Answer is INCORRECT",
                answer="Visionary International Visual Art",
                is_correct=False,
                reflections=[
                    "My reasoning for not providing an answer to the question about the new acronym for VIVA Media AG in 2004 failed because I did not attempt to deduce or estimate the new name based on the given information. In the future, I should try to make an educated guess or use logical deductions to provide a more informed response. To improve, I will try to infer potential names based on common practices in company name changes or industry trends to offer a more insightful answer.",
                    'The reasoning for the acronym of VIVACOM being "VIVA Communication" failed because it did not consider other possible words that could fit the acronym. In the future, when attempting this question, I should broaden the scope of potential words that could be used in the acronym to avoid limiting the possibilities. New Plan: When trying to determine the acronym for VIVACOM, consider a wider range of words that could fit the letters in the acronym while also taking into account the company\'s industry, focus, and branding to ensure a more accurate answer.',
                ],
                thought_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                reflection_metrics=PromptMetrics(
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
        "I'm not sure what VIVA Media AG changed their name to, but I can reason that the new acronym must represent their new name.\nThought 1: The new acronym is related to their new name, so let's find out what VIVA Media AG changed their name to in 2004.\nAnswer: Finish[I'm sorry, I can't provide the answer as the information is not available in the provided context.]",
        "Finish[I'm sorry, I can't provide the answer as the information is not available in the provided context.]",
        "My reasoning for not providing an answer to the question about the new acronym for VIVA Media AG in 2004 failed because I did not attempt to deduce or estimate the new name based on the given information. In the future, I should try to make an educated guess or use logical deductions to provide a more informed response. To improve, I will try to infer potential names based on common practices in company name changes or industry trends to offer a more insightful answer.",
        'Let\'s think step by step. VIVA Media AG was a German music television channel. In 2004, they changed their name to VIVACOM. Based on industry trends and common practices, they might have chosen a name that reflects their focus on media and communication.\nThought 2: By considering their core business and the importance of communication in the media industry, the new acronym might stand for something like "VIVA Communication" or "VIVA Media Communication."\nFinish[VIVA Communication]',
        "Finish[VIVA Communication]",
        'The reasoning for the acronym of VIVACOM being "VIVA Communication" failed because it did not consider other possible words that could fit the acronym. In the future, when attempting this question, I should broaden the scope of potential words that could be used in the acronym to avoid limiting the possibilities. \n\nNew Plan: When trying to determine the acronym for VIVACOM, consider a wider range of words that could fit the letters in the acronym while also taking into account the company\'s industry, focus, and branding to ensure a more accurate answer.',
        "Let's approach this systematically. Given that VIVA Media AG was a media company, the new name could potentially reflect their focus on media or communication. Considering the previous failed attempts, I should broaden my thinking to include various possibilities beyond just one interpretation. By combining industry knowledge and common naming practices, I can make an educated guess at what the new acronym might stand for.\n\nAction: Finish[Visual Interactive Video and Audio]",
        "Finish[Visionary International Visual Art]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionCoTQAStrategy(llm=llm, testing=True)
    out = strategy.generate(
        question=question,
        key=key,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        prompt=REFLEXION_COT_INSTRUCTION_HOTPOTQA,
        reflect_prompt=REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
        reflect_strategy="reflexion",
        additional_keys={},
        reflect_additional_keys={},
        patience=3,
        reset=True,
    )
    assert out == gt_out


def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTQAStrategy generate_action."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    responses = ["Finish[Verwaltung von Internet Video und Audio]"]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionCoTQAStrategy(llm=llm)
    scratchpad, action_type, query, action_metrics = strategy.generate_action(
        scratchpad="",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert action_type == "Finish"
    assert query == "Verwaltung von Internet Video und Audio"
    assert scratchpad == "\nAction: Finish[Verwaltung von Internet Video und Audio]"
    assert action_metrics == PromptMetrics(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTQAStrategy generate_observation."""
    # Case 1: action_type is "Finish" and answer is correct.
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm)
    scratchpad, answer, is_correct, obs = strategy.generate_observation(
        scratchpad="",
        action_type="Finish",
        query="correct_answer",
        key="correct_answer",
    )
    assert is_correct == True
    assert obs == "Answer is CORRECT"
    assert "Observation: Answer is CORRECT" in scratchpad
    assert answer == "correct_answer"

    # Case 2: action_type is "Finish" and answer is incorrect.
    strategy = ReflexionCoTQAStrategy(llm=llm)
    scratchpad, answer, is_correct, obs = strategy.generate_observation(
        scratchpad="",
        action_type="Finish",
        query="incorrect_answer",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == "Answer is INCORRECT"
    assert "Observation: Answer is INCORRECT" in scratchpad
    assert answer == "incorrect_answer"

    # Case 3: action_type is not "Finish".
    strategy = ReflexionCoTQAStrategy(llm=llm)
    scratchpad, answer, is_correct, obs = strategy.generate_observation(
        scratchpad="",
        action_type="Calculate",
        query="some_query",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == "Invalid action type, please try again."
    assert "Observation: Invalid action type, please try again." in scratchpad
    assert answer == ""


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTQAStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm, max_trials=3)

    assert strategy.halting_condition(3, "correct_answer", "incorrect_answer") == True

    assert strategy.halting_condition(2, "correct_answer", "correct_answer") == True

    assert strategy.halting_condition(2, "correct_answer", "incorrect_answer") == False


def test_reflexion_cot_reflect_condition() -> None:
    """Tests ReflexionCoTQAStrategy reflect_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTQAStrategy(llm)

    assert not strategy.reflect_condition(0, "strategy1", "key1", "key2")
    assert not strategy.reflect_condition(1, "strategy1", "key1", "key1")
    assert strategy.reflect_condition(1, "strategy1", "key2", "key1")
    assert not strategy.reflect_condition(1, "", "key2", "key2")


def test_reflexion_cot_instantiate_strategies() -> None:
    """Test instantiate all ReflexionCoT QA strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    hotqa_strategy = ReflexionCoTHotQAStrategy(llm=llm)
    triviaqa_strategy = ReflexionCoTTriviaQAStrategy(llm=llm)
    ambignq_strategy = ReflexionCoTAmbigNQStrategy(llm=llm)
    fever_strategy = ReflexionCoTFEVERStrategy(llm=llm)

    assert isinstance(hotqa_strategy, ReflexionCoTHotQAStrategy)
    assert isinstance(triviaqa_strategy, ReflexionCoTTriviaQAStrategy)
    assert isinstance(ambignq_strategy, ReflexionCoTAmbigNQStrategy)
    assert isinstance(fever_strategy, ReflexionCoTFEVERStrategy)


def test_reflexion_react_init() -> None:
    """Test ReflexionReActStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActQAStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy.reflector, ReflexionReActReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 3
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""
    assert strategy._prompt_metrics == {"reflection": None}
    assert strategy._prompt_metrics_react == {"thought": None, "action": None}


def test_reflexion_react_generate() -> None:
    """Tests ReflexionReActQAStrategy generate."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    gt_scratchpad = "\nThought: I need to search for VIVA Media AG and find out their new acronym after changing their name in 2004."
    gt_out = "I need to search for VIVA Media AG and find out their new acronym after changing their name in 2004."
    responses = [
        "I need to search for VIVA Media AG and find out their new acronym after changing their name in 2004."
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionReActQAStrategy(llm=llm)
    out = strategy.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflections="",
        prompt=REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        max_steps=5,
    )
    assert out == gt_out
    assert strategy._scratchpad == gt_scratchpad
    assert strategy._prompt_metrics == {"reflection": None}
    assert strategy._prompt_metrics_react == {
        "thought": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_cost": 1.5e-05,
            "completion_tokens_cost": 3.9999999999999996e-05,
            "total_tokens_cost": 5.4999999999999995e-05,
            "time_sec": 0.5,
        },
        "action": None,
    }


def test_reflexion_react_generate_action() -> None:
    """Tests ReflexionReActQAStrategy generate_action."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    gt_scratchpad = "\nAction: Search[VIVA Media AG]"
    responses = [
        "Search[VIVA Media AG]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionReActQAStrategy(llm=llm)
    action_type, query = strategy.generate_action(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflections="",
        prompt=REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        max_steps=5,
    )
    assert action_type == "Search"
    assert query == "VIVA Media AG"
    assert strategy._scratchpad == gt_scratchpad
    assert strategy._prompt_metrics_react == {
        "thought": None,
        "action": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_cost": 1.5e-05,
            "completion_tokens_cost": 3.9999999999999996e-05,
            "total_tokens_cost": 5.4999999999999995e-05,
            "time_sec": 0.5,
        },
    }
    assert strategy._prompt_metrics == {"reflection": None}


def test_reflexion_react_generate_observation() -> None:
    """Tests ReflexionReActQAStrategy generate_observation."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActQAStrategy(llm=llm)
    strategy.docstore.search = lambda x: "Search result"
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=1,
        action_type="Search",
        query="VIVA Media AG",
        key="key1",
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert strategy._scratchpad != ""
    assert not strategy._finished
    assert strategy._answer == ""
    assert external_tool_info == {"search_result": "Search result", "lookup_result": ""}

    strategy.docstore.lookup = lambda x: "Lookup result"
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=1,
        action_type="Lookup",
        query="VIVA Media AG",
        key="key1",
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert strategy._scratchpad != ""
    assert not strategy._finished
    assert strategy._answer == ""
    assert external_tool_info == {"search_result": "", "lookup_result": "Lookup result"}

    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=1,
        action_type="Finish",
        query="VIVA Media AG",
        key="key1",
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert strategy._scratchpad != ""
    assert strategy._finished
    assert strategy._answer == "VIVA Media AG"
    assert external_tool_info == {"search_result": "", "lookup_result": ""}


def test_reflexion_react_create_output_dict() -> None:
    """Tests ReflexionReActQAStrategy create_output_dict."""
    strategy = ReflexionReActQAStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))

    # Test case 1: Valid output creation
    react_out = [
        {
            "thought": "First thought",
            "action_type": "Query",
            "query": "What is the capital of France?",
            "observation": "Observation: Answer is CORRECT",
            "is_correct": True,
        }
    ]
    reflections = "Reflection on the first thought."
    output = strategy.create_output_dict(react_out, reflections)
    expected_output = {
        "react_output": react_out,
        "reflections": reflections,
        "prompt_metrics": {"reflection": None},
    }
    assert output == expected_output

    # Test case 2: Multiple steps in react_out
    react_out = [
        {
            "thought": "First thought",
            "action_type": "Query",
            "query": "What is the capital of France?",
            "observation": "Observation: Answer is CORRECT",
            "is_correct": True,
            "prompt_metrics": {"thought": [], "action": []},
        },
        {
            "thought": "Second thought",
            "action_type": "Validate",
            "query": "Is 2+2=4?",
            "observation": "Observation: Answer is CORRECT",
            "is_correct": True,
            "prompt_metrics": {"thought": [], "action": []},
        },
    ]
    reflections = "Reflection on the second thought."
    output = strategy.create_output_dict(react_out, reflections)
    expected_output = {
        "react_output": react_out,
        "reflections": reflections,
        "prompt_metrics": {"reflection": None},
    }
    assert output == expected_output

    # Test case 3: Empty react_out
    react_out = []
    reflections = "No reflections since no actions were taken."
    output = strategy.create_output_dict(react_out, reflections)
    expected_output = {
        "react_output": react_out,
        "reflections": reflections,
        "prompt_metrics": {"reflection": None},
    }
    assert output == expected_output


def test_reflexion_react_react_create_output_dict() -> None:
    """Tests ReflexionReActQAStrategy react_create_output_dict."""
    strategy = ReflexionReActQAStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))

    # Test case 1: Valid output creation
    output = strategy.react_create_output_dict(
        thought="Initial thought",
        action_type="Query",
        query="What is the capital of France?",
        obs="Observation: Answer is CORRECT",
        external_tool_info={"search_result": "", "lookup_result": ""},
        is_correct=True,
    )
    expected_output = {
        "thought": "Initial thought",
        "action_type": "Query",
        "query": "What is the capital of France?",
        "observation": "Observation: Answer is CORRECT",
        "answer": "",
        "external_tool_info": {"search_result": "", "lookup_result": ""},
        "is_correct": True,
        "prompt_metrics": {"thought": None, "action": None},
    }
    assert output == expected_output

    # Test case 2: Another valid output creation
    output = strategy.react_create_output_dict(
        thought="Second thought",
        action_type="Validate",
        query="Is 2+2=4?",
        obs="Observation: Answer is CORRECT",
        external_tool_info={"search_result": "", "lookup_result": ""},
        is_correct=True,
    )
    expected_output = {
        "thought": "Second thought",
        "action_type": "Validate",
        "query": "Is 2+2=4?",
        "observation": "Observation: Answer is CORRECT",
        "answer": "",
        "external_tool_info": {"search_result": "", "lookup_result": ""},
        "is_correct": True,
        "prompt_metrics": {"thought": None, "action": None},
    }
    assert output == expected_output

    # Test case 3: Incorrect answer handling
    output = strategy.react_create_output_dict(
        thought="Final thought",
        action_type="Answer",
        query="What is the square root of 16?",
        obs="Observation: Answer is INCORRECT",
        external_tool_info={"search_result": "", "lookup_result": ""},
        is_correct=False,
    )
    expected_output = {
        "thought": "Final thought",
        "action_type": "Answer",
        "query": "What is the square root of 16?",
        "observation": "Observation: Answer is INCORRECT",
        "answer": "",
        "external_tool_info": {"search_result": "", "lookup_result": ""},
        "is_correct": False,
        "prompt_metrics": {"thought": None, "action": None},
    }
    assert output == expected_output


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReActQAStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    # Test case 1: Halting condition met because answer is incorrect and index is less than max_trials.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer") == False

    # Test case 2: Halting condition not met because answer is correct.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=5)
    strategy._answer = "correct_answer"
    assert strategy.halting_condition(3, "correct_answer") == True

    # Test case 3: Halting condition not met because index is greater than or equal to max_trials.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=3)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(4, "correct_answer") == True

    # Test case 4: Halting condition met using max_trials from kwargs.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer", max_trials=4) == False

    # Test case 5: Halting condition not met using max_trials from kwargs.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(4, "correct_answer", max_trials=3) == True


def test_reflexion_react_react_halting_condition() -> None:
    """Tests ReflexionReActQAStrategy react_halting_condition."""
    strategy = ReflexionReActQAStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))

    idx = 0
    question = "What is the capital of France?"
    examples = ""
    reflections = ""
    prompt = "Answer the question."

    assert not strategy.react_halting_condition(
        idx, question, examples, reflections, prompt, {}
    )


def test_reflexion_react_reset() -> None:
    """Tests ReflexionReActQAStrategy reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActQAStrategy(llm=llm)
    strategy._scratchpad = "Some previous state"
    strategy._finished = True

    strategy.reset()

    assert strategy._scratchpad == ""
    assert not strategy._finished
    assert strategy._prompt_metrics == {"reflection": None}
    assert strategy._prompt_metrics_react == {"thought": None, "action": None}


def test_reflexion_react_reflect() -> None:
    """Tests ReflexionReActQAStrategy reflect."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    gt_reflections = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    llm = MockLLM("gpt-3.5-turbo", responses=["1"])
    strategy = ReflexionReActQAStrategy(llm=llm)
    _, reflections = strategy.reflect(
        reflect_strategy="reflexion",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert reflections == gt_reflections
    assert strategy._prompt_metrics == {
        "reflection": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_cost": 1.5e-05,
            "completion_tokens_cost": 3.9999999999999996e-05,
            "total_tokens_cost": 5.4999999999999995e-05,
            "time_sec": 0.5,
        }
    }
    assert strategy._prompt_metrics_react == {"thought": None, "action": None}


def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReActQAStrategy reflect_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=["1"])
    strategy = ReflexionReActQAStrategy(llm=llm)
    out = strategy.reflect_condition(
        step_idx=1,
        reflect_strategy="reflexion",
        question="VIVA Media AG changed it's name in 2004. What does their new acronym stand for?",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        key="key",
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert not out


def test_reflexion_react_instantiate_strategies() -> None:
    """Test instantiate all ReflexionReAct QA strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    hotqa_strategy = ReflexionReActHotQAStrategy(llm=llm)
    triviaqa_strategy = ReflexionReActTriviaQAStrategy(llm=llm)
    ambignq_strategy = ReflexionReActAmbigNQStrategy(llm=llm)
    fever_strategy = ReflexionReActFEVERStrategy(llm=llm)

    assert isinstance(hotqa_strategy, ReflexionReActHotQAStrategy)
    assert isinstance(triviaqa_strategy, ReflexionReActTriviaQAStrategy)
    assert isinstance(ambignq_strategy, ReflexionReActAmbigNQStrategy)
    assert isinstance(fever_strategy, ReflexionReActFEVERStrategy)
