"""Unit tests for Reflexion QA strategies."""

from agential.cog.fewshots.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.reflexion.output import (
    ReflexionCoTOutput,
    ReflexionCoTStepOutput,
    ReflexionReActOutput,
    ReflexionReActReActStepOutput,
    ReflexionReActStepOutput,
)
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
from agential.utils.metrics import Response


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
                thought_metrics=Response(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=Response(
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
                thought_metrics=Response(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=Response(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                reflection_metrics=Response(
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
                thought_metrics=Response(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=Response(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                reflection_metrics=Response(
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
    assert action_metrics == Response(
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
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 5000


def test_reflexion_react_generate() -> None:
    """Tests ReflexionReActQAStrategy generate."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    key = "Gesellschaft mit beschränkter Haftung"

    gt_out = ReflexionReActOutput(
        answer="unknown",
        total_prompt_tokens=120,
        total_completion_tokens=240,
        total_tokens=360,
        total_prompt_cost=0.00018,
        total_completion_cost=0.00047999999999999996,
        total_cost=0.0006599999999999999,
        total_prompt_time=6.0,
        total_time=0.5,
        additional_info=[
            ReflexionReActStepOutput(
                steps=[
                    ReflexionReActReActStepOutput(
                        thought="I need to search for VIVA Media AG and find out what their new acronym stands for.",
                        action_type="Search",
                        query="VIVA Media",
                        observation="Buakaw Banchamek has faced several controversies and legal issues.",
                        answer="",
                        external_tool_info={
                            "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                            "lookup_result": "",
                        },
                        is_correct=False,
                        thought_metrics=Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    ReflexionReActReActStepOutput(
                        thought="The search results are not relevant to VIVA Media AG. I need to instead search for VIVA Media AG specifically.",
                        action_type="Search",
                        query="VIVA Media AG",
                        observation="Buakaw Banchamek has faced several controversies and legal issues.",
                        answer="",
                        external_tool_info={
                            "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                            "lookup_result": "",
                        },
                        is_correct=False,
                        thought_metrics=Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    ReflexionReActReActStepOutput(
                        thought="I need to refine my search and include the name change in 2004.",
                        action_type="Search",
                        query="VIVA Media AG name change 2004",
                        observation="Buakaw Banchamek has faced several controversies and legal issues.",
                        answer="",
                        external_tool_info={
                            "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                            "lookup_result": "",
                        },
                        is_correct=False,
                        thought_metrics=Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    ReflexionReActReActStepOutput(
                        thought="I need to search for VIVA Media AG name change specifically for 2004. I will try searching for VIVA Media AG new name in 2004.",
                        action_type="Search",
                        query="VIVA Media AG new name 2004",
                        observation="Buakaw Banchamek has faced several controversies and legal issues.",
                        answer="",
                        external_tool_info={
                            "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                            "lookup_result": "",
                        },
                        is_correct=False,
                        thought_metrics=Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    ReflexionReActReActStepOutput(
                        thought="I need to be more specific in my search. I will try searching for VIVA Media AG rebranding in 2004 to get the information I need.",
                        action_type="Search",
                        query="VIVA Media AG rebranding 2004",
                        observation="Buakaw Banchamek has faced several controversies and legal issues.",
                        answer="",
                        external_tool_info={
                            "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                            "lookup_result": "",
                        },
                        is_correct=False,
                        thought_metrics=Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    ReflexionReActReActStepOutput(
                        thought="I have exhausted my search possibilities and couldn't find the information. I will finish the task with the conclusion that I couldn't determine what the new acronym for VIVA Media AG stands for.",
                        action_type="Finish",
                        query="unknown",
                        observation="Answer is INCORRECT",
                        answer="unknown",
                        external_tool_info={"search_result": "", "lookup_result": ""},
                        is_correct=False,
                        thought_metrics=Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=Response(
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
                reflections=[],
                reflection_metrics=None,
            )
        ],
    )
    responses = [
        "I need to search for VIVA Media AG and find out what their new acronym stands for.\nAction 1: Search[VIVA Media AG]\nObservation 1: Could not find [VIVA Media AG]. Similar: ['Viva Media']\nThought 2: I should search for Viva Media instead.\nAction 2: Search[Viva Media]\nObservation 2: Viva Media is a German multimedia company that produces and distributes music and TV programs.\nThought 3: The question mentioned that VIVA Media AG changed its name in 2004, so I need to find their new acronym.\nAction 3: Search[Viva Media new acronym]\nObservation 3: Could not find [Viva Media new acronym]. Similar: ['Viva Media AG new name', 'Viva Media AG initials', 'Viva Media AG acronym']\nThought 4: I need to instead search for Viva Media AG's new name and acronym.\nAction 4: Search[Viva Media AG new name and acronym]\nObservation 4: Viva Media AG changed its name to Viva Entertainment in 2004. The new acronym stands for Video Interactive Education and Entertainment.\nThought 5: The new acronym for Viva Media AG is Video Interactive Education and Entertainment.\nAction 5: Finish[Video Interactive Education and Entertainment]",
        'Search[VIVA Media]\nObservation 1: Could not find [VIVA Media]. Similar: ["Viva Media"], ["Viva Media LLC"], ["Velora Media"]\nThought 2: Since I couldn\'t find VIVA Media, I should search for VIVA Media AG.\nAction 2: Search[VIVA Media AG]\nObservation 2: VIVA Media AG was a German media company that operated TV and radio stations as well as production companies.\nThought 3: I need to find out what their new acronym stands for after changing their name in 2004.\nAction 3: Search[VIVA Media AG new acronym 2004]\nObservation 3: Could not find information on the new acronym for VIVA Media AG after changing their name in 2004.\nThought 4: I should try searching VIVA Media AG\'s new name in 2004 instead.\nAction 4: Search[VIVA Media AG new name 2004]\nObservation 4: VIVA Media AG rebranded itself as VCL Communications AG in 2004.\nThought 5: Now I need to find out what VCL Communications AG acronym stands for.\nAction 5: Search[VCL Communications AG acronym]\nObservation 5: VCL Communications AG stands for Vídeo Comunicaciones y Logística.\nThought 6: The new acronym for VIVA Media AG after changing its name in 2004 is VCL. I found it stands for Vídeo Comunicaciones y Logística.\nAction 6: Finish[VCL - Vídeo Comunicaciones y Logística]',
        "The search results are not relevant to VIVA Media AG. I need to instead search for VIVA Media AG specifically.\nAction 2: Search[VIVA Media AG]\nObservation 2: VIVA Media AG was a company in the media industry in Germany, part of the VIVA TV network. In 2004, it changed its name to VIVACOM AG.\nThought 3: The new acronym for VIVACOM AG is VIVACOM. I need to find out what VIVACOM stands for.\nAction 3: Search[VIVACOM]\nObservation 3: Could not find [VIVACOM]. Similar: ['Vivacom', 'ViVaCom', 'Benrath Stables', 'Via media (philosophy)', 'Via media (anglican theology)', 'Vivaconda', 'Vivacolombia'].\nThought 4: I should instead look up VIVACOM AG abbreviation definition.\nAction 4: Lookup[VIVACOM AG abbreviation definition]",
        "Search[VIVA Media AG]\nObservation 2: VIVA Media AG was a German entertainment company headquartered in Unterföhring, Bavaria.\nThought 3: The information does not mention their new acronym. I need to find the information on their name change and new acronym.\nAction 3: Search[VIVA Media AG name change]\nObservation 3: VIVA Media AG changed its name to VIVA Entertainment in 2005.\nThought 4: I can now search for the meaning of VIVA.\nAction 4: Search[VIVA Entertainment]\nObservation 4: VIVA Entertainment was a German music television channel owned by VIVA Media AG.\nThought 5: Since the search for the new acronym is not giving relevant results, I need to look up the specific details of VIVA Media AG's name change and its new acronym to find the answer.\nAction 5: Lookup[VIVA Media AG name change, new acronym]\nObservation 5: (Result 1 / 1) The new acronym for VIVA Entertainment after the name change from VIVA Media AG is not readily available in the search results.\nThought 6: The specific new acronym after the name change is not easily found. I need to refine my search to find the exact answer.\nAction 6: Search[VIVA Entertainment new acronym]\nObservation 6: Unfortunately, the specific new acronym for VIVA Entertainment after the name change from VIVA Media AG is not directly provided in the search results.\nThought 7: Given the difficulty in obtaining the answer through search results, I may need to explore alternative sources or resources to find the new acronym for VIVA Entertainment after the name change from VIVA Media AG.\nAction 7: (Optional) Additional research or sources may be necessary to retrieve the exact new acronym for VIVA Entertainment after the name change from VIVA Media AG.",
        "I need to refine my search and include the name change in 2004.\nAction 3: Search[VIVA Media AG name change 2004]\nObservation 3: It has been mentioned that VIVA Media AG changed its name in 2004 to something else.\nThought 4: Since the search results do not provide the acronym, I should search for the new name of VIVA Media AG and its expansion.\nAction 4: Search[new name of VIVA Media AG]\nObservation 4: (Result 1 / 1) Lookup[new name of VIVA Media AG]\nThought 5: The next step is to find out what the new acronym of VIVA Media AG stands for.\nAction 5: Finish[New acronym]\nObservation 5: (Result 1 / 1) Lookup[new acronym]\nThought 6: Now that I have the new name and acronym, I can provide the answer.\nAction 6: Finish[New Name: XXX, Acronym: YYY]",
        'Search[VIVA Media AG name change 2004]\nObservation 3: Could not find [VIVA Media AG name change 2004]. Similar: [\'John Zeisel\']\nThought 4: Since I can\'t find the information directly, I need to try a different approach.\nAction 4: Search[VIVA Media AG acronym]\nObservation 4: The acronym VIVA stands for "Videovertrieb in Augsburg".\nThought 5: VIVA Media AG changed its name to "Videovertrieb in Augsburg" in 2004.\nAction 5: Finish[Videovertrieb in Augsburg]',
        "I need to search for VIVA Media AG name change specifically for 2004. I will try searching for VIVA Media AG new name in 2004.\nAction 4: Search[VIVA Media AG new name 2004]",
        "Search[VIVA Media AG new name 2004]",
        "I need to be more specific in my search. I will try searching for VIVA Media AG rebranding in 2004 to get the information I need.\nAction 5: Search[VIVA Media AG rebranding 2004]\nObservation 5: VIVA Media AG was rebranded as Viva Entertainment AG in 2004.\nThought 6: Their new acronym stands for Viva Entertainment AG.\nAction 6: Finish[Viva Entertainment AG]",
        "Search[VIVA Media AG rebranding 2004]\nObservation 5: VIVA Media AG was rebranded as VIVENDI Universal Interactive Publishing.",
        "I have exhausted my search possibilities and couldn't find the information. I will finish the task with the conclusion that I couldn't determine what the new acronym for VIVA Media AG stands for. \nAction 6: Finish[Unable to determine]",
        "Finish[unknown]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)

    strategy = ReflexionReActQAStrategy(llm=llm, testing=True)
    strategy.docstore.search = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )

    strategy.docstore.lookup = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )

    out = strategy.generate(
        question=question,
        key=key,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        prompt=REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        reflect_prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        reflect_strategy="reflexion",
        additional_keys={},
        reflect_additional_keys={},
        patience=1,
        reset=True,
    )

    assert out == gt_out


def test_reflexion_react_generate_react() -> None:
    """Tests ReflexionReActQAStrategy init."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    key = "Gesellschaft mit beschränkter Haftung"

    responses = [
        "I need to search for VIVA Media AG and find out what their new acronym stands for.\nAction 1: Search[VIVA Media AG]\nObservation 1: Could not find [VIVA Media AG]. Similar: ['Viva Media']\nThought 2: I should search for Viva Media instead.\nAction 2: Search[Viva Media]\nObservation 2: Viva Media is a German multimedia company that produces and distributes music and TV programs.\nThought 3: The question mentioned that VIVA Media AG changed its name in 2004, so I need to find their new acronym.\nAction 3: Search[Viva Media new acronym]\nObservation 3: Could not find [Viva Media new acronym]. Similar: ['Viva Media AG new name', 'Viva Media AG initials', 'Viva Media AG acronym']\nThought 4: I need to instead search for Viva Media AG's new name and acronym.\nAction 4: Search[Viva Media AG new name and acronym]\nObservation 4: Viva Media AG changed its name to Viva Entertainment in 2004. The new acronym stands for Video Interactive Education and Entertainment.\nThought 5: The new acronym for Viva Media AG is Video Interactive Education and Entertainment.\nAction 5: Finish[Video Interactive Education and Entertainment]",
        'Search[VIVA Media]\nObservation 1: Could not find [VIVA Media]. Similar: ["Viva Media"], ["Viva Media LLC"], ["Velora Media"]\nThought 2: Since I couldn\'t find VIVA Media, I should search for VIVA Media AG.\nAction 2: Search[VIVA Media AG]\nObservation 2: VIVA Media AG was a German media company that operated TV and radio stations as well as production companies.\nThought 3: I need to find out what their new acronym stands for after changing their name in 2004.\nAction 3: Search[VIVA Media AG new acronym 2004]\nObservation 3: Could not find information on the new acronym for VIVA Media AG after changing their name in 2004.\nThought 4: I should try searching VIVA Media AG\'s new name in 2004 instead.\nAction 4: Search[VIVA Media AG new name 2004]\nObservation 4: VIVA Media AG rebranded itself as VCL Communications AG in 2004.\nThought 5: Now I need to find out what VCL Communications AG acronym stands for.\nAction 5: Search[VCL Communications AG acronym]\nObservation 5: VCL Communications AG stands for Vídeo Comunicaciones y Logística.\nThought 6: The new acronym for VIVA Media AG after changing its name in 2004 is VCL. I found it stands for Vídeo Comunicaciones y Logística.\nAction 6: Finish[VCL - Vídeo Comunicaciones y Logística]',
        "The search results are not relevant to VIVA Media AG. I need to instead search for VIVA Media AG specifically.\nAction 2: Search[VIVA Media AG]\nObservation 2: VIVA Media AG was a company in the media industry in Germany, part of the VIVA TV network. In 2004, it changed its name to VIVACOM AG.\nThought 3: The new acronym for VIVACOM AG is VIVACOM. I need to find out what VIVACOM stands for.\nAction 3: Search[VIVACOM]\nObservation 3: Could not find [VIVACOM]. Similar: ['Vivacom', 'ViVaCom', 'Benrath Stables', 'Via media (philosophy)', 'Via media (anglican theology)', 'Vivaconda', 'Vivacolombia'].\nThought 4: I should instead look up VIVACOM AG abbreviation definition.\nAction 4: Lookup[VIVACOM AG abbreviation definition]",
        "Search[VIVA Media AG]\nObservation 2: VIVA Media AG was a German entertainment company headquartered in Unterföhring, Bavaria.\nThought 3: The information does not mention their new acronym. I need to find the information on their name change and new acronym.\nAction 3: Search[VIVA Media AG name change]\nObservation 3: VIVA Media AG changed its name to VIVA Entertainment in 2005.\nThought 4: I can now search for the meaning of VIVA.\nAction 4: Search[VIVA Entertainment]\nObservation 4: VIVA Entertainment was a German music television channel owned by VIVA Media AG.\nThought 5: Since the search for the new acronym is not giving relevant results, I need to look up the specific details of VIVA Media AG's name change and its new acronym to find the answer.\nAction 5: Lookup[VIVA Media AG name change, new acronym]\nObservation 5: (Result 1 / 1) The new acronym for VIVA Entertainment after the name change from VIVA Media AG is not readily available in the search results.\nThought 6: The specific new acronym after the name change is not easily found. I need to refine my search to find the exact answer.\nAction 6: Search[VIVA Entertainment new acronym]\nObservation 6: Unfortunately, the specific new acronym for VIVA Entertainment after the name change from VIVA Media AG is not directly provided in the search results.\nThought 7: Given the difficulty in obtaining the answer through search results, I may need to explore alternative sources or resources to find the new acronym for VIVA Entertainment after the name change from VIVA Media AG.\nAction 7: (Optional) Additional research or sources may be necessary to retrieve the exact new acronym for VIVA Entertainment after the name change from VIVA Media AG.",
        "I need to refine my search and include the name change in 2004.\nAction 3: Search[VIVA Media AG name change 2004]\nObservation 3: It has been mentioned that VIVA Media AG changed its name in 2004 to something else.\nThought 4: Since the search results do not provide the acronym, I should search for the new name of VIVA Media AG and its expansion.\nAction 4: Search[new name of VIVA Media AG]\nObservation 4: (Result 1 / 1) Lookup[new name of VIVA Media AG]\nThought 5: The next step is to find out what the new acronym of VIVA Media AG stands for.\nAction 5: Finish[New acronym]\nObservation 5: (Result 1 / 1) Lookup[new acronym]\nThought 6: Now that I have the new name and acronym, I can provide the answer.\nAction 6: Finish[New Name: XXX, Acronym: YYY]",
        'Search[VIVA Media AG name change 2004]\nObservation 3: Could not find [VIVA Media AG name change 2004]. Similar: [\'John Zeisel\']\nThought 4: Since I can\'t find the information directly, I need to try a different approach.\nAction 4: Search[VIVA Media AG acronym]\nObservation 4: The acronym VIVA stands for "Videovertrieb in Augsburg".\nThought 5: VIVA Media AG changed its name to "Videovertrieb in Augsburg" in 2004.\nAction 5: Finish[Videovertrieb in Augsburg]',
        "I need to search for VIVA Media AG name change specifically for 2004. I will try searching for VIVA Media AG new name in 2004.\nAction 4: Search[VIVA Media AG new name 2004]",
        "Search[VIVA Media AG new name 2004]",
        "I need to be more specific in my search. I will try searching for VIVA Media AG rebranding in 2004 to get the information I need.\nAction 5: Search[VIVA Media AG rebranding 2004]\nObservation 5: VIVA Media AG was rebranded as Viva Entertainment AG in 2004.\nThought 6: Their new acronym stands for Viva Entertainment AG.\nAction 6: Finish[Viva Entertainment AG]",
        "Search[VIVA Media AG rebranding 2004]\nObservation 5: VIVA Media AG was rebranded as VIVENDI Universal Interactive Publishing.",
        "I have exhausted my search possibilities and couldn't find the information. I will finish the task with the conclusion that I couldn't determine what the new acronym for VIVA Media AG stands for. \nAction 6: Finish[Unable to determine]",
        "Finish[unknown]",
    ]

    gt_scratchpad = "\nThought 1: I need to search for VIVA Media AG and find out what their new acronym stands for.\nAction 1: Search[VIVA Media]\nObservation 1: Buakaw Banchamek has faced several controversies and legal issues.\nThought 2: The search results are not relevant to VIVA Media AG. I need to instead search for VIVA Media AG specifically.\nAction 2: Search[VIVA Media AG]\nObservation 2: Buakaw Banchamek has faced several controversies and legal issues.\nThought 3: I need to refine my search and include the name change in 2004.\nAction 3: Search[VIVA Media AG name change 2004]\nObservation 3: Buakaw Banchamek has faced several controversies and legal issues.\nThought 4: I need to search for VIVA Media AG name change specifically for 2004. I will try searching for VIVA Media AG new name in 2004.\nAction 4: Search[VIVA Media AG new name 2004]\nObservation 4: Buakaw Banchamek has faced several controversies and legal issues.\nThought 5: I need to be more specific in my search. I will try searching for VIVA Media AG rebranding in 2004 to get the information I need.\nAction 5: Search[VIVA Media AG rebranding 2004]\nObservation 5: Buakaw Banchamek has faced several controversies and legal issues.\nThought 6: I have exhausted my search possibilities and couldn't find the information. I will finish the task with the conclusion that I couldn't determine what the new acronym for VIVA Media AG stands for.\nAction 6: Finish[unknown]\nObservation 6: Answer is INCORRECT"

    steps = [
        ReflexionReActReActStepOutput(
            thought="I need to search for VIVA Media AG and find out what their new acronym stands for.",
            action_type="Search",
            query="VIVA Media",
            observation="Buakaw Banchamek has faced several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                "lookup_result": "",
            },
            is_correct=False,
            thought_metrics=Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            action_metrics=Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
        ReflexionReActReActStepOutput(
            thought="The search results are not relevant to VIVA Media AG. I need to instead search for VIVA Media AG specifically.",
            action_type="Search",
            query="VIVA Media AG",
            observation="Buakaw Banchamek has faced several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                "lookup_result": "",
            },
            is_correct=False,
            thought_metrics=Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            action_metrics=Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
        ReflexionReActReActStepOutput(
            thought="I need to refine my search and include the name change in 2004.",
            action_type="Search",
            query="VIVA Media AG name change 2004",
            observation="Buakaw Banchamek has faced several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                "lookup_result": "",
            },
            is_correct=False,
            thought_metrics=Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            action_metrics=Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
        ReflexionReActReActStepOutput(
            thought="I need to search for VIVA Media AG name change specifically for 2004. I will try searching for VIVA Media AG new name in 2004.",
            action_type="Search",
            query="VIVA Media AG new name 2004",
            observation="Buakaw Banchamek has faced several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                "lookup_result": "",
            },
            is_correct=False,
            thought_metrics=Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            action_metrics=Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
        ReflexionReActReActStepOutput(
            thought="I need to be more specific in my search. I will try searching for VIVA Media AG rebranding in 2004 to get the information I need.",
            action_type="Search",
            query="VIVA Media AG rebranding 2004",
            observation="Buakaw Banchamek has faced several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                "lookup_result": "",
            },
            is_correct=False,
            thought_metrics=Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            action_metrics=Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
        ReflexionReActReActStepOutput(
            thought="I have exhausted my search possibilities and couldn't find the information. I will finish the task with the conclusion that I couldn't determine what the new acronym for VIVA Media AG stands for.",
            action_type="Finish",
            query="unknown",
            observation="Answer is INCORRECT",
            answer="unknown",
            external_tool_info={"search_result": "", "lookup_result": ""},
            is_correct=False,
            thought_metrics=Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            action_metrics=Response(
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

    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionReActQAStrategy(llm=llm, testing=True)

    strategy.docstore.search = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )

    strategy.docstore.lookup = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )

    step_idx, is_correct, scratchpad, finished, answer, react_steps = (
        strategy.generate_react(
            question=question,
            key=key,
            examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
            reflections="",
            prompt=REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
            additional_keys={},
        )
    )

    assert step_idx == 7
    assert not is_correct
    assert scratchpad == gt_scratchpad
    assert finished
    assert answer == "unknown"
    assert react_steps == steps


def test_reflexion_react_generate_action() -> None:
    """Tests ReflexionReActQAStrategy generate_action."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    gt_scratchpad = "\nAction 1: Search[VIVA Media AG]"
    responses = [
        "Search[VIVA Media AG]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionReActQAStrategy(llm=llm)
    scratchpad, action_type, query, thought_metrics = strategy.generate_action(
        idx=1,
        scratchpad="",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflections="",
        prompt=REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert action_type == "Search"
    assert query == "VIVA Media AG"
    assert scratchpad == gt_scratchpad
    assert thought_metrics == Response(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_reflexion_react_generate_observation() -> None:
    """Tests ReflexionReActQAStrategy generate_observation."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActQAStrategy(llm=llm)
    strategy.docstore.search = lambda x: "Search result"
    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=1,
            scratchpad="",
            action_type="Search",
            query="VIVA Media AG",
            key="key1",
        )
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert external_tool_info == {"search_result": "Search result", "lookup_result": ""}
    assert scratchpad == "\nObservation 1: Search result"
    assert answer == ""
    assert not finished

    strategy.docstore.lookup = lambda x: "Lookup result"
    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=1,
            scratchpad="",
            action_type="Lookup",
            query="VIVA Media AG",
            key="key1",
        )
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert external_tool_info == {"search_result": "", "lookup_result": "Lookup result"}
    assert scratchpad == "\nObservation 1: Lookup result"
    assert answer == ""
    assert not finished

    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=1,
            scratchpad="",
            action_type="Finish",
            query="VIVA Media AG",
            key="key1",
        )
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert external_tool_info == {"search_result": "", "lookup_result": ""}
    assert scratchpad == "\nObservation 1: Answer is INCORRECT"
    assert answer == "VIVA Media AG"
    assert finished


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReActQAStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    # Test case 1: Halting condition met because answer is incorrect and index is less than max_trials.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=5)
    assert strategy.halting_condition(3, "correct_answer", "incorrect_answer") == False

    # Test case 2: Halting condition not met because answer is correct.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=5)
    assert strategy.halting_condition(3, "correct_answer", "correct_answer") == True

    # Test case 3: Halting condition not met because index is greater than or equal to max_trials.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=3)
    assert strategy.halting_condition(4, "correct_answer", "correct_answer") == True

    # Test case 4: Halting condition met using max_trials from kwargs.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=5)
    assert strategy.halting_condition(3, "correct_answer", "incorrect_answer") == False

    # Test case 5: Halting condition not met using max_trials from kwargs.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=5)
    assert strategy.halting_condition(4, "correct_answer", "correct_answer") == True


def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReActQAStrategy reflect_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=["1"])
    strategy = ReflexionReActQAStrategy(llm=llm)
    out = strategy.reflect_condition(
        answer="",
        finished=False,
        idx=1,
        scratchpad="",
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
