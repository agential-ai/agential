"""Unit tests for CLIN functional methods."""

import tiktoken

from agential.agents.clin.functional import (
    _build_meta_summary_prompt,
    _build_react_agent_prompt,
    _build_summary_prompt,
    _is_halted,
    _prompt_meta_summary,
    _prompt_react_agent,
    _prompt_summary,
    parse_qa_action,
)
from agential.agents.clin.output import (
    CLINOutput,
    CLINReActStepOutput,
    CLINStepOutput,
)
from agential.agents.clin.prompts import (
    CLIN_INSTRUCTION_HOTPOTQA,
    CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
)
from agential.core.fewshots.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.core.llm import MockLLM, Response


def test_build_react_agent_prompt() -> None:
    """Test build_react_agent_prompt function."""
    gt_out = "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: \n(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.\n(3) Finish[answer], which returns the answer and finishes the task.\nYou have a maximum of  steps.\n\nHere are some examples:\n\n(END OF EXAMPLES)\n\n\nThese learnings capture important pre-conditions and mistakes: \n- X MAY BE NECESSARY to Y\n- X SHOULD BE NECESSARY to Y\n- X MAY NOT CONTRIBUTE to Y\n- X DOES NOT CONTRIBUTE to Y\n\nThese can be useful for predicting your next action:\n\n\nQuestion: "
    out = _build_react_agent_prompt(
        question="",
        examples="",
        summaries="",
        scratchpad="",
        max_steps="",
        summary_system="",
        prompt=CLIN_INSTRUCTION_HOTPOTQA,
    )

    assert out == gt_out


def test_prompt_react_agent() -> None:
    """Test _prompt_cot_agent function."""
    q = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    out = _prompt_react_agent(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        question="",
        summaries="",
        scratchpad="",
        max_steps="",
        summary_system="",
        prompt=CLIN_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, Response)
    assert out.output_text == "1"

    # Test simple case (no reflection).
    gt_out = 'Thought: Let\'s think step by step. The new acronym for VIVA Media AG after changing its name in 2004 is "Vivendi Visual and Interactive." \nAction: Finish[Vivendi Visual and Interactive]'
    responses = [
        (
            "Thought: Let's think step by step. The new acronym for VIVA Media AG after changing its name in 2004 "
            'is "Vivendi Visual and Interactive." \nAction: Finish[Vivendi Visual and Interactive]'
        )
    ]
    out = _prompt_react_agent(
        llm=MockLLM("gpt-3.5-turbo", responses=responses),
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        question="",
        summaries="",
        scratchpad="",
        max_steps="",
        summary_system="",
        prompt=CLIN_INSTRUCTION_HOTPOTQA,
    )
    assert out.output_text == gt_out

    # Test simple case (reflection).
    reflections = (
        "You have attempted to answer the following question before and failed. Below is the last trial "
        "you attempted to answer the question.\nQuestion: VIVA Media AG changed it's name in 2004. "
        "What does their new acronym stand for?\nThought: Let's think step by step. VIVA Media AG "
        'changed its name to Constantin Film AG in 2004. The new acronym stands for "Constantin Film."'
        "Action: Finish[Constantin Film]\nAction: Finish[Constantin Film]\n"
        "Observation: Answer is INCORRECT\n(END PREVIOUS TRIAL)\n"
    )
    scratchpad = (
        "\nThought: Let's think step by step. VIVA Media AG changed its name to Constantin Film AG in 2004. "
        'The new acronym stands for "Constantin Film."Action: Finish[Constantin Film]\n'
        "Action: Finish[Constantin Film]\nObservation: Answer is INCORRECT\nThought:"
    )
    responses = [
        (
            "I made a mistake in my previous attempts. Let's think more carefully this time. "
            'The acronym "AG" stands for "Aktiengesellschaft" in German, which translates to '
            '"stock corporation" in English. So the new acronym for VIVA Media AG after the name '
            'change is "Constantin Film Aktiengesellschaft."\nFinish[Constantin Film Aktiengesellschaft]'
        )
    ]
    gt_out = 'I made a mistake in my previous attempts. Let\'s think more carefully this time. The acronym "AG" stands for "Aktiengesellschaft" in German, which translates to "stock corporation" in English. So the new acronym for VIVA Media AG after the name change is "Constantin Film Aktiengesellschaft."\nFinish[Constantin Film Aktiengesellschaft]'
    out = _prompt_react_agent(
        llm=MockLLM("gpt-3.5-turbo", responses=responses),
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        question="",
        summaries="",
        scratchpad="",
        max_steps="",
        summary_system="",
        prompt=CLIN_INSTRUCTION_HOTPOTQA,
    )
    assert out.output_text == gt_out


def test_build_summary_prompt() -> None:
    """Test _build_summary_prompt function."""
    q = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    out = _build_summary_prompt(
        question=q,
        previous_trials="",
        scratchpad="",
        prompt=CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
    )

    print(repr(out))
    assert (
        out
        == "Generate a summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.\nEach numbered item in the summary can ONLY be of the form:\n- X MAY BE NECESSARY to Y.\n- X SHOULD BE NECESSARY to Y.\n- X MAY CONTRIBUTE to Y.\n- X DOES NOT CONTRIBUTE to Y.\n\nPREVIOUS LEARNINGS:\n\n\nCURRENT TRIAL:\nQuestion: VIVA Media AG changed it's name in 2004. What does their new acronym stand for?\n\nSummary of learnings as a numbered list:"
    )


def test_prompt_summary() -> None:
    """Test _summary_prompt function."""
    q = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    out = _prompt_summary(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
        question=q,
        previous_trials="",
        scratchpad="",
        prompt=CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
    )

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


def test_build_meta_summary_prompt() -> None:
    """Test _build_meta_summary_prompt function."""
    q = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    out = _build_meta_summary_prompt(
        question=q,
        meta_summary_system="",
        meta_summaries="",
        previous_trials="",
        scratchpad="",
        prompt=CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
    )

    print(repr(out))
    assert (
        out
        == "Generate a summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.\nEach numbered item in the summary can ONLY be of the form:\n- X MAY BE NECESSARY to Y.\n- X SHOULD BE NECESSARY to Y.\n- X MAY CONTRIBUTE to Y.\n- X DOES NOT CONTRIBUTE to Y.\n\nPREVIOUS LEARNINGS:\n\n\nCURRENT TRIAL:\nQuestion: VIVA Media AG changed it's name in 2004. What does their new acronym stand for?\n\nSummary of learnings as a numbered list:"
    )


def test_prompt_meta_summary() -> None:
    """Test _prompt_meta_summary function."""
    q = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    out = _prompt_meta_summary(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
        question=q,
        meta_summary_system="",
        meta_summaries="",
        previous_trials="",
        scratchpad="",
        prompt=CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
    )

    print(repr(out))
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


def test__is_halted() -> None:
    """Test _is_halted function."""
    gpt3_5_turbo_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Test when finish is true.
    assert _is_halted(
        True,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "",
        "",
        10,
        100,
        gpt3_5_turbo_enc,
        CLIN_INSTRUCTION_HOTPOTQA,
    )

    # Test when step_n exceeds max_steps.
    assert _is_halted(
        False,
        11,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "",
        "",
        10,
        100,
        gpt3_5_turbo_enc,
        CLIN_INSTRUCTION_HOTPOTQA,
    )

    # Test when encoded prompt exceeds max_tokens.
    assert _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "",
        "",
        10,
        10,
        gpt3_5_turbo_enc,
        CLIN_INSTRUCTION_HOTPOTQA,
    )

    # Test when none of the conditions for halting are met.
    assert not _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "",
        "",
        10,
        100000,
        gpt3_5_turbo_enc,
        CLIN_INSTRUCTION_HOTPOTQA,
    )

    # Test edge case when step_n equals max_steps.
    assert _is_halted(
        False,
        10,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "",
        "",
        10,
        100,
        gpt3_5_turbo_enc,
        CLIN_INSTRUCTION_HOTPOTQA,
    )

    # Test edge case when encoded prompt equals max_tokens.
    assert _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "",
        "",
        10,
        1603,
        gpt3_5_turbo_enc,
        CLIN_INSTRUCTION_HOTPOTQA,
    )

    # Test with custom prompt template string.
    assert not _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "",
        "",
        10,
        1603,
        gpt3_5_turbo_enc,
        "{question} {scratchpad} {examples} {max_steps}",
    )


def test_parse_qa_action() -> None:
    """Tests parse_qa_action."""
    action = "Calculate[sum = 4 + 6]"
    action_type, argument = parse_qa_action(action)
    assert action_type == "Calculate"
    assert argument == "sum = 4 + 6"

    action = "Finish[result = 7 - 2]"
    action_type, argument = parse_qa_action(action)
    assert action_type == "Finish"
    assert argument == "result = 7 - 2"

    action = "InvalidAction[result = 10 / 2]"
    action_type, argument = parse_qa_action(action)
    assert action_type == "InvalidAction"
    assert argument == "result = 10 / 2"

    action = "NoBrackets"
    action_type, argument = parse_qa_action(action)
    assert action_type == ""
    assert argument == ""

    action = "EmptyBrackets[]"
    action_type, argument = parse_qa_action(action)
    assert action_type == ""
    assert argument == ""
