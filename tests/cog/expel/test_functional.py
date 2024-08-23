"""Unit tests for ExpeL functional module."""

import joblib

from agential.cog.expel.functional import (
    _build_all_success_prompt,
    _build_compare_prompt,
    _prompt_all_success_critique,
    _prompt_compare_critique,
    categorize_experiences,
    gather_experience,
    get_folds,
    parse_insights,
    remove_err_operations,
    retrieve_insight_index,
)
from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.reflexion.agent import ReflexionReActAgent
from agential.cog.reflexion.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.llm.llm import MockLLM, Response


def test_gather_experience() -> None:
    """Test gather_experience."""
    agent = ReflexionReActAgent(
        llm=MockLLM("gpt-3.5-turbo", responses=[]), benchmark="hotpotqa"
    )
    questions = [""]
    keys = [""]
    experiences = gather_experience(
        reflexion_react_agent=agent,
        questions=questions,
        keys=keys,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        reflect_prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        reflect_strategy="reflexion",
    )
    gt_experiences = [
        {
            "question": "",
            "key": "",
            "trajectory": [],
            "reflections": [],
        }
    ]
    print(repr(experiences))
    assert experiences == gt_experiences


def test_categorize_experiences(expel_experiences_10_fake_path: str) -> None:
    """Test categorize_experiences."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    categories = categorize_experiences(experiences)
    gt_categories = {"compare": [], "success": [1, 3], "fail": [0, 2, 4]}
    assert categories == gt_categories


def test_get_folds() -> None:
    """Test get_folds."""
    gt_folds = {0: [1, 4, 5, 8, 11, 14], 1: [0, 2, 3, 6, 7, 9, 10, 12, 13]}

    categories = {
        "compare": [10, 11, 12, 13, 14],
        "success": [1, 3, 6, 7, 8],
        "fail": [0, 2, 4, 5, 9],
    }
    folds = get_folds(categories, n_instances=15)
    assert folds == gt_folds


def test__build_compare_prompt() -> None:
    """Test _build_compare_prompt."""
    # Test is_full=True, empty insights.
    gt_prompt = "You are an advanced reasoning agent that can critique past task trajectories of youself. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps.\n\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n\n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"
    prompt = _build_compare_prompt(
        insights=[], question="", success_trial="", failed_trial="", is_full=True
    )
    assert prompt == gt_prompt

    # Test is_full=True, non-empty rules.
    gt_prompt = "You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps.\n\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n0. a\n1. b\n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"
    prompt = _build_compare_prompt(
        insights=[{"insight": "a", "score": 0}, {"insight": "b", "score": 0}],
        question="",
        success_trial="",
        failed_trial="",
        is_full=True,
    )
    assert prompt == gt_prompt

    # Test is_full=False, empty rules.
    gt_prompt = "You are an advanced reasoning agent that can critique past task trajectories of youself. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps.\n\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n\n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:"
    prompt = _build_compare_prompt(
        insights=[], question="", success_trial="", failed_trial="", is_full=False
    )
    assert prompt == gt_prompt

    # Test is_full=False, non-empty rules.
    gt_prompt = "You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps.\n\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n0. a\n1. b\n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:"
    prompt = _build_compare_prompt(
        insights=[{"insight": "a", "score": 0}, {"insight": "b", "score": 0}],
        question="",
        success_trial="",
        failed_trial="",
        is_full=False,
    )
    assert prompt == gt_prompt


def test__build_all_success_prompt() -> None:
    """Test _build_all_success_prompt."""
    # Test is_full=True, empty rules.
    gt_prompt = "You are an advanced reasoning agent that can critique past task trajectories of youself. You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer.\n\nHere are the trials:\n\n\nHere are the EXISTING RULES:\n\n\nBy examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"
    prompt = _build_all_success_prompt(insights=[], success_trajs_str="", is_full=True)
    assert prompt == gt_prompt

    # Test is_full=True, non-empty rules.
    gt_prompt = "You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer.\n\nHere are the trials:\n\n\nHere are the EXISTING RULES:\n0. a\n1. b\n\nBy examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"
    prompt = _build_all_success_prompt(
        insights=[{"insight": "a", "score": 0}, {"insight": "b", "score": 0}],
        success_trajs_str="",
        is_full=True,
    )
    assert prompt == gt_prompt

    # Test is_full=False, empty rules.
    gt_prompt = "You are an advanced reasoning agent that can critique past task trajectories of youself. You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer.\n\nHere are the trials:\n\n\nHere are the EXISTING RULES:\n\n\nBy examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:"
    prompt = _build_all_success_prompt(insights=[], success_trajs_str="", is_full=False)
    assert prompt == gt_prompt

    # Test is_full=False, non-empty rules.
    gt_prompt = "You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer.\n\nHere are the trials:\n\n\nHere are the EXISTING RULES:\n0. a\n1. b\n\nBy examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:"
    prompt = _build_all_success_prompt(
        insights=[{"insight": "a", "score": 0}, {"insight": "b", "score": 0}],
        success_trajs_str="",
        is_full=False,
    )
    assert prompt == gt_prompt


def test__prompt_compare_critique() -> None:
    """Test _prompt_compare_critique."""
    llm = MockLLM("gpt-3.5-turbo", responses=["1"])

    insights = [
        {"insight": "Insight 1", "score": 0.8},
        {"insight": "Insight 2", "score": 0.6},
    ]
    question = "Sample question"
    success_trial = "Successful trial"
    failed_trial = "Failed trial"
    is_full = True

    result = _prompt_compare_critique(
        llm=llm,
        insights=insights,
        question=question,
        success_trial=success_trial,
        failed_trial=failed_trial,
        is_full=is_full,
    )
    assert isinstance(result, Response)
    assert result.output_text == "1"


def test__prompt_all_success_critique() -> None:
    """Test _prompt_all_success_critique."""
    llm = MockLLM("gpt-3.5-turbo", responses=["1"])

    insights = [
        {"insight": "Insight 1", "score": 0.8},
        {"insight": "Insight 2", "score": 0.6},
    ]
    success_trajs_str = "Successful trajectories"
    is_full = True

    result = _prompt_all_success_critique(
        llm=llm,
        insights=insights,
        success_trajs_str=success_trajs_str,
        is_full=is_full,
    )
    assert isinstance(result, Response)
    assert result.output_text == "1"


def test_parse_insights() -> None:
    """Test parse_insights."""
    gt_rules = [
        ("REMOVE 1", "Rule to remove."),
        ("EDIT 2", "Rule to edit."),
        ("ADD", "Rule to add."),
    ]
    llm_output = "REMOVE 1: Rule to remove.\nEDIT 2: Rule to edit.\nADD 3: Rule to add."
    rules = parse_insights(llm_output)
    assert rules == gt_rules

    gt_rules = [("AGREE 1", "This is a valid rule.")]
    llm_output = "AGREE 1: This is a valid rule."
    rules = parse_insights(llm_output)
    assert rules == gt_rules


def test_retrieve_insight_index() -> None:
    """Tests retrieve_insight_index."""
    rules = [
        {"insight": "Rule1", "score": 1},
        {"insight": "Rule2", "score": 2},
        {"insight": "Rule3", "score": 3},
    ]

    idx = retrieve_insight_index(rules, "Operation on Rule1")
    assert idx == 0

    idx = retrieve_insight_index(rules, "Changes to Rule2")
    assert idx == 1

    idx = retrieve_insight_index(rules, "Modification of Rule3")
    assert idx == 2

    idx = retrieve_insight_index(rules, "No such rule")
    assert idx == -1


def test_remove_err_operations() -> None:
    """Test remove_err_operations."""
    rules = [{"insight": "Rule1", "score": 1}, {"insight": "Rule2", "score": 2}]

    operations = [
        ("ADD 1", "Rule1"),
        ("ADD 2", "Rule3"),
        ("EDIT 1", "Rule1"),
        ("EDIT 3", "Rule3"),
        ("REMOVE", "Rule1"),
        ("REMOVE", "Rule3"),
        ("AGREE", "Rule1"),
        ("AGREE", "Rule3"),
    ]

    expected_operations = [
        ("ADD 2", "Rule3"),
        ("AGREE 0", "Rule1"),
        ("REMOVE", "Rule1"),
        ("AGREE", "Rule1"),
    ]

    out = remove_err_operations(rules, operations)
    assert out == expected_operations
