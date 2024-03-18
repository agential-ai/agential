"""Unit tests for ExpeL functional module."""

import joblib

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.messages.human import HumanMessage

from discussion_agents.cog.agent.reflexion import ReflexionReActAgent
from discussion_agents.cog.functional.expel import (
    _build_all_success_prompt,
    _build_compare_prompt,
    categorize_experiences,
    create_rules,
    gather_experience,
    get_folds,
    is_existing_rule,
    parse_insights,
    remove_err_operations,
    retrieve_insight_index,
    update_rules,
)


def test_gather_experience(reflexion_react_agent: ReflexionReActAgent) -> None:
    """Test gather_experience."""
    questions = [""]
    keys = [""]
    experiences = gather_experience(
        reflexion_react_agent,
        questions,
        keys,
    )
    gt_experiences = {
        "idxs": [0],
        "questions": [""],
        "keys": [""],
        "trajectories": [[]],
        "reflections": [[]],
    }
    assert experiences == gt_experiences


def test_categorize_experiences(expel_experiences_10_fake_path: str) -> None:
    """Test categorize_experiences."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    categories = categorize_experiences(experiences)
    gt_categories = {"compare": [6, 7, 8, 9], "success": [3, 5], "fail": [0, 1, 2, 4]}
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
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps."
        ),
        HumanMessage(
            content="\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n1. \n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_compare_prompt(
        insights=[], question="", success_trial="", failed_trial="", is_full=True
    )
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])

    # Test is_full=True, non-empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps."
        ),
        HumanMessage(
            content="\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n1. a\n2. b\n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_compare_prompt(
        insights=[("a", 0), ("b", 0)],
        question="",
        success_trial="",
        failed_trial="",
        is_full=True,
    )
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])

    # Test is_full=False, empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps."
        ),
        HumanMessage(
            content="\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n1. \n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_compare_prompt(
        insights=[], question="", success_trial="", failed_trial="", is_full=False
    )
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])

    # Test is_full=False, non-empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps."
        ),
        HumanMessage(
            content="\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n1. a\n2. b\n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_compare_prompt(
        insights=[("a", 0), ("b", 0)],
        question="",
        success_trial="",
        failed_trial="",
        is_full=False,
    )
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])


def test__build_all_success_prompt() -> None:
    """Test _build_all_success_prompt."""
    # Test is_full=True, empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer."
        ),
        HumanMessage(
            content="\nHere are the trials:\n\n\nHere are the EXISTING RULES:\n1. \n\nBy examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_all_success_prompt(insights=[], success_trajs_str="", is_full=True)
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])

    # Test is_full=True, non-empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer."
        ),
        HumanMessage(
            content="\nHere are the trials:\n\n\nHere are the EXISTING RULES:\n1. a\n2. b\n\nBy examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_all_success_prompt(
        insights=[("a", 0), ("b", 0)], success_trajs_str="", is_full=True
    )
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])

    # Test is_full=False, empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer."
        ),
        HumanMessage(
            content="\nHere are the trials:\n\n\nHere are the EXISTING RULES:\n1. \n\nBy examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_all_success_prompt(insights=[], success_trajs_str="", is_full=False)
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])

    # Test is_full=False, non-empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer."
        ),
        HumanMessage(
            content="\nHere are the trials:\n\n\nHere are the EXISTING RULES:\n1. a\n2. b\n\nBy examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_all_success_prompt(
        insights=[("a", 0), ("b", 0)], success_trajs_str="", is_full=False
    )
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])


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
    rules = [("Rule1", 1), ("Rule2", 2), ("Rule3", 3)]

    idx = retrieve_insight_index(rules, "Operation on Rule1")
    assert idx == 0

    idx = retrieve_insight_index(rules, "Changes to Rule2")
    assert idx == 1

    idx = retrieve_insight_index(rules, "Modification of Rule3")
    assert idx == 2

    idx = retrieve_insight_index(rules, "No such rule")
    assert idx == -1


def test_is_existing_rule() -> None:
    """Tests is_existing_rule."""
    rules = [("Rule1", 1), ("Rule2", 2), ("Rule3", 3)]

    assert is_existing_rule(rules, "Operation on Rule1")
    assert is_existing_rule(rules, "Changes to Rule2")
    assert is_existing_rule(rules, "Modification of Rule3")
    assert not is_existing_rule(rules, "No such rule")


def test_remove_err_operations() -> None:
    """Test remove_err_operations."""
    rules = [("Rule1", 1), ("Rule2", 2)]

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
        ("AGREE 1", "Rule1"),
        ("REMOVE", "Rule1"),
        ("AGREE", "Rule1"),
    ]

    out = remove_err_operations(rules, operations)
    assert out == expected_operations


def test_update_rules() -> None:
    """Test update_rules."""
    initial_rules = [("Rule1", 1), ("Rule2", 2), ("Rule3", 3)]
    operations = [
        ("REMOVE", "Rule1"),
        ("AGREE", "Rule2"),
        ("EDIT 3", "Rule3"),
        ("ADD", "NewRule4"),
    ]

    # Expected outcomes.
    expected_rules_full = [("Rule3", 4), ("Rule2", 3), ("NewRule4", 2)]
    expected_rules_not_full = [("Rule3", 4), ("Rule2", 3), ("NewRule4", 2)]

    # Test with is_full=True.
    updated_rules_full = update_rules(initial_rules, operations, is_full=True)
    assert updated_rules_full == expected_rules_full

    # Test with is_full=False.
    updated_rules_not_full = update_rules(initial_rules, operations, is_full=False)
    assert updated_rules_not_full == expected_rules_not_full


def test_create_rules(expel_experiences_10_fake_path: str) -> None:
    """Test create_rules."""
    gt_rules = [
        ("Prioritize specific keywords in the question to guide search queries.", 2),
        (
            "Consider alternative search terms if the initial search query does not yield relevant results.",
            2,
        ),
        (
            "Break down complex search queries into smaller, more specific parts to guide the search process effectively.",
            2,
        ),
        (
            "Prioritize refining the search query based on the specific elements of the question to avoid ambiguity and ensure relevance in search results.",
            2,
        ),
        (
            "Prioritize verifying the accuracy of information obtained from search results before providing an answer.",
            2,
        ),
    ]

    max_num_rules = 20
    experiences = joblib.load(expel_experiences_10_fake_path)
    categories = categorize_experiences(experiences)
    folds = get_folds(categories, len(experiences["idxs"]))
    responses = [
        "ADD 1: Prioritize specific keywords in the question to guide search queries.\nEDIT 2: Specify the need to directly find the information requested in the question.\nREMOVE 3: The action is unclear and redundant with previous searches.\nAGREE 4: The action of narrowing down the search to find specific information is valid.",
        "ADD 2: Consider alternative search terms if the initial search query does not yield relevant results.",
        "ADD 3: Break down complex search queries into smaller, more specific parts to guide the search process effectively.",
        "ADD 4: Prioritize refining the search query based on the specific elements of the question to avoid ambiguity and ensure relevance in search results.",
        "ADD 5: Prioritize verifying the accuracy of information obtained from search results before providing an answer.",
    ]
    llm = FakeListChatModel(responses=responses)

    train_idxs = folds[0]
    rules = []
    rules = create_rules(llm, experiences, categories, train_idxs, rules, max_num_rules)
    assert rules == gt_rules
