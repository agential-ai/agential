"""Unit tests for ExpeL functional module."""

import joblib

from langchain_core.messages.human import HumanMessage
from discussion_agents.cog.agent.reflexion import ReflexionReActAgent
from discussion_agents.cog.functional.expel import (
    gather_experience,
    categorize_experiences,
    _build_compare_prompt,
    _build_all_success_prompt
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
        'idxs': [0], 
        'questions': [''], 
        'keys': [''], 
        'trajectories': [[]], 
        'reflections': [[]]
    }
    assert experiences == gt_experiences

def test_categorize_experiences(expel_15_compare_fake_path: str) -> None:
    """Test categorize_experiences."""
    experiences = joblib.load(expel_15_compare_fake_path)
    categories = categorize_experiences(experiences)
    print(repr(categories))
    gt_categories = {
        'compare': [10, 11, 12, 13, 14], 
        'success': [1, 3, 6, 7, 8], 
        'fail': [0, 2, 4, 5, 9]
    }
    assert categories == gt_categories


def test__build_compare_prompt() -> None:
    """Test _build_compare_prompt."""

    # Test is_full=True, empty rules.
    gt_prompt_msgs = [
        HumanMessage(content='You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps.'), 
        HumanMessage(content='\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n1. \n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:')
    ]
    prompt_msgs = _build_compare_prompt(
        rules=[],
        question="",
        success_trial="",
        failed_trial="",
        is_full=True
    )
    for gt_message, message in zip(gt_prompt_msgs, prompt_msgs):
        assert gt_message.content == message.content

    # Test is_full=True, non-empty rules.
    gt_prompt_msgs = [
        HumanMessage(content='You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps.'), 
        HumanMessage(content='\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n1. a\n2. b\n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:')
    ]
    prompt_msgs = _build_compare_prompt(
        rules=["a", "b"],
        question="",
        success_trial="",
        failed_trial="",
        is_full=True
    )
    for gt_message, message in zip(gt_prompt_msgs, prompt_msgs):
        assert gt_message.content == message.content

    # Test is_full=False, empty rules.
    gt_prompt_msgs = [
        HumanMessage(content='You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps.'), 
        HumanMessage(content='\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n1. \n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:')
    ]
    prompt_msgs = _build_compare_prompt(
        rules=[],
        question="",
        success_trial="",
        failed_trial="",
        is_full=False
    )
    for gt_message, message in zip(gt_prompt_msgs, prompt_msgs):
        assert gt_message.content == message.content

    # Test is_full=False, non-empty rules.
    gt_prompt_msgs = [
        HumanMessage(content='You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps.'), 
        HumanMessage(content='\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n1. a\n2. b\n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:')
    ]
    prompt_msgs = _build_compare_prompt(
        rules=["a", "b"],
        question="",
        success_trial="",
        failed_trial="",
        is_full=False
    )
    for gt_message, message in zip(gt_prompt_msgs, prompt_msgs):
        assert gt_message.content == message.content
