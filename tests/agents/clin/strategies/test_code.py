"""Test the CLIN Code strategy."""

import tiktoken

from agential.agents.clin.output import CLINOutput, CLINReActStepOutput, CLINStepOutput
from agential.agents.clin.prompts import (
    CLIN_ADAPT_META_SUMMARY_SYSTEM,
    CLIN_ADAPT_SUMMARY_SYSTEM,
    CLIN_INSTRUCTION_MBPP,
    CLIN_META_SUMMARY_INSTRUCTION_MBPP,
    CLIN_SUMMARY_INSTRUCTION_MBPP,

)
from agential.agents.clin.strategies.code import CLINCodeStrategy
from agential.core.fewshots.mbpp import MBPP_FEWSHOT_EXAMPLES_REACT
from agential.core.fewshots.tabmwp import TABMWP_FEWSHOT_EXAMPLES_REACT
from agential.core.llm import MockLLM, Response


def test_init() -> None:
    """Test CLIN Code strategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = CLINCodeStrategy(llm=llm, memory=None)
    assert strategy.max_trials == 3
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 5000
    assert strategy.enc == tiktoken.encoding_for_model("gpt-3.5-turbo")
    assert strategy.testing is False
    assert isinstance(strategy.enc, tiktoken.Encoding)



def test_generate() -> None:
    """Test CLIN Code strategy generate."""
    question = "Write a python function to find the first repeated character in a given string."
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_out = CLINOutput(answer='\n\nanswer = 20\n\n', total_prompt_tokens=50, total_completion_tokens=100, total_tokens=150, total_prompt_cost=7.500000000000001e-05, total_completion_cost=0.00019999999999999998, total_cost=0.00027499999999999996, total_prompt_time=2.5, total_time=0.5, additional_info=[CLINStepOutput(steps=[CLINReActStepOutput(thought="I need to find the difference between Mike's and Irma's scores.", action_type='Calculate', query='\n\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n\n', observation='\n\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n\nExecution Status: Done\nOutput: answer = 20', answer='\n\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n\n', external_tool_info={'execution_status': 'Done', 'code_answer': 20}, is_correct=True, thought_response=Response(input_text='', output_text="I need to find the difference between Mike's and Irma's scores. \n\nAction 1: Calculate[\n\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n\n]\n\nObservation 1: \n\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n\n\nExecution Status: Done\n\nOutput: answer = 20\n\nThought 2: The difference in score between Mike and Irma is 20 points. \n\nAction 2: Finish[\n\nanswer = 20\n\n]\n\nObservation 2: \n\nanswer = 20\n", prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5), action_response=Response(input_text='', output_text='Calculate[\n\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n\n]', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)), CLINReActStepOutput(thought="The difference between Mike's and Irma's scores is 20 points.", action_type='Finish', query='\n\nanswer = 20\n\n', observation='Answer is CORRECT', answer='\n\nanswer = 20\n\n', external_tool_info={'execution_status': 'Done', 'code_answer': 20}, is_correct=True, thought_response=Response(input_text='', output_text="The difference between Mike's and Irma's scores is 20 points.\nAction 2: Finish[\n\nanswer = 20\n\n]\nObservation 2: \n\nanswer = 20\n", prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5), action_response=Response(input_text='', output_text='Finish[\n\nanswer = 20\n\n]\nObservation 2: \n\nanswer = 20\n', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5))], summaries='1. Knowing how to calculate the difference between two values MAY BE NECESSARY to comparing scores accurately.\n2. Understanding how to assign values to variables MAY CONTRIBUTE to performing mathematical operations in Python.', summaries_response=Response(input_text='', output_text='1. Knowing how to calculate the difference between two values MAY BE NECESSARY to comparing scores accurately.\n2. Understanding how to assign values to variables MAY CONTRIBUTE to performing mathematical operations in Python.', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5), meta_summaries='', previous_trials='')])
    responses = ['I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in the string.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```',
                 'Implement[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]',
                 'I need to test the function to ensure it works correctly with different test cases.',
                 'Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```',
                 'The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```',
                 'Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```',
                 '1. Keeping track of characters seen so far in a set may be necessary to find the first repeated character in a given string.\n2. Testing the function with different test cases may contribute to ensuring its correctness.',
                 'I need to write a function that can find the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in a given string for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```',
                 'Implement[\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```\n]',
                 'I need to test the function to ensure it works correctly with different test cases.',
                 'Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]',
                 'The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```',
                 'Finish[\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```',
                 '1. Keeping track of characters seen so far in a set may be necessary to find the first repeated character in a given string.\n2. Testing the function with different test cases may contribute to ensuring its correctness.',
                 ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CLINCodeStrategy(llm=llm, testing=True)
    output = strategy.generate(
        question=question,
        key=key,
        examples=MBPP_FEWSHOT_EXAMPLES_REACT,
        prompt=CLIN_INSTRUCTION_MBPP,
        summary_prompt=CLIN_SUMMARY_INSTRUCTION_MBPP,
        meta_summary_prompt=CLIN_META_SUMMARY_INSTRUCTION_MBPP,
        additional_keys={},
        summary_additional_keys={},
        meta_summary_additional_keys={},
        summary_system=CLIN_ADAPT_SUMMARY_SYSTEM,
        meta_summary_system=CLIN_ADAPT_META_SUMMARY_SYSTEM,
        quadrant="adapt",
        patience=1,
        reset=False,
    )
    print(repr(output))
    assert output == gt_out

def test_generate_react() -> None:
    """Test CLIN Code strategy generate react."""
    question = """Read the following table regarding "Bowling Scores" and then write Python code to answer a question:

    Name | Score
    Amanda | 117
    Sam | 236
    Irma | 144
    Mike | 164

    Question: Some friends went bowling and kept track of their scores. How many more points did Mike score than Irma?"""
    key = "20"

    gt_react_steps = [CLINReActStepOutput(thought="I need to find the difference between Mike's and Irma's scores.", action_type='Calculate', query='\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n', observation='\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\nExecution Status: Done\nOutput: answer = 20', answer='\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n', external_tool_info={'execution_status': 'Done', 'code_answer': 20}, is_correct=True, thought_response=Response(input_text='', output_text="I need to find the difference between Mike's and Irma's scores. \n\nAction 1: Calculate[\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n]\n\nObservation 1: \n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n\nExecution Status: Done\n\nOutput: answer = 20\n\nThought 2: The difference in score between Mike and Irma is 20 points. \n\nAction 2: Finish[\n```python\nanswer = 20\n```\n]\n\nObservation 2: \n```python\nanswer = 20\n```", prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5), action_response=Response(input_text='', output_text='Calculate[\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n]', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)), CLINReActStepOutput(thought="The difference between Mike's and Irma's scores is 20 points.", action_type='Finish', query='\n```python\nanswer = 20\n```\n', observation='Answer is CORRECT', answer='\n```python\nanswer = 20\n```\n', external_tool_info={'execution_status': 'Done', 'code_answer': 20}, is_correct=True, thought_response=Response(input_text='', output_text="The difference between Mike's and Irma's scores is 20 points.\nAction 2: Finish[\n```python\nanswer = 20\n```\n]\nObservation 2: \n```python\nanswer = 20\n```", prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5), action_response=Response(input_text='', output_text='Finish[\n```python\nanswer = 20\n```\n]\nObservation 2: \n```python\nanswer = 20\n```', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5))]
    gt_scratchpad = "\nThought 1: I need to find the difference between Mike's and Irma's scores.\nAction 1: Calculate[\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n]\nObservation 1: \n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\nExecution Status: Done\nOutput: answer = 20\nThought 2: The difference between Mike's and Irma's scores is 20 points.\nAction 2: Finish[\n```python\nanswer = 20\n```\n]\nObservation 2: Answer is CORRECT"
    responses = ["I need to find the difference between Mike's and Irma's scores. \n\nAction 1: Calculate[\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n]\n\nObservation 1: \n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n\nExecution Status: Done\n\nOutput: answer = 20\n\nThought 2: The difference in score between Mike and Irma is 20 points. \n\nAction 2: Finish[\n```python\nanswer = 20\n```\n]\n\nObservation 2: \n```python\nanswer = 20\n```", 
                 'Calculate[\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n]', 
                 "The difference between Mike's and Irma's scores is 20 points.\nAction 2: Finish[\n```python\nanswer = 20\n```\n]\nObservation 2: \n```python\nanswer = 20\n```",
                 'Finish[\n```python\nanswer = 20\n```\n]\nObservation 2: \n```python\nanswer = 20\n```', 
                 '1. Knowing how to calculate the difference between two values MAY BE NECESSARY to comparing scores accurately.\n2. Understanding how to assign values to variables MAY CONTRIBUTE to performing mathematical operations in Python.', 
                 "I need to calculate the difference between Mike's and Irma's scores.\nAction 1: Calculate[\n```python\nmike_score = 164\nirma_score = 144\ndifference = mike_score - irma_score\nanswer = difference\n```\n]\nObservation 1: \n```python\nmike_score = 164\nirma_score = 144\ndifference = mike_score - irma_score\nanswer = difference\n```\nExecution Status: Done\nOutput: answer = 20\nThought 2: Mike scored 20 points more than Irma.\nAction 2: Finish[\n```python\nanswer = 20\n```\n]\nObservation 2: \n```python\nanswer = 20\n```", 
                 'Calculate[\n```python\nmike_score = 164\nirma_score = 144\ndifference = mike_score - irma_score\nanswer = difference\n```\n]', 
                 'Mike scored 20 points more than Irma.\nAction 2: Finish[\n```python\nanswer = 20\n```\n]\nObservation 2: \n```python\nanswer = 20\n```', 
                 'Finish[\n```python\nanswer = 20\n```\n]\nObservation 2: \n```python\nanswer = 20\n```', 
                 '1. Knowing how to calculate the difference between two values MAY BE NECESSARY to comparing scores accurately.\n2. Understanding how to assign values to variables MAY CONTRIBUTE to performing mathematical operations in Python.\n3. Checking the calculated difference for accuracy SHOULD BE NECESSARY to ensure the correct answer is obtained.', 
                 ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CLINCodeStrategy(llm=llm, memory=None)
    step_idx, is_correct, scratchpad, finished, answer, react_steps = (
        strategy.generate_react(
            question=question,
            key=key,
            examples=TABMWP_FEWSHOT_EXAMPLES_REACT,
            summaries="",
            summary_system=CLIN_ADAPT_SUMMARY_SYSTEM,
            meta_summaries="",
            meta_summary_system=CLIN_ADAPT_META_SUMMARY_SYSTEM,
            prompt=CLIN_INSTRUCTION_TABMWP,
            additional_keys={},
        )
    )
    assert step_idx == 3
    assert is_correct == True
    assert scratchpad == gt_scratchpad
    assert finished == True
    assert answer == '\n```python\nanswer = 20\n```\n'
    assert react_steps == gt_react_steps


def test_generate_action() -> None:
    """Tests CLIN Code strategy generate action."""
    question = """Read the following table regarding "Bowling Scores" and then write Python code to answer a question:

    Name | Score
    Amanda | 117
    Sam | 236
    Irma | 144
    Mike | 164

    Question: Some friends went bowling and kept track of their scores. How many more points did Mike score than Irma?"""

    gt_scratchpad = '\nAction 1: Calculate[\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n]'
    responses = [
        "I need to find the difference between Mike's and Irma's scores. \n\nAction 1: Calculate[\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n]\n\nObservation 1: \n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n\nExecution Status: Done\n\nOutput: answer = 20\n\nThought 2: The difference in score between Mike and Irma is 20 points. \n\nAction 2: Finish[\n```python\nanswer = 20\n```\n]\n\nObservation 2: \n```python\nanswer = 20\n```",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CLINCodeStrategy(llm=llm)
    scratchpad, action_type, query, thought_response = strategy.generate_action(
        idx=1,
        scratchpad="",
        question=question,
        examples=TABMWP_FEWSHOT_EXAMPLES_REACT,
        summaries="",
        summary_system=CLIN_ADAPT_SUMMARY_SYSTEM,
        meta_summaries="",
        meta_summary_system=CLIN_ADAPT_META_SUMMARY_SYSTEM,
        prompt=CLIN_INSTRUCTION_TABMWP,
        additional_keys={},
    )

    assert action_type == "Calculate"
    assert query == '\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n'
    assert scratchpad == gt_scratchpad
    assert thought_response == Response(input_text='', output_text="I need to find the difference between Mike's and Irma's scores. \n\nAction 1: Calculate[\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n]\n\nObservation 1: \n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n\nExecution Status: Done\n\nOutput: answer = 20\n\nThought 2: The difference in score between Mike and Irma is 20 points. \n\nAction 2: Finish[\n```python\nanswer = 20\n```\n]\n\nObservation 2: \n```python\nanswer = 20\n```", prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)


def test_generate_observation() -> None:
    """Tests CLIN Code strategy generate observation."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = CLINCodeStrategy(llm=llm)
    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=1,
            scratchpad="",
            action_type="Calculate",
            query='\n\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n\n',
            key="key1",
        )
    )

    assert not is_correct
    assert isinstance(obs, str)
    assert external_tool_info == {'execution_status': 'Done', 'code_answer': 20}
    assert scratchpad == '\nObservation 1: \n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\nExecution Status: Done\nOutput: answer = 20'
    assert answer == '\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n'
    assert not finished


def test_generate_summary() -> None:
    """Test CLIN Code strategy generate summary."""
    gt_summary = "I need to find the difference between Mike's and Irma's scores. \n\nAction 1: Calculate[\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n]\n\nObservation 1: \n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n\nExecution Status: Done\n\nOutput: answer = 20\n\nThought 2: The difference in score between Mike and Irma is 20 points. \n\nAction 2: Finish[\n```python\nanswer = 20\n```\n]\n\nObservation 2: \n```python\nanswer = 20\n```"
    gt_summary_response = Response(input_text='', output_text="I need to find the difference between Mike's and Irma's scores. \n\nAction 1: Calculate[\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n]\n\nObservation 1: \n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n\nExecution Status: Done\n\nOutput: answer = 20\n\nThought 2: The difference in score between Mike and Irma is 20 points. \n\nAction 2: Finish[\n```python\nanswer = 20\n```\n]\n\nObservation 2: \n```python\nanswer = 20\n```", prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)
    llm = MockLLM(
        "gpt-3.5-turbo", responses = ["I need to find the difference between Mike's and Irma's scores. \n\nAction 1: Calculate[\n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n]\n\nObservation 1: \n```python\nmike_score = 164\nirma_score = 144\nanswer = mike_score - irma_score\n```\n\nExecution Status: Done\n\nOutput: answer = 20\n\nThought 2: The difference in score between Mike and Irma is 20 points. \n\nAction 2: Finish[\n```python\nanswer = 20\n```\n]\n\nObservation 2: \n```python\nanswer = 20\n```",]
    )
    strat = CLINCodeStrategy(llm=llm)
    summary, summary_response = strat.generate_summary(
        question="Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        previous_trials="",
        scratchpad="",
        is_correct=False,
        prompt=CLIN_SUMMARY_INSTRUCTION_TABMWP,
        additional_keys={},
    )

    assert summary == gt_summary
    assert summary_response == gt_summary_response

def test_halting_condition() -> None:
    """Test CLIN Code strategy halting condition."""
    strategy = CLINCodeStrategy(llm=None, memory=None)
    
    assert (
        strategy.halting_condition(
            idx=0,
            key="",
            answer="",
        )
        is False
    )
