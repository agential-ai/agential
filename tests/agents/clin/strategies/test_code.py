"""Test the CLIN Code strategy."""

import tiktoken

from agential.agents.clin.output import CLINOutput, CLINReActStepOutput, CLINStepOutput
from agential.agents.clin.prompts import (
    CLIN_ADAPT_META_SUMMARY_SYSTEM,
    CLIN_ADAPT_SUMMARY_SYSTEM,
    CLIN_INSTRUCTION_HUMANEVAL,
    CLIN_META_SUMMARY_INSTRUCTION_HUMANEVAL,
    CLIN_SUMMARY_INSTRUCTION_HUMANEVAL,
    CLIN_SUMMARY_INSTRUCTION_MBPP,
)
from agential.agents.clin.strategies.code import CLINCodeStrategy, CLINMBPPStrategy
from agential.core.fewshots.humaneval import HUMANEVAL_FEWSHOT_EXAMPLES_REACT
from agential.core.fewshots.mbpp import MBPP_FEWSHOT_EXAMPLES_REACT
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

    gt_out = CLINOutput(
        answer="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n",
        total_prompt_tokens=70,
        total_completion_tokens=140,
        total_tokens=210,
        total_prompt_cost=0.000105,
        total_completion_cost=0.00028,
        total_cost=0.000385,
        total_prompt_time=3.5,
        total_time=0.5,
        additional_info=[
            CLINStepOutput(
                steps=[
                    CLINReActStepOutput(
                        thought="I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.",
                        action_type="Implement",
                        query="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n",
                        observation="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\nExecution Status: ",
                        answer="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n",
                        external_tool_info={"execution_status": "Done"},
                        is_correct=False,
                        thought_response=Response(
                            input_text="",
                            output_text='I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in the string.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```',
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_response=Response(
                            input_text="",
                            output_text="Implement[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]",
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    CLINReActStepOutput(
                        thought="I need to test the function to ensure it works correctly with different test cases.",
                        action_type="Test",
                        query='\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n',
                        observation='\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done',
                        answer="\n```python\n\n```\n",
                        external_tool_info={"execution_status": "Done"},
                        is_correct=True,
                        thought_response=Response(
                            input_text="",
                            output_text="I need to test the function to ensure it works correctly with different test cases.",
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_response=Response(
                            input_text="",
                            output_text='Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```',
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    CLINReActStepOutput(
                        thought="The function works correctly for the provided test cases.",
                        action_type="Finish",
                        query="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n",
                        observation="Answer is INCORRECT",
                        answer="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n",
                        external_tool_info={
                            "execution_status": "IndentationError('unexpected indent', ('<string>', 10, 4, '    assert first_repeated_char(\"abc\") == None\\n', 10, -1))"
                        },
                        is_correct=False,
                        thought_response=Response(
                            input_text="",
                            output_text="The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_response=Response(
                            input_text="",
                            output_text="Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
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
                summaries="1. Keeping track of characters seen so far in a set may be necessary to find the first repeated character in a given string.\n2. Testing the function with different test cases may contribute to ensuring its correctness.",
                summaries_response=Response(
                    input_text="",
                    output_text="1. Keeping track of characters seen so far in a set may be necessary to find the first repeated character in a given string.\n2. Testing the function with different test cases may contribute to ensuring its correctness.",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                meta_summaries="",
                previous_trials="",
            )
        ],
    )
    responses = [
        'I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in the string.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```',
        "Implement[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]",
        "I need to test the function to ensure it works correctly with different test cases.",
        'Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```',
        "The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
        "Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
        "1. Keeping track of characters seen so far in a set may be necessary to find the first repeated character in a given string.\n2. Testing the function with different test cases may contribute to ensuring its correctness.",
        'I need to write a function that can find the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in a given string for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```',
        "Implement[\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```\n]",
        "I need to test the function to ensure it works correctly with different test cases.",
        'Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]',
        "The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```",
        "Finish[\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```",
        "1. Keeping track of characters seen so far in a set may be necessary to find the first repeated character in a given string.\n2. Testing the function with different test cases may contribute to ensuring its correctness.",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CLINCodeStrategy(llm=llm, testing=True)
    output = strategy.generate(
        question=question,
        key=key,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
        prompt=CLIN_INSTRUCTION_HUMANEVAL,
        summary_prompt=CLIN_SUMMARY_INSTRUCTION_HUMANEVAL,
        meta_summary_prompt=CLIN_META_SUMMARY_INSTRUCTION_HUMANEVAL,
        additional_keys={"tests": key},
        summary_additional_keys={"tests": key},
        meta_summary_additional_keys={"tests": key},
        summary_system=CLIN_ADAPT_SUMMARY_SYSTEM,
        meta_summary_system=CLIN_ADAPT_META_SUMMARY_SYSTEM,
        quadrant="adapt",
        patience=1,
        reset=False,
    )

    assert output == gt_out


def test_generate_react() -> None:
    """Test CLIN Code strategy generate react."""
    question = "Write a python function to find the first repeated character in a given string."
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_react_steps = [
        CLINReActStepOutput(
            thought="I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.",
            action_type="Implement",
            query="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n",
            observation="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\nExecution Status: ",
            answer="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n",
            external_tool_info={"execution_status": "Done"},
            is_correct=False,
            thought_response=Response(
                input_text="",
                output_text='I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in the string.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```',
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            action_response=Response(
                input_text="",
                output_text="Implement[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
        CLINReActStepOutput(
            thought="I need to test the function to ensure it works correctly with different test cases.",
            action_type="Test",
            query='\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n',
            observation='\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done',
            answer="\n```python\n\n```\n",
            external_tool_info={"execution_status": "Done"},
            is_correct=True,
            thought_response=Response(
                input_text="",
                output_text="I need to test the function to ensure it works correctly with different test cases.",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            action_response=Response(
                input_text="",
                output_text='Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```',
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
        CLINReActStepOutput(
            thought="The function works correctly for the provided test cases.",
            action_type="Finish",
            query="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n",
            observation="Answer is INCORRECT",
            answer="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n",
            external_tool_info={
                "execution_status": "IndentationError('unexpected indent', ('<string>', 10, 4, '    assert first_repeated_char(\"abc\") == None\\n', 10, -1))"
            },
            is_correct=False,
            thought_response=Response(
                input_text="",
                output_text="The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            action_response=Response(
                input_text="",
                output_text="Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
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
    gt_scratchpad = '\nThought 1: I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\nExecution Status: \nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3: Answer is INCORRECT'
    responses = [
        'I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in the string.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```',
        "Implement[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]",
        "I need to test the function to ensure it works correctly with different test cases.",
        'Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```',
        "The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
        "Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
        "1. Keeping track of characters seen so far in a set may be necessary to find the first repeated character in a given string.\n2. Testing the function with different test cases may contribute to ensuring its correctness.",
        'I need to write a function that can find the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in a given string for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(input_str):\n    seen = set()\n    for char in input_str:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```',
        "Implement[\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```\n]",
        "I need to test the function to ensure it works correctly with different test cases.",
        'Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]',
        "The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```",
        "Finish[\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    chars_seen = set()\n    for char in s:\n        if char in chars_seen:\n            return char\n        chars_seen.add(char)\n    return None\n```",
        "1. Keeping track of characters seen so far in a set may be necessary to find the first repeated character in a given string.\n2. Testing the function with different test cases may contribute to ensuring its correctness.",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CLINCodeStrategy(llm=llm, memory=None)
    step_idx, is_correct, scratchpad, finished, answer, react_steps = (
        strategy.generate_react(
            question=question,
            key=key,
            examples=HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
            summaries="",
            summary_system=CLIN_ADAPT_SUMMARY_SYSTEM,
            meta_summaries="",
            meta_summary_system=CLIN_ADAPT_META_SUMMARY_SYSTEM,
            prompt=CLIN_INSTRUCTION_HUMANEVAL,
            additional_keys={"tests": key},
        )
    )
    assert step_idx == 4
    assert is_correct == False
    assert scratchpad == gt_scratchpad
    assert finished == True
    assert (
        answer
        == "\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n"
    )
    assert react_steps == gt_react_steps


def test_generate_action() -> None:
    """Tests CLIN Code strategy generate action."""
    question = "Write a python function to find the first repeated character in a given string."
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_scratchpad = "\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]"
    responses = [
        'I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in the string.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```',
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CLINCodeStrategy(llm=llm)
    scratchpad, action_type, query, thought_response = strategy.generate_action(
        idx=1,
        scratchpad="",
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
        summaries="",
        summary_system=CLIN_ADAPT_SUMMARY_SYSTEM,
        meta_summaries="",
        meta_summary_system=CLIN_ADAPT_META_SUMMARY_SYSTEM,
        prompt=CLIN_INSTRUCTION_HUMANEVAL,
        additional_keys={"tests": key},
    )

    assert action_type == "Implement"
    assert (
        query
        == "\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n"
    )
    assert scratchpad == gt_scratchpad
    assert thought_response == Response(
        input_text="",
        output_text='I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in the string.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```',
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_generate_observation() -> None:
    """Tests CLIN Code strategy generate observation."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = CLINCodeStrategy(llm=llm)
    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=1,
            scratchpad="",
            action_type="Calculate",
            query="\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n",
            key="key1",
        )
    )

    assert not is_correct
    assert isinstance(obs, str)
    assert external_tool_info == {"execution_status": ""}
    assert (
        scratchpad
        == "\nObservation 1: Invalid Action. Valid Actions are Implement[\\n```python\\n<code>\\n```\\n], Test[\\n```python\\n<code>\\n```\\n], and Finish[\\n```python\\n<answer>\\n```\\n]."
    )
    assert answer == "\n```python\n\n```\n"
    assert not finished


def test_generate_summary() -> None:
    """Test CLIN Code strategy generate summary."""
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_summary = 'I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in the string.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```'
    gt_summary_response = Response(
        input_text="",
        output_text='I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in the string.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```',
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )
    llm = MockLLM(
        "gpt-3.5-turbo",
        responses=[
            'I need to write a function that finds the first repeated character in a given string by keeping track of characters seen so far.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function now correctly finds the first repeated character in the string.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```',
        ],
    )
    strat = CLINCodeStrategy(llm=llm)
    summary, summary_response = strat.generate_summary(
        question="Write a python function to find the first repeated character in a given string.",
        previous_trials="",
        scratchpad="",
        is_correct=False,
        prompt=CLIN_SUMMARY_INSTRUCTION_HUMANEVAL,
        additional_keys={"tests": key},
    )

    assert summary == gt_summary
    assert summary_response == gt_summary_response


def test_mbpp_generate_summary() -> None:
    """Test CLIN MBPP strategy generate summary."""
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_summary = "Thought: I need to find the capital of France."
    gt_summary_response = Response(
        input_text="",
        output_text="Thought: I need to find the capital of France.",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )
    llm = MockLLM(
        "gpt-3.5-turbo", responses=["Thought: I need to find the capital of France."]
    )
    strat = CLINMBPPStrategy(llm=llm)
    summary, summary_response = strat.generate_summary(
        question="What is the capital of France?",
        previous_trials="",
        scratchpad="",
        is_correct=False,
        prompt=CLIN_SUMMARY_INSTRUCTION_MBPP,
        additional_keys={"tests": key},
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
        is True
    )
