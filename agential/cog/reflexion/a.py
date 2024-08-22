from agential.cog.reflexion.output import (
    ReflexionReActOutput,
    ReflexionReActReActStepOutput,
    ReflexionReActStepOutput,
)
from agential.utils.metrics import PromptMetrics

ReflexionReActOutput(
    answer="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
    total_prompt_tokens=60,
    total_completion_tokens=120,
    total_tokens=180,
    total_prompt_cost=9e-05,
    total_completion_cost=0.00023999999999999998,
    total_cost=0.00033,
    total_prompt_time=3.0,
    total_time=0.5,
    additional_info=[
        ReflexionReActStepOutput(
            steps=[
                ReflexionReActReActStepOutput(
                    thought="I need to write a function that finds the first repeated character in a given string by iterating through the characters and checking for duplicates.",
                    action_type="Implement",
                    query="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
                    observation="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\nExecution Status: ",
                    answer="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
                    external_tool_info={"execution_status": "Done"},
                    is_correct=False,
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
                ),
                ReflexionReActReActStepOutput(
                    thought="I need to test the function to ensure it works correctly with different test cases.",
                    action_type="Test",
                    query='assert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                    observation='\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done',
                    answer="",
                    external_tool_info={"execution_status": "Done"},
                    is_correct=True,
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
                ),
                ReflexionReActReActStepOutput(
                    thought="The function works correctly for the provided test cases.",
                    action_type="Finish",
                    query="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
                    observation="Answer is CORRECT",
                    answer="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
                    external_tool_info={"execution_status": "Done"},
                    is_correct=True,
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
                ),
            ],
            reflections=[],
            reflection_metrics=None,
        )
    ],
)


responses = [
    'I need to write a function that finds the first repeated character in a given string by iterating through the characters and checking for duplicates.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```',
    "Implement[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]",
    "I need to test the function to ensure it works correctly with different test cases.",
    'Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3: \n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```',
    "The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
    "Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
]