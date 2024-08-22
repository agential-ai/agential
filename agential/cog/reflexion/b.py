from agential.cog.reflexion.output import (
    ReflexionReActOutput,
    ReflexionReActReActStepOutput,
    ReflexionReActStepOutput,
)
from agential.utils.metrics import PromptMetrics

ReflexionReActOutput(
    answer="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
    total_prompt_tokens=240,
    total_completion_tokens=480,
    total_tokens=720,
    total_prompt_cost=0.0003600000000000001,
    total_completion_cost=0.0009599999999999999,
    total_cost=0.00132,
    total_prompt_time=12.0,
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
                    observation="Answer is INCORRECT",
                    answer="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
                    external_tool_info={
                        "execution_status": "IndentationError('unexpected indent', ('<string>', 10, 4, '    assert first_repeated_char(\"abc\") == None\\n', 10, -1))"
                    },
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
            ],
            reflections=[],
            reflection_metrics=None,
        ),
        ReflexionReActStepOutput(
            steps=[
                ReflexionReActReActStepOutput(
                    thought="Implement[```pythondef first_repeated_char(s):    seen = set()    for char in s:        if char in seen:            return char        seen.add(char)    return None```]",
                    action_type="",
                    query="",
                    observation="Invalid Action. Valid Actions are Implement[code] Test[code] and Finish[answer].",
                    answer="",
                    external_tool_info={"execution_status": ""},
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
                    thought='Test[```pythonassert first_repeated_char("abcabc") == "a"assert first_repeated_char("abc") == Noneassert first_repeated_char("123123") == "1"```]Observation 2: ```pythondef first_repeated_char(s):    seen = set()    for char in s:        if char in seen:            return char        seen.add(char)    return Noneassert first_repeated_char("abcabc") == "a"assert first_repeated_char("abc") == Noneassert first_repeated_char("123123") == "1"```Execution Status: DoneThought 3: The function works correctly for the provided test cases.',
                    action_type="Test",
                    query="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
                    observation="\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\nExecution Status: Done",
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
                    thought="Finish[```pythondef first_repeated_char(s):    seen = set()    for char in s:        if char in seen:            return char        seen.add(char)    return None```]Observation 3:```pythondef first_repeated_char(s):    seen = set()    for char in s:        if char in seen:            return char        seen.add(char)    return None```",
                    action_type="Implement",
                    query="def first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None",
                    observation="\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: ",
                    answer="def first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None",
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
                    thought="Implement[```pythondef first_repeated_char(s):    seen = set()    for char in s:        if char in seen:            return char        seen.add(char)    return None```]",
                    action_type="",
                    query="",
                    observation="Invalid Action. Valid Actions are Implement[code] Test[code] and Finish[answer].",
                    answer="",
                    external_tool_info={"execution_status": ""},
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
                    thought='Test[```pythonassert first_repeated_char("abcabc") == "a"assert first_repeated_char("abc") == Noneassert first_repeated_char("123123") == "1"```]Observation 2: ```pythondef first_repeated_char(s):    seen = set()    for char in s:        if char in seen:            return char        seen.add(char)    return Noneassert first_repeated_char("abcabc") == "a"assert first_repeated_char("abc") == Noneassert first_repeated_char("123123") == "1"```Execution Status: DoneThought 3: The function works correctly for the provided test cases.',
                    action_type="Test",
                    query="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
                    observation="\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\nExecution Status: Done",
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
                    thought="Finish[```pythondef first_repeated_char(s):    seen = set()    for char in s:        if char in seen:            return char        seen.add(char)    return None```]Observation 3:```pythondef first_repeated_char(s):    seen = set()    for char in s:        if char in seen:            return char        seen.add(char)    return None```",
                    action_type="Implement",
                    query="def first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None",
                    observation="\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: ",
                    answer="def first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None",
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
            ],
            reflections=[
                'I need to write a function that finds the first repeated character in a given string by iterating through the characters and checking for duplicates.Action 1: Implement[```pythondef first_repeated_char(s):    seen_chars = set()    for char in s:        if char in seen_chars:            return char        seen_chars.add(char)    return None```]Observation 1: ```pythondef first_repeated_char(s):    seen_chars = set()    for char in s:        if char in seen_chars:            return char        seen_chars.add(char)    return None```Execution Status: DoneThought 2: I need to test the function to ensure it works correctly with different test cases.Action 2: Test[```pythonassert first_repeated_char("abcabc") == "a"assert first_repeated_char("abc") == Noneassert first_repeated_char("123123") == "1"```]Observation 2: ```pythondef first_repeated_char(s):    seen_chars = set()    for char in s:        if char in seen_chars:            return char        seen_chars.add(char)    return Noneassert first_repeated_char("abcabc") == "a"assert first_repeated_char("abc") == Noneassert first_repeated_char("123123") == "1"```Execution Status: DoneThought 3: The function works correctly for the provided test cases.Action 3: Finish[```pythondef first_repeated_char(s):    seen_chars = set()    for char in s:        if char in seen_chars:            return char        seen_chars.add(char)    return None```]Observation 3:```pythondef first_repeated_char(s):    seen_chars = set()    for char in s:        if char in seen_chars:            return char        seen_chars.add(char)    return None```'
            ],
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
        ReflexionReActStepOutput(
            steps=[
                ReflexionReActReActStepOutput(
                    thought="I need to test the function to ensure it works correctly with different test cases.",
                    action_type="Test",
                    query='assert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                    observation='\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done',
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
                    observation="Answer is INCORRECT",
                    answer="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
                    external_tool_info={
                        "execution_status": "IndentationError('unexpected indent', ('<string>', 10, 4, '    assert first_repeated_char(\"abc\") == None\\n', 10, -1))"
                    },
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
            ],
            reflections=[
                'I need to write a function that finds the first repeated character in a given string by iterating through the characters and checking for duplicates.Action 1: Implement[```pythondef first_repeated_char(s):    seen_chars = set()    for char in s:        if char in seen_chars:            return char        seen_chars.add(char)    return None```]Observation 1: ```pythondef first_repeated_char(s):    seen_chars = set()    for char in s:        if char in seen_chars:            return char        seen_chars.add(char)    return None```Execution Status: DoneThought 2: I need to test the function to ensure it works correctly with different test cases.Action 2: Test[```pythonassert first_repeated_char("abcabc") == "a"assert first_repeated_char("abc") == Noneassert first_repeated_char("123123") == "1"```]Observation 2: ```pythondef first_repeated_char(s):    seen_chars = set()    for char in s:        if char in seen_chars:            return char        seen_chars.add(char)    return Noneassert first_repeated_char("abcabc") == "a"assert first_repeated_char("abc") == Noneassert first_repeated_char("123123") == "1"```Execution Status: DoneThought 3: The function works correctly for the provided test cases.Action 3: Finish[```pythondef first_repeated_char(s):    seen_chars = set()    for char in s:        if char in seen_chars:            return char        seen_chars.add(char)    return None```]Observation 3:```pythondef first_repeated_char(s):    seen_chars = set()    for char in s:        if char in seen_chars:            return char        seen_chars.add(char)    return None```',
                "Implement[```pythondef first_repeated_char(s):    seen = set()    for char in s:        if char in seen:            return char        seen.add(char)    return None```]",
            ],
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
    'I need to write a function that finds the first repeated character in a given string by iterating through the characters and checking for duplicates.\nAction 1: Implement[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 1: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\nExecution Status: Done\nThought 2: I need to test the function to ensure it works correctly with different test cases.\nAction 2: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen_chars = set()\n    for char in s:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```',
    "Implement[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]",
    "I need to test the function to ensure it works correctly with different test cases.",
    'Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation 2: \n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\nExecution Status: Done\nThought 3: The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3: \n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```',
    "The function works correctly for the provided test cases.\nAction 3: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
    "Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation 3:\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
]