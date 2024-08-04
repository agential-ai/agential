"""Unit tests for LATS Code strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from agential.cog.lats.strategies.code import (
    LATSCodeStrategy,
    LATSHEvalStrategy,
    LATSMBPPStrategy,
    parse_latest_implement,
    get_node_trajectory_code,
    parse_code_action,
    parse_code_value,
)
from agential.cog.react.output import ReActOutput
from agential.cog.lats.node import Node
from agential.cog.fewshots.humaneval import HUMANEVAL_FEWSHOT_EXAMPLES_REACT
from agential.cog.lats.prompts import (
    HUMANEVAL_FEWSHOT_EXAMPLES_LATS_REFLECT,
    LATS_INSTRUCTION_HUMANEVAL,
    LATS_REFLECT_INSTRUCTION_HUMANEVAL,
)

def test_parse_latest_implement() -> None:
    """Test parse_latest_implement function."""
    
    # Test case with single implementation.
    single_impl = """
    Some text
    Implement[```python
    def add(a, b):
        return a + b
    ```]
    More text
    """
    assert parse_latest_implement(single_impl) == "def add(a, b):\n        return a + b"

    # Test case with multiple implementations.
    multiple_impl = """
    Implement[```python
    def subtract(a, b):
        return a - b
    ```]
    Some text
    Implement[```python
    def multiply(a, b):
        return a * b
    ```]
    """
    assert parse_latest_implement(multiple_impl) == "def multiply(a, b):\n        return a * b"

    # Test case with no implementation.
    no_impl = "Some text without any implementation"
    assert parse_latest_implement(no_impl) == ""

    # Test case with empty implementation.
    empty_impl = "Implement[```python\n```]"
    assert parse_latest_implement(empty_impl) == ""

    # Test case with multiple lines in implementation.
    multi_line_impl = """
    Implement[```python
    def complex_function(x):
        if x > 0:
            return x * 2
        else:
            return x * -1
    ```]
    """
    expected_multi_line = """def complex_function(x):
        if x > 0:
            return x * 2
        else:
            return x * -1"""
    assert parse_latest_implement(multi_line_impl) == expected_multi_line


def test_get_node_trajectory_code() -> None:
    """Tests the get_node_trajectory_code() function."""
    root = Node(
        state=ReActOutput(
            **{
                "thought": "Root thought",
                "action_type": "",
                "query": "",
                "observation": "",
                "answer": "",
                "external_tool_info": {},
            }
        )
    )
    child1 = Node(
        state=ReActOutput(
            **{
                "thought": "Child1 thought",
                "action_type": "Lookup",
                "query": "topic",
                "observation": "",
                "answer": "",
                "external_tool_info": {},
            }
        ),
        parent=root,
    )
    child2 = Node(
        state=ReActOutput(
            **{
                "thought": "Child2 thought",
                "action_type": "Finish",
                "query": "answer",
                "observation": "Answer correct",
                "answer": "",
                "external_tool_info": {},
            }
        ),
        parent=child1,
    )

    expected_trajectory = "\nThought 1: Child1 thought\nAction 1: Lookup[\n```python\ntopic\n```\n]\nThought 2: Child2 thought\nAction 2: Finish[\n```python\nanswer\n```\n]\nObservation 2: Answer correct"
    assert get_node_trajectory_code(child2) == expected_trajectory

    # Test root node.
    root = Node()
    assert get_node_trajectory_code(root) == ""


def test_parse_code_action() -> None:
    """Test parse_code_action function."""
    test_cases = [
        {
            "input": "Implement[```python\ndef add(a, b): return a + b\n```]",
            "expected": ("Implement", "def add(a, b): return a + b"),
        },
        {
            "input": "TEST[```python\nassert add(2, 3) == 5\n```]",
            "expected": ("Test", "assert add(2, 3) == 5"),
        },
        {
            "input": "finish[```python\nprint('Done')\n```]",
            "expected": ("Finish", "print('Done')"),
        },
        {
            "input": "Invalid[```python\nThis should not match\n```]",
            "expected": ("", ""),
        },
        {
            "input": "Implement[```python\n \n```]",
            "expected": ("Implement", ""),
        },
        {
            "input": "Something else entirely",
            "expected": ("", ""),
        },
    ]

    for case in test_cases:
        result = parse_code_action(case["input"])
        assert result == case["expected"]

    exception_case = "Implement[```python\nincomplete code"
    result = parse_code_action(exception_case)
    assert result == ('Implement', 'incomplete code')


def test_parse_code_value() -> None:
    """Test the parse_code_value function."""
    # Test valid value strings.
    valid_input = (
        "Some text. Explanation: This is the explanation. Correctness score: 5"
    )
    assert parse_code_value(valid_input) == ("This is the explanation.", 5)

    # Test invalid value strings.
    assert parse_code_value("No explanation or score") == ("Explanation not found", 0)
    assert parse_code_value("Explanation: Only explanation") == (
        "Explanation not found",
        0,
    )
    assert parse_code_value("Correctness score: 5") == ("Explanation not found", 0)

    # Test edge cases.
    assert parse_code_value("Explanation: Empty. Correctness score: 0") == ("Empty.", 0)
    assert parse_code_value(
        "Explanation: Multi-line\nexplanation. Correctness score: 10"
    ) == ("Multi-line\nexplanation.", 10)

    # Test with unexpected format.
    assert parse_code_value("Explanation: Tricky: score. Correctness score: 7") == (
        "Tricky: score.",
        7,
    )


def test_init() -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSCodeStrategy(
        llm=llm,
        n_samples=5,
        max_reflections=4,
        depth_limit=7,
        max_unique=5,
        cache_values=True,
    )

    assert strategy.llm == llm
    assert strategy.n_samples == 5
    assert strategy.max_reflections == 4
    assert strategy.depth_limit == 7
    assert strategy.max_unique == 5
    assert strategy.cache_values is True
    assert strategy.root is None
    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == []
    assert strategy.value_cache == {}


def test_initialize() -> None:
    """Test the initialize method."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSCodeStrategy(llm=llm)

    node = strategy.initialize()

    assert strategy.root == node
    assert strategy.root is not None
    assert isinstance(strategy.root, Node)
    assert strategy.root.state.thought == ""
    assert strategy.root.state.action_type == ""
    assert strategy.root.state.query == ""
    assert strategy.root.state.observation == ""
    assert strategy.root.state.external_tool_info == {}


def test_generate_thought() -> None:
    """Test the generate_thought method."""
    llm = FakeListChatModel(
        responses=["I should search for information about the topic."]
    )
    strategy = LATSCodeStrategy(llm=llm)

    question = "What is the capital of France?"
    examples = "Example 1\nExample 2"
    trajectory = "Previous thought"
    reflections = "Reflection 1\nReflection 2"
    depth = 1
    prompt = "Generate a thought"
    additional_keys = {"key": "value"}

    updated_trajectory, thought = strategy.generate_thought(
        question, examples, trajectory, reflections, depth, prompt, additional_keys
    )

    assert thought == "I should search for information about the topic."
    assert (
        updated_trajectory
        == "Previous thought\nThought 2: I should search for information about the topic."
    )


def test_generate_action() -> None:
    """Test the generate_action method."""
    llm = FakeListChatModel(responses=["Implement[```python\nresult = 2 + 2\n```]"])
    strategy = LATSCodeStrategy(llm=llm)

    question = "What is 2 + 2?"
    examples = "Example 1\nExample 2"
    trajectory = "Thought 1: I need to calculate 2 + 2."
    reflections = "Reflection 1\nReflection 2"
    depth = 0
    prompt = "Generate an action"
    additional_keys = {"key": "value"}

    trajectory, action_type, query = strategy.generate_action(
        question, examples, trajectory, reflections, depth, prompt, additional_keys
    )

    assert (
        trajectory
        == "Thought 1: I need to calculate 2 + 2.\nAction 1: Implement[\n```python\nresult = 2 + 2\n```\n]"
    )
    assert action_type == "Implement"
    assert query == "result = 2 + 2"


def test_generate_observation() -> None:
    """Test the generate_observation method."""
    strategy = LATSCodeStrategy(llm=FakeListChatModel(responses=[]))

    # Test Finish action.
    finish_result = strategy.generate_observation("assert x == 10", "Finish", "x = 10", "Previous trajectory", 1)
    assert finish_result == ('Previous trajectory\nObservation 2: Answer is CORRECT', 1, 'Answer is CORRECT', True, {'execution_status': 'Done'})

    # Test Implement action.
    implement_result = strategy.generate_observation("", "Implement", "def add(a, b): return a + b", "Previous trajectory", 2)
    assert implement_result == ('Previous trajectory\nObservation 3: \n```python\ndef add(a, b): return a + b\n```\nExecution Status: ', 0, '\n```python\ndef add(a, b): return a + b\n```\nExecution Status: ', False, {'execution_status': 'Done'})

    # Test Test action.
    test_result = strategy.generate_observation("", "Test", "assert add(2, 3) == 5", "Previous trajectory\nImplement[```python\ndef add(a, b): return a + b\n```]", 3)
    assert test_result == ('Previous trajectory\nImplement[```python\ndef add(a, b): return a + b\n```]\nObservation 4: \n```python\ndef add(a, b): return a + b\n\nassert add(2, 3) == 5\n```\nExecution Status: Done', 0, '\n```python\ndef add(a, b): return a + b\n\nassert add(2, 3) == 5\n```\nExecution Status: Done', False, {'execution_status': 'Done'})

    # Test invalid action.
    invalid_result = strategy.generate_observation("", "Invalid", "query", "Previous trajectory", 4)
    assert invalid_result == ('Previous trajectory\nObservation 5: Invalid Action. Valid Actions are Implement[code] Test[code] and Finish[answer].', 0, 'Invalid Action. Valid Actions are Implement[code] Test[code] and Finish[answer].', False, {'execution_status': ''})


def test_generate() -> None:
    """Test the generate method."""
    gt_states = [
        ReActOutput(thought='We need to iterate through the list of numbers and check if any two numbers are closer to each other than the given threshold.', action_type='Implement', query='from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False', observation='\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\nExecution Status: ', answer='', external_tool_info={'execution_status': 'Done'}),
        ReActOutput(thought='I need to iterate through the list of numbers and compare each pair to see if they are closer to each other than the threshold.', action_type='Implement', query='from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False', observation='\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\nExecution Status: ', answer='', external_tool_info={'execution_status': 'Done'}),
        ReActOutput(thought='We need to iterate through the list of numbers and check if any two numbers are closer to each other than the given threshold.', action_type='Implement', query='from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False', observation='\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\nExecution Status: ', answer='', external_tool_info={'execution_status': 'Done'}),
        ReActOutput(thought='To solve this problem, I need to iterate through the list of numbers and compare each pair of numbers to see if they are closer to each other than the threshold.', action_type='Implement', query='from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False', observation='\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\nExecution Status: ', answer='', external_tool_info={'execution_status': 'Done'}),
    ]
    responses = [
        'We need to iterate through the list of numbers and check if any two numbers are closer to each other than the given threshold.\n\nAction 1: Implement\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation 1: The implementation checks for each pair of numbers in the list if they are closer than the threshold and returns True if found, otherwise False.\n\n\nThought 2: We should test the implemented function with some test cases.\n\nAction 2: Test\n\n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\n\nObservation 2: The test cases pass successfully, indicating that the implementation is correct.\n\n\nThought 3: We have successfully implemented and tested the function. Now we can finish the task.\n\nAction 3: Finish\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'Implement the function to check if any two numbers in the list are closer to each other than the given threshold.\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation 1: The implementation checks all pairs of numbers in the list and returns True if any two numbers are closer to each other than the threshold.\n\nThought 2: We need to test the implementation with some test cases to verify if it works correctly.\nAction 2: \n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert has_close_elements([], 0.5) == False\n    assert has_close_elements([1.0, 2.0, 3.0], 2.0) == True\n    assert has_close_elements([1.0, 2.0, 3.0], 3.0) == True\n\ntest_has_close_elements()\n```\n\nObservation 2: All test cases passed successfully, indicating that the implementation is correct.\n\nFinish: \n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'We need to iterate through the list of numbers and check if any two numbers are closer to each other than the given threshold.\n\nAction 1: Implement\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation 1: The implementation checks each pair of numbers in the list and returns True if the absolute difference between them is less than the threshold.\n\nThought 2: We should test the implemented function with some test cases to verify its correctness.\n\nAction 2: Test\n\n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\n\nObservation 2: The test cases passed successfully, indicating that the implemented function is working correctly.\n\nAction 3: Finish\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'Implement the function to check if any two numbers are closer to each other than the given threshold.\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nThought 2: Now, we need to test the implemented function with some test cases.\nAction 2: \n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n\ntest_has_close_elements()\n```\n\nThought 3: The function seems to be working correctly based on the test cases.\nAction 3: \n\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nFinish: The function to check if any two numbers are closer to each other than the given threshold has been implemented successfully.',
        'I need to iterate through the list of numbers and compare each pair to see if they are closer to each other than the threshold.\n\nAction 1:\nImplement the function to check for close elements in the list.\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation 1: \nThe function compares each pair of numbers in the list and returns True if any pair is closer than the threshold.\n\nThought 2:\nI need to test the function to make sure it works correctly.\n\nAction 2:\nImplement test cases to check the function.\n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert has_close_elements([1.0, 1.1, 1.2], 0.1) == True\n    assert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.5) == False\n\ntest_has_close_elements()\n```\n\nObservation 2:\nThe test cases pass, and the function correctly identifies close elements in the list.\n\nFinish:\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        ' Implement the function to check if any two numbers are closer than the threshold.\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nThought 2: Now I need to test the implemented function with test cases.\nAction 2: Test the implemented function.\n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n\ntest_has_close_elements()\n```\n\nThought 3: The function passed the test cases successfully. I can now finish and submit the code.\nAction 3: Finish and provide the final code.\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'We need to iterate through the list of numbers and check if any two numbers are closer to each other than the given threshold.\n\nAction 1: Implement\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation 1: The function has been implemented to check for close elements in the list.\n\nThought 2: We should test the implemented function with some test cases to verify its correctness.\n\nAction 2: Test\n\n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\n\nObservation 2: The test cases have passed successfully.\n\nThought 3: The implementation is correct and the function is working as expected.\n\nAction 3: Finish\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'Implement the code to check for close elements in the list.\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nThought 2: Now that we have implemented the code, we should test it with some test cases to ensure it works correctly.\nAction 2:\n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n\ntest_has_close_elements()\n```\n\nThought 3: The code passed the test cases successfully, so we can consider it finished.\nAction 3:\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'To solve this problem, I need to iterate through the list of numbers and compare each pair of numbers to see if they are closer to each other than the threshold.\n\nAction 1:\n\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation 1: I have implemented the function to iterate through the list of numbers and check if any two numbers are closer to each other than the given threshold.\n\nAction 2:\n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n\ntest_has_close_elements()\n```\n\nObservation 2: The test cases passed successfully, indicating that the function is working correctly.\n\nFinish:\n\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'Implement the has_close_elements function:\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation 1: The implementation of the has_close_elements function seems correct as it iterates through the list of numbers and compares each pair of numbers to check if they are closer than the threshold.\n\nThought 2: Now, I need to test the implemented function with some test cases to verify its correctness.\nAction 2:',
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = LATSCodeStrategy(llm=llm)

    inst = {"task_id": "HumanEval/0", "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n", "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"}
    question = inst['prompt']
    key = f"{inst['test']}\ncheck({inst['entry_point']})"

    root = strategy.initialize()

    children_nodes = strategy.generate(
        node=root,
        question=question,
        key=key,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=HUMANEVAL_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_HUMANEVAL,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_HUMANEVAL,
        additional_keys={},
        reflect_additional_keys={},
    )
    assert len(children_nodes) == 4
    for gt_state, node in zip(gt_states, children_nodes):
        assert node.state == gt_state
        assert node.depth == 1
        assert node.reward == 0
        assert node.value == 0
        assert node.is_terminal is False
        assert node.visits == 0

    # Test generate with reflections.
    gt_states = [

    ]
    responses = [
        'My reasoning failed in the previous trial because I did not iterate through the list to compare each pair of numbers against the threshold. To mitigate this failure, I should implement a nested loop to compare all possible pairs of numbers in the list and return True if any pair is closer than the threshold.\n\nHigh-level plan:\n1. Implement a nested loop to iterate through all possible pairs of numbers in the list.\n2. Calculate the absolute difference between each pair of numbers.\n3. Check if the absolute difference is less than the threshold.\n4. If any pair meets the condition, return True.\n5. If no pair meets the condition, return False.',
        'My reasoning potentially failed because I did not provide an implementation for the `has_close_elements` function, leaving it with a `pass` statement. To mitigate this failure, I should ensure to complete the implementation of the function by iterating over the list of numbers and comparing each pair to check if their difference falls below the given threshold.'
        'I should implement the `has_close_elements` function to iterate through all pairs of numbers in the list and check if their absolute difference is less than the threshold.\n\nAction 1:\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation 1: I have implemented the `has_close_elements` function to iterate through all pairs of numbers in the list and check if their absolute difference is less than the threshold.\n\nThought 2: I need to test the implemented `has_close_elements` function with different test cases to ensure it works correctly.\n\nAction 2:\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n\ntest_has_close_elements()\n```\n\nObservation 2: The tests passed successfully, indicating that the `has_close_elements` function works as expected.\n\nFinish:\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'Implement the `has_close_elements` function with nested loops to compare all pairs of numbers.\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nThought 2: I need to test the implemented function with test cases to ensure it works correctly.\nAction 2: \n\nTest the implemented `has_close_elements` function with assert statement test cases.\n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n\ntest_has_close_elements()\n```\n\nThought 3: The function implementation and test cases are successful, and I can now provide the final answer.\nAction 3: \n\nFinish the task by returning the implemented function and test cases.\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n\ntest_has_close_elements()\n```',
        'I need to implement the `has_close_elements` function with a nested loop to compare each pair of numbers in the list against the threshold.\n\nAction 1:\nImplementing the `has_close_elements` function with the nested loop:\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation 1:\nI have implemented the `has_close_elements` function with the nested loop to compare each pair of numbers in the list against the threshold.\n\nThought 2:\nNow, I need to test the implemented function with assert statement test cases to verify its correctness.\n\nAction 2:\nTesting the implemented function with assert statement test cases:\n\n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\n\nObservation 2:\nThe test cases passed successfully, indicating that the implemented function is correct.\n\nFinish:\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'Implement the `has_close_elements` function with the nested loop logic.\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nThought 2: I need to test the implemented `has_close_elements` function with some test cases to ensure it works as expected.\nAction 2: \n\nTest the implemented function with test cases.\n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 1.0) == True\n\ntest_has_close_elements()\n```\n\nThought 3: The `has_close_elements` function implementation seems correct as it passed the test cases. Now, I can finish this task.\nAction 3: \n\nFinish with the implemented function.\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'I need to implement the `has_close_elements` function to check if there are any two numbers in the list that are closer to each other than the given threshold.\n\nAction 1:\nImplement the `has_close_elements` function with the nested loop to compare all pairs of numbers against the threshold.\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nThought 2:\nNow, I need to test the implemented function to ensure it works correctly for different test cases.\n\nAction 2:\nTest the `has_close_elements` function with assert statements for the provided test cases.\n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n\ntest_has_close_elements()\n```\n\nFinish:\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'Implement the `has_close_elements` function with nested loops to compare each pair of numbers in the list against the threshold.\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nThought 2: I should test the implemented `has_close_elements` function with some test cases to ensure it works correctly.\nAction 2:\n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == False\n    assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0], 1.5) == True\n\ntest_has_close_elements()\n```\n\nThought 3: The implementation is correct, and all test cases passed successfully.\nAction 3:\n\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nFinish 1: The `has_close_elements` function has been successfully implemented and tested with various test cases to check if any two numbers in the list are closer than the given threshold.',
        'I need to implement the `has_close_elements` function to check if any two numbers in the list are closer than the given threshold.\n\nAction 1: Implement\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nThought 2:\nNow that I have implemented the function, I should test it with some test cases to ensure it works correctly.\n\nAction 2: Test\n\n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\n\nThought 3:\nThe function implementation seems to be correct based on the test cases. Now, I can finalize my answer.\n\nAction 3: Finish\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'Implement the `has_close_elements` function.\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    \n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nThought 2: I now need to test the `has_close_elements` function with different test cases to ensure it works correctly.\nAction 2: Test the `has_close_elements` function.\n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert has_close_elements([1.0, 5.0, 10.0, 15.0], 4.0) == False\n    assert has_close_elements([1.0, 5.0, 5.5, 10.0], 0.6) == True\n\ntest_has_close_elements()\n```\n\nThought 3: The implementation of the `has_close_elements` function and the test cases have been successfully completed.\nAction 3: \n\n```python\nFinish the task.\n\n```',
        'I need to implement the `has_close_elements` function to compare each pair of numbers in the list and check if they are closer than the given threshold.\n\nAction 1:\nImplement the `has_close_elements` function with a nested loop to compare each pair of numbers.\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nThought 2:\nI should now test the implemented function with different test cases to ensure it works correctly.\n\nAction 2:\nTest the `has_close_elements` function with assert statement tests.\n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n\ntest_has_close_elements()\n```\n\nFinish:\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'Implement the `has_close_elements` function with the nested loop approach.\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nThought 2: Now that the implementation is done, I need to test the function with some test cases to ensure it works correctly.\nAction 2: \n\n```python\ndef test_has_close_elements():\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.1) == True\n    assert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.01) == False\n\ntest_has_close_elements()\n```\n\nThought 3: The function implementation passed all the test cases successfully.\nAction 3: \n\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\ntest_has_close_elements()\n```\n\nFinish: The `has_close_elements` function has been successfully implemented and tested.',
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = LATSCodeStrategy(llm=llm)

    strategy.failed_trajectories = [
        {"trajectory": "Failed trajectory 1", "final_answer": "Incorrect answer 1"},
        {"trajectory": "Failed trajectory 2", "final_answer": "Incorrect answer 2"},
        {
            "trajectory": "Failed trajectory 1",
            "final_answer": "Incorrect answer 1",
        },  # Duplicate, should be ignored
    ]

    root = strategy.initialize()
    children_nodes = strategy.generate(
        node=root,
        question=question,
        key=key,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=HUMANEVAL_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_HUMANEVAL,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_HUMANEVAL,
        additional_keys={},
        reflect_additional_keys={},
    )

    assert len(children_nodes) == 5
    for gt_state, node in zip(gt_states, children_nodes):
        assert node.state == gt_state
        assert node.depth == 1
        assert node.reward == 0
        assert node.value == 0
        assert node.is_terminal is False
        assert node.visits == 0

    # Test case with a terminal child node (reward 0)
    responses = [
        'We need to iterate through the list of numbers and check if any two numbers are closer to each other than the threshold.\n\nAction 1:\nImplement the has_close_elements function.\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation 1: We have implemented the function to check if any two numbers are closer to each other than the threshold.\n\nAction 2:\nTest the implemented function with test cases.\n\n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\n\nObservation 2: The implemented function passed the test cases.\n\nAction 3:\nFinish the task.\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        'Implement\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nThought 2: Now that we have implemented the function, we need to test it with some test cases.\nAction 2: \n\n```python\nTest\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\n\nThought 3: The function passed the test cases successfully. Now we can finish by providing the final implementation.\nAction 3: \n\n```python\nFinish\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = LATSCodeStrategy(llm=llm, n_samples=1)

    root = strategy.initialize()
    children_nodes = strategy.generate(
        node=root,
        question=question,
        key=key,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=HUMANEVAL_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_HUMANEVAL,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_HUMANEVAL,
        additional_keys={},
        reflect_additional_keys={},
    )
    assert len(children_nodes) == 1
    assert (
        children_nodes[0].state.thought
        == 'We need to iterate through the list of numbers and check if any two numbers are closer to each other than the threshold.'
    )
    assert children_nodes[0].state.action_type == "Implement"
    assert (
        children_nodes[0].state.query
        == 'from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False'
    )
    assert not children_nodes[0].is_terminal
    assert children_nodes[0].reward == 0


def test_select_node() -> None:
    """Test the select_node method."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSCodeStrategy(llm=llm)

    # Create a tree structure.
    root = Node(state={})
    child1 = Node(state={}, parent=root)
    child2 = Node(state={}, parent=root)
    grandchild1 = Node(state={}, parent=child1)
    grandchild2 = Node(state={}, parent=child1)

    root.children = [child1, child2]
    child1.children = [grandchild1, grandchild2]

    # Test selection of non-terminal node with highest UCT.
    child1.visits = 10
    child1.value = 0.6
    child2.visits = 5
    child2.value = 0.4
    selected_node = strategy.select_node(root)
    assert (
        selected_node == grandchild1
    )  # child2 should have higher UCT due to fewer visits

    # Test pruning of fully expanded terminal node.
    grandchild2.is_terminal = True
    grandchild2.reward = 0
    selected_node = strategy.select_node(root)
    assert selected_node == grandchild1

    # Test selection when all children are terminal.
    root = Node(state={})
    child1 = Node(state={}, parent=root)
    child2 = Node(state={}, parent=root)
    root.add_children([child1, child2])
    child1.is_terminal = True
    child2.is_terminal = True
    selected_node = strategy.select_node(root)
    assert selected_node == root


def test_expand_node() -> None:
    """Test the expand_node method."""
    gt_states = [
        ReActOutput(
            thought="I need to calculate how much money Janet makes daily at the farmers' market.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day",
            observation="\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day\n```\nExecution Status: Done\nOutput: answer = -9867630",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": -9867630},
        ),
        ReActOutput(
            thought="I need to calculate how much money Janet makes daily at the farmers' market by selling the remaining eggs after breakfast and baking muffins for her friends.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg",
            observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg\n```\nExecution Status: Done\nOutput: answer = -9867630",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": -9867630},
        ),
        ReActOutput(
            thought="First, I need to calculate the total number of eggs Janet has available to sell at the farmers' market after accounting for her breakfast consumption and muffin baking.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins",
            observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: answer = None",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": None},
        ),
        ReActOutput(
            thought="First, I need to calculate how many eggs Janet has left after eating three for breakfast every day.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast",
            observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\nExecution Status: Done\nOutput: answer = None",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": None},
        ),
        ReActOutput(
            thought="First, I need to calculate how many eggs Janet has left after eating three eggs for breakfast and baking muffins.",
            action_type="Calculate",
            query="eggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins",
            observation="\n```python\neggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins\n```\nExecution Status: Done\nOutput: answer = None",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": None},
        ),
    ]

    responses = [
        "I need to calculate how much money Janet makes daily at the farmers' market.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nanswer = money_made_per_day\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867650\nThought 2: The calculation resulted in a negative value, which doesn't make sense for money made. I need to review the calculation.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nmoney_made_per_day = abs(money_made_per_day)  # Take the absolute value\nanswer = money_made_per_day\n```\n]\nObservation 2:\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nmoney_made_per_day = abs(money_made_per_day)  # Take the absolute value\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = 9867650\nThought 3: Janet makes $9867650 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 9867650\n```\n]\nObservation 3:\n```python\nanswer = 9867650\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day\n```\n]",
        "I need to calculate how much money Janet makes daily at the farmers' market by selling the remaining eggs after breakfast and baking muffins for her friends.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867649\nThought 2: The calculation resulted in a negative number, which doesn't make sense in this context. I need to review the subtraction steps.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg\n```\n]",
        "First, I need to calculate the total number of eggs Janet has available to sell at the farmers' market after accounting for her breakfast consumption and muffin baking.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - muffin_egg_usage\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - muffin_egg_usage\n```\nExecution Status: Done\nOutput: eggs_remaining = -4933815\nThought 2: The negative result indicates an error in the calculations. I need to correct the computation to determine the correct number of eggs left for sale.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\n```\nExecution Status: Done\nOutput: eggs_for_sale = -4933807\nThought 3: Another error occurred in the calculation. I need to correct the computation to determine the correct number of eggs available for sale.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = eggs_for_sale * price_per_egg\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = eggs_for_sale * price_per_egg\n```\nExecution Status: Done\nOutput: profit_per_day = -9867614\nThought 4: The negative result indicates that there was an error in the calculation. I need to correct the computation to determine the accurate daily profit from selling duck eggs at the farmers' market.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = abs(eggs_for_sale) * price_per_egg\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = abs(eggs_for_sale) * price_per_egg\n```\nExecution Status: Done\nOutput: profit_per_day = 9867614\nThought 5: Janet makes $9867614 every day at the farmers' market.\nAction 5: Finish[\n```python\nprofit_per_day = 9867614\n```\n]\nObservation 5: \n```python\nprofit_per_day = 9867614\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins\n```\n]",
        "First, I need to calculate how many eggs Janet has left after eating three for breakfast every day.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_per_day = 3\neggs_remaining_per_day = eggs_laid_per_day - eggs_consumed_per_day\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_consumed_per_day = 3\neggs_remaining_per_day = eggs_laid_per_day - eggs_consumed_per_day\n```\nExecution Status: Done\nOutput: eggs_remaining_per_day = 13\nThought 2: Now, I need to find out how much Janet earns from selling the remaining eggs at the farmers' market.\nAction 2: Calculate[\n```python\neggs_remaining_per_day = 13\nearnings_per_egg = 2\ndaily_earnings = eggs_remaining_per_day * earnings_per_egg\n```\n]\nObservation 2:\n```python\neggs_remaining_per_day = 13\nearnings_per_egg = 2\ndaily_earnings = eggs_remaining_per_day * earnings_per_egg\n```\nExecution Status: Done\nOutput: daily_earnings = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3:\n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\n]",
        "First, I need to calculate how many eggs Janet has left after eating three eggs for breakfast and baking muffins.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: eggs_sold = -4933815\nThought 2: The calculation is incorrect because Janet cannot have negative eggs to sell. I need to review the subtraction.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\n]\nObservation 2:\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: eggs_sold = 5\nThought 3: Janet sells 5 fresh duck eggs every day at the farmers' market.\nAction 3: Finish[\n```python\neggs_sold = 5\n```\n]\nObservation 3:\n```python\neggs_sold = 5\n```",
        "Calculate[\n```python\neggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins\n```\n]",
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = LATSMathStrategy(llm=llm)

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    key = -9867630

    root = strategy.initialize()

    children_nodes = strategy.expand_node(
        node=root,
        question=question,
        key=key,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_GSM8K,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_GSM8K,
        additional_keys={},
        reflect_additional_keys={},
    )
    assert len(children_nodes) == 5
    for gt_state, node in zip(gt_states, children_nodes):
        assert node.state == gt_state
        assert node.depth == 1
        assert node.reward == 0
        assert node.value == 0
        assert node.is_terminal is False
        assert node.visits == 0
    assert strategy.root.children == children_nodes