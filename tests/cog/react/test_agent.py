"""Unit tests for ReAct."""

import pytest

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.fewshots.humaneval import HUMANEVAL_FEWSHOT_EXAMPLES_REACT
from agential.cog.react.agent import (
    ReActAgent,
)
from agential.cog.react.prompts import (
    REACT_INSTRUCTION_HOTPOTQA,
    REACT_INSTRUCTION_HUMANEVAL,
)
from agential.cog.react.strategies.base import ReActBaseStrategy
from agential.llm.llm import BaseLLM, MockLLM
from agential.cog.constants import Benchmarks
from agential.cog.react.strategies.code import (
    ReActHEvalStrategy,
    ReActMBPPStrategy,
)
from agential.cog.react.strategies.math import (
    ReActGSM8KStrategy,
    ReActSVAMPStrategy,
    ReActTabMWPStrategy,
)
from agential.cog.react.strategies.qa import (
    ReActAmbigNQStrategy,
    ReActFEVERStrategy,
    ReActHotQAStrategy,
    ReActTriviaQAStrategy,
)

def test_init() -> None:
    """Test initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    agent = ReActAgent(llm=llm, benchmark="hotpotqa")
    assert isinstance(agent, ReActAgent)
    assert isinstance(agent.llm, BaseLLM)
    assert agent.benchmark == "hotpotqa"
    assert isinstance(agent.strategy, ReActBaseStrategy)

def test_react_factory_get_strategy() -> None:
    """Tests ReActAgent get_strategy method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    # QA benchmarks.
    assert isinstance(
        ReActAgent.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        ReActHotQAStrategy,
    )
    assert isinstance(
        ReActAgent.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        ReActTriviaQAStrategy,
    )
    assert isinstance(
        ReActAgent.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        ReActAmbigNQStrategy,
    )
    assert isinstance(
        ReActAgent.get_strategy(Benchmarks.FEVER, llm=llm),
        ReActFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        ReActAgent.get_strategy(Benchmarks.GSM8K, llm=llm),
        ReActGSM8KStrategy,
    )
    assert isinstance(
        ReActAgent.get_strategy(Benchmarks.SVAMP, llm=llm),
        ReActSVAMPStrategy,
    )
    assert isinstance(
        ReActAgent.get_strategy(Benchmarks.TABMWP, llm=llm),
        ReActTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        ReActAgent.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        ReActHEvalStrategy,
    )
    assert isinstance(
        ReActAgent.get_strategy(Benchmarks.MBPP, llm=llm),
        ReActMBPPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent ReAct"
    ):
        ReActAgent.get_strategy("unknown", llm=llm)


def test_react_factory_get_fewshots() -> None:
    """Tests ReActAgent get_fewshots method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = ReActAgent.get_fewshots(benchmark, fewshot_type="react")
    assert isinstance(result, dict)
    assert result == {"examples": HOTPOTQA_FEWSHOT_EXAMPLES_REACT}

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for ReAct."
    ):
        ReActAgent.get_fewshots("unknown", fewshot_type="react")

    # Test unsupported fewshot_type.
    with pytest.raises(
        ValueError, match="Benchmark 'hotpotqa' few-shot type not supported for ReAct."
    ):
        ReActAgent.get_fewshots("hotpotqa", fewshot_type="pot")


def test_react_factory_get_prompts() -> None:
    """Tests ReActAgent get_prompts method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = ReActAgent.get_prompts(benchmark)
    assert result == {"prompt": REACT_INSTRUCTION_HOTPOTQA}

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for ReAct."
    ):
        ReActAgent.get_prompts("unknown")

def test_generate() -> None:
    """Test generate."""
    # Test QA.
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    responses = [
        "I need to search for the best kickboxer in the world who has been involved in controversies and crimes.\nAction: Search[best kickboxer in the world]\nObservation: Could not find exact match. Similar: ['List of male kickboxers', 'List of female kickboxers', 'Kickboxing in Japan', 'Kickboxing in the United Kingdom', 'Kickboxing in the United States', 'Kickboxing in Thailand', 'Kickboxing in the Netherlands'].\nThought: I couldn't find the exact information, so I will try to find a famous kickboxer known for controversies.\nAction: Search[kickboxer controversies]\nObservation: Could not find exact match. Similar: ['List of male kickboxers', 'List of female kickboxers', 'Kickboxing in Japan', 'Kickboxing in the United Kingdom', 'Kickboxing in the United States', 'Kickboxing in Thailand', 'Kickboxing in the Netherlands'].\nThought: I need to search for famous kickboxers involved in controversies.\nAction: Search[famous kickboxers controversies]\nObservation: Could not find exact match. Similar: ['List of male kickboxers', 'List of female kickboxers', 'Kickboxing in Japan', 'Kickboxing in the United Kingdom', 'Kickboxing in the United States', 'Kickboxing in Thailand', 'Kickboxing in the Netherlands'].\nThought: I need to try a different approach to find the answer.\nAction: Search[kickboxer controversies crimes]\nObservation: Could not find exact match. Similar: ['List of male kickboxers', 'List of female kickboxers', 'Kickboxing in Japan', 'Kickboxing in the United Kingdom', 'Kickboxing in the United States', 'Kickboxing in Thailand', 'Kickboxing in the Netherlands'].\nThought: I need to refine my search terms to find the specific kickboxer mentioned in the question.\nAction: Search[kickboxer crimes violence unsportsmanlike conduct]\nObservation: Could not find exact match. Similar: ['List of male kickboxers', 'List of female kickboxers', 'Kickboxing in Japan', 'Kickboxing in the United Kingdom', 'Kickboxing in the United States', 'Kickboxing in Thailand', 'Kickboxing in the Netherlands'].\nThought: I'm unable to find the specific kickboxer mentioned in the question. \nAction: Finish[unable to find specific kickboxer]",
        "Search[best kickboxer in the world controversies crimes]\nObservation: Could not find exact match. Similar: ['List of male kickboxers', 'List of female kickboxers', 'List of kickboxing organizations', 'List of kickboxing events', 'List of kickboxers', 'List of female kickboxers', 'List of male kickboxers', 'Kickboxing', 'Muay Thai', 'Mixed martial arts']\nThought: I need to search for a list of kickboxers and look for the one who fits the description.\nAction: Search[List of kickboxers]\nObservation: The list includes many kickboxers. I need to look up each one to find the one involved in controversies and crimes.\nThought: I should start with the most famous kickboxers on the list and look for information on controversies and crimes.\nAction: Lookup[controversies crimes]\nObservation: (Result 1 / 1) The kickboxer known as Badr Hari was once considered the best kickboxer in the world but has been involved in controversies and crimes.\nThought: Badr Hari fits the description, so the answer is Badr Hari.\nAction: Finish[Badr Hari]",
        "This passage does not mention the name of the kickboxer. I need to search for another source or keyword that could lead me to the answer.\nAction: Search[Emory Andrew Tate III kickboxing controversies]\nObservation 2: Could not find [Emory Andrew Tate III kickboxing controversies]. Similar: ['Emory Andrew Tate III']\nThought: I should search for Emory Andrew Tate III's kickboxing career to see if that provides more information about his reputation in the sport.\nAction: Search[Emory Andrew Tate III kickboxing career]\nObservation 3: Emory Andrew Tate III first began to kickbox in 2005, winning several kickboxing titles in the late 2000s and early 2010s.\nThought: Since the previous searches did not provide clear information about the best kickboxer in the world involved in controversies and crimes, I need to reevaluate my approach and consider searching for other well-known kickboxers who fit the description given in the question.\nAction: Search[most controversial kickboxers crimes]\nObservation 4: Could not find [most controversial kickboxers crimes]. Similar: ['most controversial kickboxers of all time', 'most controversial boxers', 'most controversial UFC fighters']\nThought: I should narrow down my search to specific well-known kickboxers who have been involved in controversies and crimes to find the answer.\nAction: Search[well-known kickboxers controversies crimes]\nObservation 5: Could not find [well-known kickboxers controversies crimes]. Similar: ['well-known kickboxers 2024', 'well-known kickboxers 2023', 'well-known kickboxers 2022']\nThought: I need to find a more specific keyword that can lead me to the answer. Maybe I can search for \"kickboxer controversies crimes\" to find a list of kickboxers known for such behavior.\nAction: Search[kickboxer controversies crimes]\nObservation 6: Could not find [kickboxer controversies crimes]. Similar: ['kickboxer crimes', 'kickboxer controversies', 'boxer controversies crimes']\nThought: The searches are not yielding the desired results. I may have to broaden my search to include more general information about kickboxers known for controversies and crimes to identify the individual in question.\nAction: Search[kickboxers with criminal records controversies]\nObservation 7: Could not find [kickboxers with criminal records controversies]. Similar: ['famous kickboxers', 'kickboxers', 'kickboxers 2023']\nThought: The information available is not specific enough to pinpoint the kickboxer mentioned in the question. I may need to reconsider the keywords I am using in my searches or seek additional sources to find the answer.\nAction: Finish[unable to determine the answer based on the available information]",
        "Search[former best kickboxer controversies crimes]",
        "This passage still does not mention the name of the kickboxer. I need to search for a different keyword or source that could lead me to the answer.\nAction: Search[kickboxer controversies crimes]\nObservation 3: Sorry, I couldn’t find any information on the kickboxer involved in controversies and crimes. Would you like me to suggest some similar entities to search for?\nThought: Since the search did not yield the answer, I need to try a different approach. I will search for famous kickboxers and look for the one involved in controversies and crimes.\nAction: Search[famous kickboxers]\nObservation 4: Famous kickboxers include: Joe Lewis, Benny Urquidez, Bill Wallace, Peter Aerts, Badr Hari, Ernesto Hoost, and more. \nThought: None of these famous kickboxers are matching the description of being involved in controversies and crimes. I need to try a different search strategy or keyword.\nAction: Search[kickboxers involved in controversies crimes]\nObservation 5: Sorry, I couldn’t find any information on kickboxers specifically involved in controversies and crimes. Would you like me to suggest some similar entities to search for?\nThought: Since the direct search is not yielding results, I need to think of another way to approach this question. I will try to search for kickboxers known for violent behavior outside the ring.\nAction: Search[kickboxers violent behavior outside ring]\nObservation 6: Sorry, I couldn’t find any information on kickboxers known for violent behavior outside the ring. Would you like me to suggest some similar entities to search for?\nThought: This question seems difficult to answer based on the available information. I am unable to find the specific kickboxer mentioned in the question involved in controversies and crimes. \nAction: Finish[unable to find answer]",
        "Search[kickboxing controversies crimes famous]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    agent = ReActAgent(llm=llm, benchmark="hotpotqa")
    agent.strategy.docstore.search = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )

    out = agent.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        reset=True,
        max_steps=3,
    )
    assert isinstance(out, list)
    assert len(out) == 3

    # Test Code.
    inst = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    question = inst["prompt"]

    responses = [
        "To implement this function, we need to iterate through the list of numbers and check if the absolute difference between any two numbers is less than the given threshold.\n\nAction: Implement\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation:",
        "Implement\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```",
        "We should test the implemented function with different test cases to verify its correctness.\nAction: Test[\n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\n]\nObservation 2: \n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\nExecution Status: Done\nThought: The implemented function passed the test cases successfully. We can now finish the task.\nAction: Finish[\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n]",
        "Test[\n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.1) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.01) == False\n```\n]\nObservation 2: \n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.1) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.01) == False\n```\nExecution Status: Done\nThought: The function implementation is correct and the test cases have passed. We can finalize the implementation.\nAction: Finish[\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n]",
        "The implemented function has passed the test cases, and it correctly identifies if there are any two numbers in the list that are closer to each other than the given threshold. \nAction: Finish[\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n]",
        "Finish[\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    agent = ReActAgent(llm=llm, benchmark="humaneval")
    out = agent.generate(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HUMANEVAL,
        max_steps=3,
    )
    assert isinstance(out, list)
    assert len(out) == 3

    # Test auto-select prompts and few-shots.
    responses = [
        "To implement this function, we need to iterate through the list of numbers and check if the absolute difference between any two numbers is less than the given threshold.\n\nAction: Implement\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation:",
        "Implement\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```",
        "We should test the implemented function with different test cases to verify its correctness.\nAction: Test[\n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\n]\nObservation 2: \n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\nExecution Status: Done\nThought: The implemented function passed the test cases successfully. We can now finish the task.\nAction: Finish[\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n]",
        "Test[\n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.1) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.01) == False\n```\n]\nObservation 2: \n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.1) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.01) == False\n```\nExecution Status: Done\nThought: The function implementation is correct and the test cases have passed. We can finalize the implementation.\nAction: Finish[\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n]",
        "The implemented function has passed the test cases, and it correctly identifies if there are any two numbers in the list that are closer to each other than the given threshold. \nAction: Finish[\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n]",
        "Finish[\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    agent = ReActAgent(llm=llm, benchmark="humaneval")
    out = agent.generate(
        question=question,
        max_steps=3,
    )
    assert isinstance(out, list)
    assert len(out) == 3

    # Test auto-select prompts and few-shots.
    responses = [
        "To implement this function, we need to iterate through the list of numbers and check if the absolute difference between any two numbers is less than the given threshold.\n\nAction: Implement\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nObservation:",
        "Implement\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```",
        "We should test the implemented function with different test cases to verify its correctness.\nAction: Test[\n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\n]\nObservation 2: \n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n```\nExecution Status: Done\nThought: The implemented function passed the test cases successfully. We can now finish the task.\nAction: Finish[\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n]",
        "Test[\n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.1) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.01) == False\n```\n]\nObservation 2: \n```python\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.1) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.01) == False\n```\nExecution Status: Done\nThought: The function implementation is correct and the test cases have passed. We can finalize the implementation.\nAction: Finish[\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n]",
        "The implemented function has passed the test cases, and it correctly identifies if there are any two numbers in the list that are closer to each other than the given threshold. \nAction: Finish[\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n]",
        "Finish[\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    agent = ReActAgent(llm=llm, benchmark="humaneval")
    out = agent.generate(
        question=question,
        fewshot_type="react",
        max_steps=3,
    )
    assert isinstance(out, list)
    assert len(out) == 3

    # Test auto-select prompts and few-shots.
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    agent = ReActAgent(llm=llm, benchmark="humaneval")
    with pytest.raises(
        ValueError,
        match="Benchmark 'humaneval' few-shot type not supported for ReAct.",
    ):
        out = agent.generate(
            question=question,
            fewshot_type="pot",
            max_steps=3,
        )
    assert isinstance(out, list)
    assert len(out) == 3
