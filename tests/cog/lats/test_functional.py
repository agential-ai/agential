"""Unit tests for LATS functional module."""

from litellm.types.utils import ModelResponse

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.lats.functional import (
    _accumulate_metric,
    _build_agent_prompt,
    _build_failed_trajectory_format,
    _build_reflection_format,
    _build_reflection_prompt,
    _build_value_prompt,
    _prompt_agent,
    _prompt_reflection,
    _prompt_value,
    accumulate_metrics,
    get_node_trajectory_code,
    get_node_trajectory_math,
    get_node_trajectory_qa,
    get_unique_trajectories,
    parse_code_action,
    parse_code_value,
    parse_latest_implement,
    parse_math_action,
    parse_math_value,
    parse_qa_action,
    parse_qa_value,
)
from agential.cog.lats.node import Node
from agential.cog.lats.output import (
    LATSEvaluateMetrics,
    LATSGenerateMetrics,
    LATSReActStepOutput,
    LATSSimulationMetrics,
    LATSSimulationOutput,
    LATSStepOutput,
)
from agential.cog.lats.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    LATS_INSTRUCTION_HOTPOTQA,
    LATS_REFLECT_INSTRUCTION_HOTPOTQA,
    LATS_VALUE_INSTRUCTION_HOTPOTQA,
)
from agential.llm.llm import MockLLM
from agential.utils.general import PromptMetrics


def test__build_reflection_format() -> None:
    """Tests the _build_reflection_format() function."""
    gt_reflection = "Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]\n\nReflection: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    reflection = _build_reflection_format(
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        reflection="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
    )
    assert reflection == gt_reflection


def test__build_failed_trajectory_format() -> None:
    """Tests the _build_failed_trajectory_format() function."""
    gt_failed_trajectory = "Question: What is the capital of France?\nRoot thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]\n\nExplanation: This trajectory is incorrect as The trajectory failed to provide the correct answer. I should have looked up information about France instead.\nCorrectness score: 1"
    failed_trajectory = _build_failed_trajectory_format(
        question="What is the capital of France?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        reflection="The trajectory failed to provide the correct answer. I should have looked up information about France instead.",
    )
    assert failed_trajectory == gt_failed_trajectory


def test__build_reflection_prompt() -> None:
    """Tests the _build_reflection_prompt() function."""
    prompt = _build_reflection_prompt(
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(prompt, str)
    assert "Colorado orogeny" in prompt
    assert "elevation range" in prompt


def test__prompt_reflection() -> None:
    """Tests the _prompt_reflection() function."""
    out = _prompt_reflection(
        llm=MockLLM("gpt-3.5-turbo", responses=["Reflection Output"]),
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, ModelResponse)
    assert out.choices[0].message.content == "Reflection Output"


def test__build_value_prompt() -> None:
    """Tests the _build_value_prompt() function."""
    prompt = _build_value_prompt(
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        failed_trajectories="Failed Trajectories",
        prompt=LATS_VALUE_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(prompt, str)
    assert "Colorado orogeny" in prompt
    assert "elevation range" in prompt


def test__prompt_value() -> None:
    """Tests the _prompt_value() function."""
    out = _prompt_value(
        llm=MockLLM("gpt-3.5-turbo", responses=["Value Output"]),
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        failed_trajectories="Failed Trajectories",
        prompt=LATS_VALUE_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, ModelResponse)
    assert out.choices[0].message.content == "Value Output"


def test__build_agent_prompt() -> None:
    """Tests the _build_agent_prompt() function."""
    prompt = _build_agent_prompt(
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=LATS_INSTRUCTION_HOTPOTQA,
        reflections="Reflections",
    )
    assert isinstance(prompt, str)
    assert "Colorado orogeny" in prompt
    assert "elevation range" in prompt


def test__prompt_agent() -> None:
    """Tests the _prompt_agent() function."""
    out = _prompt_agent(
        llm=MockLLM("gpt-3.5-turbo", responses=["Agent Output"]),
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflections="Reflections",
        prompt=LATS_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, ModelResponse)
    assert out.choices[0].message.content == "Agent Output"


def test_get_unique_trajectories() -> None:
    """Tests the get_unique_trajectories() function."""
    failed_trajectories = [
        {"trajectory": "Path1", "final_answer": "Answer1"},
        {"trajectory": "Path2", "final_answer": "Answer1"},  # Duplicate answer
        {"trajectory": "Path3", "final_answer": "Answer2"},
        {"trajectory": "Path4", "final_answer": "Answer3"},
        {"trajectory": "Path5", "final_answer": "Answer2"},  # Duplicate answer
        {"trajectory": "Path6", "final_answer": "Answer4"},
    ]

    # Test with max_unique=5.
    result = get_unique_trajectories(failed_trajectories, max_unique=5)
    assert result == ["Path1", "Path3", "Path4", "Path6"]

    # Test with max_unique=2.
    result = get_unique_trajectories(failed_trajectories, max_unique=2)
    assert result == ["Path1", "Path3"]

    # Test with empty list.
    result = get_unique_trajectories([], max_unique=5)
    assert result == []

    # Test with all unique answers.
    unique_trajectories = [
        {"trajectory": f"Path{i}", "final_answer": f"Answer{i}"} for i in range(1, 7)
    ]
    result = get_unique_trajectories(unique_trajectories, max_unique=5)
    assert result == [f"Path{i}" for i in range(1, 6)]


def test_get_node_trajectory_qa() -> None:
    """Tests the get_node_trajectory_qa() function."""
    root = Node(
        state=LATSReActStepOutput(
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
        state=LATSReActStepOutput(
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
        state=LATSReActStepOutput(
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

    expected_trajectory = "\nThought 1: Child1 thought\nAction 1: Lookup[topic]\nThought 2: Child2 thought\nAction 2: Finish[answer]\nObservation 2: Answer correct"
    assert get_node_trajectory_qa(child2) == expected_trajectory

    # Test root node.
    root = Node()
    assert get_node_trajectory_qa(root) == ""


def test_parse_qa_action():
    """Test the parse_qa_action function."""
    # Test valid action strings.
    assert parse_qa_action("Search[query]") == ("Search", "query")
    assert parse_qa_action("Lookup[term]") == ("Lookup", "term")
    assert parse_qa_action("Finish[answer]") == ("Finish", "answer")

    # Test invalid action strings.
    assert parse_qa_action("InvalidAction") == ("", "")
    assert parse_qa_action("") == ("", "")
    assert parse_qa_action("Action[]") == ("", "")


def test_parse_qa_value():
    """Test the parse_qa_value function."""
    # Test valid value strings.
    valid_input = (
        "Some text. Explanation: This is the explanation. Correctness score: 5"
    )
    assert parse_qa_value(valid_input) == ("This is the explanation.", 5)

    # Test invalid value strings.
    assert parse_qa_value("No explanation or score") == ("Explanation not found", 0)
    assert parse_qa_value("Explanation: Only explanation") == (
        "Explanation not found",
        0,
    )
    assert parse_qa_value("Correctness score: 5") == ("Explanation not found", 0)

    # Test edge cases.
    assert parse_qa_value("Explanation: Empty. Correctness score: 0") == ("Empty.", 0)
    assert parse_qa_value(
        "Explanation: Multi-line\nexplanation. Correctness score: 10"
    ) == ("Multi-line\nexplanation.", 10)

    # Test with unexpected format.
    assert parse_qa_value("Explanation: Tricky: score. Correctness score: 7") == (
        "Tricky: score.",
        7,
    )


def test_get_node_trajectory_math() -> None:
    """Tests the get_node_trajectory_math() function."""
    root = Node(
        state=LATSReActStepOutput(
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
        state=LATSReActStepOutput(
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
        state=LATSReActStepOutput(
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
    assert get_node_trajectory_math(child2) == expected_trajectory

    # Test root node.
    root = Node()
    assert get_node_trajectory_math(root) == ""


def test_parse_math_action():
    """Test the parse_math_action function."""
    test_cases = [
        {
            "input": "Calculate[```python\ndef add(a, b): return a + b\n```]",
            "expected": ("Calculate", "def add(a, b): return a + b"),
        },
        {
            "input": "FINISH[```python\nassert add(2, 3) == 5\n```]",
            "expected": ("Finish", "assert add(2, 3) == 5"),
        },
        {
            "input": "calculate[```python\ndef subtract(a, b): return a - b\n```]",
            "expected": ("Calculate", "def subtract(a, b): return a - b"),
        },
        {
            "input": "Invalid[```python\nThis should not match\n```]",
            "expected": ("", ""),
        },
        {
            "input": "Calculate[```python\n \n```]",
            "expected": ("Calculate", ""),
        },
        {
            "input": "Something else entirely",
            "expected": ("", ""),
        },
    ]

    for case in test_cases:
        result = parse_math_action(case["input"])
        assert result == case["expected"]


def test_parse_math_value():
    """Test the parse_math_value function."""
    # Test valid value strings.
    valid_input = (
        "Some text. Explanation: This is the explanation. Correctness score: 5"
    )
    assert parse_math_value(valid_input) == ("This is the explanation.", 5)

    # Test invalid value strings.
    assert parse_math_value("No explanation or score") == ("Explanation not found", 0)
    assert parse_math_value("Explanation: Only explanation") == (
        "Explanation not found",
        0,
    )
    assert parse_math_value("Correctness score: 5") == ("Explanation not found", 0)

    # Test edge cases.
    assert parse_math_value("Explanation: Empty. Correctness score: 0") == ("Empty.", 0)
    assert parse_math_value(
        "Explanation: Multi-line\nexplanation. Correctness score: 10"
    ) == ("Multi-line\nexplanation.", 10)

    # Test with unexpected format.
    assert parse_math_value("Explanation: Tricky: score. Correctness score: 7") == (
        "Tricky: score.",
        7,
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
    assert (
        parse_latest_implement(multiple_impl)
        == "def multiply(a, b):\n        return a * b"
    )

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
        state=LATSReActStepOutput(
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
        state=LATSReActStepOutput(
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
        state=LATSReActStepOutput(
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
    assert result == ("Implement", "incomplete code")


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


def test__accumulate_metric() -> None:
    """Test the _accumulate_metric function."""
    step = LATSStepOutput(
        iteration=0,
        current_node={},
        children_nodes=[],
        generate_metrics=LATSGenerateMetrics(
            thoughts_metrics=[
                PromptMetrics(
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=5,
                    prompt_cost=5,
                    completion_cost=5,
                    total_cost=5,
                    prompt_time=5,
                ),
            ],
            actions_metrics=[
                PromptMetrics(
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=5,
                    prompt_cost=5,
                    completion_cost=5,
                    total_cost=5,
                    prompt_time=5,
                ),
            ],
            reflections_metrics=[
                PromptMetrics(
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=5,
                    prompt_cost=5,
                    completion_cost=5,
                    total_cost=5,
                    prompt_time=5,
                ),
            ],
        ),
        evaluate_metrics=LATSEvaluateMetrics(
            values_metrics=[
                None,
                PromptMetrics(
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=5,
                    prompt_cost=5,
                    completion_cost=5,
                    total_cost=5,
                    prompt_time=5,
                ),
            ]
        ),
        simulation_results=LATSSimulationOutput(
            simulation_reward=0.5,
            simulation_terminal_node=None,
            simulation_current_nodes=[],
            simulation_children_nodes=[],
            simulation_values=[],
        ),
        simulation_metrics=LATSSimulationMetrics(
            
            simulation_reward=0.5,
            simulation_terminal_node=None,
            simulation_current_nodes=[],
            simulation_children_nodes=[],
            simulation_values=[],
        )
    )

    metric_types = [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cost",
        "completion_cost",
        "total_cost",
        "prompt_time",
    ]

    for metric_type in metric_types:
        assert _accumulate_metric(step, metric_type) == 50


def test_accumulate_metrics() -> None:
    """Test the accumulate_metrics function."""
    # Test with empty input.

    step = LATSStepOutput(
        iteration=0,
        current_node={},
        children_nodes=[],
        thoughts_metrics=[
            PromptMetrics(
                prompt_tokens=5,
                completion_tokens=5,
                total_tokens=5,
                prompt_cost=5,
                completion_cost=5,
                total_cost=5,
                prompt_time=5,
            ),
            PromptMetrics(
                prompt_tokens=5,
                completion_tokens=5,
                total_tokens=5,
                prompt_cost=5,
                completion_cost=5,
                total_cost=5,
                prompt_time=5,
            ),
        ],
        actions_metrics=[
            PromptMetrics(
                prompt_tokens=5,
                completion_tokens=5,
                total_tokens=5,
                prompt_cost=5,
                completion_cost=5,
                total_cost=5,
                prompt_time=5,
            ),
            PromptMetrics(
                prompt_tokens=5,
                completion_tokens=5,
                total_tokens=5,
                prompt_cost=5,
                completion_cost=5,
                total_cost=5,
                prompt_time=5,
            ),
        ],
        values=[],
        values_metrics=[
            None,
            PromptMetrics(
                prompt_tokens=5,
                completion_tokens=5,
                total_tokens=5,
                prompt_cost=5,
                completion_cost=5,
                total_cost=5,
                prompt_time=5,
            ),
        ],
        simulation_results=LATSSimulationOutput(
            simulation_reward=0.5,
            simulation_terminal_node=None,
            simulation_current_nodes=[],
            simulation_children_nodes=[],
            simulation_thoughts_metrics=[
                [
                    PromptMetrics(
                        prompt_tokens=5,
                        completion_tokens=5,
                        total_tokens=5,
                        prompt_cost=5,
                        completion_cost=5,
                        total_cost=5,
                        prompt_time=5,
                    ),
                    PromptMetrics(
                        prompt_tokens=5,
                        completion_tokens=5,
                        total_tokens=5,
                        prompt_cost=5,
                        completion_cost=5,
                        total_cost=5,
                        prompt_time=5,
                    ),
                ]
            ],
            simulation_actions_metrics=[
                [
                    PromptMetrics(
                        prompt_tokens=5,
                        completion_tokens=5,
                        total_tokens=5,
                        prompt_cost=5,
                        completion_cost=5,
                        total_cost=5,
                        prompt_time=5,
                    ),
                    PromptMetrics(
                        prompt_tokens=5,
                        completion_tokens=5,
                        total_tokens=5,
                        prompt_cost=5,
                        completion_cost=5,
                        total_cost=5,
                        prompt_time=5,
                    ),
                ]
            ],
            simulation_values=[],
            simulation_values_metrics=[
                [
                    None,
                    PromptMetrics(
                        prompt_tokens=5,
                        completion_tokens=5,
                        total_tokens=5,
                        prompt_cost=5,
                        completion_cost=5,
                        total_cost=5,
                        prompt_time=5,
                    ),
                ]
            ],
        ),
    )

    assert accumulate_metrics([step, step]) == {
        "total_prompt_tokens": 100,
        "total_completion_tokens": 100,
        "total_tokens": 100,
        "total_prompt_cost": 100.0,
        "total_completion_cost": 100.0,
        "total_cost": 100.0,
        "total_prompt_time": 100.0,
    }
