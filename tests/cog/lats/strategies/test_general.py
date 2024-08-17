"""Unit tests for LATS general strategies."""

import pytest

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.lats.node import Node
from agential.cog.lats.output import LATSStepOutput
from agential.cog.lats.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    LATS_INSTRUCTION_HOTPOTQA,
    LATS_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.lats.strategies.general import LATSGeneralStrategy
from agential.llm.llm import MockLLM, ModelResponse, Usage
from agential.utils.general import PromptMetrics


def test_init() -> None:
    """Test initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSGeneralStrategy(
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
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    strategy = LATSGeneralStrategy(llm=llm)
    node = strategy.initialize()
    assert strategy.root == node
    assert strategy.root is not None
    assert isinstance(strategy.root, Node)
    assert strategy.root.state.thought == ""
    assert strategy.root.state.action_type == ""
    assert strategy.root.state.query == ""
    assert strategy.root.state.observation == ""
    assert strategy.root.state.external_tool_info == {}


def test_generate_children_nodes() -> None:
    """Test the generate_children_nodes method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSGeneralStrategy(llm=llm)

    question = "What is the capital of France?"
    examples = "Example content"
    prompt = "Generate an action to answer the question."
    additional_keys = {}

    with pytest.raises(NotImplementedError):
        strategy.generate_children_nodes(
            node=Node(),
            question=question,
            key="",
            examples=examples,
            reflect_examples=examples,
            prompt=prompt,
            reflect_prompt=prompt,
            additional_keys=additional_keys,
            reflect_additional_keys=additional_keys,
        )


def test_generate_thought() -> None:
    """Test the generate_thought method."""
    llm = MockLLM(
        "gpt-3.5-turbo",
        responses=[
            "I should search for information about the topic. Action: Search[topic]"
        ],
    )
    strategy = LATSGeneralStrategy(llm=llm)

    question = "What is the capital of France?"
    examples = "Example 1\nExample 2"
    trajectory = "Previous thought"
    reflections = "Reflection 1\nReflection 2"
    depth = 1
    prompt = "Generate a thought"
    additional_keys = {"key": "value"}

    updated_trajectory, thought, out = strategy.generate_thought(
        question,
        examples,
        trajectory,
        reflections,
        depth,
        prompt,
        additional_keys,
    )

    assert thought == "I should search for information about the topic."
    assert (
        updated_trajectory
        == "Previous thought\nThought 2: I should search for information about the topic."
    )
    assert (
        out.choices[0].message.content
        == "I should search for information about the topic. Action: Search[topic]"
    )


def test_generate_action() -> None:
    """Test the generate_action method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[""])
    strategy = LATSGeneralStrategy(llm=llm)

    # Define test inputs
    question = "What is the capital of France?"
    examples = "Example content"
    prompt = "Generate an action to answer the question."
    additional_keys = {}

    # Check for NotImplementedError
    with pytest.raises(NotImplementedError):
        strategy.generate_action(
            question=question,
            examples=examples,
            trajectory="",
            reflections="",
            depth=1,
            prompt=prompt,
            additional_keys=additional_keys,
        )


def test_generate_observation() -> None:
    """Test the generate_observation method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSGeneralStrategy(llm=llm)

    with pytest.raises(NotImplementedError):
        strategy.generate_observation(
            key="test_key",
            action_type="Search",
            query="test query",
            trajectory="test trajectory",
            depth=0,
        )


def test_select_node() -> None:
    """Test the select_node method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSGeneralStrategy(llm=llm)

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


def test_evaluate_node() -> None:
    """Test the evaluate_node method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSGeneralStrategy(llm=llm)
    node = Node()

    with pytest.raises(NotImplementedError):
        strategy.evaluate_node(
            node=node,
            question="test question",
            examples="test examples",
            prompt="test prompt",
            additional_keys={},
        )


def test_simulate_node() -> None:
    """Test the simulate_node method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSGeneralStrategy(llm=llm)
    node = Node()

    with pytest.raises(NotImplementedError):
        strategy.simulate_node(
            node=node,
            question="test question",
            key="test key",
            examples="test examples",
            reflect_examples="test reflect examples",
            value_examples="test value examples",
            prompt="test prompt",
            reflect_prompt="test reflect prompt",
            value_prompt="test value prompt",
            additional_keys={},
            reflect_additional_keys={},
            value_additional_keys={},
        )


def test_backpropagate_node() -> None:
    """Test the backpropagate_node method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSGeneralStrategy(llm=llm)

    # Create a simple tree structure.
    root = Node(state={})
    child = Node(state={}, parent=root)
    grandchild = Node(state={}, parent=child)
    grandchild.is_terminal = True

    # Test backpropagation for a successful terminal node.
    grandchild.reward = 1
    strategy.backpropagate_node(grandchild, 1.0)

    assert root.visits == 1
    assert child.visits == 1
    assert grandchild.visits == 1
    assert root.value == 1.0
    assert child.value == 1.0
    assert grandchild.value == 1.0

    # Test backpropagation for a failed terminal node.
    grandchild.reward = 0
    strategy.backpropagate_node(grandchild, 1.0)

    assert root.visits == 2
    assert child.visits == 2
    assert grandchild.visits == 2
    assert root.value == 1.0
    assert child.value == 1.0
    assert grandchild.value == 0.0

    # Test backpropagation for a non-terminal node.
    child.is_terminal = False
    strategy.backpropagate_node(child, 0.5)

    assert root.visits == 3
    assert child.visits == 3
    assert root.value == 5 / 6
    assert child.value == 5 / 6


def test_halting_condition() -> None:
    """Test the halting_condition method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSGeneralStrategy(llm=llm)

    # Test with a terminal node and reward of 1.
    terminal_node = Node(state={})
    terminal_node.is_terminal = True
    terminal_node.reward = 1
    assert strategy.halting_condition(terminal_node) is True

    # Test with a non-terminal node.
    non_terminal_node = Node(state={})
    assert strategy.halting_condition(non_terminal_node) is False

    # Test with a terminal node but reward is not 1.
    incorrect_terminal_node = Node(state={})
    incorrect_terminal_node.is_terminal = True
    incorrect_terminal_node.reward = 0
    assert strategy.halting_condition(incorrect_terminal_node) is False


def test_reflect_condition() -> None:
    """Test the reflect_condition method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSGeneralStrategy(llm=llm, max_unique=3, max_reflections=5)

    # Test when there are fewer unique trajectories than reflections
    strategy.failed_trajectories = [
        {"trajectory": f"t{i}", "final_answer": "answer"} for i in range(2)
    ]
    strategy.reflection_map = {}
    assert strategy.reflect_condition() is True

    # Test when there are more unique trajectories than reflections but less than max_reflections
    strategy.failed_trajectories = [
        {"trajectory": f"t{i}", "final_answer": f"answer{i}"} for i in range(4)
    ]
    strategy.reflection_map = {"r1": "reflection1"}
    assert strategy.reflect_condition() is True

    # Test when there are max_reflections unique trajectories
    strategy.failed_trajectories = [
        {"trajectory": f"t{i}", "final_answer": "answer"} for i in range(5)
    ]
    strategy.reflection_map = {
        "r1": "reflection1",
        "r2": "reflection2",
        "r3": "reflection3",
        "r4": "reflection4",
    }
    assert strategy.reflect_condition() is False


def test_reflect() -> None:
    """Test the reflect method."""
    llm = MockLLM("gpt-3.5-turbo", responses=["Reflection 1", "Reflection 2"])
    strategy = LATSGeneralStrategy(llm=llm, max_unique=2)

    strategy.failed_trajectories = [
        {"trajectory": "Failed trajectory 1", "final_answer": "Incorrect answer 1"},
        {"trajectory": "Failed trajectory 2", "final_answer": "Incorrect answer 2"},
        {
            "trajectory": "Failed trajectory 1",
            "final_answer": "Incorrect answer 1",
        },  # Duplicate, should be ignored
    ]

    question = "What is the capital of France?"
    examples = "Example 1\nExample 2"
    prompt = "Reflect on the failed trajectory"
    additional_keys = {"key": "value"}

    reflections = strategy.reflect(question, examples, prompt, additional_keys)

    assert len(reflections) == 2
    assert reflections[0]["trajectory"] == "Failed trajectory 1"
    assert reflections[0]["reflection"] == "Reflection 1"
    assert reflections[1]["trajectory"] == "Failed trajectory 2"
    assert reflections[1]["reflection"] == "Reflection 2"

    assert strategy.reflection_map == reflections


def test_format_output() -> None:
    """Test the format_output method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSGeneralStrategy(llm=llm)
    # Test with minimal input
    iteration = 1
    current_node = Node()
    children_nodes = []
    thought_model_responses = []
    action_model_responses = []
    result = strategy.format_output(
        iteration,
        current_node,
        children_nodes,
        thought_model_responses,
        action_model_responses,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert isinstance(result, LATSStepOutput)
    assert result.iteration == 1
    assert result.current_node == current_node.to_dict()
    assert result.children_nodes == []
    assert result.thoughts_metrics == []
    assert result.actions_metrics == []
    assert result.values == []
    assert result.values_metrics == []
    assert result.simulation_results.simulation_reward == 0.0
    assert result.simulation_results.simulation_terminal_node is None
    assert result.simulation_results.simulation_current_nodes == []
    assert result.simulation_results.simulation_children_nodes == []
    assert result.simulation_results.simulation_thoughts_metrics == []
    assert result.simulation_results.simulation_actions_metrics == []
    assert result.simulation_results.simulation_values == []
    assert result.simulation_results.simulation_values_metrics == []

    # Test with full input
    strategy = LATSGeneralStrategy(llm=llm)
    iteration = 2
    current_node = Node()
    children_nodes = [Node()]

    # Test with sample token counts and model.
    prompt_tokens = 100
    completion_tokens = 50
    model = "gpt-3.5-turbo"

    usage = Usage()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    response = ModelResponse()
    response.choices = []
    response.usage = usage
    response.model = model
    response.time_taken = 0.5

    thought_model_responses = [response]

    action_model_responses = [response]
    values = [{"value": 0.5}]
    values_responses = [response]
    simulation_reward = 1.0
    simulation_terminal_node = Node()
    simulation_current_nodes = [Node()]
    simulation_children_nodes = [[Node()]]
    simulation_thought_model_responses = [[response]]
    simulation_action_model_responses = [[response]]
    simulation_values = [[{"value": 0.7}]]
    simulation_values_model_responses = [[response]]

    result = strategy.format_output(
        iteration,
        current_node,
        children_nodes,
        thought_model_responses,
        action_model_responses,
        values,
        values_responses,
        simulation_reward,
        simulation_terminal_node,
        simulation_current_nodes,
        simulation_children_nodes,
        simulation_thought_model_responses,
        simulation_action_model_responses,
        simulation_values,
        simulation_values_model_responses,
    )

    expect_prompt_metrics = PromptMetrics(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_cost=0.00015000000000000001,
        completion_cost=9.999999999999999e-05,
        total_cost=0.00025,
        prompt_time=0.5,
    )

    assert isinstance(result, LATSStepOutput)
    assert result.iteration == 2
    assert result.current_node == current_node.to_dict()
    assert result.children_nodes == [children_nodes[0].to_dict()]
    assert result.thoughts_metrics == [expect_prompt_metrics]

    assert result.actions_metrics == [expect_prompt_metrics]
    assert result.values == [{"value": 0.5}]
    assert result.values_metrics == [expect_prompt_metrics]
    assert result.simulation_results.simulation_reward == 1.0
    assert (
        result.simulation_results.simulation_terminal_node
        == simulation_terminal_node.to_dict()
    )
    assert result.simulation_results.simulation_current_nodes == [
        simulation_current_nodes[0].to_dict()
    ]
    assert result.simulation_results.simulation_children_nodes == [
        [simulation_children_nodes[0][0].to_dict()]
    ]
    assert result.simulation_results.simulation_thoughts_metrics == [
        [expect_prompt_metrics]
    ]
    assert result.simulation_results.simulation_actions_metrics == [
        [expect_prompt_metrics]
    ]
    assert result.simulation_results.simulation_values == [[{"value": 0.7}]]
    assert result.simulation_results.simulation_values_metrics == [
        [expect_prompt_metrics]
    ]

    # Test with paritial simulation data
    iteration = 3
    current_node = Node()
    children_nodes = [Node()]
    thought_model_responses = [response]
    action_model_responses = [response]
    simulation_reward = 0.5
    simulation_terminal_node = None
    simulation_current_nodes = [Node()]
    simulation_children_nodes = None
    simulation_thought_model_responses = [[response]]
    simulation_action_model_responses = None
    simulation_values = None
    simulation_values_model_responses = None

    result = strategy.format_output(
        iteration,
        current_node,
        children_nodes,
        thought_model_responses,
        action_model_responses,
        None,
        None,
        simulation_reward,
        simulation_terminal_node,
        simulation_current_nodes,
        simulation_children_nodes,
        simulation_thought_model_responses,
        simulation_action_model_responses,
        simulation_values,
        simulation_values_model_responses,
    )

    assert isinstance(result, LATSStepOutput)
    assert result.iteration == 3
    assert result.current_node == current_node.to_dict()
    assert result.children_nodes == [children_nodes[0].to_dict()]
    assert result.thoughts_metrics == [expect_prompt_metrics]
    assert result.actions_metrics == [expect_prompt_metrics]
    assert result.values == []
    assert result.values_metrics == []
    assert result.simulation_results.simulation_reward == 0.5
    assert result.simulation_results.simulation_terminal_node is None
    assert result.simulation_results.simulation_current_nodes == [
        simulation_current_nodes[0].to_dict()
    ]
    assert result.simulation_results.simulation_children_nodes == []
    assert result.simulation_results.simulation_thoughts_metrics == [
        [expect_prompt_metrics]
    ]
    assert result.simulation_results.simulation_actions_metrics == []
    assert result.simulation_results.simulation_values == []
    assert result.simulation_results.simulation_values_metrics == []

    # Test with empty lists
    iteration = 4
    current_node = Node()
    children_nodes = [Node()]
    thought_model_responses = [response]
    action_model_responses = [response]
    simulation_reward = 0.5
    simulation_terminal_node = None
    simulation_current_nodes = [Node()]
    simulation_children_nodes = None
    simulation_thought_model_responses = [[response]]
    simulation_action_model_responses = None
    simulation_values = None
    simulation_values_model_responses = None

    result = strategy.format_output(
        iteration,
        current_node,
        children_nodes,
        thought_model_responses,
        action_model_responses,
        None,
        None,
        simulation_reward,
        simulation_terminal_node,
        simulation_current_nodes,
        simulation_children_nodes,
        simulation_thought_model_responses,
        simulation_action_model_responses,
        simulation_values,
        simulation_values_model_responses,
    )

    assert isinstance(result, LATSStepOutput)
    assert result.iteration == 4
    assert result.current_node == current_node.to_dict()
    assert result.children_nodes == [children_nodes[0].to_dict()]
    assert result.thoughts_metrics == [expect_prompt_metrics]
    assert result.actions_metrics == [expect_prompt_metrics]
    assert result.values == []
    assert result.values_metrics == []
    assert result.simulation_results.simulation_reward == 0.5
    assert result.simulation_results.simulation_terminal_node is None
    assert result.simulation_results.simulation_current_nodes == [
        simulation_current_nodes[0].to_dict()
    ]
    assert result.simulation_results.simulation_children_nodes == []
    assert result.simulation_results.simulation_thoughts_metrics == [
        [expect_prompt_metrics]
    ]
    assert result.simulation_results.simulation_actions_metrics == []
    assert result.simulation_results.simulation_values == []
    assert result.simulation_results.simulation_values_metrics == []


def test_reset() -> None:
    """Test the reset method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSGeneralStrategy(llm=llm)

    strategy.root = "some_root"
    strategy.reflection_map = ["reflection1", "reflection2"]
    strategy.value_cache = {"value1": "value2"}
    strategy.failed_trajectories = ["trajectory1", "trajectory2"]

    # Call reset.
    strategy.reset()

    # Check if the state has been reset.
    assert strategy.root is None
    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == []
    assert strategy.value_cache == {}
