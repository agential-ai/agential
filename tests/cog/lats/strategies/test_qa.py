"""Unit tests for LATS QA strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.docstore.wikipedia import Wikipedia

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.lats.node import Node
from agential.cog.lats.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    LATS_INSTRUCTION_HOTPOTQA,
    LATS_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.lats.strategies.qa import (
    LATSAmbigNQStrategy,
    LATSFEVERStrategy,
    LATSHotQAStrategy,
    LATSTriviaQAStrategy,
    parse_qa_action,
    parse_qa_value,
)
from agential.cog.react.output import ReActOutput
from agential.utils.docstore import DocstoreExplorer


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


def test_init() -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=[])
    docstore = DocstoreExplorer(Wikipedia())
    strategy = LATSHotQAStrategy(
        llm=llm,
        docstore=docstore,
        n_samples=5,
        max_reflections=4,
        depth_limit=7,
        max_unique=5,
        cache_values=True,
    )

    assert strategy.llm == llm
    assert isinstance(strategy.docstore, DocstoreExplorer)
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
    strategy = LATSHotQAStrategy(llm=llm)

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
        responses=[
            "I should search for information about the topic. Action: Search[topic]"
        ]
    )
    strategy = LATSHotQAStrategy(llm=llm)

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
    llm = FakeListChatModel(responses=["Search[capital of France]"])
    strategy = LATSHotQAStrategy(llm=llm)

    question = "What is the capital of France?"
    examples = "Example 1\nExample 2"
    trajectory = (
        "Thought 2: I should search for information about the capital of France."
    )
    reflections = "Reflection 1\nReflection 2"
    depth = 1
    prompt = "Generate an action"
    additional_keys = {"key": "value"}

    trajectory, action_type, query = strategy.generate_action(
        question, examples, trajectory, reflections, depth, prompt, additional_keys
    )
    assert (
        trajectory
        == "Thought 2: I should search for information about the capital of France.\nAction 2: Search[capital of France]"
    )
    assert action_type == "Search"
    assert query == "capital of France"


def test_generate_observation() -> None:
    """Test the generate_observation method."""
    llm = FakeListChatModel(responses=[])
    docstore = DocstoreExplorer(None)
    docstore.search = lambda x: "Paris is the capital of France."
    docstore.lookup = lambda x: "Paris is a city in France."
    strategy = LATSHotQAStrategy(llm=llm, docstore=docstore)

    key = "Paris"
    trajectory = "Previous trajectory"

    # Test Finish action.
    finish_result = strategy.generate_observation(key, "Finish", "Paris", trajectory, 1)
    assert finish_result[0] == "Previous trajectory\nObservation 2: Answer is CORRECT"
    assert finish_result[1] == 1
    assert finish_result[2] == "Answer is CORRECT"
    assert finish_result[3] is True
    assert finish_result[4] == {"search_result": "", "lookup_result": ""}

    # Test Search action.
    search_result = strategy.generate_observation(
        key, "Search", "capital of France", trajectory, 2
    )
    assert (
        search_result[0]
        == "Previous trajectory\nObservation 3: Paris is the capital of France."
    )
    assert search_result[1] == 0
    assert search_result[2] == "Paris is the capital of France."
    assert search_result[3] is False
    assert search_result[4] == {
        "search_result": "Badr Hari is the best kick boxer in the world.",
        "lookup_result": "",
    }

    # Test Lookup action.
    lookup_result = strategy.generate_observation(key, "Lookup", "Paris", trajectory, 3)
    assert lookup_result[0].endswith("Observation 4: Paris is a city in France.")
    assert lookup_result[1] == 0
    assert lookup_result[2] == "Paris is a city in France."
    assert lookup_result[3] is False
    assert lookup_result[4] == {
        "search_result": "Badr Hari is the best kick boxer in the world.",
        "lookup_result": "Paris is a city in France.",
    }

    # Test invalid action.
    invalid_result = strategy.generate_observation(
        key, "Invalid", "query", trajectory, 4
    )
    assert (
        invalid_result[0]
        == "Previous trajectory\nObservation 5: Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
    )
    assert invalid_result[1] == 0
    assert (
        invalid_result[2]
        == "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
    )
    assert invalid_result[3] is False
    assert invalid_result[4] == {"search_result": "", "lookup_result": ""}


def test_generate() -> None:
    """Test the generate method."""

    gt_states = [
        ReActOutput(thought='I need to search for the name of the kick boxer who was once considered the best but has been involved in controversies and crimes', action_type='Search', query='best kick boxer controversies crimes', observation='Badr Hari is the best kick boxer in the world.', answer='', external_tool_info={'search_result': 'Badr Hari is the best kick boxer in the world.', 'lookup_result': ''}),
        ReActOutput(thought='I need to search for the best kickboxer who has been involved in controversies and crimes of violence', action_type='Search', query='best kick boxer controversies crimes', observation='Badr Hari is the best kick boxer in the world.', answer='', external_tool_info={'search_result': 'Badr Hari is the best kick boxer in the world.', 'lookup_result': ''}),
        ReActOutput(thought='I need to search for the name of the kick boxer who was once considered the best in the world and has been involved in controversies', action_type='Search', query='best kick boxer controversies', observation='Badr Hari is the best kick boxer in the world.', answer='', external_tool_info={'search_result': 'Badr Hari is the best kick boxer in the world.', 'lookup_result': ''}),
        ReActOutput(thought='I need to search for the best kick boxer who has been involved in controversies relating to unsportsmanlike conduct and crimes of violence outside the ring', action_type='Search', query='best kick boxer controversies violence', observation='Badr Hari is the best kick boxer in the world.', answer='', external_tool_info={'search_result': 'Badr Hari is the best kick boxer in the world.', 'lookup_result': ''}),
        ReActOutput(thought='I need to search for the kickboxer who was once considered the best in the world but has been involved in controversies', action_type='Search', query='best kickboxer controversies', observation='Badr Hari is the best kick boxer in the world.', answer='', external_tool_info={'search_result': 'Badr Hari is the best kick boxer in the world.', 'lookup_result': ''}),
    ]

    responses = [
        "I need to search for the name of the kick boxer who was once considered the best but has been involved in controversies and crimes",
        "Search[best kick boxer controversies crimes]",
        "I need to search for the best kickboxer who has been involved in controversies and crimes of violence",
        "Search[best kick boxer controversies crimes]\nObservation 0: No exact matches found",
        "I need to search for the name of the kick boxer who was once considered the best in the world and has been involved in controversies",
        "Search[best kick boxer controversies]\nObservation 0: Could not find [best kick boxer controversies]",
        "I need to search for the best kick boxer who has been involved in controversies relating to unsportsmanlike conduct and crimes of violence outside the ring",
        "Search[best kick boxer controversies violence]\nObservation 0: Could not find [best kick boxer controversies violence]",
        "I need to search for the kickboxer who was once considered the best in the world but has been involved in controversies",
        "Search[best kickboxer controversies]\nObservation 0: The search results show multiple kickboxers who have been involved in controversies",
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = LATSHotQAStrategy(llm=llm)
    strategy.docstore.search = (
        lambda x: "Badr Hari is the best kick boxer in the world."
    )

    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'
    key = "Badr Hari"

    root = strategy.initialize()

    children_nodes = strategy.generate(
        node=root,
        question=question,
        key=key,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_HOTPOTQA,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_HOTPOTQA,
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


def test_select_node() -> None:
    """Test the select_node method."""
    pass


def test_expand_node() -> None:
    """Test the expand_node method."""
    pass


def test_evaluate_node() -> None:
    """Test the evaluate_node method."""
    pass


def test_simulate_node() -> None:
    """Test the simulate_node method."""
    pass


def test_backpropagate_node() -> None:
    """Test the backpropagate_node method."""
    pass


def test_halting_condition():
    """Test the halting_condition method."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSHotQAStrategy(llm=llm)

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

def test_reflect_condition():
    """Test the reflect_condition method."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSHotQAStrategy(llm=llm, max_unique=3, max_reflections=5)

    # Test when there are fewer unique trajectories than reflections
    strategy.failed_trajectories = [{"trajectory": f"t{i}", "final_answer": "answer"} for i in range(2)]
    strategy.reflection_map = {}
    assert strategy.reflect_condition() is True

    # Test when there are more unique trajectories than reflections but less than max_reflections
    strategy.failed_trajectories = [{"trajectory": f"t{i}", "final_answer": f"answer{i}"} for i in range(4)]
    strategy.reflection_map = {"r1": "reflection1"}
    assert strategy.reflect_condition() is True

    # Test when there are max_reflections unique trajectories
    strategy.failed_trajectories = [{"trajectory": f"t{i}", "final_answer": "answer"} for i in range(5)]
    strategy.reflection_map = {"r1": "reflection1", "r2": "reflection2", "r3": "reflection3", "r4": "reflection4"}
    assert strategy.reflect_condition() is False


def test_reflect():
    """Test the reflect method."""
    llm = FakeListChatModel(responses=["Reflection 1", "Reflection 2"])
    strategy = LATSHotQAStrategy(llm=llm, max_unique=2)
    
    strategy.failed_trajectories = [
        {"trajectory": "Failed trajectory 1", "final_answer": "Incorrect answer 1"},
        {"trajectory": "Failed trajectory 2", "final_answer": "Incorrect answer 2"},
        {"trajectory": "Failed trajectory 1", "final_answer": "Incorrect answer 1"},  # Duplicate, should be ignored
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


def test_reset() -> None:
    """Test the reset method."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSHotQAStrategy(llm=llm)

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


def test_instantiate_strategies() -> None:
    """Test the instantiation of various LATS QA strategies."""
    llm = FakeListChatModel(responses=[])
    hotqa_strategy = LATSHotQAStrategy(llm=llm)
    triviaqa_strategy = LATSTriviaQAStrategy(llm=llm)
    ambignq_strategy = LATSAmbigNQStrategy(llm=llm)
    fever_strategy = LATSFEVERStrategy(llm=llm)

    assert isinstance(hotqa_strategy, LATSHotQAStrategy)
    assert isinstance(triviaqa_strategy, LATSTriviaQAStrategy)
    assert isinstance(ambignq_strategy, LATSAmbigNQStrategy)
    assert isinstance(fever_strategy, LATSFEVERStrategy)
