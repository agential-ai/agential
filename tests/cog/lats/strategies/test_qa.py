"""Unit tests for LATS QA strategies."""

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.lats.strategies.qa import (
    parse_qa_action,
    parse_qa_value,
    LATSHotQAStrategy,
    LATSTriviaQAStrategy,
    LATSAmbigNQStrategy,
    LATSFEVERStrategy,
)
from agential.utils.docstore import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from agential.cog.lats.node import Node

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
    valid_input = "Some text. Explanation: This is the explanation. Correctness score: 5"
    assert parse_qa_value(valid_input) == ("This is the explanation.", 5)

    # Test invalid value strings.
    assert parse_qa_value("No explanation or score") == ("Explanation not found", 0)
    assert parse_qa_value("Explanation: Only explanation") == ("Explanation not found", 0)
    assert parse_qa_value("Correctness score: 5") == ("Explanation not found", 0)

    # Test edge cases.
    assert parse_qa_value("Explanation: Empty. Correctness score: 0") == ("Empty.", 0)
    assert parse_qa_value("Explanation: Multi-line\nexplanation. Correctness score: 10") == ("Multi-line\nexplanation.", 10)

    # Test with unexpected format.
    assert parse_qa_value("Explanation: Tricky: score. Correctness score: 7") == ("Tricky: score.", 7)


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
        cache_values=True
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
    llm = FakeListChatModel(responses=["I should search for information about the topic. Action: Search[topic]"])
    strategy = LATSHotQAStrategy(llm=llm)
    
    question = "What is the capital of France?"
    examples = "Example 1\nExample 2"
    trajectory = "Previous thought"
    reflections = "Reflection 1\nReflection 2"
    depth = 1
    prompt = "Generate a thought"
    additional_keys = {"key": "value"}

    updated_trajectory, thought  = strategy.generate_thought(
        question, examples, trajectory, reflections, depth, prompt, additional_keys
    )

    assert thought == "I should search for information about the topic."
    assert updated_trajectory == "Previous thought\nThought 1: I should search for information about the topic."


def test_generate_action() -> None:
    """Test the generate_action method."""
    llm = FakeListChatModel(responses=["Search[capital of France]"])
    strategy = LATSHotQAStrategy(llm=llm)
    
    question = "What is the capital of France?"
    examples = "Example 1\nExample 2"
    trajectory = "Thought 1: I should search for information about the capital of France."
    reflections = "Reflection 1\nReflection 2"
    depth = 1
    prompt = "Generate an action"
    additional_keys = {"key": "value"}

    trajectory, action_type, query = strategy.generate_action(
        question, examples, trajectory, reflections, depth, prompt, additional_keys
    )
    assert trajectory == 'Thought 1: I should search for information about the capital of France.\nAction 1: Search[capital of France]'
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
    assert finish_result[0] == 'Previous trajectory\nObservation 1: Answer is CORRECT'     
    assert finish_result[1] == 1
    assert finish_result[2] == 'Answer is CORRECT'
    assert finish_result[3] is True
    assert finish_result[4] == {"search_result": "", "lookup_result": ""}

    # Test Search action.
    search_result = strategy.generate_observation(key, "Search", "capital of France", trajectory, 2)
    assert search_result[0] == 'Previous trajectory\nObservation 2: Paris is the capital of France.'
    assert search_result[1] == 0
    assert search_result[2] == 'Paris is the capital of France.'
    assert search_result[3] is False
    assert search_result[4] == {'search_result': 'Paris is the capital of France.', 'lookup_result': ''}

    # Test Lookup action.
    lookup_result = strategy.generate_observation(key, "Lookup", "Paris", trajectory, 3)
    assert lookup_result[0].endswith("Observation 3: Paris is a city in France.")
    assert lookup_result[1] == 0
    assert lookup_result[2] == 'Paris is a city in France.'
    assert lookup_result[3] is False
    assert lookup_result[4] == {'search_result': '', 'lookup_result': 'Paris is a city in France.'}

    # Test invalid action.
    invalid_result = strategy.generate_observation(key, "Invalid", "query", trajectory, 4)
    assert invalid_result[0] == 'Previous trajectory\nObservation 4: Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'
    assert invalid_result[1] == 0
    assert invalid_result[2] == 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'
    assert invalid_result[3] is False
    assert invalid_result[4] == {'search_result': '', 'lookup_result': ''}


def test_generate() -> None:
    """Test the generate method."""
    pass


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


def test_halting_condition() -> None:
    """Test the halting_condition method."""
    pass


def test_reflect_condition() -> None:
    """Test the reflect_condition method."""
    pass


def test_reflect() -> None:
    """Test the reflect method."""
    pass


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
    