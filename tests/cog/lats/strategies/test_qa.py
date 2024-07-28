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
    assert updated_trajectory == "Previous thought\nThought 2: I should search for information about the topic."

def test_generate_action() -> None:
    """Test the generate_action method."""
    pass


def test_generate_observation() -> None:
    """Test the generate_observation method."""
    pass


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
    