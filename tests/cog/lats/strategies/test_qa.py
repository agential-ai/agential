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
    pass


def test_initialize() -> None:
    """Test the initialize method."""
    pass


def test_generate_thought() -> None:
    """Test the generate_thought method."""
    pass


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
    