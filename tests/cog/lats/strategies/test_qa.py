"""Unit tests for LATS QA strategies."""

from langchain_community.docstore.wikipedia import Wikipedia

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.lats.node import Node
from agential.cog.lats.output import LATSSimulationOutput, LATSReActOutput
from agential.cog.lats.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    LATS_INSTRUCTION_HOTPOTQA,
    LATS_REFLECT_INSTRUCTION_HOTPOTQA,
    LATS_VALUE_INSTRUCTION_HOTPOTQA,
)
from agential.cog.lats.strategies.qa import (
    LATSAmbigNQStrategy,
    LATSFEVERStrategy,
    LATSHotQAStrategy,
    LATSQAStrategy,
    LATSTriviaQAStrategy,
    get_node_trajectory_qa,
    parse_qa_action,
    parse_qa_value,
)
from agential.llm.llm import MockLLM
from agential.utils.docstore import DocstoreExplorer


def test_get_node_trajectory_qa() -> None:
    """Tests the get_node_trajectory_qa() function."""
    root = Node(
        state=LATSReActOutput(
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
        state=LATSReActOutput(
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
        state=LATSReActOutput(
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


def test_init() -> None:
    """Test initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    docstore = DocstoreExplorer(Wikipedia())
    strategy = LATSQAStrategy(
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
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSQAStrategy(llm=llm)

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
    llm = MockLLM(
        "gpt-3.5-turbo",
        responses=[
            "I should search for information about the topic. Action: Search[topic]"
        ],
    )
    strategy = LATSQAStrategy(llm=llm)

    question = "What is the capital of France?"
    examples = "Example 1\nExample 2"
    trajectory = "Previous thought"
    reflections = "Reflection 1\nReflection 2"
    depth = 1
    prompt = "Generate a thought"
    additional_keys = {"key": "value"}

    updated_trajectory, thought = strategy.generate_thought(
        question, examples, trajectory, reflections, depth, prompt, additional_keys, is_simulate=False
    )

    assert thought == "I should search for information about the topic."
    assert (
        updated_trajectory
        == "Previous thought\nThought 2: I should search for information about the topic."
    )


def test_generate_action() -> None:
    """Test the generate_action method."""
    llm = MockLLM("gpt-3.5-turbo", responses=["Search[capital of France]"])
    strategy = LATSQAStrategy(llm=llm)

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
        question, examples, trajectory, reflections, depth, prompt, additional_keys, is_simulate=False
    )
    assert (
        trajectory
        == "Thought 2: I should search for information about the capital of France.\nAction 2: Search[capital of France]"
    )
    assert action_type == "Search"
    assert query == "capital of France"


def test_generate_observation() -> None:
    """Test the generate_observation method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    docstore = DocstoreExplorer(None)
    docstore.search = lambda x: "Paris is the capital of France."
    docstore.lookup = lambda x: "Paris is a city in France."
    strategy = LATSQAStrategy(llm=llm, docstore=docstore)

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
        "search_result": "Paris is the capital of France.",
        "lookup_result": "",
    }

    # Test Lookup action.
    lookup_result = strategy.generate_observation(key, "Lookup", "Paris", trajectory, 3)
    assert lookup_result[0].endswith("Observation 4: Paris is a city in France.")
    assert lookup_result[1] == 0
    assert lookup_result[2] == "Paris is a city in France."
    assert lookup_result[3] is False
    assert lookup_result[4] == {
        "search_result": "",
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
        LATSReActOutput(
            thought="I need to search for the name of the kick boxer who was once considered the best but has been involved in controversies and crimes",
            action_type="Search",
            query="best kick boxer controversies crimes",
            observation="Badr Hari is the best kick boxer in the world.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari is the best kick boxer in the world.",
                "lookup_result": "",
            },
        ),
        LATSReActOutput(
            thought="I need to search for the best kickboxer who has been involved in controversies and crimes of violence",
            action_type="Search",
            query="best kick boxer controversies crimes",
            observation="Badr Hari is the best kick boxer in the world.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari is the best kick boxer in the world.",
                "lookup_result": "",
            },
        ),
        LATSReActOutput(
            thought="I need to search for the name of the kick boxer who was once considered the best in the world and has been involved in controversies",
            action_type="Search",
            query="best kick boxer controversies",
            observation="Badr Hari is the best kick boxer in the world.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari is the best kick boxer in the world.",
                "lookup_result": "",
            },
        ),
        LATSReActOutput(
            thought="I need to search for the best kick boxer who has been involved in controversies relating to unsportsmanlike conduct and crimes of violence outside the ring",
            action_type="Search",
            query="best kick boxer controversies violence",
            observation="Badr Hari is the best kick boxer in the world.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari is the best kick boxer in the world.",
                "lookup_result": "",
            },
        ),
        LATSReActOutput(
            thought="I need to search for the kickboxer who was once considered the best in the world but has been involved in controversies",
            action_type="Search",
            query="best kickboxer controversies",
            observation="Badr Hari is the best kick boxer in the world.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari is the best kick boxer in the world.",
                "lookup_result": "",
            },
        ),
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
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = LATSQAStrategy(llm=llm)
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
        is_simulate=False
    )
    assert len(children_nodes) == 5
    for gt_state, node in zip(gt_states, children_nodes):
        assert node.state == gt_state
        assert node.depth == 1
        assert node.reward == 0
        assert node.value == 0
        assert node.is_terminal is False
        assert node.visits == 0

    # Test generate with reflections.
    gt_states = [
        LATSReActOutput(
            thought="I need to search for the best kick boxer in the world who has been involved in controversies related to unsportsmanlike conduct and crimes of violence outside the ring",
            action_type="Search",
            query="best kickboxer controversies violence",
            observation="Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
                "lookup_result": "",
            },
        ),
        LATSReActOutput(
            thought="I need to search for the best kick boxer in the world and then look into his controversies related to unsportsmanlike conduct and crimes of violence",
            action_type="Search",
            query="best kick boxer in the world",
            observation="Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
                "lookup_result": "",
            },
        ),
        LATSReActOutput(
            thought="I need to search for the best kick boxer in the world who has been involved in controversies related to unsportsmanlike conduct and violence outside of the ring",
            action_type="Search",
            query="best kick boxer in the world controversies",
            observation="Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
                "lookup_result": "",
            },
        ),
        LATSReActOutput(
            thought="I need to search for the best kickboxer in the world who has been involved in controversies regarding unsportsmanlike conduct and crimes of violence outside the ring",
            action_type="Search",
            query="best kickboxer controversies",
            observation="Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
                "lookup_result": "",
            },
        ),
        LATSReActOutput(
            thought="I need to search for the best kick boxer in the world and his controversies regarding unsportsmanlike conducts and crimes of violence",
            action_type="Search",
            query="best kick boxer in the world controversies",
            observation="Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
                "lookup_result": "",
            },
        ),
    ]
    responses = [
        "My reasoning for this question failed because I did not narrow down the search to focus on kick boxers and instead ended up with unrelated information",
        "My reasoning failed because I did not focus on gathering specific information related to the individual's kickboxing career and controversies, leading to an incorrect answer",
        "I need to search for the best kick boxer in the world who has been involved in controversies related to unsportsmanlike conduct and crimes of violence outside the ring",
        "Search[best kickboxer controversies violence]\nObservation 1: Could not find [best kickboxer controversies violence]",
        "I need to search for the best kick boxer in the world and then look into his controversies related to unsportsmanlike conduct and crimes of violence",
        "Search[best kick boxer in the world]\nObservation 1: There have been several renowned kickboxers throughout history, such as Buakaw Banchamek, Ernesto Hoost, and Ramon Dekkers",
        "I need to search for the best kick boxer in the world who has been involved in controversies related to unsportsmanlike conduct and violence outside of the ring",
        "Search[best kick boxer in the world controversies]\nObservation 1: Could not find [best kick boxer in the world controversies]",
        "I need to search for the best kickboxer in the world who has been involved in controversies regarding unsportsmanlike conduct and crimes of violence outside the ring",
        "Search[best kickboxer controversies]\nObservation 1: Could not find [best kickboxer controversies]",
        "I need to search for the best kick boxer in the world and his controversies regarding unsportsmanlike conducts and crimes of violence",
        "Search[best kick boxer in the world controversies]\nObservation 1: Could not find [best kick boxer in the world controversies]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = LATSQAStrategy(llm=llm)
    strategy.docstore.search = (
        lambda x: "Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues."
    )
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
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_HOTPOTQA,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        reflect_additional_keys={},
        is_simulate=False
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
        "I think the answer is Mike Tyson.",
        "Finish[Mike Tyson]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = LATSQAStrategy(llm=llm, n_samples=1)

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
        is_simulate=False
    )
    assert len(children_nodes) == 1
    assert children_nodes[0].state.thought == "I think the answer is Mike Tyson."
    assert children_nodes[0].state.action_type == "Finish"
    assert children_nodes[0].state.query == "Mike Tyson"
    assert children_nodes[0].is_terminal
    assert children_nodes[0].reward == 0


def test_select_node() -> None:
    """Test the select_node method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSQAStrategy(llm=llm)

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
        LATSReActOutput(
            thought="I need to search for the name of the kick boxer who was once considered the best but has been involved in controversies and crimes",
            action_type="Search",
            query="best kick boxer controversies crimes",
            observation="Badr Hari is the best kick boxer in the world.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari is the best kick boxer in the world.",
                "lookup_result": "",
            },
        ),
        LATSReActOutput(
            thought="I need to search for the best kickboxer who has been involved in controversies and crimes of violence",
            action_type="Search",
            query="best kick boxer controversies crimes",
            observation="Badr Hari is the best kick boxer in the world.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari is the best kick boxer in the world.",
                "lookup_result": "",
            },
        ),
        LATSReActOutput(
            thought="I need to search for the name of the kick boxer who was once considered the best in the world and has been involved in controversies",
            action_type="Search",
            query="best kick boxer controversies",
            observation="Badr Hari is the best kick boxer in the world.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari is the best kick boxer in the world.",
                "lookup_result": "",
            },
        ),
        LATSReActOutput(
            thought="I need to search for the best kick boxer who has been involved in controversies relating to unsportsmanlike conduct and crimes of violence outside the ring",
            action_type="Search",
            query="best kick boxer controversies violence",
            observation="Badr Hari is the best kick boxer in the world.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari is the best kick boxer in the world.",
                "lookup_result": "",
            },
        ),
        LATSReActOutput(
            thought="I need to search for the kickboxer who was once considered the best in the world but has been involved in controversies",
            action_type="Search",
            query="best kickboxer controversies",
            observation="Badr Hari is the best kick boxer in the world.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari is the best kick boxer in the world.",
                "lookup_result": "",
            },
        ),
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
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = LATSQAStrategy(llm=llm)
    strategy.docstore.search = (
        lambda x: "Badr Hari is the best kick boxer in the world."
    )

    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'
    key = "Badr Hari"

    root = strategy.initialize()

    children_nodes = strategy.expand_node(
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
    assert strategy.root.children == children_nodes


def test_evaluate_node() -> None:
    """Test the evaluate_node method."""
    llm = MockLLM(
        "gpt-3.5-turbo",
        responses=["Explanation: Good trajectory. Correctness score: 8"],
    )
    strategy = LATSQAStrategy(llm=llm)

    root = strategy.initialize()
    child1 = Node(
        state=LATSReActOutput(
            thought="Child 1",
            action_type="",
            query="",
            observation="",
            answer="",
            external_tool_info={},
        ),
        parent=root,
    )
    child2 = Node(
        state=LATSReActOutput(
            thought="Child 2",
            action_type="",
            query="",
            observation="",
            answer="",
            external_tool_info={},
        ),
        parent=root,
        is_terminal=True,
    )

    root.children = [child1, child2]

    question = "What is the capital of France?"
    examples = "Example 1\nExample 2"
    prompt = "Evaluate this trajectory"

    strategy.reflection_map = [
        {
            "trajectory": "Failed trajectory",
            "reflection": "This trajectory failed because...",
        }
    ]

    values = strategy.evaluate_node(root, question, examples, prompt, {})

    assert len(values) == 1  # Only one non-terminal child.
    assert "explanation" in values[0]
    assert "value" in values[0]
    assert values[0]["explanation"] == "Good trajectory."
    assert values[0]["value"] == 0.8  # 8 / 10

    assert child1.value == 0.8
    assert child2.value == 0  # Terminal node, value not updated.

    # Test caching.
    strategy.cache_values = True
    cached_values = strategy.evaluate_node(root, question, examples, prompt, {})
    assert cached_values == values

    # Test with empty reflection_map.
    strategy.reflection_map = []
    empty_reflection_values = strategy.evaluate_node(
        root, question, examples, prompt, {}
    )
    assert empty_reflection_values == values


def test_simulate_node() -> None:
    """Test the simulate_node method."""
    responses = [
        "I need to search for the capital of France",
        "Search[capital of France]",
        "I need to search for the capital of France",
        "Search[capital of France]",
        "The trajectory provided is completely incorrect as the observation received does not relate to the search query at all, indicating that the search term might have been mistyped or confused",
        "The search results did not return the information needed",
        "Search[capital of France]\nObservation 2: The capital of France is Paris, known for its art, fashion, gastronomy, and culture",
        "The search did not return relevant information",
        "Search[capital of France Wikipedia]\nObservation 2: The capital of France is Paris, the largest city in France and its capital since the 4th century",
        "The trajectory provided is incorrect because the environmental observation does not relate to the question asked",
        "This trajectory is incorrect as it did not provide any relevant information regarding the capital of France",
        "There seems to be an issue with the search results",
        "Search[similar entities to the capital of France]\nObservation 3: Similar: [Paris, Marseille, Lyon, Toulouse, Lille]\nThought 4: The capital of France is Paris",
        "The search results seem to be incorrect",
        "Search[capital of France]\nObservation 3: The capital of France is Paris",
        "The trajectory is incorrect as the observations did not provide any relevant information related to the question",
        "This trajectory is incorrect as the focus should have been on verifying the information related to the capital of France, rather than repeatedly trying the same search query that does not provide the desired information",
    ]

    qa_strategy = LATSQAStrategy(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), depth_limit=3, n_samples=2
    )
    root_node = qa_strategy.initialize()

    question = "What is the capital of France?"
    key = "Paris"
    examples = HOTPOTQA_FEWSHOT_EXAMPLES_REACT
    reflect_examples = HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT
    value_examples = HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE
    prompt = LATS_INSTRUCTION_HOTPOTQA
    reflect_prompt = LATS_REFLECT_INSTRUCTION_HOTPOTQA
    value_prompt = LATS_VALUE_INSTRUCTION_HOTPOTQA
    additional_keys = {}
    reflect_additional_keys = {}
    value_additional_keys = {}

    reward, final_node, simulation_results = qa_strategy.simulate_node(
        node=root_node,
        question=question,
        key=key,
        examples=examples,
        reflect_examples=reflect_examples,
        value_examples=value_examples,
        prompt=prompt,
        reflect_prompt=reflect_prompt,
        value_prompt=value_prompt,
        additional_keys=additional_keys,
        reflect_additional_keys=reflect_additional_keys,
        value_additional_keys=value_additional_keys,
    )

    assert isinstance(reward, float)
    assert isinstance(final_node, Node)
    assert isinstance(simulation_results, list)

    assert final_node.depth <= qa_strategy.depth_limit

    assert len(simulation_results) > 0

    assert -1 <= reward <= 1


def test_backpropagate_node() -> None:
    """Test the backpropagate_node method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSQAStrategy(llm=llm)

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
    strategy = LATSQAStrategy(llm=llm)

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
    strategy = LATSQAStrategy(llm=llm, max_unique=3, max_reflections=5)

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
    strategy = LATSQAStrategy(llm=llm, max_unique=2)

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


def test_create_output_dict() -> None:
    """Test create_output_dict method."""
    llm = MockLLM("gpt-3.5-turbo", responses=["1"])
    strategy = LATSQAStrategy(llm=llm, max_unique=2)

    gt_out = {'iteration': 1, 'current_node': {'state': LATSReActOutput(thought='', action_type='', query='', observation='', answer='', external_tool_info={}), 'visits': 0, 'value': 0, 'depth': 0, 'is_terminal': False, 'reward': 0}, 'children_nodes': [{'state': LATSReActOutput(thought='', action_type='', query='', observation='', answer='', external_tool_info={}), 'visits': 0, 'value': 0, 'depth': 0, 'is_terminal': False, 'reward': 0}], 'values': [{}], 'simulation_reward': 1.0, 'simulation_terminal_node': {'state': LATSReActOutput(thought='', action_type='', query='', observation='', answer='', external_tool_info={}), 'visits': 0, 'value': 0, 'depth': 0, 'is_terminal': False, 'reward': 0}, 'simulation_results': [LATSSimulationOutput(current_node={'state': LATSReActOutput(thought='', action_type='', query='', observation='', answer='', external_tool_info={}), 'visits': 0, 'value': 0, 'depth': 0, 'is_terminal': False, 'reward': 0}, children_nodes=[], values=[{}])], 'prompt_metrics': {'thought': [], 'action': [], 'value': [], 'simulate_thought': [], 'simulate_action': [], 'simulate_value': [], 'reflection': []}}
    simulation_results = [
        {"current_node": Node(), "children_nodes": [], "values": [{}]}
    ]
    out = strategy.create_output_dict(
        iteration=1,
        current_node=Node(),
        children_nodes=[Node()],
        values=[{}],
        simulation_reward=1.0,
        simulation_terminal_node=Node(),
        simulation_results=simulation_results,
    )
    assert out == gt_out

    # Test half empty.
    gt_out = {'iteration': 1, 'current_node': {'state': LATSReActOutput(thought='', action_type='', query='', observation='', answer='', external_tool_info={}), 'visits': 0, 'value': 0, 'depth': 0, 'is_terminal': False, 'reward': 0}, 'children_nodes': [{'state': LATSReActOutput(thought='', action_type='', query='', observation='', answer='', external_tool_info={}), 'visits': 0, 'value': 0, 'depth': 0, 'is_terminal': False, 'reward': 0}], 'values': [], 'simulation_reward': 0, 'simulation_terminal_node': {}, 'simulation_results': [], 'prompt_metrics': {'thought': [], 'action': [], 'value': [], 'simulate_thought': [], 'simulate_action': [], 'simulate_value': [], 'reflection': []}}
    out = strategy.create_output_dict(
        iteration=1,
        current_node=Node(),
        children_nodes=[Node()],
        values=None,
        simulation_reward=None,
        simulation_terminal_node=None,
        simulation_results=None,
    )
    assert out == gt_out


def test_reset() -> None:
    """Test the reset method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSQAStrategy(llm=llm)

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
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    hotqa_strategy = LATSHotQAStrategy(llm=llm)
    triviaqa_strategy = LATSTriviaQAStrategy(llm=llm)
    ambignq_strategy = LATSAmbigNQStrategy(llm=llm)
    fever_strategy = LATSFEVERStrategy(llm=llm)

    assert isinstance(hotqa_strategy, LATSHotQAStrategy)
    assert isinstance(triviaqa_strategy, LATSTriviaQAStrategy)
    assert isinstance(ambignq_strategy, LATSAmbigNQStrategy)
    assert isinstance(fever_strategy, LATSFEVERStrategy)
