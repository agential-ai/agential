"""Unit tests for LATS QA strategies."""

from langchain_community.docstore.wikipedia import Wikipedia

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.lats.node import Node
from agential.cog.lats.output import LATSReActStepOutput, LATSSimulationOutput
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
    assert strategy._prompt_metrics == {
        "thought": [],
        "action": [],
        "value": [],
        "simulate_thought": [],
        "simulate_action": [],
        "simulate_value": [],
        "reflection": [],
    }


def test_generate() -> None:
    """Test the generate method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSQAStrategy(
        llm=llm,
        n_samples=5,
        max_reflections=4,
        depth_limit=7,
        max_unique=5,
        cache_values=True,
    )


def test_generate_children_nodes() -> None:
    """Test the generate method."""
    gt_states = [
        LATSReActStepOutput(
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
        LATSReActStepOutput(
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
        LATSReActStepOutput(
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
        LATSReActStepOutput(
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
        LATSReActStepOutput(
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
    gt_prompt_metrics = {
        "thought": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
        ],
        "action": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
        ],
        "value": [],
        "simulate_thought": [],
        "simulate_action": [],
        "simulate_value": [],
        "reflection": [],
    }

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

    children_nodes = strategy.generate_children_nodes(
        node=root,
        question=question,
        key=key,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_HOTPOTQA,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        reflect_additional_keys={},
        is_simulate=False,
    )
    assert len(children_nodes) == 5
    for gt_state, node in zip(gt_states, children_nodes):
        assert node.state == gt_state
        assert node.depth == 1
        assert node.reward == 0
        assert node.value == 0
        assert node.is_terminal is False
        assert node.visits == 0
    assert strategy._prompt_metrics == gt_prompt_metrics

    # Test generate with reflections.
    gt_states = [
        LATSReActStepOutput(
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
        LATSReActStepOutput(
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
        LATSReActStepOutput(
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
        LATSReActStepOutput(
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
        LATSReActStepOutput(
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
    gt_prompt_metrics = {
        "thought": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
        ],
        "action": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
        ],
        "value": [],
        "simulate_thought": [],
        "simulate_action": [],
        "simulate_value": [],
        "reflection": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
        ],
    }
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
    children_nodes = strategy.generate_children_nodes(
        node=root,
        question=question,
        key=key,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_HOTPOTQA,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        reflect_additional_keys={},
        is_simulate=False,
    )
    assert len(children_nodes) == 5
    for gt_state, node in zip(gt_states, children_nodes):
        assert node.state == gt_state
        assert node.depth == 1
        assert node.reward == 0
        assert node.value == 0
        assert node.is_terminal is False
        assert node.visits == 0
    assert strategy._prompt_metrics == gt_prompt_metrics

    # Test case with a terminal child node (reward 0)
    gt_prompt_metrics = {
        "thought": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            }
        ],
        "action": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            }
        ],
        "value": [],
        "simulate_thought": [],
        "simulate_action": [],
        "simulate_value": [],
        "reflection": [],
    }

    responses = [
        "I think the answer is Mike Tyson.",
        "Finish[Mike Tyson]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = LATSQAStrategy(llm=llm, n_samples=1)

    root = strategy.initialize()
    children_nodes = strategy.generate_children_nodes(
        node=root,
        question=question,
        key=key,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_HOTPOTQA,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        reflect_additional_keys={},
        is_simulate=False,
    )
    assert len(children_nodes) == 1
    assert children_nodes[0].state.thought == "I think the answer is Mike Tyson."
    assert children_nodes[0].state.action_type == "Finish"
    assert children_nodes[0].state.query == "Mike Tyson"
    assert children_nodes[0].is_terminal
    assert children_nodes[0].reward == 0

    assert strategy._prompt_metrics == gt_prompt_metrics


def test_generate_action() -> None:
    """Test the generate_action method."""
    gt_prompt_metrics = {
        "thought": [],
        "action": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            }
        ],
        "value": [],
        "simulate_thought": [],
        "simulate_action": [],
        "simulate_value": [],
        "reflection": [],
    }

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
        question,
        examples,
        trajectory,
        reflections,
        depth,
        prompt,
        additional_keys,
        is_simulate=False,
    )
    assert (
        trajectory
        == "Thought 2: I should search for information about the capital of France.\nAction 2: Search[capital of France]"
    )
    assert action_type == "Search"
    assert query == "capital of France"

    assert strategy._prompt_metrics == gt_prompt_metrics


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


def test_evaluate_node() -> None:
    """Test the evaluate_node method."""
    gt_prompt_metrics = {
        "thought": [],
        "action": [],
        "value": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            }
        ],
        "simulate_thought": [],
        "simulate_action": [],
        "simulate_value": [],
        "reflection": [],
    }

    llm = MockLLM(
        "gpt-3.5-turbo",
        responses=["Explanation: Good trajectory. Correctness score: 8"],
    )
    strategy = LATSQAStrategy(llm=llm)

    root = strategy.initialize()
    child1 = Node(
        state=LATSReActStepOutput(
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
        state=LATSReActStepOutput(
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

    assert strategy._prompt_metrics == gt_prompt_metrics

    assert len(values) == 1  # Only one non-terminal child.
    assert "explanation" in values[0]
    assert "value" in values[0]
    assert values[0]["explanation"] == "Good trajectory."
    assert values[0]["value"] == 0.8  # 8 / 10

    assert child1.value == 0.8
    assert child2.value == 0  # Terminal node, value not updated.

    # Test caching.
    gt_prompt_metrics = {
        "thought": [],
        "action": [],
        "value": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
        ],
        "simulate_thought": [],
        "simulate_action": [],
        "simulate_value": [],
        "reflection": [],
    }
    strategy.cache_values = True
    cached_values = strategy.evaluate_node(root, question, examples, prompt, {})
    assert cached_values == values

    # Test with empty reflection_map.
    strategy.reflection_map = []
    empty_reflection_values = strategy.evaluate_node(
        root, question, examples, prompt, {}
    )
    assert empty_reflection_values == values

    assert strategy._prompt_metrics == gt_prompt_metrics


def test_simulate_node() -> None:
    """Test the simulate_node method."""
    gt_prompt_metrics = {
        "thought": [],
        "action": [],
        "value": [],
        "simulate_thought": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
        ],
        "simulate_action": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
        ],
        "simulate_value": [
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_cost": 1.5e-05,
                "completion_tokens_cost": 3.9999999999999996e-05,
                "total_tokens_cost": 5.4999999999999995e-05,
                "time_sec": 0.5,
            },
        ],
        "reflection": [],
    }

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

    assert qa_strategy._prompt_metrics == gt_prompt_metrics


def test_expand_node() -> None:
    """Test the expand_node method."""
    gt_states = [
        LATSReActStepOutput(
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
        LATSReActStepOutput(
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
        LATSReActStepOutput(
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
        LATSReActStepOutput(
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
        LATSReActStepOutput(
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
