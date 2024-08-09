"""Unit tests for ReAct QA strategies."""

from tiktoken import Encoding

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.react.prompts import (
    REACT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.react.strategies.qa import (
    ReActAmbigNQStrategy,
    ReActFEVERStrategy,
    ReActHotQAStrategy,
    ReActQAStrategy,
    ReActTriviaQAStrategy,
    parse_qa_action,
)
from agential.llm.llm import BaseLLM, MockLLM
from agential.utils.docstore import DocstoreExplorer


def test_parse_qa_action() -> None:
    """Test parse_qa_action function."""
    # Test with a valid action string.
    valid_string = "ActionType[Argument]"
    assert parse_qa_action(valid_string) == ("ActionType", "Argument")

    # Test with an invalid action string (missing brackets).
    invalid_string = "ActionType Argument"
    assert parse_qa_action(invalid_string) == ("", "")

    # Test with an invalid action string (no action type).
    invalid_string = "[Argument]"
    assert parse_qa_action(invalid_string) == ("", "")

    # Test with an invalid action string (no argument).
    invalid_string = "ActionType[]"
    assert parse_qa_action(invalid_string) == ("", "")


def test_init() -> None:
    """Test ReActQAStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActQAStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseLLM)
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 5000
    assert isinstance(strategy.docstore, DocstoreExplorer)
    assert isinstance(strategy.enc, Encoding)
    assert strategy._scratchpad == ""
    assert strategy._finished == False


def test_generate() -> None:
    """Tests ReActQAStrategy generate."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    gt_result = "I need to search for the best kickboxer in the world who has been involved in controversies and crimes."
    gt_scratchpad = "\nThought: I need to search for the best kickboxer in the world who has been involved in controversies and crimes."
    responses = [
        "I need to search for the best kickboxer in the world who has been involved in controversies and crimes.\nAction 1: Search[best kickboxer in the world controversies crimes]\nObservation 1: Could not find exact match. Similar: ['List of kickboxers', 'Kickboxing', 'List of controversies involving Kickboxing']\nThought 2: I should try searching for the best kickboxer in the world and then look for any controversies or crimes related to him.\nAction 2: Search[best kickboxer in the world]\nObservation 2: Could not find exact match. Similar: ['List of best kickboxers in the world', 'List of kickboxing organizations', 'Kickboxing', 'Best Fighters in the World']\nThought 3: I can try searching for top kickboxers and then look for controversies and crimes.\nAction 3: Search[top kickboxers]\nObservation 3: Could not find exact match. Similar: ['Top 10 kickboxers', 'Top 5 kickboxers', 'Top 15 kickboxers']\nThought 4: I need to refine my search terms to find the information I need.\nAction 4: Search[most famous kickboxer controversies crimes]\nObservation 4: Could not find exact match. Similar: ['Famous kickboxers', 'Kickboxing controversies', 'Famous kickboxers in the world']\nThought 5: I should try searching for famous kickboxers involved in controversies and crimes.\nAction 5: Search[famous kickboxers controversies crimes]\nObservation 5: Could not find exact match. Similar: ['Famous kickboxers', 'Kickboxing controversies', 'Famous kickboxers in the world']\nThought 6: I am unable to find the specific information I need within the given steps. \nAction 6: Finish[unable to find answer]"
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    result = strategy.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert result == gt_result
    assert not strategy._finished
    assert strategy._scratchpad == gt_scratchpad


def test_generate_action() -> None:
    """Tests ReActQAStrategy generate_action."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    gt_action_type = "Search"
    gt_query = "best kick boxer in the world controversies crimes"
    init_scratchpad = "\nThought: I need to search for the best kickboxer in the world who has been involved in controversies and crimes."
    responses = ["Search[best kick boxer in the world controversies crimes]"]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    strategy._scratchpad = init_scratchpad
    strategy._finished = False
    action_type, query = strategy.generate_action(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert action_type == gt_action_type
    assert query == gt_query


def test_generate_observation() -> None:
    """Tests ReActQAStrategy generate_observation."""
    action_type = "Search"
    query = "best kick boxer in the world controversies crimes"
    init_scratchpad = "\nThought: I need to search for the best kickboxer in the world who has been involved in controversies and crimes.\nAction: Search[best kick boxer in the world controversies crimes]"
    responses = []
    llm = MockLLM("gpt-3.5-turbo", responses=responses)

    strategy = ReActQAStrategy(llm=llm)
    strategy._scratchpad = init_scratchpad
    strategy._finished = False
    strategy.docstore.search = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )
    obs, external_tool_info = strategy.generate_observation(
        idx=1, action_type=action_type, query=query
    )
    assert isinstance(obs, str)
    assert strategy._finished == False
    assert strategy._scratchpad != init_scratchpad
    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert (
        external_tool_info["search_result"]
        == "Buakaw Banchamek has faced several controversies and legal issues."
    )

    # Test finish.
    action_type = "Finish"
    query = "The best kickboxer is Buakaw Banchamek."
    init_scratchpad = "\nThought: I need to provide the final answer.\nAction: Finish[The best kickboxer is Buakaw Banchamek.]"
    responses = []
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    strategy._scratchpad = init_scratchpad
    strategy._finished = False
    obs, external_tool_info = strategy.generate_observation(
        idx=2, action_type=action_type, query=query
    )
    assert isinstance(obs, str)
    assert obs == "The best kickboxer is Buakaw Banchamek."
    assert strategy._finished == True
    assert strategy._scratchpad != init_scratchpad
    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert external_tool_info == {"search_result": "", "lookup_result": ""}

    # Test search success.
    action_type = "Search"
    query = "best kick boxer in the world controversies crimes"
    init_scratchpad = "\nThought: I need to search for the best kickboxer in the world who has been involved in controversies and crimes.\nAction: Search[best kick boxer in the world controversies crimes]"
    responses = ["Buakaw Banchamek has faced several controversies and legal issues."]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    strategy._scratchpad = init_scratchpad
    strategy._finished = False
    strategy.docstore.search = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )
    obs, external_tool_info = strategy.generate_observation(
        idx=3, action_type=action_type, query=query
    )
    assert isinstance(obs, str)
    assert obs == "Buakaw Banchamek has faced several controversies and legal issues."
    assert strategy._finished == False
    assert strategy._scratchpad != init_scratchpad
    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert (
        external_tool_info["search_result"]
        == "Buakaw Banchamek has faced several controversies and legal issues."
    )

    # Test search failure.
    action_type = "Search"
    query = "best kick boxer in the world controversies crimes"
    init_scratchpad = "\nThought: I need to search for the best kickboxer in the world who has been involved in controversies and crimes.\nAction: Search[best kick boxer in the world controversies crimes]"
    responses = []
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    strategy._scratchpad = init_scratchpad
    strategy._finished = False
    strategy.docstore.search = lambda x: (_ for _ in ()).throw(
        Exception("Search failed")
    )
    obs, external_tool_info = strategy.generate_observation(
        idx=4, action_type=action_type, query=query
    )
    assert isinstance(obs, str)
    assert obs == "Could not find that page, please try again."
    assert strategy._finished == False
    assert strategy._scratchpad != init_scratchpad
    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert external_tool_info["search_result"] == ""

    # Test lookup success.
    action_type = "Lookup"
    query = "controversies"
    init_scratchpad = "\nThought: I need to lookup controversies related to the best kickboxer in the world.\nAction: Lookup[controversies]"
    responses = ["Buakaw Banchamek has faced several controversies and legal issues."]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    strategy._scratchpad = init_scratchpad
    strategy._finished = False
    strategy.docstore.lookup = (
        lambda x: "Several controversies and legal issues related to Buakaw Banchamek."
    )
    obs, external_tool_info = strategy.generate_observation(
        idx=5, action_type=action_type, query=query
    )
    assert isinstance(obs, str)
    assert obs == "Several controversies and legal issues related to Buakaw Banchamek."
    assert strategy._finished == False
    assert strategy._scratchpad != init_scratchpad
    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert external_tool_info["lookup_result"] != ""

    # Test lookup failure.
    action_type = "Lookup"
    query = "controversies"
    init_scratchpad = "\nThought: I need to lookup controversies related to the best kickboxer in the world.\nAction: Lookup[controversies]"
    responses = []
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    strategy._scratchpad = init_scratchpad
    strategy._finished = False
    strategy.docstore.lookup = lambda x: (_ for _ in ()).throw(
        ValueError("Lookup failed")
    )
    obs, external_tool_info = strategy.generate_observation(
        idx=6, action_type=action_type, query=query
    )
    assert isinstance(obs, str)
    assert (
        obs
        == "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
    )
    assert strategy._finished == False
    assert strategy._scratchpad != init_scratchpad
    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert external_tool_info["lookup_result"] == ""

    # Test invalid action.
    action_type = "Invalid"
    query = "invalid action"
    init_scratchpad = "\nThought: I need to perform an invalid action.\nAction: Invalid[invalid action]"
    responses = []
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    strategy._scratchpad = init_scratchpad
    strategy._finished = False
    obs, external_tool_info = strategy.generate_observation(
        idx=7, action_type=action_type, query=query
    )
    assert isinstance(obs, str)
    assert (
        obs
        == "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
    )
    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert external_tool_info["search_result"] == ""
    assert external_tool_info["lookup_result"] == ""


def test_create_output_dict() -> None:
    """Tests ReActQAStrategy create_output_dict."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActQAStrategy(llm=llm)
    thought = "This is a thought."
    action_type = "search"
    query = "query"
    obs = "observation"
    external_tool_info = {"search_result": "", "lookup_result": ""}

    expected_output = {
        "thought": thought,
        "action_type": action_type,
        "query": query,
        "observation": obs,
        "answer": "",
        "external_tool_info": {"search_result": "", "lookup_result": ""},
        "prompt_metrics": {"thought": None, "action": None},
    }

    assert (
        strategy.create_output_dict(
            thought, action_type, query, obs, external_tool_info
        )
        == expected_output
    )


def test_halting_condition() -> None:
    """Tests ReActQAStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActQAStrategy(llm=llm)
    idx = 0
    question = "What is the capital of France?"
    examples = ""
    prompt = "Answer the question."

    assert not strategy.halting_condition(idx, question, examples, prompt, {})


def test_reset() -> None:
    """Tests ReActQAStrategy reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActQAStrategy(llm=llm)
    strategy._scratchpad = "Some previous state"
    strategy._finished = True

    strategy.reset()

    assert strategy._scratchpad == ""
    assert not strategy._finished


def test_instantiate_strategies() -> None:
    """Test instantiate all QA strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    hotqa_strategy = ReActHotQAStrategy(llm=llm)
    triviaqa_strategy = ReActTriviaQAStrategy(llm=llm)
    ambignq_strategy = ReActAmbigNQStrategy(llm=llm)
    fever_strategy = ReActFEVERStrategy(llm=llm)

    assert isinstance(hotqa_strategy, ReActHotQAStrategy)
    assert isinstance(triviaqa_strategy, ReActTriviaQAStrategy)
    assert isinstance(ambignq_strategy, ReActAmbigNQStrategy)
    assert isinstance(fever_strategy, ReActFEVERStrategy)
