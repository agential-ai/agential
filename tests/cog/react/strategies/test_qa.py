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
)
from agential.llm.llm import BaseLLM, MockLLM
from agential.utils.docstore import DocstoreExplorer
from agential.utils.metrics import PromptMetrics


def test_init() -> None:
    """Test ReActQAStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActQAStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseLLM)
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 5000
    assert isinstance(strategy.docstore, DocstoreExplorer)
    assert isinstance(strategy.enc, Encoding)


def test_generate_action() -> None:
    """Tests ReActQAStrategy generate_action."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'
    gt_scratchpad = (
        "\nAction 0: Search[best kick boxer in the world controversies crimes]"
    )
    gt_action_type = "Search"
    gt_query = "best kick boxer in the world controversies crimes"
    responses = ["Search[best kick boxer in the world controversies crimes]"]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)

    scratchpad, action_type, query, action_metrics = strategy.generate_action(
        idx=0,
        scratchpad="",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )

    assert scratchpad == gt_scratchpad
    assert action_type == gt_action_type
    assert query == gt_query
    assert action_metrics == PromptMetrics(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_generate_observation() -> None:
    """Tests ReActQAStrategy generate_observation."""
    action_type = "Search"
    gt_answer = ""
    gt_scratchpad = "\nObservation 1: Buakaw Banchamek has faced several controversies and legal issues."
    gt_obs = "Buakaw Banchamek has faced several controversies and legal issues."
    query = "best kick boxer in the world controversies crimes"
    responses = []
    llm = MockLLM("gpt-3.5-turbo", responses=responses)

    strategy = ReActQAStrategy(llm=llm)

    strategy.docstore.search = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=1, scratchpad="", action_type=action_type, query=query
        )
    )
    assert isinstance(obs, str)
    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert (
        external_tool_info["search_result"]
        == "Buakaw Banchamek has faced several controversies and legal issues."
    )

    assert answer == gt_answer
    assert obs == gt_obs
    assert scratchpad == gt_scratchpad
    assert finished == False

    # Test finish.
    action_type = "Finish"
    gt_answer = "The best kickboxer is Buakaw Banchamek."
    gt_scratchpad = "\nObservation 2: The best kickboxer is Buakaw Banchamek."
    gt_obs = "The best kickboxer is Buakaw Banchamek."
    query = "The best kickboxer is Buakaw Banchamek."
    responses = []
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=2, scratchpad="", action_type=action_type, query=query
        )
    )
    assert isinstance(obs, str)
    assert obs == "The best kickboxer is Buakaw Banchamek."
    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert external_tool_info == {"search_result": "", "lookup_result": ""}

    assert answer == gt_answer
    assert obs == gt_obs
    assert scratchpad == gt_scratchpad
    assert finished == True

    # Test search success.
    action_type = "Search"
    gt_answer = ""
    gt_scratchpad = "\nObservation 3: Buakaw Banchamek has faced several controversies and legal issues."
    gt_obs = "Buakaw Banchamek has faced several controversies and legal issues."
    query = "best kick boxer in the world controversies crimes"
    responses = ["Buakaw Banchamek has faced several controversies and legal issues."]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    strategy.docstore.search = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=3, scratchpad="", action_type=action_type, query=query
        )
    )
    assert isinstance(obs, str)
    assert obs == "Buakaw Banchamek has faced several controversies and legal issues."
    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert (
        external_tool_info["search_result"]
        == "Buakaw Banchamek has faced several controversies and legal issues."
    )

    assert answer == gt_answer
    assert obs == gt_obs
    assert scratchpad == gt_scratchpad
    assert finished == False

    # Test search failure.
    action_type = "Search"
    gt_answer = ""
    gt_scratchpad = "\nObservation 4: Could not find that page, please try again."
    gt_obs = "Could not find that page, please try again."
    query = "best kick boxer in the world controversies crimes"
    responses = []
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    strategy.docstore.search = lambda x: (_ for _ in ()).throw(
        Exception("Search failed")
    )
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=4, scratchpad="", action_type=action_type, query=query
        )
    )
    assert isinstance(obs, str)
    assert obs == "Could not find that page, please try again."
    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert external_tool_info["search_result"] == ""

    assert answer == gt_answer
    assert obs == gt_obs
    assert scratchpad == gt_scratchpad
    assert finished == False

    # Test lookup success.
    action_type = "Lookup"
    gt_answer = ""
    gt_scratchpad = "\nObservation 5: Several controversies and legal issues related to Buakaw Banchamek."
    gt_obs = "Several controversies and legal issues related to Buakaw Banchamek."

    query = "controversies"
    responses = ["Buakaw Banchamek has faced several controversies and legal issues."]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    strategy.docstore.lookup = (
        lambda x: "Several controversies and legal issues related to Buakaw Banchamek."
    )
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=5, scratchpad="", action_type=action_type, query=query
        )
    )
    assert isinstance(obs, str)
    assert obs == "Several controversies and legal issues related to Buakaw Banchamek."
    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert external_tool_info["lookup_result"] != ""

    assert answer == gt_answer
    assert obs == gt_obs
    assert scratchpad == gt_scratchpad
    assert finished == False

    # Test lookup failure.
    action_type = "Lookup"
    gt_answer = ""
    gt_scratchpad = "\nObservation 6: The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
    gt_obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
    query = "controversies"
    responses = []
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    strategy.docstore.lookup = lambda x: (_ for _ in ()).throw(
        ValueError("Lookup failed")
    )
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=6, scratchpad="", action_type=action_type, query=query
        )
    )
    assert isinstance(obs, str)
    assert (
        obs
        == "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
    )

    assert "search_result" in external_tool_info
    assert "lookup_result" in external_tool_info
    assert external_tool_info["lookup_result"] == ""

    assert answer == gt_answer
    assert obs == gt_obs
    assert scratchpad == gt_scratchpad
    assert finished == False

    # Test invalid action.
    action_type = "Invalid"
    gt_answer = ""
    gt_scratchpad = "\nObservation 7: Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
    gt_obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
    query = "invalid action"
    responses = []
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActQAStrategy(llm=llm)

    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=7, scratchpad="", action_type=action_type, query=query
        )
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

    assert answer == gt_answer
    assert obs == gt_obs
    assert scratchpad == gt_scratchpad
    assert finished == False


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
