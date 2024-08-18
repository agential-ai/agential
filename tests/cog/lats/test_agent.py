"""Test LATS agent."""

import pytest

from agential.cog.constants import Benchmarks
from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.lats.agent import LATSAgent
from agential.cog.lats.node import Node
from agential.cog.lats.output import (
    LATSEvaluateMetrics,
    LATSGenerateMetrics,
    LATSOutput,
    LATSReActStepOutput,
    LATSSimulationMetrics,
    LATSSimulationOutput,
    LATSSimulationStepMetrics,
    LATSStepOutput,
)
from agential.cog.lats.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    LATS_INSTRUCTION_HOTPOTQA,
    LATS_REFLECT_INSTRUCTION_HOTPOTQA,
    LATS_VALUE_INSTRUCTION_HOTPOTQA,
)
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.cog.lats.strategies.code import (
    LATSHEvalStrategy,
    LATSMBPPStrategy,
)
from agential.cog.lats.strategies.math import (
    LATSGSM8KStrategy,
    LATSSVAMPStrategy,
    LATSTabMWPStrategy,
)
from agential.cog.lats.strategies.qa import (
    LATSAmbigNQStrategy,
    LATSFEVERStrategy,
    LATSHotQAStrategy,
    LATSTriviaQAStrategy,
)
from agential.llm.llm import BaseLLM, MockLLM
from agential.utils.general import PromptMetrics


def test_init() -> None:
    """Test initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    agent = LATSAgent(llm=llm, benchmark="hotpotqa")
    assert isinstance(agent, LATSAgent)
    assert isinstance(agent.llm, BaseLLM)
    assert isinstance(agent.strategy, LATSBaseStrategy)
    assert agent.benchmark == "hotpotqa"


def test_get_strategy() -> None:
    """Tests LATSAgent get_strategy method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    # QA benchmarks.
    assert isinstance(
        LATSAgent.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        LATSHotQAStrategy,
    )
    assert isinstance(
        LATSAgent.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        LATSTriviaQAStrategy,
    )
    assert isinstance(
        LATSAgent.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        LATSAmbigNQStrategy,
    )
    assert isinstance(
        LATSAgent.get_strategy(Benchmarks.FEVER, llm=llm),
        LATSFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        LATSAgent.get_strategy(Benchmarks.GSM8K, llm=llm),
        LATSGSM8KStrategy,
    )
    assert isinstance(
        LATSAgent.get_strategy(Benchmarks.SVAMP, llm=llm),
        LATSSVAMPStrategy,
    )
    assert isinstance(
        LATSAgent.get_strategy(Benchmarks.TABMWP, llm=llm),
        LATSTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        LATSAgent.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        LATSHEvalStrategy,
    )
    assert isinstance(
        LATSAgent.get_strategy(Benchmarks.MBPP, llm=llm),
        LATSMBPPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent LATS"
    ):
        LATSAgent.get_strategy("unknown", llm=llm)


def test_get_fewshots() -> None:
    """Tests LATSAgent get_fewshots method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = LATSAgent.get_fewshots(benchmark, fewshot_type="react")
    assert isinstance(result, dict)
    assert result == {
        "examples": HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "reflect_examples": HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    }

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for LATS."
    ):
        LATSAgent.get_fewshots("unknown", fewshot_type="react")

    # Test unsupported fewshot_type.
    with pytest.raises(
        ValueError, match="Benchmark 'hotpotqa' few-shot type not supported for LATS."
    ):
        LATSAgent.get_fewshots("hotpotqa", fewshot_type="pot")


def test_get_prompts() -> None:
    """Tests LATSAgent get_prompts method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = LATSAgent.get_prompts(benchmark)
    assert result == {
        "prompt": LATS_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_HOTPOTQA,
        "value_prompt": LATS_VALUE_INSTRUCTION_HOTPOTQA,
    }

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for LATS."
    ):
        LATSAgent.get_prompts("unknown")


def test_generate() -> None:
    """Test generate."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    key = "Gesellschaft mit beschränkter Haftung"

    gt_terminal_node_state = {
        "state": LATSReActStepOutput(
            thought="Since direct searches for VIVA Media AG and its new acronym after the name change in 2004 did not provide relevant information, I should consider looking for industry reports, press releases, or official announcements related to the company's rebranding to uncover the acronym.",
            action_type="Search",
            query="VIVA Media AG rebranding press release",
            observation="Badr Hari is the best kick boxer in the world.",
            answer="",
            external_tool_info={
                "search_result": "Badr Hari is the best kick boxer in the world.",
                "lookup_result": "",
            },
        ),
        "visits": 1,
        "value": -1.0,
        "depth": 5,
        "is_terminal": False,
        "reward": 0,
    }
    gt_additional_info = [
        LATSStepOutput(
            iteration=0,
            current_node={
                "state": LATSReActStepOutput(
                    thought="",
                    action_type="",
                    query="",
                    observation="",
                    answer="",
                    external_tool_info={},
                ),
                "visits": 0,
                "value": 0,
                "depth": 0,
                "is_terminal": False,
                "reward": 0,
            },
            children_nodes=[
                {
                    "state": LATSReActStepOutput(
                        thought="I need to search for VIVA Media AG and find out its new acronym after changing its name in 2004.",
                        action_type="Search",
                        query="VIVA Media AG",
                        observation="Badr Hari is the best kick boxer in the world.",
                        answer="",
                        external_tool_info={
                            "search_result": "Badr Hari is the best kick boxer in the world.",
                            "lookup_result": "",
                        },
                    ),
                    "visits": 0,
                    "value": 0.0,
                    "depth": 1,
                    "is_terminal": False,
                    "reward": 0,
                },
                {
                    "state": LATSReActStepOutput(
                        thought="I need to search for VIVA Media AG to find out what their new acronym stands for after changing their name in 2004.",
                        action_type="Search",
                        query="VIVA Media AG",
                        observation="Badr Hari is the best kick boxer in the world.",
                        answer="",
                        external_tool_info={
                            "search_result": "Badr Hari is the best kick boxer in the world.",
                            "lookup_result": "",
                        },
                    ),
                    "visits": 0,
                    "value": 0.0,
                    "depth": 1,
                    "is_terminal": False,
                    "reward": 0,
                },
            ],
            generate_metrics=LATSGenerateMetrics(
                thoughts_metrics=[
                    PromptMetrics(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    PromptMetrics(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ],
                actions_metrics=[
                    PromptMetrics(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    PromptMetrics(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ],
                reflections_metrics=[],
            ),
            values=[
                {"explanation": "Explanation not found", "value": 0.0},
                {"explanation": "Explanation not found", "value": 0.0},
            ],
            evaluate_metrics=LATSEvaluateMetrics(
                values_metrics=[
                    PromptMetrics(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    PromptMetrics(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ]
            ),
            simulation_results=LATSSimulationOutput(
                simulation_reward=-1.0,
                simulation_terminal_node={
                    "state": LATSReActStepOutput(
                        thought="Since direct searches for VIVA Media AG and its new acronym after the name change in 2004 did not provide relevant information, I should consider looking for industry reports, press releases, or official announcements related to the company's rebranding to uncover the acronym.",
                        action_type="Search",
                        query="VIVA Media AG rebranding press release",
                        observation="Badr Hari is the best kick boxer in the world.",
                        answer="",
                        external_tool_info={
                            "search_result": "Badr Hari is the best kick boxer in the world.",
                            "lookup_result": "",
                        },
                    ),
                    "visits": 0,
                    "value": 0,
                    "depth": 5,
                    "is_terminal": False,
                    "reward": 0,
                },
                simulation_current_nodes=[
                    {
                        "state": LATSReActStepOutput(
                            thought="I need to search for VIVA Media AG and find out its new acronym after changing its name in 2004.",
                            action_type="Search",
                            query="VIVA Media AG",
                            observation="Badr Hari is the best kick boxer in the world.",
                            answer="",
                            external_tool_info={
                                "search_result": "Badr Hari is the best kick boxer in the world.",
                                "lookup_result": "",
                            },
                        ),
                        "visits": 0,
                        "value": 0.0,
                        "depth": 1,
                        "is_terminal": False,
                        "reward": 0,
                    },
                    {
                        "state": LATSReActStepOutput(
                            thought="The search for VIVA Media AG did not return relevant results. I should try searching for the new name of the company directly.",
                            action_type="Search",
                            query="VIVA Media AG new name",
                            observation="Badr Hari is the best kick boxer in the world.",
                            answer="",
                            external_tool_info={
                                "search_result": "Badr Hari is the best kick boxer in the world.",
                                "lookup_result": "",
                            },
                        ),
                        "visits": 0,
                        "value": 0,
                        "depth": 2,
                        "is_terminal": False,
                        "reward": 0,
                    },
                    {
                        "state": LATSReActStepOutput(
                            thought="Since direct searches for VIVA Media AG and its new name did not yield results, I should try to search for the company's name change history or any related news articles to find out the acronym.",
                            action_type="Search",
                            query="VIVA Media AG name change history",
                            observation="Badr Hari is the best kick boxer in the world.",
                            answer="",
                            external_tool_info={
                                "search_result": "Badr Hari is the best kick boxer in the world.",
                                "lookup_result": "",
                            },
                        ),
                        "visits": 0,
                        "value": 0,
                        "depth": 3,
                        "is_terminal": False,
                        "reward": 0,
                    },
                    {
                        "state": LATSReActStepOutput(
                            thought="The search results are still not providing the information needed. I should try to find a different angle to approach this question.",
                            action_type="Search",
                            query="VIVA Media AG rebranding 2004",
                            observation="Badr Hari is the best kick boxer in the world.",
                            answer="",
                            external_tool_info={
                                "search_result": "Badr Hari is the best kick boxer in the world.",
                                "lookup_result": "",
                            },
                        ),
                        "visits": 0,
                        "value": 0,
                        "depth": 4,
                        "is_terminal": False,
                        "reward": 0,
                    },
                ],
                simulation_children_nodes=[
                    [
                        {
                            "state": LATSReActStepOutput(
                                thought="The search for VIVA Media AG did not return relevant results. I should try searching for the new name of the company directly.",
                                action_type="Search",
                                query="VIVA Media AG new name",
                                observation="Badr Hari is the best kick boxer in the world.",
                                answer="",
                                external_tool_info={
                                    "search_result": "Badr Hari is the best kick boxer in the world.",
                                    "lookup_result": "",
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 2,
                            "is_terminal": False,
                            "reward": 0,
                        },
                        {
                            "state": LATSReActStepOutput(
                                thought="I couldn't find VIVA Media AG. Let me try searching for VIVA Media AG (acronym) instead.",
                                action_type="Search",
                                query="VIVA Media AG (acronym)",
                                observation="Badr Hari is the best kick boxer in the world.",
                                answer="",
                                external_tool_info={
                                    "search_result": "Badr Hari is the best kick boxer in the world.",
                                    "lookup_result": "",
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 2,
                            "is_terminal": False,
                            "reward": 0,
                        },
                    ],
                    [
                        {
                            "state": LATSReActStepOutput(
                                thought="Since direct searches for VIVA Media AG and its new name did not yield results, I should try to search for the company's name change history or any related news articles to find out the acronym.",
                                action_type="Search",
                                query="VIVA Media AG name change history",
                                observation="Badr Hari is the best kick boxer in the world.",
                                answer="",
                                external_tool_info={
                                    "search_result": "Badr Hari is the best kick boxer in the world.",
                                    "lookup_result": "",
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 3,
                            "is_terminal": False,
                            "reward": 0,
                        },
                        {
                            "state": LATSReActStepOutput(
                                thought="It seems the direct search for the new name of VIVA Media AG is not yielding results. I should try a different approach to find the acronym.",
                                action_type="Search",
                                query="VIVA Media AG acronym 2004",
                                observation="Badr Hari is the best kick boxer in the world.",
                                answer="",
                                external_tool_info={
                                    "search_result": "Badr Hari is the best kick boxer in the world.",
                                    "lookup_result": "",
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 3,
                            "is_terminal": False,
                            "reward": 0,
                        },
                    ],
                    [
                        {
                            "state": LATSReActStepOutput(
                                thought="The search results are still not providing the information needed. I should try to find a different angle to approach this question.",
                                action_type="Search",
                                query="VIVA Media AG rebranding 2004",
                                observation="Badr Hari is the best kick boxer in the world.",
                                answer="",
                                external_tool_info={
                                    "search_result": "Badr Hari is the best kick boxer in the world.",
                                    "lookup_result": "",
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 4,
                            "is_terminal": False,
                            "reward": 0,
                        },
                        {
                            "state": LATSReActStepOutput(
                                thought="As the search results are not providing relevant information, I should consider looking up the company's history or press releases to find out the acronym of VIVA Media AG after the name change in 2004.",
                                action_type="Search",
                                query="VIVA Media AG press releases 2004",
                                observation="Badr Hari is the best kick boxer in the world.",
                                answer="",
                                external_tool_info={
                                    "search_result": "Badr Hari is the best kick boxer in the world.",
                                    "lookup_result": "",
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 4,
                            "is_terminal": False,
                            "reward": 0,
                        },
                    ],
                    [
                        {
                            "state": LATSReActStepOutput(
                                thought="Since direct searches for VIVA Media AG and its new acronym after the name change in 2004 did not provide relevant information, I should consider looking for industry reports, press releases, or official announcements related to the company's rebranding to uncover the acronym.",
                                action_type="Search",
                                query="VIVA Media AG rebranding press release",
                                observation="Badr Hari is the best kick boxer in the world.",
                                answer="",
                                external_tool_info={
                                    "search_result": "Badr Hari is the best kick boxer in the world.",
                                    "lookup_result": "",
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 5,
                            "is_terminal": False,
                            "reward": 0,
                        },
                        {
                            "state": LATSReActStepOutput(
                                thought="Since the search results are not yielding the required information, I should try a more general search for VIVA Media AG's name change history or company information to find the acronym.",
                                action_type="Search",
                                query="VIVA Media AG company information",
                                observation="Badr Hari is the best kick boxer in the world.",
                                answer="",
                                external_tool_info={
                                    "search_result": "Badr Hari is the best kick boxer in the world.",
                                    "lookup_result": "",
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 5,
                            "is_terminal": False,
                            "reward": 0,
                        },
                    ],
                ],
                simulation_values=[
                    [
                        {"explanation": "Explanation not found", "value": 0.0},
                        {"explanation": "Explanation not found", "value": 0.0},
                    ],
                    [
                        {"explanation": "Explanation not found", "value": 0.0},
                        {"explanation": "Explanation not found", "value": 0.0},
                    ],
                    [
                        {"explanation": "Explanation not found", "value": 0.0},
                        {"explanation": "Explanation not found", "value": 0.0},
                    ],
                    [
                        {"explanation": "Explanation not found", "value": 0.0},
                        {"explanation": "Explanation not found", "value": 0.0},
                    ],
                ],
            ),
            simulation_metrics=LATSSimulationMetrics(
                simulation_step_metrics=[
                    LATSSimulationStepMetrics(
                        generate_metrics=LATSGenerateMetrics(
                            thoughts_metrics=[
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            actions_metrics=[
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            reflections_metrics=[],
                        ),
                        evaluate_metrics=LATSEvaluateMetrics(
                            values_metrics=[
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ]
                        ),
                    ),
                    LATSSimulationStepMetrics(
                        generate_metrics=LATSGenerateMetrics(
                            thoughts_metrics=[
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            actions_metrics=[
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            reflections_metrics=[],
                        ),
                        evaluate_metrics=LATSEvaluateMetrics(
                            values_metrics=[
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ]
                        ),
                    ),
                    LATSSimulationStepMetrics(
                        generate_metrics=LATSGenerateMetrics(
                            thoughts_metrics=[
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            actions_metrics=[
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            reflections_metrics=[],
                        ),
                        evaluate_metrics=LATSEvaluateMetrics(
                            values_metrics=[
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ]
                        ),
                    ),
                    LATSSimulationStepMetrics(
                        generate_metrics=LATSGenerateMetrics(
                            thoughts_metrics=[
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            actions_metrics=[
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            reflections_metrics=[],
                        ),
                        evaluate_metrics=LATSEvaluateMetrics(
                            values_metrics=[
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                PromptMetrics(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ]
                        ),
                    ),
                ]
            ),
        )
    ]

    responses = [
        "I need to search for VIVA Media AG and find out its new acronym after changing its name in 2004.\nAction 1: Search[VIVA Media AG]\nObservation 1: VIVA Media AG was a German media company that operated several television channels.\nThought 2: I need to find the new acronym for VIVA Media AG after its name change in 2004.\nAction 2: Lookup[new name of VIVA Media AG in 2004]\nObservation 2: (Result 1 / 1) The new acronym for VIVA Media AG after changing its name in 2004 was VIVA Entertainment AG.\nThought 3: The new acronym for VIVA Media AG after its name change in 2004 was VIVA Entertainment AG. \nAction 3: Finish[VIVA Entertainment AG]",
        "Search[VIVA Media AG]\nObservation 1: VIVA Media AG was a German company in the media industry that was rebranded in 2004.\nThought 2: Since VIVA Media AG was rebranded in 2004, I need to find out what their new acronym stands for.\nAction 2: Lookup[new acronym of VIVA Media AG]\nObservation 2: (Result 1 / 1) Their new acronym stands for VSG Entertainment.\nThought 3: VIVA Media AG changed its name to VSG Entertainment in 2004, so the new acronym stands for VSG Entertainment.\nAction 3: Finish[VSG Entertainment]",
        "I need to search for VIVA Media AG to find out what their new acronym stands for after changing their name in 2004.\nAction 1: Search[VIVA Media AG]\nObservation 1: VIVA Media AG was a German music television network.\nThought 2: Since the search did not provide the information I need, I should look for the new acronym after their name change in 2004.\nAction 2: Lookup[new acronym for VIVA Media AG]\nObservation 2: (Result 1 / 1) The new acronym for VIVA Media AG after changing its name in 2004 is VIVA.\nThought 3: The new acronym for VIVA Media AG after changing its name in 2004 is VIVA.\nAction 3: Finish[VIVA]",
        "Search[VIVA Media AG]\nObservation 1: VIVA Media AG was a German media company and television broadcaster founded on December 1, 1993. It was known for its music television channels.\nThought 2: The search did not provide information about the acronym after the name change. I need to look up the new acronym of VIVA Media AG after 2004.\nAction 2: Lookup[new acronym of VIVA Media AG]\nObservation 2: (Result 1/1) The new name of VIVA Media AG after 2004 was VIVA Media Group.\nThought 3: The new acronym for VIVA Media AG after 2004 is VIVA Media Group. I need to find out what the acronym stands for.\nAction 3: Search[VIVA Media Group acronym]\nObservation 3: (No relevant information found)\nThought 4: Since I couldn't find information on the acronym for VIVA Media Group, I can try searching for the meaning of VIVA in a business context.\nAction 4: Search[VIVA acronym business meaning]\nObservation 4: (No relevant information found)\nThought 5: Without concrete information on the acronym's meaning, I will have to conclude that the meaning of the new acronym for VIVA Media AG after 2004 is unknown.\nAction 5: Finish[unknown]",
        "I need to search for VIVA Media AG to find out what their new acronym stands for after changing their name in 2004.\nAction 1: Search[VIVA Media AG]\nObservation 1: VIVA Media AG was a German media company that operated several television channels.\nThought 2: Since the search did not provide the information I need, I should look for the new acronym after their name change in 2004.\nAction 2: Lookup[new acronym'The trajectory is incorrect because the search query did not yield results for VIVA Media AG. This indicates that the initial search was not specific enough or possibly the entity has limited online presence. Future attempts should consider refining the search terms or looking for alternative sources of information.\nCorrectness score: 2",
        "This trajectory is incorrect because the search did not yield results for VIVA Media AG. The action taken was appropriate, but the lack of relevant information hindered progress towards finding the acronym. In the future, it would be beneficial to explore alternative sources or search for related entities that might provide the necessary information.\nCorrectness score: 2",
        "The search for VIVA Media AG did not return relevant results. I should try searching for the new name of the company directly.\nAction 2: Search[new name of VIVA Media AG]\nObservation 2: Could not find [new name of VIVA Media AG]. Similar: ['Media AG', 'New Media Investment Group', 'Media (communication)', 'Media (region)', 'Media (2017 film)', 'Media (Bulgaria)', 'Media (2012 film)', 'Media (artist)', 'Media (communication)', 'Media (region)']\nThought 3: Since I couldn't find the new acronym directly, I should try searching for the history of VIVA Media AG to see if the new name and acronym are mentioned there.\nAction 3: Search[history of VIVA Media AG]\nObservation 3: The history of VIVA Media AG involves its establishment in 1993 as a German music television network and its subsequent growth into a multimedia company. A significant event in 2004 was a change in name, but the new acronym is not explicitly mentioned.\nThought 4: I will need to look up the specific details of the name change in 2004 to find out the new acronym.\nAction 4: Search[VIVA Media AG name change 2004]\nObservation 4: The name change in 2004 transformed VIVA Media AG into VIVA Media GmbH. The new acronym \"GmbH\" stands for \"Gesellschaft mit beschränkter Haftung\", which is a type of German legal entity similar to a limited liability company.\nThought 5: The new acronym for VIVA Media AG after changing its name in 2004 is \"GmbH\", which stands for \"Gesellschaft mit beschränkter Haftung\".\nAction 5: Finish[Gesellschaft mit beschränkter Haftung]",
        "Search[VIVA Media AG new name]\nObservation 2: Could not find [VIVA Media AG new name]. Similar: ['List of company name etymologies', 'New York City Fire Department', 'New York City Police Department', 'New York City Subway', 'New York City', 'New York City mayoral elections', 'New York City Transit Authority', 'New York City Marathon', 'New York City Department of Education', 'New York City Department of Transportation']\nThought 3: Since I couldn't find the new acronym directly, I should try searching for the company's name change in 2004 to get more information.\nAction 3: Search[VIVA Media AG name change 2004]\nObservation 3: Could not find [VIVA Media AG name change 2004]. Similar: ['List of company name etymologies', 'VIVA (German TV channel)', 'VIVA (Polish TV channel)', 'Media AG (disambiguation)', 'Media AG', 'Taurus Media', 'ZDF Enterprises', 'Viva Entertainment', 'Viva Films', 'Viva Films (Philippines)']\nThought 4: It seems challenging to directly find the new acronym for VIVA Media AG after its name change in 2004. I should try searching for the company's history to see if the acronym is mentioned.\nAction 4: Search[VIVA Media AG history]\nObservation 4: VIVA Media AG was a German music television network and production company, headquartered in Cologne.\nThought 5: The search for history did not provide the acronym. I should attempt a broader search to find any mentions of the acronym.\nAction 5: Search[VIVA Media AG acronym]\nObservation 5: Could not find [VIVA Media AG acronym]. Similar: ['List of company name etymologies', 'VIVA (German TV channel)', 'VIVA (Polish TV channel)', 'Media AG (disambiguation)', 'Media AG', 'Taurus Media', 'ZDF Enterprises', 'Viva Entertainment', 'Viva Films', 'Viva Films (Philippines)']\nThought 6: Due to the lack of direct information, I may need to search for any articles or press releases related to the name change of VIVA Media AG in 2004 to find out the new acronym.\nAction 6: Search[VIVA Media AG name change 2004 articles]\nObservation 6: Found relevant articles discussing the name change of VIVA Media AG in 2004.\nThought 7: By reading the articles, I should be able to find the new acronym for VIVA Media AG after the name change in 2004.\nAction 7: Finish[new acronym not found]",
        "I couldn't find VIVA Media AG. Let me try searching for VIVA Media AG (acronym) instead.\nAction 2: Search[VIVA Media AG (acronym)]",
        "Search[VIVA Media AG (acronym)]",
        "The trajectory is incorrect because the initial search did not return relevant results, and subsequent searches did not directly address the specific information needed to answer the question. Future attempts should focus on searching for the new acronym directly related to the name change of VIVA Media AG in 2004.\nCorrectness score: 2",
        "The trajectory is incorrect because the search for VIVA Media AG did not yield relevant results, and the subsequent search for the acronym did not directly address the name change in 2004. The trajectory should have focused on finding information specifically related to the name change and the new acronym. \nCorrectness score: 2",
        "Since direct searches for VIVA Media AG and its new name did not yield results, I should try to search for the company's name change history or any related news articles to find out the acronym. \nAction 3: Search[VIVA Media AG name change history]\nObservation 3: (No relevant results found)\nThought 4: Since I am unable to find the information through direct searches, I should broaden my search to include any articles or sources discussing the name change of VIVA Media AG in 2004.\nAction 4: Search[VIVA Media AG 2004 name change]\nObservation 4: (Relevant information found) VIVA Media AG changed its name to Star Division in 2004.\nThought 5: The new name for VIVA Media AG after the name change in 2004 is Star Division. I need to find out what the acronym \"STAR\" stands for in Star Division.\nAction 5: Search[Star Division acronym]\nObservation 5: Could not find [Star Division acronym]. Similar: ['Star Tribune', 'Star Alliance', 'Star Plus', 'Star Fox', 'Star Trek', 'Star Wars', 'Star Trek: Discovery', 'Star Trek: The Original Series', 'Star Wars: The Clone Wars (2008 TV series)', 'Star Wars: The Force Awakens']\nThought 6: Since searching for the acronym directly did not yield results, I should try to search for the meaning of the acronym within the context of Star Division to find out what it stands for.\nAction 6: Search[Star Division meaning of STAR]\nObservation 6: (Relevant information found) The acronym \"STAR\" stands for \"Software Technology for Application and Research\" in the context of Star Division.\nThought 7: The acronym \"STAR\" in Star Division stands for \"Software Technology for Application and Research.\" \nAction 7: Finish[Software Technology for Application and Research]",
        "Search[VIVA Media AG name change history]\nObservation 3: (Result 1 / 1) Lookup[VIVA Media AG name change 2004 acronym]\nThought 4: By searching for the name change history, I can find the acronym for VIVA Media AG after its name change in 2004.\nAction 4: Finish[no results found]",
        "It seems the direct search for the new name of VIVA Media AG is not yielding results. I should try a different approach to find the acronym.\nAction 3: Search[new name of VIVA Media AG 2004]\nObservation 3: Unfortunately, the search results are still not providing the information needed. I may need to look for historical sources or company records to determine the acronym.\nThought 4: To find the acronym of VIVA Media AG after changing its name in 2004, I should look for historical records or official announcements from the company.\nAction 4: Finish[unable to find the answer]",
        "Search[VIVA Media AG acronym 2004]\nObservation 3: Could not find [VIVA Media AG acronym 2004]. Similar: ['VIVA Media AG', 'VIVA (German TV channel)', 'MTV Music (Polish TV channel)', 'VIVA Plus', 'Viacom (1952–2005)', 'Vauxhall Viva', 'Sartorius AG', 'GfK Entertainment charts', 'Spellbound Entertainment', 'Ag-gag']\nThought 4: Since direct searches for the new name and acronym of VIVA Media AG are not yielding results, I should try searching for VIVA Media AG's history in 2004 to find the information.\nAction 4: Search[VIVA Media AG history 2004]\nObservation 4: VIVA Media AG was a German media company that operated music channels and other media services in Europe. It was acquired by Viacom in 2005.\nThought 5: The search for VIVA Media AG's history in 2004 did not provide the acronym. I might need to broaden the search to include information about the name change.\nAction 5: Search[VIVA Media AG name change 2004]\nObservation 5: VIVA Media AG changed its name to VIVA Entertainment in 2004.\nThought 6: The new acronym for VIVA Media AG after changing its name in 2004 is VIVA Entertainment. \nAction 6: Finish[VIVA Entertainment]",
        "The trajectory is incorrect because the focus shifted to unrelated entities and topics instead of directly verifying the new acronym of VIVA Media AG after its name change in 2004. Future attempts should ensure to verify specific details related to the company's name change and the resulting acronym directly before exploring other avenues.\nCorrectness score: 2",
        "The trajectory is incorrect because the search queries did not directly address the new acronym of VIVA Media AG after changing its name in 2004. The searches were too broad and did not specifically target the information needed to answer the question. In the future, it would be more effective to focus on finding the direct information related to the new acronym of the company.\nCorrectness score: 2",
        "The search results are still not providing the information needed. I should try to find a different angle to approach this question.\nAction 4: Search[VIVA Media AG rebranding 2004]\nObservation 4: Rebranding refers to a marketing agent.strategy in which a new name, term, symbol, design, or combination thereof is created for an established brand with the intention of developing a new, differentiated identity in the minds of consumers, investors, competitors, and other stakeholders.\nThought 5: The search for rebranding information did not specify the new acronym. I should consider looking up industry news or press releases related to VIVA Media AG's name change in 2004.\nAction 5: Search[VIVA Media AG new acronym 2004]\nObservation 5: Could not find [VIVA Media AG new acronym 2004]. Similar: ['VIVA Media AG', 'MTV Music (Polish TV channel)', 'VIVA Plus', 'Paramount International Networks', 'VIVA (German TV channel)', 'Viacom (1952–2005)', 'Vauxhall Viva', 'Sartorius AG', 'GfK Entertainment charts', 'Spellbound Entertainment']\nThought 6: The search results are still not providing the specific information needed. I should look for any official announcements or company reports regarding the name change and new acronym.\nAction 6: Search[VIVA Media AG official announcement name change 2004]\nObservation 6: (Result 1 / 1) Could not find [VIVA Media AG official announcement name change 2004]. Similar: ['VIVA Media AG', 'MTV Music (Polish TV channel)', 'VIVA Plus', 'Paramount International Networks', 'VIVA (German TV channel)', 'Viacom (1952–2005)', 'Vauxhall Viva', 'Sartorius AG', 'GfK Entertainment charts', 'Spellbound Entertainment']\nThought 7: Despite various search attempts, the specific information about VIVA Media AG's new acronym after the 2004 name change remains elusive. It may require further investigation through industry databases or contacting the company directly.\nAction 7: Finish[unable to determine new acronym]",
        "Search[VIVA Media AG rebranding 2004]",
        "As the search results are not providing relevant information, I should consider looking up the company's history or press releases to find out the acronym of VIVA Media AG after the name change in 2004.\nAction 4: Search[VIVA Media AG press releases]\nObservation 4: Could not find [VIVA Media AG press releases]. Similar: ['Vauxhall Viva', 'GfK Entertainment charts', 'Lindt', 'Channel 21 (German TV network)', 'Puma (brand)', 'Sigg', 'Opel Astra', 'Sildenafil', 'About You Now', 'Schneider Electric']\nThought 5: Since direct searches for the company and its name change have not been successful, I may need to expand the search to include industry news sources or company announcements to uncover the new acronym.\nAction 5: Search[VIVA Media AG name change news]\nObservation 5: (Result 1 / 1) Lookup[VIVA Media AG name change]\nObservation 6: VIVA Media AG changed its name to VIVA Entertainment in 2004.\nThought 6: The new acronym for VIVA Entertainment after the name change is \"VE.\"\nAction 6: Finish[VE]",
        "Search[VIVA Media AG press releases 2004]",
        "This trajectory is incorrect because the focus shifted to general searches about VIVA Media AG and its name change history instead of directly searching for the new acronym after the name change in 2004. The trajectory should have focused on finding the specific acronym related to the company's name change in 2004.\nCorrectness score: 3",
        "The trajectory is incorrect because the focus shifted towards searching for the company's history and press releases instead of directly searching for the new acronym after the name change in 2004. Future attempts should focus on finding direct information related to the specific details mentioned in the question.\nCorrectness score: 3",
        'Since direct searches for VIVA Media AG and its new acronym after the name change in 2004 did not provide relevant information, I should consider looking for industry reports, press releases, or official announcements related to the company\'s rebranding to uncover the acronym.\nAction 5: Search[VIVA Media AG rebranding news]\nObservation 5: Found relevant information that VIVA Media AG changed its name to "VIAVAC AG" in 2004, but the acronym expansion is not readily available.\nThought 6: To find the expansion of the new acronym "VIAVAC AG," I should search for official announcements or press releases from the company regarding the name change.\nAction 6: Search[VIAVAC AG acronym expansion]\nObservation 6: Found an official press release stating that the new acronym "VIAVAC" stands for "Video Audio Vision And Communication."\nThought 7: The new acronym "VIAVAC" stands for Video Audio Vision And Communication.\nAction 7: Finish[Video Audio Vision And Communication]',
        "Search[VIVA Media AG rebranding press release]\nObservation 5: (Result 1 / 1) The company announced its rebranding in 2004 to VIVA Entertainment Group, Inc.\nThought 6: The new acronym for VIVA Media AG after changing its name in 2004 is VIVA Entertainment Group, Inc.\nAction 6: Finish[VIVA Entertainment Group, Inc]",
        "Since the search results are not yielding the required information, I should try a more general search for VIVA Media AG's name change history or company information to find the acronym.\nAction 5: Search[VIVA Media AG company information]\nObservation 5: VIVA Media AG was a German media company that operated several television channels.\nThought 6: The search results are still not providing the specific information about the acronym. I should try to look for press releases or official announcements regarding the name change.\nAction 6: Search[VIVA Media AG official name change announcement 2004]\nObservation 6: Could not find [VIVA Media AG official name change announcement 2004]. Similar: ['Paramount International Networks', 'Veolia Transport', 'Voice of America', 'Universal Music Group', 'Virgin Interactive Entertainment', '2024 in Philippine television', 'Viacom (1952–2005)', 'PolyGram', 'British Midland International', 'Renault']\nThought 7: The search results are still not helping in finding the acronym. I should consider looking for industry news articles or contacting the company directly to obtain the information.\nAction 7: Finish[unable to determine without further information]",
        "Search[VIVA Media AG company information]\nObservation 5: VIVA Media AG was a German media company that operated television channels and other media outlets. The company underwent a name change in 2004 but the new acronym is not directly mentioned in the search results.\nThought 6: The search results have provided background information about VIVA Media AG but not the specific acronym after the name change in 2004. I should try to find a source that directly mentions the new acronym.\nAction 6: Search[VIVA Media AG new acronym 2004]\nObservation 6: Could not find [VIVA Media AG new acronym 2004]. Similar: ['Paramount International Networks', 'Voice of America', 'Universal Music Group', 'British Midland International', 'Renault', 'Virgin Interactive Entertainment', '2024 in Philippine television', 'PolyGram', 'Veolia Transport', 'Eros International']\nThought 7: The direct search for the new acronym after VIVA Media AG's name change in 2004 is not yielding any results. I should try to search for specific news articles or company announcements from that time period to find the acronym.\nAction 7: Search[VIVA Media AG name change 2004 acronym]\nObservation 7: The search results are still not providing the required information about the new acronym for VIVA Media AG after the name change in 2004.\nThought 8: Since the direct searches are not successful, I should try to broaden the search terms to include variations of the company name change and acronym to find the information.\nAction 8: Search[VIVA Media AG 2004 rebranding acronym]\nObservation 8: Could not find [VIVA Media AG 2004 rebranding acronym]. Similar: ['Universal Music Group', 'Renault', 'PolyGram', 'Paramount International Networks', 'Virgin Interactive Entertainment', 'Veolia Transport', 'British Midland International', '2024 in Philippine television', 'Voice of America', 'Eros International']\nThought 9: The broader search terms are still not yielding the specific information needed. I should try to search for industry reports or company archives that may mention the new acronym for VIVA Media AG after the name change in 2004.\nAction 9: Search[VIVA Media AG 2004 name change acronym industry reports]\nObservation 9: The search results are not providing the required information about the new acronym for VIVA Media AG after the name change in 2004.\nThought 10: Despite various search attempts, the specific acronym for VIVA Media AG after changing its name in 2004 remains elusive. Additional research or access to company records may be necessary to find the exact acronym.\nAction 10: Finish[unable to find the new acronym]",
        "The trajectory is incorrect because the search queries did not directly target the specific information needed to answer the question. Instead of searching for the new acronym directly, the user attempted various related searches that did not yield the required information. Future attempts should focus on refining search queries to directly address the specific details required to answer the question.\nCorrectness score: 3",
        "This trajectory is incorrect because the focus shifted towards general searches and unrelated information instead of directly attempting to find the specific acronym for VIVA Media AG after its name change in 2004. Future attempts should ensure to focus on the specific details related to the question and avoid getting sidetracked by unrelated search results.\nCorrectness score: 3",
    ]

    agent = LATSAgent(
        MockLLM("gpt-3.5-turbo", responses=responses),
        benchmark="hotpotqa",
        n_samples=2,
        depth_limit=5,
        testing=True,
    )
    agent.strategy.docstore.search = (
        lambda x: "Badr Hari is the best kick boxer in the world."
    )

    out = agent.generate(
        question=question,
        key=key,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        value_examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
        prompt=LATS_INSTRUCTION_HOTPOTQA,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_HOTPOTQA,
        value_prompt=LATS_VALUE_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        reflect_additional_keys={},
        value_additional_keys={},
        max_iterations=1,
        reset=True,
    )

    assert len(agent.strategy.failed_trajectories) == 0
    assert len(agent.strategy.reflection_map) == 0
    assert agent.strategy.value_cache == {
        "\nThought 1: I need to search for VIVA Media AG and find out its new acronym after changing its name in 2004.\nAction 1: Search[VIVA Media AG]\nObservation 1: Badr Hari is the best kick boxer in the world.::": "I need to search for VIVA Media AG to find out what their new acronym stands for after changing their name in 2004.\nAction 1: Search[VIVA Media AG]\nObservation 1: VIVA Media AG was a German media company that operated several television channels.\nThought 2: Since the search did not provide the information I need, I should look for the new acronym after their name change in 2004.\nAction 2: Lookup[new acronym'The trajectory is incorrect because the search query did not yield results for VIVA Media AG. This indicates that the initial search was not specific enough or possibly the entity has limited online presence. Future attempts should consider refining the search terms or looking for alternative sources of information.\nCorrectness score: 2",
        "\nThought 1: I need to search for VIVA Media AG to find out what their new acronym stands for after changing their name in 2004.\nAction 1: Search[VIVA Media AG]\nObservation 1: Badr Hari is the best kick boxer in the world.::": "This trajectory is incorrect because the search did not yield results for VIVA Media AG. The action taken was appropriate, but the lack of relevant information hindered progress towards finding the acronym. In the future, it would be beneficial to explore alternative sources or search for related entities that might provide the necessary information.\nCorrectness score: 2",
    }

    assert out.answer.to_dict() == gt_terminal_node_state
    assert out.total_completion_cost == 0.0012
    assert out.total_completion_tokens == 600
    assert out.total_prompt_cost == 0.00045000000000000015
    assert out.total_prompt_tokens == 300
    assert out.total_tokens == 900
    assert out.total_cost == 0.0016500000000000002
    assert out.total_prompt_time == 15.0
    assert out.total_time == 0.5
    assert out.additional_info == gt_additional_info
    assert agent.strategy.root.to_dict() == {
        "state": LATSReActStepOutput(
            thought="",
            action_type="",
            query="",
            observation="",
            answer="",
            external_tool_info={},
        ),
        "visits": 1,
        "value": -1.0,
        "depth": 0,
        "is_terminal": False,
        "reward": 0,
    }
