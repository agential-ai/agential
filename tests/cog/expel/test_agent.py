"""Unit tests for ExpeL."""

import joblib

from agential.cog.expel.agent import ExpeLAgent
from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.cog.expel.output import ExpeLOutput
from agential.cog.expel.prompts import (
    EXPEL_REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.reflexion.agent import (
    ReflexionReActAgent,
    ReflexionReActOutput,
    ReflexionReActStepOutput,
)
from agential.cog.reflexion.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.llm.llm import BaseLLM, MockLLM


def test_init(expel_experiences_10_fake_path: str) -> None:
    """Test initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    agent = ExpeLAgent(llm=llm, benchmark="hotpotqa")
    assert isinstance(agent.llm, BaseLLM)
    assert isinstance(agent.strategy.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(agent.strategy.experience_memory, ExpeLExperienceMemory)
    assert isinstance(agent.strategy.insight_memory, ExpeLInsightMemory)
    assert agent.strategy.success_batch_size == 8
    assert agent.strategy.experience_memory.experiences == []
    assert not agent.strategy.experience_memory.success_traj_docs
    assert not agent.strategy.experience_memory.vectorstore
    assert not agent.strategy.insight_memory.insights

    # Test with all parameters specified except experience memory and reflexion_react_agent.
    agent = ExpeLAgent(
        llm=llm,
        benchmark="hotpotqa",
        reflexion_react_strategy_kwargs={"max_steps": 3},
        insight_memory=ExpeLInsightMemory(
            insights=[{"insight": "blah blah", "score": 10}]
        ),
        success_batch_size=10,
    )
    assert isinstance(agent.llm, BaseLLM)
    assert isinstance(agent.strategy.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(agent.strategy.experience_memory, ExpeLExperienceMemory)
    assert isinstance(agent.strategy.insight_memory, ExpeLInsightMemory)
    assert agent.strategy.success_batch_size == 10
    assert agent.strategy.experience_memory.experiences == []
    assert not agent.strategy.experience_memory.success_traj_docs
    assert not agent.strategy.experience_memory.vectorstore
    assert agent.strategy.insight_memory.insights == [
        {"insight": "blah blah", "score": 10}
    ]

    # Test with custom reflexion_react_agent (verify it overrides reflexion_react_kwargs)
    agent = ExpeLAgent(
        llm=llm,
        benchmark="hotpotqa",
        reflexion_react_strategy_kwargs={"max_steps": 100},
        reflexion_react_agent=ReflexionReActAgent(llm=llm, benchmark="hotpotqa"),
    )
    assert isinstance(agent.strategy.reflexion_react_agent, ReflexionReActAgent)
    assert agent.strategy.reflexion_react_agent.benchmark == "hotpotqa"

    # Test with custom experience memory (verify correct initialization).
    experiences = joblib.load(expel_experiences_10_fake_path)
    experiences = experiences[:1]

    agent = ExpeLAgent(
        llm=llm,
        benchmark="hotpotqa",
        experience_memory=ExpeLExperienceMemory(experiences),
    )
    assert agent.strategy.experience_memory.experiences == experiences
    assert agent.strategy.insight_memory.insights == []


def test_reset() -> None:
    """Test reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    agent = ExpeLAgent(llm=llm, benchmark="hotpotqa")
    agent.strategy.reflexion_react_agent.strategy._scratchpad == "cat"
    agent.strategy.experience_memory.experiences == "dog"
    agent.strategy.insight_memory.insights = ["turtle"]
    agent.reset()
    assert agent.strategy.reflexion_react_agent.strategy._scratchpad == ""
    assert agent.strategy.experience_memory.experiences == []
    assert agent.strategy.insight_memory.insights == []


def test_generate(expel_experiences_10_fake_path: str) -> None:
    """Test generate."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    question = "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?"
    key = "Oneida Limited"

    gt_out = ExpeLOutput(
        question="What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?",
        key="Oneida Limited",
        trajectory=[
            ReflexionReActOutput(
                react_output=[
                    ReflexionReActStepOutput(
                        thought="I need to find out the name of the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes for many years. I should search for the company's name and its history to gather more information.",
                        action_type="Search",
                        query="giant silverware company religious Utopian group Pierrepont Noyes",
                        observation="Search result",
                        answer="",
                        external_tool_info={
                            "search_result": "Search result",
                            "lookup_result": "",
                        },
                        is_correct=False,
                    ),
                    ReflexionReActStepOutput(
                        thought="I need to search for more information about Oneida Limited to see if it matches the description of a giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes. I should look for details about the company's history and any relevant connections to Noyes.",
                        action_type="Search",
                        query="Oneida Limited giant silverware company religious Utopian group Pierrepont Noyes",
                        observation="Search result",
                        answer="",
                        external_tool_info={
                            "search_result": "Search result",
                            "lookup_result": "",
                        },
                        is_correct=False,
                    ),
                    ReflexionReActStepOutput(
                        thought='It seems like the search results are not providing specific information about Oneida Limited as a giant silverware company. I should try to look up the keyword "silverware" in the last passage successfully found and see if it provides any relevant details about Oneida Limited\'s products.',
                        action_type="Lookup",
                        query="silverware",
                        observation="Lookup result",
                        answer="",
                        external_tool_info={
                            "search_result": "",
                            "lookup_result": "Lookup result",
                        },
                        is_correct=False,
                    ),
                    ReflexionReActStepOutput(
                        thought="The passage mentions that Oneida Limited specialized in silverware and stainless steel cutlery under Pierrepont Noyes' leadership. This aligns with the description of a giant silverware company. I can confidently conclude that Oneida Limited is the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes for many years. I need to finish with this answer.",
                        action_type="Finish",
                        query="Oneida Limited",
                        observation="Answer is CORRECT",
                        answer="Oneida Limited",
                        external_tool_info={
                            "search_result": "",
                            "lookup_result": "",
                        },
                        is_correct=True,
                    ),
                ],
                reflections=[],
            )
        ],
        reflections=[],
    )

    gt_insights = [
        {
            "insight": "When searching for information, if the initial search query does not return relevant results, try using different keywords or search terms to refine the search.",
            "score": 2,
        },
        {
            "insight": "When searching for information, consider searching for the creator or key figures related to the topic to gather more relevant details.",
            "score": 2,
        },
        {
            "insight": "When searching for information, consider looking up keywords or terms mentioned in the search results to gather more specific details.",
            "score": 2,
        },
    ]

    responses = [
        "ADD 9: When searching for information, if the initial search query does not return relevant results, try using different keywords or search terms to refine the search.",
        "ADD 1: When searching for information, consider searching for the creator or key figures related to the topic to gather more relevant details.",
        "I need to find out the name of the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes for many years. I should search for the company's name and its history to gather more information. \nAction: Search[giant silverware company religious Utopian Pierrepont Noyes]\nObservation: The search did not return relevant results. I should try refining my search terms to find the information I need.\nAction: Search[silverware company Pierrepont Noyes history]\nObservation: The search results mention Oneida Limited as the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes for many years. \nAction: Lookup[Oneida Limited]\nObservation: Oneida Limited was a company that originated in the 19th century as a religious Utopian community, the Oneida Community, founded by John Humphrey Noyes in Oneida, New York. The company initially produced animal traps and then shifted to the production of silverware. It became known for its high-quality silverware and cutlery products. \nAction: Finish[Oneida Limited]\nObservation: Answer is CORRECT.",
        "Search[giant silverware company religious Utopian group Pierrepont Noyes]",
        "I need to search for more information about Oneida Limited to see if it matches the description of a giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes. I should look for details about the company's history and any relevant connections to Noyes. \nAction: Search[Oneida Limited history Pierrepont Noyes]\nObservation 2: Oneida Limited is an American manufacturer of tableware and cutlery. The company originated in a utopian community, the Oneida Community, established in the 19th century. Pierrepont Burt Noyes, a descendant of the community's founder, became the head of Oneida Limited and led the company for many years.\nThought: It seems like Oneida Limited fits the description of the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes. I should look for more specific details to confirm this information.\nAction: Lookup[silverware]\nObservation 3: Oneida Limited is known for its production of silverware, flatware, and tableware products. The company has a long history of manufacturing high-quality silverware and cutlery items.\nThought: Based on the information gathered, Oneida Limited is indeed the giant silverware company that matches the description provided in the question. I can now finish with this answer.\nAction: Finish[Oneida Limited] \nObservation: Answer is CORRECT",
        "Search[Oneida Limited giant silverware company religious Utopian group Pierrepont Noyes]",
        'It seems like the search results are not providing specific information about Oneida Limited as a giant silverware company. I should try to look up the keyword "silverware" in the last passage successfully found and see if it provides any relevant details about Oneida Limited\'s products.\nAction: Lookup[silverware]',
        "Lookup[silverware]",
        "The passage mentions that Oneida Limited specialized in silverware and stainless steel cutlery under Pierrepont Noyes' leadership. This aligns with the description of a giant silverware company. I can confidently conclude that Oneida Limited is the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes for many years. I need to finish with this answer.\nAction: Finish[Oneida Limited]\nObservation: Answer is CORRECT",
        "Finish[Oneida Limited]",
        "ADD 2: When searching for information, consider looking up keywords or terms mentioned in the search results to gather more specific details.",
    ]

    agent = ExpeLAgent(
        llm=MockLLM("gpt-3.5-turbo", responses=responses),
        benchmark="hotpotqa",
        experience_memory=ExpeLExperienceMemory(experiences),
    )
    agent.strategy.reflexion_react_agent.strategy.docstore.search = (
        lambda x: "Search result"
    )
    agent.strategy.reflexion_react_agent.strategy.docstore.lookup = (
        lambda x: "Lookup result"
    )
    out = agent.generate(
        question=question,
        key=key,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=EXPEL_REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        reflect_prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert out == gt_out
    assert len(agent.strategy.experience_memory.experiences) == 6
    assert agent.strategy.experience_memory.experiences[5].question == question
    assert agent.strategy.experience_memory.experiences[5].key == key
    assert agent.strategy.experience_memory.experiences[5].reflections == []

    assert agent.strategy.insight_memory.insights == gt_insights
    assert len(agent.strategy.experience_memory.success_traj_docs) == 36
    assert agent.strategy.experience_memory.vectorstore
