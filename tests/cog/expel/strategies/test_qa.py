"""Unit tests for ExpeL QA strategies."""

import joblib
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models.fake import FakeListChatModel
from agential.cog.expel.strategies.qa import ExpeLQAStrategy
from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.cog.reflexion.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.reflexion.agent import (
    ReflexionReActAgent,
    ReflexionReActOutput,
    ReflexionReActStepOutput,
)
from agential.cog.reflexion.agent import ReflexionReActAgent


def test_init(expel_experiences_10_fake_path: str) -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLQAStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(strategy.experience_memory, ExpeLExperienceMemory)
    assert isinstance(strategy.insight_memory, ExpeLInsightMemory)
    assert strategy.success_batch_size == 8
    assert strategy.experience_memory.experiences == {
        "idxs": [],
        "questions": [],
        "keys": [],
        "trajectories": [],
        "reflections": [],
    }
    assert not strategy.experience_memory.success_traj_docs
    assert not strategy.experience_memory.vectorstore
    assert not strategy.insight_memory.insights

    # Test with all parameters specified except experience memory and reflexion_react_agent.
    strategy = ExpeLQAStrategy(
        llm=llm,
        reflexion_react_agent=ReflexionReActAgent(llm=llm, benchmark="hotpotqa", max_trials=3),
        insight_memory=ExpeLInsightMemory(
            insights=[{"insight": "blah blah", "score": 10}]
        ),
        success_batch_size=10,
    )
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(strategy.experience_memory, ExpeLExperienceMemory)
    assert isinstance(strategy.insight_memory, ExpeLInsightMemory)
    assert strategy.success_batch_size == 10
    assert strategy.experience_memory.experiences == {
        "idxs": [],
        "questions": [],
        "keys": [],
        "trajectories": [],
        "reflections": [],
    }
    assert not strategy.experience_memory.success_traj_docs
    assert not strategy.experience_memory.vectorstore
    assert strategy.insight_memory.insights == [{"insight": "blah blah", "score": 10}]

    # Test with custom reflexion_react_agent (verify it overrides reflexion_react_kwargs)
    strategy = ExpeLQAStrategy(
        llm=llm,
        reflexion_react_agent=ReflexionReActAgent(llm=llm, benchmark="hotpotqa", max_steps=100),
    )
    assert isinstance(strategy.reflexion_react_agent, ReflexionReActAgent)
    assert strategy.reflexion_react_agent.benchmark == "hotpotqa"

    # Test with custom experience memory (verify correct initialization).
    experiences = joblib.load(expel_experiences_10_fake_path)
    experiences = {key: value[:1] for key, value in experiences.items()}

    strategy = ExpeLQAStrategy(
        llm=llm,
        reflexion_react_agent=ReflexionReActAgent(llm=llm, benchmark="hotpotqa"),
        experience_memory=ExpeLExperienceMemory(experiences),
    )
    assert strategy.experience_memory.experiences == experiences
    assert strategy.insight_memory.insights == []


def test_generate():
    pass


def test_get_dynamic_examples():
    pass


def test_gather_experience(hotpotqa_distractor_sample_path: str) -> None:
    """Test gather_experience."""
    hotpotqa = joblib.load(hotpotqa_distractor_sample_path)

    gt_new_experiences = {
        "idxs": [0],
        "questions": [
            "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?"
        ],
        "keys": ["Oneida Limited"],
        "trajectories": [
            [
                ReflexionReActOutput(
                    react_output=[
                        ReflexionReActStepOutput(
                            thought="I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.",
                            action_type="Search",
                            query="giant silverware company started as religious Utopian group Pierrepont Noyes",
                            observation="Search result",
                            answer="",
                            external_tool_info={
                                "search_result": "Search result",
                                "lookup_result": "",
                            },
                            is_correct=False,
                        ),
                        ReflexionReActStepOutput(
                            thought="The search query was too specific. I should try searching for the silverware company and then look for information about its history and founder.",
                            action_type="Search",
                            query="silverware company Pierrepont Noyes",
                            observation="Search result",
                            answer="",
                            external_tool_info={
                                "search_result": "Search result",
                                "lookup_result": "",
                            },
                            is_correct=False,
                        ),
                        ReflexionReActStepOutput(
                            thought="Pierrepont Noyes was the head of Oneida Limited, a silverware company. I need to confirm if Oneida Limited was indeed started as a religious Utopian group.",
                            action_type="Search",
                            query="Oneida Limited religious Utopian group",
                            observation="Search result",
                            answer="",
                            external_tool_info={
                                "search_result": "Search result",
                                "lookup_result": "",
                            },
                            is_correct=False,
                        ),
                        ReflexionReActStepOutput(
                            thought="Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.",
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
            ]
        ],
        "reflections": [[]],
    }

    action_responses = [
        "I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.\nAction: Search[Pierrepont Noyes]\nObservation: Could not find [Pierrepont Noyes]. Similar: ['Noyes Academy', 'Penn Noyes', 'Pierrepont Noyes', 'Pierrepont Noyes House', 'Pierrepont Noyes Mausoleum', 'Pierrepont Noyes Memorial Foundation']\nThought: I need to search for the company directly instead.\nAction: Search[giant silverware company religious Utopian group]\nObservation: Could not find [giant silverware company religious Utopian group]. Similar: ['Oneida Limited', 'Oneida Community', 'Oneida Silver']\nThought: I should search for Oneida Limited.\nAction: Search[Oneida Limited]\nObservation: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silver-plated cutlery and tableware for the consumer and foodservice industries.\nThought: Oneida Limited is the giant silverware company that was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Finish[Oneida Limited]",
        "Search[giant silverware company started as religious Utopian group Pierrepont Noyes]\nObservation: Could not find [giant silverware company started as religious Utopian group Pierrepont Noyes]. Similar: ['Oneida Community', 'Gorham Silver Manufacturing Company', 'F. B. Rogers Silver Company']\nThought: Let's search for Oneida Community and see if it was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Search[Oneida Community]\nObservation: The Oneida Community was a religious perfectionist group founded by John Humphrey Noyes in 1848 in Oneida, New York.\nThought: The Oneida Community was indeed a religious Utopian group, but it was founded by John Humphrey Noyes, not Pierrepont Noyes. Let's look up Pierrepont Noyes to see if he was involved with the Oneida Community.\nAction: Lookup[Pierrepont Noyes]\nObservation: (Result 1 / 1) Pierrepont Noyes was an American industrialist and president of Oneida Community from 1877 to 1881.\nThought: Pierrepont Noyes was indeed involved with the Oneida Community. So the answer to the question is Oneida Community.\nAction: Finish[Oneida Community]",
        "The search query was too specific. I should try searching for the silverware company and then look for information about its history and founder.\nAction: Search[silverware company Pierrepont Noyes]\nObservation 2: Oneida Limited is an American manufacturer of tableware and cutlery. The company originated in a utopian community founded in the 19th century.\nThought: Oneida Limited was the silverware company started as a religious Utopian group. Now I need to find out if it was run by Pierrepont Noyes.\nAction: Lookup[Pierrepont Noyes]\nObservation 3: (Result 1 / 1) Pierrepont Noyes was a member of the Oneida Community, a religious communal society founded by his father John Humphrey Noyes.\nThought: Oneida Limited was indeed run by Pierrepont Noyes, so the answer is Oneida Limited.\nAction: Finish[Oneida Limited]",
        "Search[silverware company Pierrepont Noyes]\nObservation 2: Could not find [silverware company Pierrepont Noyes]. Similar: []\nThought: I should try searching for Pierrepont Noyes to see if I can find any information about the company he was associated with.\nAction: Search[Pierrepont Noyes]\nObservation 3: Pierrepont Noyes (1868–1933) was an American industrialist, socialist, and founder of the Oneida Community.\nThought: Pierrepont Noyes was the founder of the Oneida Community. I should search for the history of the Oneida Community to see if it is the silverware company in question.\nAction: Search[Oneida Community silverware company]\nObservation 4: The Oneida Community was a religiously-centered communal society founded by John Humphrey Noyes in 1848 in Oneida, New York. The Oneida Community practiced communalism, complex marriage, male continence, mutual criticism, and asceticism.\nThought: The Oneida Community is the religious Utopian group mentioned in the question. I should now search for the connection between the Oneida Community and the silverware company.\nAction: Search[Oneida Community silverware company connection]\nObservation 5: Could not find [Oneida Community silverware company connection]. Similar: []\nThought: Since I couldn't find a direct connection, I should try searching for the history of the Oneida Community to see if it eventually led to the creation of a silverware company.\nAction: Search[Oneida Community history]\nObservation 6: The Oneida Community dissolved in 1881, and some of its members formed the Oneida Community Limited to continue the manufacture of silverware.\nThought: The Oneida Community eventually led to the formation of the Oneida Community Limited, a silverware company. The answer to the question is Oneida Community Limited.\nAction: Finish[Oneida Community Limited]",
        "Pierrepont Noyes was the head of Oneida Limited, a silverware company. I need to confirm if Oneida Limited was indeed started as a religious Utopian group.\nAction: Search[Oneida Limited history]\nObservation 3: Oneida Limited was founded by John Humphfrey Noyes in the mid-19th century as a religious commune known as the Oneida Community.\nThought: Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.\nAction: Finish[Oneida Limited]",
        "Search[Oneida Limited religious Utopian group]\nObservation 3: Could not find [Oneida Limited religious Utopian group]. Similar: ['Oneida, New York', 'Oneida Community', 'Oneida Silverware']\nThought: I can search for the history of Oneida Limited to confirm if it was started as a religious Utopian group.\nAction: Search[Oneida Limited history]\nObservation 4: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silverplated cutlery and tableware for the consumer and foodservice industries.\nThought: The search results do not provide information about the religious Utopian origins of Oneida Limited. I should look up the history of the Oneida Community to find the connection.\nAction: Lookup[Oneida Community]\nObservation 5: (Result 1 / 1) The Oneida Community was a religious communal society founded by John Humphrey Noyes in 1848 in Oneida, New York.\nThought: The Oneida Community was a religious communal society founded by John Humphrey Noyes in 1848. Pierrepont Noyes, who later ran Oneida Limited, was brought up in the Oneida Community. Therefore, the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes is Oneida Limited.\nAction: Finish[Oneida Limited]",
        "Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.\nAction: Finish[Oneida Limited]",
        "Finish[Oneida Limited]",
    ]
    llm = FakeListChatModel(responses=action_responses)
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLQAStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
    strategy.reflexion_react_agent.strategy.docstore.search = lambda x: "Search result"
    strategy.reflexion_react_agent.strategy.docstore.lookup = lambda x: "Lookup result"
    new_experiences = strategy.gather_experience(
        questions=hotpotqa.question.values[-1:],
        keys=hotpotqa.answer.values[-1:],
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        reflect_prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        reflect_strategy="reflexion",
        additional_keys={},
        reflect_additional_keys={},
        patience=1
    )

    assert new_experiences == gt_new_experiences
    assert new_experiences == strategy.experience_memory.experiences
    assert len(strategy.experience_memory.success_traj_docs) == 13
    assert strategy.experience_memory.vectorstore



def test_extract_insights(expel_experiences_10_fake_path: str) -> None:
    """Test extract_insights."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    selected_indices = [3]
    selected_dict = {
        key: [value[i] for i in selected_indices] for key, value in experiences.items()
    }
    selected_dict["idxs"] = list(range(len(selected_indices)))

    gt_insights = [
        {
            "insight": "Always try multiple variations of search terms when looking for specific information.",
            "score": 2,
        },
        {
            "insight": "If unable to find relevant information through initial searches, consider looking for official announcements or press releases from the company.",
            "score": 2,
        },
    ]
    responses = [
        "ADD 11: Always try multiple variations of search terms when looking for specific information.\nADD 12: If unable to find relevant information through initial searches, consider looking for official announcements or press releases from the company.\nREMOVE 3: Always use the exact search term provided in the question, do not try variations.\nEDIT 7: Make sure to exhaust all possible search options before concluding that the information is unavailable.",
    ]
    llm = FakeListChatModel(responses=responses)
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLQAStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
    
    strategy.extract_insights(selected_dict)
    assert strategy.insight_memory.insights == gt_insights

def test_update_insights() -> None:
    """Test update_insights."""
    insights = [
        {"insight": "Test 1", "score": 1},
        {"insight": "Test 2", "score": 2},
        {"insight": "Test 3", "score": 3},
    ]
    memory = ExpeLInsightMemory(insights, max_num_insights=3)
    llm = FakeListChatModel(responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLQAStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent, insight_memory=memory)

    # Valid remove.
    gt_insights = [{"insight": "Test 2", "score": 2}, {"insight": "Test 3", "score": 3}]
    strategy.update_insights(
        [
            ("REMOVE 0", "Test 1"),
        ]
    )
    assert strategy.insight_memory.insights == gt_insights

    # Invalid remove.
    strategy.update_insights(
        [
            ("REMOVE 0", "Test askdasf"),
        ]
    )
    assert strategy.insight_memory.insights == gt_insights

    # Valid agree.
    gt_insights = [{"insight": "Test 2", "score": 3}, {"insight": "Test 3", "score": 3}]
    strategy.update_insights([("AGREE 0", "Test 2")])
    assert strategy.insight_memory.insights == gt_insights

    # Invalid agree.
    strategy.update_insights([("AGREE 0", "Test asdjafh")])
    assert strategy.insight_memory.insights == gt_insights

    # Edit.
    gt_insights = [{"insight": "Test 2", "score": 3}, {"insight": "Test 4", "score": 4}]
    strategy.update_insights([("EDIT 1", "Test 4")])
    assert strategy.insight_memory.insights == gt_insights

    # Add.
    gt_insights = [
        {"insight": "Test 2", "score": 3},
        {"insight": "Test 4", "score": 4},
        {"insight": "Another insight", "score": 2},
    ]
    strategy.update_insights([("ADD", "Another insight")])
    assert strategy.insight_memory.insights == gt_insights


def test_reset() -> None:
    """Test reset."""
    llm = FakeListChatModel(responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLQAStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
    
    strategy.reflexion_react_agent.strategy._scratchpad = "cat"
    strategy.experience_memory.experiences = "dog"
    strategy.insight_memory.insights = ["turtle"]
    strategy.reset()
    assert strategy.reflexion_react_agent.strategy._scratchpad == ""
    assert strategy.experience_memory.experiences == {
        "idxs": [],
        "questions": [],
        "keys": [],
        "trajectories": [],
        "reflections": [],
    }
    assert strategy.insight_memory.insights == []

    # Test only_reflexion=True.
    llm = FakeListChatModel(responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLQAStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
    
    strategy.reflexion_react_agent.strategy._scratchpad = "cat"
    strategy.experience_memory.experiences = "dog"
    strategy.insight_memory.insights = ["turtle"]
    strategy.reset(only_reflexion=True)
    assert strategy.reflexion_react_agent.strategy._scratchpad == ""
    assert strategy.experience_memory.experiences == "dog"
    assert strategy.insight_memory.insights == ["turtle"]