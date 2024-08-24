"""Unit tests for ExpeL strategies."""

import joblib

from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.cog.expel.strategies.general import ExpeLGeneralStrategy
from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.reflexion.agent import (
    ReflexionReActAgent,
    ReflexionReActOutput,
)
from agential.cog.reflexion.output import ReflexionReActStepOutput
from agential.cog.reflexion.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.llm.llm import BaseLLM, MockLLM


def test_init(expel_experiences_10_fake_path: str) -> None:
    """Test initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLGeneralStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(strategy.experience_memory, ExpeLExperienceMemory)
    assert isinstance(strategy.insight_memory, ExpeLInsightMemory)
    assert strategy.success_batch_size == 8
    assert strategy.experience_memory.experiences == []
    assert not strategy.experience_memory.success_traj_docs
    assert not strategy.experience_memory.vectorstore
    assert not strategy.insight_memory.insights
    assert strategy._prompt_metrics == {"compare": [], "success": []}

    # Test with all parameters specified except experience memory and reflexion_react_agent.
    strategy = ExpeLGeneralStrategy(
        llm=llm,
        reflexion_react_agent=ReflexionReActAgent(
            llm=llm, benchmark="hotpotqa", max_trials=3
        ),
        insight_memory=ExpeLInsightMemory(
            insights=[{"insight": "blah blah", "score": 10}]
        ),
        success_batch_size=10,
    )
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(strategy.experience_memory, ExpeLExperienceMemory)
    assert isinstance(strategy.insight_memory, ExpeLInsightMemory)
    assert strategy.success_batch_size == 10
    assert strategy.experience_memory.experiences == []
    assert not strategy.experience_memory.success_traj_docs
    assert not strategy.experience_memory.vectorstore
    assert strategy.insight_memory.insights == [{"insight": "blah blah", "score": 10}]
    assert strategy._prompt_metrics == {"compare": [], "success": []}

    # Test with custom reflexion_react_agent (verify it overrides reflexion_react_kwargs)
    strategy = ExpeLGeneralStrategy(
        llm=llm,
        reflexion_react_agent=ReflexionReActAgent(
            llm=llm, benchmark="hotpotqa", max_steps=100
        ),
    )
    assert isinstance(strategy.reflexion_react_agent, ReflexionReActAgent)
    assert strategy.reflexion_react_agent.benchmark == "hotpotqa"
    assert strategy._prompt_metrics == {"compare": [], "success": []}

    # Test with custom experience memory (verify correct initialization).
    experiences = joblib.load(expel_experiences_10_fake_path)
    experiences = experiences[:1]

    strategy = ExpeLGeneralStrategy(
        llm=llm,
        reflexion_react_agent=ReflexionReActAgent(llm=llm, benchmark="hotpotqa"),
        experience_memory=ExpeLExperienceMemory(experiences),
    )
    assert strategy.experience_memory.experiences == experiences
    assert strategy.insight_memory.insights == []
    assert strategy._prompt_metrics == {"compare": [], "success": []}


def test_generate() -> None:
    """Test generate."""
    question = "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?"
    key = "Oneida Limited"

    # gt_new_experiences = [
    #     {
    #         "question": "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?",
    #         "key": "Oneida Limited",
    #         "trajectory": [
    #             ReflexionReActOutput(
    #                 react_output=[
    #                     ReflexionReActStepOutput(
    #                         thought="I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.",
    #                         action_type="Search",
    #                         query="giant silverware company started as religious Utopian group Pierrepont Noyes",
    #                         observation="Search result",
    #                         answer="",
    #                         external_tool_info={
    #                             "search_result": "Search result",
    #                             "lookup_result": "",
    #                         },
    #                         is_correct=False,
    #                         prompt_metrics={
    #                             "thought": {
    #                                 "prompt_tokens": 10,
    #                                 "completion_tokens": 20,
    #                                 "total_tokens": 30,
    #                                 "prompt_tokens_cost": 1.5e-05,
    #                                 "completion_tokens_cost": 3.9999999999999996e-05,
    #                                 "total_tokens_cost": 5.4999999999999995e-05,
    #                                 "time_sec": 0.5,
    #                             },
    #                             "action": {
    #                                 "prompt_tokens": 10,
    #                                 "completion_tokens": 20,
    #                                 "total_tokens": 30,
    #                                 "prompt_tokens_cost": 1.5e-05,
    #                                 "completion_tokens_cost": 3.9999999999999996e-05,
    #                                 "total_tokens_cost": 5.4999999999999995e-05,
    #                                 "time_sec": 0.5,
    #                             },
    #                         },
    #                     ),
    #                     ReflexionReActStepOutput(
    #                         thought="The search query was too specific. I should try searching for the silverware company and then look for information about its history and founder.",
    #                         action_type="Search",
    #                         query="silverware company Pierrepont Noyes",
    #                         observation="Search result",
    #                         answer="",
    #                         external_tool_info={
    #                             "search_result": "Search result",
    #                             "lookup_result": "",
    #                         },
    #                         is_correct=False,
    #                         prompt_metrics={
    #                             "thought": {
    #                                 "prompt_tokens": 10,
    #                                 "completion_tokens": 20,
    #                                 "total_tokens": 30,
    #                                 "prompt_tokens_cost": 1.5e-05,
    #                                 "completion_tokens_cost": 3.9999999999999996e-05,
    #                                 "total_tokens_cost": 5.4999999999999995e-05,
    #                                 "time_sec": 0.5,
    #                             },
    #                             "action": {
    #                                 "prompt_tokens": 10,
    #                                 "completion_tokens": 20,
    #                                 "total_tokens": 30,
    #                                 "prompt_tokens_cost": 1.5e-05,
    #                                 "completion_tokens_cost": 3.9999999999999996e-05,
    #                                 "total_tokens_cost": 5.4999999999999995e-05,
    #                                 "time_sec": 0.5,
    #                             },
    #                         },
    #                     ),
    #                     ReflexionReActStepOutput(
    #                         thought="Pierrepont Noyes was the head of Oneida Limited, a silverware company. I need to confirm if Oneida Limited was indeed started as a religious Utopian group.",
    #                         action_type="Search",
    #                         query="Oneida Limited religious Utopian group",
    #                         observation="Search result",
    #                         answer="",
    #                         external_tool_info={
    #                             "search_result": "Search result",
    #                             "lookup_result": "",
    #                         },
    #                         is_correct=False,
    #                         prompt_metrics={
    #                             "thought": {
    #                                 "prompt_tokens": 10,
    #                                 "completion_tokens": 20,
    #                                 "total_tokens": 30,
    #                                 "prompt_tokens_cost": 1.5e-05,
    #                                 "completion_tokens_cost": 3.9999999999999996e-05,
    #                                 "total_tokens_cost": 5.4999999999999995e-05,
    #                                 "time_sec": 0.5,
    #                             },
    #                             "action": {
    #                                 "prompt_tokens": 10,
    #                                 "completion_tokens": 20,
    #                                 "total_tokens": 30,
    #                                 "prompt_tokens_cost": 1.5e-05,
    #                                 "completion_tokens_cost": 3.9999999999999996e-05,
    #                                 "total_tokens_cost": 5.4999999999999995e-05,
    #                                 "time_sec": 0.5,
    #                             },
    #                         },
    #                     ),
    #                     ReflexionReActStepOutput(
    #                         thought="Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.",
    #                         action_type="Finish",
    #                         query="Oneida Limited",
    #                         observation="Answer is CORRECT",
    #                         answer="Oneida Limited",
    #                         external_tool_info={
    #                             "search_result": "",
    #                             "lookup_result": "",
    #                         },
    #                         is_correct=True,
    #                         prompt_metrics={
    #                             "thought": {
    #                                 "prompt_tokens": 10,
    #                                 "completion_tokens": 20,
    #                                 "total_tokens": 30,
    #                                 "prompt_tokens_cost": 1.5e-05,
    #                                 "completion_tokens_cost": 3.9999999999999996e-05,
    #                                 "total_tokens_cost": 5.4999999999999995e-05,
    #                                 "time_sec": 0.5,
    #                             },
    #                             "action": {
    #                                 "prompt_tokens": 10,
    #                                 "completion_tokens": 20,
    #                                 "total_tokens": 30,
    #                                 "prompt_tokens_cost": 1.5e-05,
    #                                 "completion_tokens_cost": 3.9999999999999996e-05,
    #                                 "total_tokens_cost": 5.4999999999999995e-05,
    #                                 "time_sec": 0.5,
    #                             },
    #                         },
    #                     ),
    #                 ],
    #                 reflections=[],
    #                 prompt_metrics={"reflection": None},
    #             )
    #         ],
    #         "reflections": [],
    #     }
    # ]
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
    llm = MockLLM("gpt-3.5-turbo", responses=action_responses)
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLGeneralStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
    strategy.reflexion_react_agent.strategy.docstore.search = lambda x: "Search result"
    strategy.reflexion_react_agent.strategy.docstore.lookup = lambda x: "Lookup result"
    new_experiences = strategy.generate(
        question=question,
        key=key,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        reflect_prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        reflect_strategy="reflexion",
        additional_keys={},
        reflect_additional_keys={},
        patience=1,
        use_dynamic_examples = False,
        extract_insights = False,
        k_docs = 1,
        num_fewshots = 1,
        max_fewshot_tokens = 100,
        reranker_strategy = "bm25",
        reset = False,
    )
    print(repr(new_experiences))
    assert False
    assert new_experiences == gt_new_experiences
    assert new_experiences == strategy.experience_memory.experiences
    assert len(strategy.experience_memory.success_traj_docs) == 13
    assert strategy.experience_memory.vectorstore


def test_get_dynamic_examples(expel_experiences_10_fake_path: str) -> None:
    """Test get_dynamic_examples."""
    question = "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?"
    experiences = joblib.load(expel_experiences_10_fake_path)

    gt_new_examples = "Which of Jonny Craig and Pete Doherty has been a member of more bands ?\nThought: I need to search for the band memberships of Jonny Craig and Pete Doherty to determine who has been in more bands.\nAction: Search[Jonny Craig band memberships]\nObservation: Could not find [Jonny Craig band memberships]. Similar: ['Coldplay', 'Gerard Butler', 'List of musical supergroups', 'List of musical acts from Western Australia', 'Johnny Carson', '2024 in American television', 'Garage rock', 'University of Leeds', 'North Korea', 'Grammy Award for Album of the Year']\nThought: I should try searching for Jonny Craig's band history instead.\nAction: Search[Jonny Craig band history]\nObservation: Jonathan Monroe Craig (born March 26, 1986) is an American-Canadian singer, widely recognised for his tenure as the former lead vocalist of renowned post-hardcore bands such as Dance Gavin Dance, Emarosa, and then Slaves, as well as his role as co-lead vocalist in the short-lived supergroup Isles & Glaciers. Presently, he serves as the lead vocalist for the American post-hardcore band Old Flame. As a solo artist, Craig has released two studio albums, two EPs, and a live album, garnering acclaim for his distinctive soul-based singing style.\nThought: Jonny Craig has been a member of Dance Gavin Dance, Emarosa, Slaves, Isle & Glaciers, and Old Flame. I should now search for Pete Doherty's band memberships.\nAction: Search[Pete Doherty band memberships]\nObservation: Could not find [Pete Doherty band memberships]. Similar: ['Rush (band)', 'List of musical supergroups', \"The Lovin' Spoonful\", 'Bert Jansch', 'John McEnroe', '2024 in American television', 'The Clancy Brothers', 'List of bands that played at Dagenham Roundhouse', 'List of people from the London Borough of Hackney', 'Ibiza']\nThought: I should try searching for Pete Doherty's band history instead.\nAction: Search[Pete Doherty band history]\nObservation: Peter Doherty (born 12 March 1979) is an English musician. He is best known for being co-frontman of The Libertines, which he formed with Carl Barât in 1997. His other musical projects are indie bands Babyshambles and Peter Doherty and the Puta Madres.\nThought: Jonny Craig has been a member of more bands than Pete Doherty.\nAction: Finish[Jonny Craig]\nObservation: Answer is CORRECT\n"
    gt_additional_keys = {
        "insights": "0. Focus on identifying key creators or individuals related to the topic in order to gather relevant information quickly.\n1. Prioritize gathering information about key individuals related to the topic to quickly acquire relevant details.\n"
    }
    responses = [
        "ADD 13: Focus on identifying key creators or individuals related to the topic in order to gather relevant information quickly.",
        "ADD 1: Prioritize gathering information about key individuals related to the topic to quickly acquire relevant details.",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ExpeLGeneralStrategy(
        llm=llm,
        reflexion_react_agent=ReflexionReActAgent(
            llm=MockLLM("gpt-3.5-turbo", responses=[]), benchmark="hotpotqa"
        ),
        experience_memory=ExpeLExperienceMemory(experiences),
    )

    new_examples, additional_keys = strategy.get_dynamic_examples(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        k_docs=24,
        num_fewshots=6,
        max_fewshot_tokens=1500,
        reranker_strategy=None,
        additional_keys={},
    )
    print(repr(new_examples))
    assert new_examples == gt_new_examples
    assert additional_keys == gt_additional_keys


# TODO:
def test_gather_experience(hotpotqa_distractor_sample_path: str) -> None:
    """Test gather_experience."""
    hotpotqa = joblib.load(hotpotqa_distractor_sample_path)

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

    llm = MockLLM("gpt-3.5-turbo", responses=action_responses)
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLGeneralStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
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
        patience=1,
    )

    print(repr(new_experiences))
    assert new_experiences == False
    assert new_experiences == strategy.experience_memory.experiences
    assert len(strategy.experience_memory.success_traj_docs) == 13
    assert strategy.experience_memory.vectorstore


def test_extract_insights(expel_experiences_10_fake_path: str) -> None:
    """Test extract_insights."""
    experiences = joblib.load(expel_experiences_10_fake_path)[3:4]
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
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLGeneralStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)

    strategy.extract_insights(experiences)
    assert strategy.insight_memory.insights == gt_insights
    # assert strategy._prompt_metrics == {
    #     "compare": [],
    #     "success": [
    #         {
    #             "prompt_tokens": 10,
    #             "completion_tokens": 20,
    #             "total_tokens": 30,
    #             "prompt_tokens_cost": 1.5e-05,
    #             "completion_tokens_cost": 3.9999999999999996e-05,
    #             "total_tokens_cost": 5.4999999999999995e-05,
    #             "time_sec": 0.5,
    #         }
    #     ],
    # }


def test_update_insights() -> None:
    """Test update_insights."""
    insights = [
        {"insight": "Test 1", "score": 1},
        {"insight": "Test 2", "score": 2},
        {"insight": "Test 3", "score": 3},
    ]
    memory = ExpeLInsightMemory(insights, max_num_insights=3)
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLGeneralStrategy(
        llm=llm, reflexion_react_agent=reflexion_react_agent, insight_memory=memory
    )

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


def test_create_output_dict() -> None:
    """Test create_output_dict method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLGeneralStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)

    # Set up test data
    strategy.insight_memory.insights = [
        {"insight": "Insight 1", "score": 3},
        {"insight": "Insight 2", "score": 4},
    ]
    strategy.experience_memory.experiences = [
        {"experience": "Experience 1"},
        {"experience": "Experience 2"},
    ]

    gt_output = {
        "examples": "",
        "insights": "some insight.",
        "experience": {"other": "Other"},
        "experience_memory": {
            "experiences": [
                {"experience": "Experience 1"},
                {"experience": "Experience 2"},
            ]
        },
        "insight_memory": {
            "insights": [
                {"insight": "Insight 1", "score": 3},
                {"insight": "Insight 2", "score": 4},
            ]
        },
        "prompt_metrics": {"compare": [], "success": []},
    }
    output = strategy.create_output_dict(
        examples="",
        additional_keys={"insights": "some insight.", "other": "other"},
        experience=[{"question": "question", "key": "key", "other": "Other"}],
    )
    assert output == gt_output


# def test_reset() -> None:
#     """Test reset."""
#     llm = MockLLM("gpt-3.5-turbo", responses=[])
#     reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
#     strategy = ExpeLGeneralStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)

#     strategy.reflexion_react_agent.strategy._scratchpad = "cat"
#     strategy.experience_memory.experiences = "dog"
#     strategy.insight_memory.insights = ["turtle"]
#     strategy.reset()
#     assert strategy.reflexion_react_agent.strategy._scratchpad == ""
#     assert strategy.experience_memory.experiences == []
#     assert strategy.insight_memory.insights == []
#     assert strategy._prompt_metrics == {"compare": [], "success": []}

#     # Test only_reflexion=True.
#     llm = MockLLM("gpt-3.5-turbo", responses=[])
#     reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
#     strategy = ExpeLGeneralStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)

#     strategy.reflexion_react_agent.strategy._scratchpad = "cat"
#     strategy.experience_memory.experiences = "dog"
#     strategy.insight_memory.insights = ["turtle"]
#     strategy.reset(only_reflexion=True)
#     assert strategy.reflexion_react_agent.strategy._scratchpad == ""
#     assert strategy.experience_memory.experiences == "dog"
#     assert strategy.insight_memory.insights == ["turtle"]
#     assert strategy._prompt_metrics == {"compare": [], "success": []}
