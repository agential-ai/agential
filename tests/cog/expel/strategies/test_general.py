"""Unit tests for ExpeL strategies."""

import joblib

from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.cog.expel.strategies.general import ExpeLStrategy
from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.reflexion.agent import (
    ReflexionReActAgent,
    ReflexionReActOutput,
    ReflexionReActStepOutput,
)
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
    strategy = ExpeLStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(strategy.experience_memory, ExpeLExperienceMemory)
    assert isinstance(strategy.insight_memory, ExpeLInsightMemory)
    assert strategy.success_batch_size == 8
    assert strategy.experience_memory.experiences == []
    assert not strategy.experience_memory.success_traj_docs
    assert not strategy.experience_memory.vectorstore
    assert not strategy.insight_memory.insights
    assert strategy._prompt_metrics == {'compare': [], 'success': []}

    # Test with all parameters specified except experience memory and reflexion_react_agent.
    strategy = ExpeLStrategy(
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
    assert strategy._prompt_metrics == {'compare': [], 'success': []}

    # Test with custom reflexion_react_agent (verify it overrides reflexion_react_kwargs)
    strategy = ExpeLStrategy(
        llm=llm,
        reflexion_react_agent=ReflexionReActAgent(
            llm=llm, benchmark="hotpotqa", max_steps=100
        ),
    )
    assert isinstance(strategy.reflexion_react_agent, ReflexionReActAgent)
    assert strategy.reflexion_react_agent.benchmark == "hotpotqa"
    assert strategy._prompt_metrics == {'compare': [], 'success': []}

    # Test with custom experience memory (verify correct initialization).
    experiences = joblib.load(expel_experiences_10_fake_path)
    experiences = experiences[:1]

    strategy = ExpeLStrategy(
        llm=llm,
        reflexion_react_agent=ReflexionReActAgent(llm=llm, benchmark="hotpotqa"),
        experience_memory=ExpeLExperienceMemory(experiences),
    )
    assert strategy.experience_memory.experiences == experiences
    assert strategy.insight_memory.insights == []
    assert strategy._prompt_metrics == {'compare': [], 'success': []}


def test_generate() -> None:
    """Test generate."""
    question = "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?"
    key = "Oneida Limited"

    gt_new_experiences = [
        {
            "question": "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?",
            "key": "Oneida Limited",
            "trajectory": [
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
                            prompt_metrics={
                                "thought": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                                "action": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                            },
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
                            prompt_metrics={
                                "thought": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                                "action": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                            },
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
                            prompt_metrics={
                                "thought": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                                "action": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                            },
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
                            prompt_metrics={
                                "thought": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                                "action": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                            },
                        ),
                    ],
                    reflections=[],
                    prompt_metrics={"reflection": None},
                )
            ],
            "reflections": [],
        }
    ]
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
    strategy = ExpeLStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
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
    )

    assert new_experiences == gt_new_experiences
    assert new_experiences == strategy.experience_memory.experiences
    assert len(strategy.experience_memory.success_traj_docs) == 13
    assert strategy.experience_memory.vectorstore


def test_get_dynamic_examples(expel_experiences_10_fake_path: str) -> None:
    """Test get_dynamic_examples."""
    question = "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?"
    experiences = joblib.load(expel_experiences_10_fake_path)

    gt_new_examples = 'The creator of "Wallace and Gromit" also created what animation comedy that matched animated zoo animals with a soundtrack of people talking about their homes? \nThought: I need to search for the creator of "Wallace and Gromit" and find out what animation comedy they also created that involves animated zoo animals and people talking about their homes in the soundtrack.\nAction: Search[creator of "Wallace and Gromit"]\nObservation: Wallace and Gromit is a British stop-motion animated comedy franchise created by Nick Park and produced by Aardman Animations. It consists of four short films, two feature-length films and has spawned numerous spin-offs and TV adaptations. The series centres on Wallace, a good-natured, eccentric, cheese-loving inventor, and Gromit, his loyal and intelligent anthropomorphic beagle. The first short film, A Grand Day Out, was finished and released in 1989. Wallace was voiced by actor Peter Sallis until 2010 when he was succeeded by Ben Whitehead. While Wallace speaks very often, Gromit is largely silent and has no dialogue, communicating through facial expressions and body language.Because of their popularity, the characters have been described as positive international cultural icons of both modern British culture and British people in general. BBC News called them "some of the best-known and best-loved stars to come out of the UK". Icons has said they have done "more to improve the image of the English world-wide than any officially appointed ambassadors". Although not overtly set in any particular town, Park has hinted that it was inspired by 1950s Wigan in Northern England. Wallace\'s accent comes from the Holme Valley of West Yorkshire. Wallace is fond of Wensleydale cheese (from Wensleydale, North Yorkshire).Their films have been widely praised, with the first three short films, A Grand Day Out (1989), The Wrong Trousers (1993) and A Close Shave (1995) earning 100% on Rotten Tomatoes; the feature film Wallace & Gromit: The Curse of the Were-Rabbit (2005) has also received acclaim. The feature film is the second-highest-grossing stop-motion animated film, outgrossed by Chicken Run (2000), another creation of Park\'s. A fourth short film, A Matter of Loaf and Death, was released in 2008. A second feature film, Wallace & Gromit: Vengeance Most Fowl, marking the return of the villainous penguin Feathers McGraw, will be released in 2024. The franchise has received numerous accolades, including five British Academy Film Awards, three Academy Awards and a Peabody Award.The Wallace and Gromit characters spearhead the fundraising for two children\'s charities: Wallace & Gromit\'s Children\'s Foundation, which supports children\'s hospices and hospitals in the United Kingdom, and Wallace and Gromit\'s Grand Appeal, the charity for Bristol Children\'s Hospital in Bristol. In December 2010, Wallace and Gromit featured on a festive themed series of UK postage stamps issued by the Royal Mail.\nThought: Since the creator of "Wallace and Gromit" is Nick Park, I need to find out what animation comedy he created that matches animated zoo animals with a soundtrack of people talking about their homes.\nAction: Search[Nick Park animated zoo animals movie]\nObservation: Nicholas Wulstan Park  (born 6 December 1958) is an  English filmmaker and animator who created Wallace and Gromit, Creature Comforts, Chicken Run, Shaun the Sheep, and Early Man. Park has been nominated for an Academy Award a total of six times and won four with Creature Comforts (1989), The Wrong Trousers (1993), A Close Shave (1995) and Wallace & Gromit: The Curse of the Were-Rabbit (2005).He has also received five BAFTA Awards, including the BAFTA for Best Short Animation for A Matter of Loaf and Death, which was also the most watched television programme in the United Kingdom in 2008. His 2000 film Chicken Run is the highest-grossing stop motion animated film.In 1985, Park joined Aardman Animations based in Bristol, and for his work in animation he was among the British cultural icons selected by artist Peter Blake to appear in a 2012 version of Blake\'s most famous artwork—the Beatles\' Sgt. Pepper\'s Lonely Hearts Club Band album cover—to celebrate the British cultural figures of his life.Park was appointed a CBE by Queen Elizabeth II in the 1997 Birthday Honours for "services to the animated film industry".\nThought: Nick Park created "Creature Comforts" which matches animated zoo animals with a soundtrack of people talking about their homes.\nAction: Finish[Creature Comforts]\nObservation: Answer is CORRECT\n\n\n---\n\nWhich of Jonny Craig and Pete Doherty has been a member of more bands ?\nThought: I need to search for the bands Jonny Craig and Pete Doherty have been a part of, then compare the number of bands each has been in.\nAction: Search[Jonny Craig]\nObservation: Jenny Craig, Inc., often known simply as Jenny Craig, is an American weight loss, weight management, and nutrition company. The company had more than 700 weight management centers in Australia, the United States, Canada, and New Zealand. The program combined individual weight management counseling with a menu of frozen meals and other foods which were distributed through its centers or shipped directly to clients. On 4 May 2023, the company announced that it was going out of business after 40 years. It was revived as an e-commerce store by Wellful sometime after this announcement in fall 2023.\nThought: The search results for Jonny Craig are incorrect. I need to search for Jonny Craig musician to get accurate information about the bands he has been a part of.\nAction: Search[Jonny Craig musician]\nObservation: Jonathan Monroe Craig (born March 26, 1986) is an American-Canadian singer, widely recognised for his tenure as the former lead vocalist of renowned post-hardcore bands such as Dance Gavin Dance, Emarosa, and then Slaves, as well as his role as co-lead vocalist in the short-lived supergroup Isles & Glaciers. Presently, he serves as the lead vocalist for the American post-hardcore band Old Flame. As a solo artist, Craig has released two studio albums, two EPs, and a live album, garnering acclaim for his distinctive soul-based singing style.\nThought: Jonny Craig has been a member of several bands such as Dance Gavin Dance, Emarosa, Slaves, and Isles & Glaciers. Now, I need to search for the bands Pete Doherty has been a part of.\nAction: Search[Pete Doherty]\nObservation: Peter Doherty (born 12 March 1979) is an English musician, songwriter, actor, poet, writer and artist. He is best known for being co-frontman of The Libertines, which he formed with Carl Barât in 1997. His other musical projects are indie bands Babyshambles and Peter Doherty and the Puta Madres.\nThought: Pete Doherty has been a member of The Libertines, Babyshambles, and Peter Doherty and the Puta Madres. I need to compare the number of bands Jonny Craig and Pete Doherty have been a part of to determine who has been in more bands.\nAction: Finish[Jonny Craig]\nObservation: Answer is CORRECT\n'
    gt_additional_keys = {
        "insights": "0. Focus on identifying key creators or individuals related to the topic in order to gather relevant information quickly.\n1. Prioritize gathering information about key individuals related to the topic to quickly acquire relevant details.\n"
    }
    responses = [
        "ADD 13: Focus on identifying key creators or individuals related to the topic in order to gather relevant information quickly.",
        "ADD 1: Prioritize gathering information about key individuals related to the topic to quickly acquire relevant details.",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ExpeLStrategy(
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
    assert new_examples == gt_new_examples
    assert additional_keys == gt_additional_keys


def test_gather_experience(hotpotqa_distractor_sample_path: str) -> None:
    """Test gather_experience."""
    hotpotqa = joblib.load(hotpotqa_distractor_sample_path)

    gt_new_experiences = [
        {
            "question": "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?",
            "key": "Oneida Limited",
            "trajectory": [
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
                            prompt_metrics={
                                "thought": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                                "action": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                            },
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
                            prompt_metrics={
                                "thought": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                                "action": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                            },
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
                            prompt_metrics={
                                "thought": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                                "action": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                            },
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
                            prompt_metrics={
                                "thought": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                                "action": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30,
                                    "prompt_tokens_cost": 1.5e-05,
                                    "completion_tokens_cost": 3.9999999999999996e-05,
                                    "total_tokens_cost": 5.4999999999999995e-05,
                                    "time_sec": 0.5,
                                },
                            },
                        ),
                    ],
                    reflections=[],
                    prompt_metrics={"reflection": None},
                )
            ],
            "reflections": [],
        }
    ]

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
    strategy = ExpeLStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)
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

    assert new_experiences == gt_new_experiences
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
    strategy = ExpeLStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)

    strategy.extract_insights(experiences)
    assert strategy.insight_memory.insights == gt_insights
    print(strategy._prompt_metrics)
    assert strategy._prompt_metrics == {
        "compare": [],
        "success": [
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
    }


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
    strategy = ExpeLStrategy(
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
    strategy = ExpeLStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)

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


def test_reset() -> None:
    """Test reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)

    strategy.reflexion_react_agent.strategy._scratchpad = "cat"
    strategy.experience_memory.experiences = "dog"
    strategy.insight_memory.insights = ["turtle"]
    strategy.reset()
    assert strategy.reflexion_react_agent.strategy._scratchpad == ""
    assert strategy.experience_memory.experiences == []
    assert strategy.insight_memory.insights == []
    assert strategy._prompt_metrics == {"compare": [], "success": []}

    # Test only_reflexion=True.
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)

    strategy.reflexion_react_agent.strategy._scratchpad = "cat"
    strategy.experience_memory.experiences = "dog"
    strategy.insight_memory.insights = ["turtle"]
    strategy.reset(only_reflexion=True)
    assert strategy.reflexion_react_agent.strategy._scratchpad == ""
    assert strategy.experience_memory.experiences == "dog"
    assert strategy.insight_memory.insights == ["turtle"]
    assert strategy._prompt_metrics == {"compare": [], "success": []}
