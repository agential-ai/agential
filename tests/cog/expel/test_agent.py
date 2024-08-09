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

    gt_examples = 'The creator of "Wallace and Gromit" also created what animation comedy that matched animated zoo animals with a soundtrack of people talking about their homes? \nThought: I need to search for the creator of "Wallace and Gromit" and find out what animation comedy they also created that involves animated zoo animals and people talking about their homes in the soundtrack.\nAction: Search[creator of "Wallace and Gromit"]\nObservation: Wallace and Gromit is a British stop-motion animated comedy franchise created by Nick Park and produced by Aardman Animations. It consists of four short films, two feature-length films and has spawned numerous spin-offs and TV adaptations. The series centres on Wallace, a good-natured, eccentric, cheese-loving inventor, and Gromit, his loyal and intelligent anthropomorphic beagle. The first short film, A Grand Day Out, was finished and released in 1989. Wallace was voiced by actor Peter Sallis until 2010 when he was succeeded by Ben Whitehead. While Wallace speaks very often, Gromit is largely silent and has no dialogue, communicating through facial expressions and body language.Because of their popularity, the characters have been described as positive international cultural icons of both modern British culture and British people in general. BBC News called them "some of the best-known and best-loved stars to come out of the UK". Icons has said they have done "more to improve the image of the English world-wide than any officially appointed ambassadors". Although not overtly set in any particular town, Park has hinted that it was inspired by 1950s Wigan in Northern England. Wallace\'s accent comes from the Holme Valley of West Yorkshire. Wallace is fond of Wensleydale cheese (from Wensleydale, North Yorkshire).Their films have been widely praised, with the first three short films, A Grand Day Out (1989), The Wrong Trousers (1993) and A Close Shave (1995) earning 100% on Rotten Tomatoes; the feature film Wallace & Gromit: The Curse of the Were-Rabbit (2005) has also received acclaim. The feature film is the second-highest-grossing stop-motion animated film, outgrossed by Chicken Run (2000), another creation of Park\'s. A fourth short film, A Matter of Loaf and Death, was released in 2008. A second feature film, Wallace & Gromit: Vengeance Most Fowl, marking the return of the villainous penguin Feathers McGraw, will be released in 2024. The franchise has received numerous accolades, including five British Academy Film Awards, three Academy Awards and a Peabody Award.The Wallace and Gromit characters spearhead the fundraising for two children\'s charities: Wallace & Gromit\'s Children\'s Foundation, which supports children\'s hospices and hospitals in the United Kingdom, and Wallace and Gromit\'s Grand Appeal, the charity for Bristol Children\'s Hospital in Bristol. In December 2010, Wallace and Gromit featured on a festive themed series of UK postage stamps issued by the Royal Mail.\nThought: Since the creator of "Wallace and Gromit" is Nick Park, I need to find out what animation comedy he created that matches animated zoo animals with a soundtrack of people talking about their homes.\nAction: Search[Nick Park animated zoo animals movie]\nObservation: Nicholas Wulstan Park  (born 6 December 1958) is an  English filmmaker and animator who created Wallace and Gromit, Creature Comforts, Chicken Run, Shaun the Sheep, and Early Man. Park has been nominated for an Academy Award a total of six times and won four with Creature Comforts (1989), The Wrong Trousers (1993), A Close Shave (1995) and Wallace & Gromit: The Curse of the Were-Rabbit (2005).He has also received five BAFTA Awards, including the BAFTA for Best Short Animation for A Matter of Loaf and Death, which was also the most watched television programme in the United Kingdom in 2008. His 2000 film Chicken Run is the highest-grossing stop motion animated film.In 1985, Park joined Aardman Animations based in Bristol, and for his work in animation he was among the British cultural icons selected by artist Peter Blake to appear in a 2012 version of Blake\'s most famous artwork—the Beatles\' Sgt. Pepper\'s Lonely Hearts Club Band album cover—to celebrate the British cultural figures of his life.Park was appointed a CBE by Queen Elizabeth II in the 1997 Birthday Honours for "services to the animated film industry".\nThought: Nick Park created "Creature Comforts" which matches animated zoo animals with a soundtrack of people talking about their homes.\nAction: Finish[Creature Comforts]\nObservation: Answer is CORRECT\n\n\n---\n\nWhich of Jonny Craig and Pete Doherty has been a member of more bands ?\nThought: I need to search for the bands Jonny Craig and Pete Doherty have been a part of, then compare the number of bands each has been in.\nAction: Search[Jonny Craig]\nObservation: Jenny Craig, Inc., often known simply as Jenny Craig, is an American weight loss, weight management, and nutrition company. The company had more than 700 weight management centers in Australia, the United States, Canada, and New Zealand. The program combined individual weight management counseling with a menu of frozen meals and other foods which were distributed through its centers or shipped directly to clients. On 4 May 2023, the company announced that it was going out of business after 40 years. It was revived as an e-commerce store by Wellful sometime after this announcement in fall 2023.\nThought: The search results for Jonny Craig are incorrect. I need to search for Jonny Craig musician to get accurate information about the bands he has been a part of.\nAction: Search[Jonny Craig musician]\nObservation: Jonathan Monroe Craig (born March 26, 1986) is an American-Canadian singer, widely recognised for his tenure as the former lead vocalist of renowned post-hardcore bands such as Dance Gavin Dance, Emarosa, and then Slaves, as well as his role as co-lead vocalist in the short-lived supergroup Isles & Glaciers. Presently, he serves as the lead vocalist for the American post-hardcore band Old Flame. As a solo artist, Craig has released two studio albums, two EPs, and a live album, garnering acclaim for his distinctive soul-based singing style.\nThought: Jonny Craig has been a member of several bands such as Dance Gavin Dance, Emarosa, Slaves, and Isles & Glaciers. Now, I need to search for the bands Pete Doherty has been a part of.\nAction: Search[Pete Doherty]\nObservation: Peter Doherty (born 12 March 1979) is an English musician, songwriter, actor, poet, writer and artist. He is best known for being co-frontman of The Libertines, which he formed with Carl Barât in 1997. His other musical projects are indie bands Babyshambles and Peter Doherty and the Puta Madres.\nThought: Pete Doherty has been a member of The Libertines, Babyshambles, and Peter Doherty and the Puta Madres. I need to compare the number of bands Jonny Craig and Pete Doherty have been a part of to determine who has been in more bands.\nAction: Finish[Jonny Craig]\nObservation: Answer is CORRECT\n'
    gt_insights = "0. When searching for information, if the initial search query does not return relevant results, try using different keywords or search terms to refine the search.\n1. When searching for information, consider searching for the creator or key figures related to the topic to gather more relevant details.\n"
    gt_experience = {
        "trajectory": [
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
                        external_tool_info={"search_result": "", "lookup_result": ""},
                        is_correct=True,
                    ),
                ],
                reflections=[],
            )
        ],
        "reflections": [],
    }
    gt_insight_memory = {
        "insights": [
            {
                "insight": "When searching for information, if the initial search query does not return relevant results, try using different keywords or search terms to refine the search.",
                "score": 2,
            },
            {
                "insight": "When searching for information, consider searching for the creator or key figures related to the topic to gather more relevant details.",
                "score": 2,
            },
        ]
    }

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

    gt_insights_insight_memory = [
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
    assert isinstance(out, ExpeLOutput)
    assert out.examples == gt_examples
    assert out.insights == gt_insights
    assert out.experience == gt_experience
    assert out.experience_memory
    assert out.insight_memory == gt_insight_memory
    assert len(agent.strategy.experience_memory.experiences) == 6
    assert agent.strategy.experience_memory.experiences[5]["question"] == question
    assert agent.strategy.experience_memory.experiences[5]["key"] == key
    assert agent.strategy.experience_memory.experiences[5]["reflections"] == []

    assert agent.strategy.insight_memory.insights == gt_insights_insight_memory
    assert len(agent.strategy.experience_memory.success_traj_docs) == 36
    assert agent.strategy.experience_memory.vectorstore
