"""Unit tests for ExpeL strategies."""

import joblib

from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.cog.expel.output import ExpeLGenerateOutput, ExpeLOutput
from agential.cog.expel.strategies.general import ExpeLGeneralStrategy
from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.reflexion.agent import (
    ReflexionReActAgent,
    ReflexionReActOutput,
)
from agential.cog.reflexion.output import (
    ReflexionReActReActStepOutput,
    ReflexionReActStepOutput,
)
from agential.cog.expel.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    EXPEL_REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.llm.llm import BaseLLM, MockLLM, Response


def test_init(expel_experiences_10_fake_path: str) -> None:
    """Test initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLGeneralStrategy(
        llm=llm, reflexion_react_agent=reflexion_react_agent
    )
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(strategy.experience_memory, ExpeLExperienceMemory)
    assert isinstance(strategy.insight_memory, ExpeLInsightMemory)
    assert strategy.success_batch_size == 8
    assert strategy.experience_memory.experiences == []
    assert not strategy.experience_memory.success_traj_docs
    assert not strategy.experience_memory.vectorstore
    assert not strategy.insight_memory.insights
    
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
   
    # Test with custom reflexion_react_agent (verify it overrides reflexion_react_kwargs)
    strategy = ExpeLGeneralStrategy(
        llm=llm,
        reflexion_react_agent=ReflexionReActAgent(
            llm=llm, benchmark="hotpotqa", max_steps=100
        ),
    )
    assert isinstance(strategy.reflexion_react_agent, ReflexionReActAgent)
    assert strategy.reflexion_react_agent.benchmark == "hotpotqa"
   
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
   

def test_generate() -> None:
    """Test generate."""
    question = "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?"
    key = "Oneida Limited"

    gt_new_experiences = ExpeLOutput(
        answer="Oneida Limited",
        total_prompt_tokens=80,
        total_completion_tokens=160,
        total_tokens=240,
        total_prompt_cost=0.00012,
        total_completion_cost=0.00031999999999999997,
        total_cost=0.00043999999999999996,
        total_prompt_time=4.0,
        total_time=0.5,
        additional_info=ExpeLGenerateOutput(
            examples="Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?\nThought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.\nAction 1: Search[Colorado orogeny]\nObservation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.\nThought 2: It does not mention the eastern sector. So I need to look up eastern sector.\nAction 2: Lookup[eastern sector]\nObservation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.\nThought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.\nAction 3: Search[High Plains]\nObservation 3: High Plains refers to one of two distinct land regions:\nThought 4: I need to instead search High Plains (United States).\nAction 4: Search[High Plains (United States)]\nObservation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]\nThought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.\nAction 5: Finish[1,800 to 7,000 ft]\n\n---\n\nQuestion: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?\nThought 1: I need to confirm who Matt Groening is and his connection to \"The Simpsons\".\nAction 1: Search[Matt Groening]\nObservation 1: Matt Groening is an American cartoonist, writer, producer, and animator, best known for creating the animated television series \"The Simpsons\".\nThought 2: Since Matt Groening created \"The Simpsons\" and named the character Milhouse, I need to find out whom he named Milhouse after.\nAction 2: Search[Milhouse inspiration]\nObservation 2: Matt Groening named the character Milhouse after Richard Milhous Nixon, the 37th President of the United States.\nThought 3: To ensure accuracy, I should verify the connection between Matt Groening naming Milhouse after Richard Nixon.\nAction 3: Search[Milhouse named after Richard Nixon]\nObservation 3: It is confirmed that Matt Groening named Milhouse after Richard Nixon, as stated in various sources discussing the origins of \"The Simpsons\" characters.\nThought 4: Allie Goertz's song about Milhouse connects back to Matt Groening's naming of the character after Richard Nixon.\nAction 4: Finish[Richard Nixon]\n\n---\n\nQuestion: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\nThought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.\nAction 1: Search[Adam Clayton Powell]\nObservation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].\nThought 2: To find the documentary, I can search Adam Clayton Powell (film).\nAction 2: Search[Adam Clayton Powell (film)]\nObservation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\nThe film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.\nThought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.\nAction 3: Finish[The Saimaa Gesture]\n\n---\n\nQuestion: What profession does Nicholas Ray and Elia Kazan have in common?\nThought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.\nAction 1: Search[Nicholas Ray]\nObservation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.\nThought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.\nAction 2: Search[Elia Kazan]\nObservation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.\nThought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.\nAction 3: Finish[director, screenwriter, actor]\n\n---\n\nQuestion: Which magazine was started first Arthur's Magazine or First for Women?\nThought 1: I need to search for the founding dates of Arthur's Magazine and First for Women to determine which was started first.\nAction 1: Search[founding date of Arthur's Magazine]\nObservation 1: Arthur's Magazine was founded in 1844.\nThought 2: Now I need to find the founding date of First for Women.\nAction 2: Search[founding date of First for Women]\nObservation 2: First for Women was founded in 1989.\nThought 3: I need to verify these dates to ensure their accuracy.\nAction 3: (Result 1 / 1) Lookup[Arthur's Magazine founding date, First for Women founding date]\nObservation 3: Arthur's Magazine is founded in 1844 and First for Women is founded in 1989.\nThought 4: Since Arthur's Magazine was founded in 1844 and First for Women was founded in 1989, Arthur's Magazine was started first.\nAction 4: Finish[Arthur's Magazine]\n\n---\n\nQuestion: Were Pavel Urysohn and Leonid Levin known for the same type of work?\nThought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.\nAction 1: Search[Pavel Urysohn]\nObservation 1: Pavel Samuilovich Urysohn (February 3, 1898 â\x80\x93 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.\nThought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.\nAction 2: Search[Leonid Levin]\nObservation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. \nThought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. \nAction 3: Finish[yes]",
            insights="",
            experience={
                "trajectory": ReflexionReActOutput(
                    answer="Oneida Limited",
                    total_prompt_tokens=80,
                    total_completion_tokens=160,
                    total_tokens=240,
                    total_prompt_cost=0.00012,
                    total_completion_cost=0.00031999999999999997,
                    total_cost=0.00043999999999999996,
                    total_prompt_time=4.0,
                    total_time=0.5,
                    additional_info=[
                        ReflexionReActStepOutput(
                            steps=[
                                ReflexionReActReActStepOutput(
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
                                    thought_response=Response(
                                        input_text="",
                                        output_text="I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.\nAction: Search[Pierrepont Noyes]\nObservation: Could not find [Pierrepont Noyes]. Similar: ['Noyes Academy', 'Penn Noyes', 'Pierrepont Noyes', 'Pierrepont Noyes House', 'Pierrepont Noyes Mausoleum', 'Pierrepont Noyes Memorial Foundation']\nThought: I need to search for the company directly instead.\nAction: Search[giant silverware company religious Utopian group]\nObservation: Could not find [giant silverware company religious Utopian group]. Similar: ['Oneida Limited', 'Oneida Community', 'Oneida Silver']\nThought: I should search for Oneida Limited.\nAction: Search[Oneida Limited]\nObservation: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silver-plated cutlery and tableware for the consumer and foodservice industries.\nThought: Oneida Limited is the giant silverware company that was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Finish[Oneida Limited]",
                                        prompt_tokens=10,
                                        completion_tokens=20,
                                        total_tokens=30,
                                        prompt_cost=1.5e-05,
                                        completion_cost=3.9999999999999996e-05,
                                        total_cost=5.4999999999999995e-05,
                                        prompt_time=0.5,
                                    ),
                                    action_response=Response(
                                        input_text="",
                                        output_text="Search[giant silverware company started as religious Utopian group Pierrepont Noyes]\nObservation: Could not find [giant silverware company started as religious Utopian group Pierrepont Noyes]. Similar: ['Oneida Community', 'Gorham Silver Manufacturing Company', 'F. B. Rogers Silver Company']\nThought: Let's search for Oneida Community and see if it was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Search[Oneida Community]\nObservation: The Oneida Community was a religious perfectionist group founded by John Humphrey Noyes in 1848 in Oneida, New York.\nThought: The Oneida Community was indeed a religious Utopian group, but it was founded by John Humphrey Noyes, not Pierrepont Noyes. Let's look up Pierrepont Noyes to see if he was involved with the Oneida Community.\nAction: Lookup[Pierrepont Noyes]\nObservation: (Result 1 / 1) Pierrepont Noyes was an American industrialist and president of Oneida Community from 1877 to 1881.\nThought: Pierrepont Noyes was indeed involved with the Oneida Community. So the answer to the question is Oneida Community.\nAction: Finish[Oneida Community]",
                                        prompt_tokens=10,
                                        completion_tokens=20,
                                        total_tokens=30,
                                        prompt_cost=1.5e-05,
                                        completion_cost=3.9999999999999996e-05,
                                        total_cost=5.4999999999999995e-05,
                                        prompt_time=0.5,
                                    ),
                                ),
                                ReflexionReActReActStepOutput(
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
                                    thought_response=Response(
                                        input_text="",
                                        output_text="The search query was too specific. I should try searching for the silverware company and then look for information about its history and founder.\nAction: Search[silverware company Pierrepont Noyes]\nObservation 2: Oneida Limited is an American manufacturer of tableware and cutlery. The company originated in a utopian community founded in the 19th century.\nThought: Oneida Limited was the silverware company started as a religious Utopian group. Now I need to find out if it was run by Pierrepont Noyes.\nAction: Lookup[Pierrepont Noyes]\nObservation 3: (Result 1 / 1) Pierrepont Noyes was a member of the Oneida Community, a religious communal society founded by his father John Humphrey Noyes.\nThought: Oneida Limited was indeed run by Pierrepont Noyes, so the answer is Oneida Limited.\nAction: Finish[Oneida Limited]",
                                        prompt_tokens=10,
                                        completion_tokens=20,
                                        total_tokens=30,
                                        prompt_cost=1.5e-05,
                                        completion_cost=3.9999999999999996e-05,
                                        total_cost=5.4999999999999995e-05,
                                        prompt_time=0.5,
                                    ),
                                    action_response=Response(
                                        input_text="",
                                        output_text="Search[silverware company Pierrepont Noyes]\nObservation 2: Could not find [silverware company Pierrepont Noyes]. Similar: []\nThought: I should try searching for Pierrepont Noyes to see if I can find any information about the company he was associated with.\nAction: Search[Pierrepont Noyes]\nObservation 3: Pierrepont Noyes (1868–1933) was an American industrialist, socialist, and founder of the Oneida Community.\nThought: Pierrepont Noyes was the founder of the Oneida Community. I should search for the history of the Oneida Community to see if it is the silverware company in question.\nAction: Search[Oneida Community silverware company]\nObservation 4: The Oneida Community was a religiously-centered communal society founded by John Humphrey Noyes in 1848 in Oneida, New York. The Oneida Community practiced communalism, complex marriage, male continence, mutual criticism, and asceticism.\nThought: The Oneida Community is the religious Utopian group mentioned in the question. I should now search for the connection between the Oneida Community and the silverware company.\nAction: Search[Oneida Community silverware company connection]\nObservation 5: Could not find [Oneida Community silverware company connection]. Similar: []\nThought: Since I couldn't find a direct connection, I should try searching for the history of the Oneida Community to see if it eventually led to the creation of a silverware company.\nAction: Search[Oneida Community history]\nObservation 6: The Oneida Community dissolved in 1881, and some of its members formed the Oneida Community Limited to continue the manufacture of silverware.\nThought: The Oneida Community eventually led to the formation of the Oneida Community Limited, a silverware company. The answer to the question is Oneida Community Limited.\nAction: Finish[Oneida Community Limited]",
                                        prompt_tokens=10,
                                        completion_tokens=20,
                                        total_tokens=30,
                                        prompt_cost=1.5e-05,
                                        completion_cost=3.9999999999999996e-05,
                                        total_cost=5.4999999999999995e-05,
                                        prompt_time=0.5,
                                    ),
                                ),
                                ReflexionReActReActStepOutput(
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
                                    thought_response=Response(
                                        input_text="",
                                        output_text="Pierrepont Noyes was the head of Oneida Limited, a silverware company. I need to confirm if Oneida Limited was indeed started as a religious Utopian group.\nAction: Search[Oneida Limited history]\nObservation 3: Oneida Limited was founded by John Humphfrey Noyes in the mid-19th century as a religious commune known as the Oneida Community.\nThought: Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.\nAction: Finish[Oneida Limited]",
                                        prompt_tokens=10,
                                        completion_tokens=20,
                                        total_tokens=30,
                                        prompt_cost=1.5e-05,
                                        completion_cost=3.9999999999999996e-05,
                                        total_cost=5.4999999999999995e-05,
                                        prompt_time=0.5,
                                    ),
                                    action_response=Response(
                                        input_text="",
                                        output_text="Search[Oneida Limited religious Utopian group]\nObservation 3: Could not find [Oneida Limited religious Utopian group]. Similar: ['Oneida, New York', 'Oneida Community', 'Oneida Silverware']\nThought: I can search for the history of Oneida Limited to confirm if it was started as a religious Utopian group.\nAction: Search[Oneida Limited history]\nObservation 4: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silverplated cutlery and tableware for the consumer and foodservice industries.\nThought: The search results do not provide information about the religious Utopian origins of Oneida Limited. I should look up the history of the Oneida Community to find the connection.\nAction: Lookup[Oneida Community]\nObservation 5: (Result 1 / 1) The Oneida Community was a religious communal society founded by John Humphrey Noyes in 1848 in Oneida, New York.\nThought: The Oneida Community was a religious communal society founded by John Humphrey Noyes in 1848. Pierrepont Noyes, who later ran Oneida Limited, was brought up in the Oneida Community. Therefore, the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes is Oneida Limited.\nAction: Finish[Oneida Limited]",
                                        prompt_tokens=10,
                                        completion_tokens=20,
                                        total_tokens=30,
                                        prompt_cost=1.5e-05,
                                        completion_cost=3.9999999999999996e-05,
                                        total_cost=5.4999999999999995e-05,
                                        prompt_time=0.5,
                                    ),
                                ),
                                ReflexionReActReActStepOutput(
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
                                    thought_response=Response(
                                        input_text="",
                                        output_text="Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.\nAction: Finish[Oneida Limited]",
                                        prompt_tokens=10,
                                        completion_tokens=20,
                                        total_tokens=30,
                                        prompt_cost=1.5e-05,
                                        completion_cost=3.9999999999999996e-05,
                                        total_cost=5.4999999999999995e-05,
                                        prompt_time=0.5,
                                    ),
                                    action_response=Response(
                                        input_text="",
                                        output_text="Finish[Oneida Limited]",
                                        prompt_tokens=10,
                                        completion_tokens=20,
                                        total_tokens=30,
                                        prompt_cost=1.5e-05,
                                        completion_cost=3.9999999999999996e-05,
                                        total_cost=5.4999999999999995e-05,
                                        prompt_time=0.5,
                                    ),
                                ),
                            ],
                            reflections=[],
                            reflection_response=None,
                        )
                    ],
                ),
                "reflections": [],
            },
            experience_memory={
                "experiences": [
                    {
                        "question": "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?",
                        "key": "Oneida Limited",
                        "trajectory": ReflexionReActOutput(
                            answer="Oneida Limited",
                            total_prompt_tokens=80,
                            total_completion_tokens=160,
                            total_tokens=240,
                            total_prompt_cost=0.00012,
                            total_completion_cost=0.00031999999999999997,
                            total_cost=0.00043999999999999996,
                            total_prompt_time=4.0,
                            total_time=0.5,
                            additional_info=[
                                ReflexionReActStepOutput(
                                    steps=[
                                        ReflexionReActReActStepOutput(
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
                                            thought_response=Response(
                                                input_text="",
                                                output_text="I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.\nAction: Search[Pierrepont Noyes]\nObservation: Could not find [Pierrepont Noyes]. Similar: ['Noyes Academy', 'Penn Noyes', 'Pierrepont Noyes', 'Pierrepont Noyes House', 'Pierrepont Noyes Mausoleum', 'Pierrepont Noyes Memorial Foundation']\nThought: I need to search for the company directly instead.\nAction: Search[giant silverware company religious Utopian group]\nObservation: Could not find [giant silverware company religious Utopian group]. Similar: ['Oneida Limited', 'Oneida Community', 'Oneida Silver']\nThought: I should search for Oneida Limited.\nAction: Search[Oneida Limited]\nObservation: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silver-plated cutlery and tableware for the consumer and foodservice industries.\nThought: Oneida Limited is the giant silverware company that was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Finish[Oneida Limited]",
                                                prompt_tokens=10,
                                                completion_tokens=20,
                                                total_tokens=30,
                                                prompt_cost=1.5e-05,
                                                completion_cost=3.9999999999999996e-05,
                                                total_cost=5.4999999999999995e-05,
                                                prompt_time=0.5,
                                            ),
                                            action_response=Response(
                                                input_text="",
                                                output_text="Search[giant silverware company started as religious Utopian group Pierrepont Noyes]\nObservation: Could not find [giant silverware company started as religious Utopian group Pierrepont Noyes]. Similar: ['Oneida Community', 'Gorham Silver Manufacturing Company', 'F. B. Rogers Silver Company']\nThought: Let's search for Oneida Community and see if it was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Search[Oneida Community]\nObservation: The Oneida Community was a religious perfectionist group founded by John Humphrey Noyes in 1848 in Oneida, New York.\nThought: The Oneida Community was indeed a religious Utopian group, but it was founded by John Humphrey Noyes, not Pierrepont Noyes. Let's look up Pierrepont Noyes to see if he was involved with the Oneida Community.\nAction: Lookup[Pierrepont Noyes]\nObservation: (Result 1 / 1) Pierrepont Noyes was an American industrialist and president of Oneida Community from 1877 to 1881.\nThought: Pierrepont Noyes was indeed involved with the Oneida Community. So the answer to the question is Oneida Community.\nAction: Finish[Oneida Community]",
                                                prompt_tokens=10,
                                                completion_tokens=20,
                                                total_tokens=30,
                                                prompt_cost=1.5e-05,
                                                completion_cost=3.9999999999999996e-05,
                                                total_cost=5.4999999999999995e-05,
                                                prompt_time=0.5,
                                            ),
                                        ),
                                        ReflexionReActReActStepOutput(
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
                                            thought_response=Response(
                                                input_text="",
                                                output_text="The search query was too specific. I should try searching for the silverware company and then look for information about its history and founder.\nAction: Search[silverware company Pierrepont Noyes]\nObservation 2: Oneida Limited is an American manufacturer of tableware and cutlery. The company originated in a utopian community founded in the 19th century.\nThought: Oneida Limited was the silverware company started as a religious Utopian group. Now I need to find out if it was run by Pierrepont Noyes.\nAction: Lookup[Pierrepont Noyes]\nObservation 3: (Result 1 / 1) Pierrepont Noyes was a member of the Oneida Community, a religious communal society founded by his father John Humphrey Noyes.\nThought: Oneida Limited was indeed run by Pierrepont Noyes, so the answer is Oneida Limited.\nAction: Finish[Oneida Limited]",
                                                prompt_tokens=10,
                                                completion_tokens=20,
                                                total_tokens=30,
                                                prompt_cost=1.5e-05,
                                                completion_cost=3.9999999999999996e-05,
                                                total_cost=5.4999999999999995e-05,
                                                prompt_time=0.5,
                                            ),
                                            action_response=Response(
                                                input_text="",
                                                output_text="Search[silverware company Pierrepont Noyes]\nObservation 2: Could not find [silverware company Pierrepont Noyes]. Similar: []\nThought: I should try searching for Pierrepont Noyes to see if I can find any information about the company he was associated with.\nAction: Search[Pierrepont Noyes]\nObservation 3: Pierrepont Noyes (1868–1933) was an American industrialist, socialist, and founder of the Oneida Community.\nThought: Pierrepont Noyes was the founder of the Oneida Community. I should search for the history of the Oneida Community to see if it is the silverware company in question.\nAction: Search[Oneida Community silverware company]\nObservation 4: The Oneida Community was a religiously-centered communal society founded by John Humphrey Noyes in 1848 in Oneida, New York. The Oneida Community practiced communalism, complex marriage, male continence, mutual criticism, and asceticism.\nThought: The Oneida Community is the religious Utopian group mentioned in the question. I should now search for the connection between the Oneida Community and the silverware company.\nAction: Search[Oneida Community silverware company connection]\nObservation 5: Could not find [Oneida Community silverware company connection]. Similar: []\nThought: Since I couldn't find a direct connection, I should try searching for the history of the Oneida Community to see if it eventually led to the creation of a silverware company.\nAction: Search[Oneida Community history]\nObservation 6: The Oneida Community dissolved in 1881, and some of its members formed the Oneida Community Limited to continue the manufacture of silverware.\nThought: The Oneida Community eventually led to the formation of the Oneida Community Limited, a silverware company. The answer to the question is Oneida Community Limited.\nAction: Finish[Oneida Community Limited]",
                                                prompt_tokens=10,
                                                completion_tokens=20,
                                                total_tokens=30,
                                                prompt_cost=1.5e-05,
                                                completion_cost=3.9999999999999996e-05,
                                                total_cost=5.4999999999999995e-05,
                                                prompt_time=0.5,
                                            ),
                                        ),
                                        ReflexionReActReActStepOutput(
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
                                            thought_response=Response(
                                                input_text="",
                                                output_text="Pierrepont Noyes was the head of Oneida Limited, a silverware company. I need to confirm if Oneida Limited was indeed started as a religious Utopian group.\nAction: Search[Oneida Limited history]\nObservation 3: Oneida Limited was founded by John Humphfrey Noyes in the mid-19th century as a religious commune known as the Oneida Community.\nThought: Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.\nAction: Finish[Oneida Limited]",
                                                prompt_tokens=10,
                                                completion_tokens=20,
                                                total_tokens=30,
                                                prompt_cost=1.5e-05,
                                                completion_cost=3.9999999999999996e-05,
                                                total_cost=5.4999999999999995e-05,
                                                prompt_time=0.5,
                                            ),
                                            action_response=Response(
                                                input_text="",
                                                output_text="Search[Oneida Limited religious Utopian group]\nObservation 3: Could not find [Oneida Limited religious Utopian group]. Similar: ['Oneida, New York', 'Oneida Community', 'Oneida Silverware']\nThought: I can search for the history of Oneida Limited to confirm if it was started as a religious Utopian group.\nAction: Search[Oneida Limited history]\nObservation 4: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silverplated cutlery and tableware for the consumer and foodservice industries.\nThought: The search results do not provide information about the religious Utopian origins of Oneida Limited. I should look up the history of the Oneida Community to find the connection.\nAction: Lookup[Oneida Community]\nObservation 5: (Result 1 / 1) The Oneida Community was a religious communal society founded by John Humphrey Noyes in 1848 in Oneida, New York.\nThought: The Oneida Community was a religious communal society founded by John Humphrey Noyes in 1848. Pierrepont Noyes, who later ran Oneida Limited, was brought up in the Oneida Community. Therefore, the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes is Oneida Limited.\nAction: Finish[Oneida Limited]",
                                                prompt_tokens=10,
                                                completion_tokens=20,
                                                total_tokens=30,
                                                prompt_cost=1.5e-05,
                                                completion_cost=3.9999999999999996e-05,
                                                total_cost=5.4999999999999995e-05,
                                                prompt_time=0.5,
                                            ),
                                        ),
                                        ReflexionReActReActStepOutput(
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
                                            thought_response=Response(
                                                input_text="",
                                                output_text="Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.\nAction: Finish[Oneida Limited]",
                                                prompt_tokens=10,
                                                completion_tokens=20,
                                                total_tokens=30,
                                                prompt_cost=1.5e-05,
                                                completion_cost=3.9999999999999996e-05,
                                                total_cost=5.4999999999999995e-05,
                                                prompt_time=0.5,
                                            ),
                                            action_response=Response(
                                                input_text="",
                                                output_text="Finish[Oneida Limited]",
                                                prompt_tokens=10,
                                                completion_tokens=20,
                                                total_tokens=30,
                                                prompt_cost=1.5e-05,
                                                completion_cost=3.9999999999999996e-05,
                                                total_cost=5.4999999999999995e-05,
                                                prompt_time=0.5,
                                            ),
                                        ),
                                    ],
                                    reflections=[],
                                    reflection_response=None,
                                )
                            ],
                        ),
                        "reflections": [],
                    }
                ]
            },
            insight_memory={"insights": []},
            compares_response=None,
            successes_response=None,
        ),
    )
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
    reflexion_react_agent = ReflexionReActAgent(
        llm=llm, benchmark="hotpotqa", testing=True
    )
    strategy = ExpeLGeneralStrategy(
        llm=llm, reflexion_react_agent=reflexion_react_agent, testing=True
    )
    strategy.reflexion_react_agent.strategy.docstore.search = lambda x: "Search result"
    strategy.reflexion_react_agent.strategy.docstore.lookup = lambda x: "Lookup result"
    new_experiences = strategy.generate(
        question=question,
        key=key,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=EXPEL_REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
        reflect_prompt=EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        reflect_strategy="reflexion",
        additional_keys={},
        reflect_additional_keys={},
        patience=1,
        use_dynamic_examples=False,
        extract_insights=False,
        k_docs=1,
        num_fewshots=1,
        max_fewshot_tokens=100,
        reranker_strategy=None,
        reset=False,
    )
    print(repr(new_experiences))
    assert new_experiences == gt_new_experiences
    assert len(strategy.experience_memory.success_traj_docs) == 13
    assert strategy.experience_memory.vectorstore


def test_get_dynamic_examples(expel_experiences_10_fake_path: str) -> None:
    """Test get_dynamic_examples."""
    question = "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?"
    experiences = joblib.load(expel_experiences_10_fake_path)

    gt_new_examples = "Which of Jonny Craig and Pete Doherty has been a member of more bands ?\nThought: I need to search for the band memberships of Jonny Craig and Pete Doherty to determine who has been in more bands.\nAction: Search[Jonny Craig band memberships]\nObservation: Could not find [Jonny Craig band memberships]. Similar: ['Coldplay', 'Gerard Butler', 'List of musical supergroups', 'List of musical acts from Western Australia', 'Johnny Carson', '2024 in American television', 'Garage rock', 'University of Leeds', 'North Korea', 'Grammy Award for Album of the Year']\nThought: I should try searching for Jonny Craig's band history instead.\nAction: Search[Jonny Craig band history]\nObservation: Jonathan Monroe Craig (born March 26, 1986) is an American-Canadian singer, widely recognised for his tenure as the former lead vocalist of renowned post-hardcore bands such as Dance Gavin Dance, Emarosa, and then Slaves, as well as his role as co-lead vocalist in the short-lived supergroup Isles & Glaciers. Presently, he serves as the lead vocalist for the American post-hardcore band Old Flame. As a solo artist, Craig has released two studio albums, two EPs, and a live album, garnering acclaim for his distinctive soul-based singing style.\nThought: Jonny Craig has been a member of Dance Gavin Dance, Emarosa, Slaves, Isle & Glaciers, and Old Flame. I should now search for Pete Doherty's band memberships.\nAction: Search[Pete Doherty band memberships]\nObservation: Could not find [Pete Doherty band memberships]. Similar: ['Rush (band)', 'List of musical supergroups', \"The Lovin' Spoonful\", 'Bert Jansch', 'John McEnroe', '2024 in American television', 'The Clancy Brothers', 'List of bands that played at Dagenham Roundhouse', 'List of people from the London Borough of Hackney', 'Ibiza']\nThought: I should try searching for Pete Doherty's band history instead.\nAction: Search[Pete Doherty band history]\nObservation: Peter Doherty (born 12 March 1979) is an English musician. He is best known for being co-frontman of The Libertines, which he formed with Carl Barât in 1997. His other musical projects are indie bands Babyshambles and Peter Doherty and the Puta Madres.\nThought: Jonny Craig has been a member of more bands than Pete Doherty.\nAction: Finish[Jonny Craig]\nObservation: Answer is CORRECT\n"
    gt_additional_keys = {"insights": ""}
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

    assert new_examples == gt_new_examples
    assert additional_keys == gt_additional_keys


def test_gather_experience(hotpotqa_distractor_sample_path: str) -> None:
    """Test gather_experience."""
    hotpotqa = joblib.load(hotpotqa_distractor_sample_path)

    gt_experience = [
        {
            "question": "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?",
            "key": "Oneida Limited",
            "trajectory": ReflexionReActOutput(
                answer="Oneida Limited",
                total_prompt_tokens=80,
                total_completion_tokens=160,
                total_tokens=240,
                total_prompt_cost=0.00012,
                total_completion_cost=0.00031999999999999997,
                total_cost=0.00043999999999999996,
                total_prompt_time=4.0,
                total_time=0.5,
                additional_info=[
                    ReflexionReActStepOutput(
                        steps=[
                            ReflexionReActReActStepOutput(
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
                                thought_response=Response(
                                    input_text="",
                                    output_text="I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.\nAction: Search[Pierrepont Noyes]\nObservation: Could not find [Pierrepont Noyes]. Similar: ['Noyes Academy', 'Penn Noyes', 'Pierrepont Noyes', 'Pierrepont Noyes House', 'Pierrepont Noyes Mausoleum', 'Pierrepont Noyes Memorial Foundation']\nThought: I need to search for the company directly instead.\nAction: Search[giant silverware company religious Utopian group]\nObservation: Could not find [giant silverware company religious Utopian group]. Similar: ['Oneida Limited', 'Oneida Community', 'Oneida Silver']\nThought: I should search for Oneida Limited.\nAction: Search[Oneida Limited]\nObservation: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silver-plated cutlery and tableware for the consumer and foodservice industries.\nThought: Oneida Limited is the giant silverware company that was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Finish[Oneida Limited]",
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                action_response=Response(
                                    input_text="",
                                    output_text="Search[giant silverware company started as religious Utopian group Pierrepont Noyes]\nObservation: Could not find [giant silverware company started as religious Utopian group Pierrepont Noyes]. Similar: ['Oneida Community', 'Gorham Silver Manufacturing Company', 'F. B. Rogers Silver Company']\nThought: Let's search for Oneida Community and see if it was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Search[Oneida Community]\nObservation: The Oneida Community was a religious perfectionist group founded by John Humphrey Noyes in 1848 in Oneida, New York.\nThought: The Oneida Community was indeed a religious Utopian group, but it was founded by John Humphrey Noyes, not Pierrepont Noyes. Let's look up Pierrepont Noyes to see if he was involved with the Oneida Community.\nAction: Lookup[Pierrepont Noyes]\nObservation: (Result 1 / 1) Pierrepont Noyes was an American industrialist and president of Oneida Community from 1877 to 1881.\nThought: Pierrepont Noyes was indeed involved with the Oneida Community. So the answer to the question is Oneida Community.\nAction: Finish[Oneida Community]",
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ),
                            ReflexionReActReActStepOutput(
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
                                thought_response=Response(
                                    input_text="",
                                    output_text="The search query was too specific. I should try searching for the silverware company and then look for information about its history and founder.\nAction: Search[silverware company Pierrepont Noyes]\nObservation 2: Oneida Limited is an American manufacturer of tableware and cutlery. The company originated in a utopian community founded in the 19th century.\nThought: Oneida Limited was the silverware company started as a religious Utopian group. Now I need to find out if it was run by Pierrepont Noyes.\nAction: Lookup[Pierrepont Noyes]\nObservation 3: (Result 1 / 1) Pierrepont Noyes was a member of the Oneida Community, a religious communal society founded by his father John Humphrey Noyes.\nThought: Oneida Limited was indeed run by Pierrepont Noyes, so the answer is Oneida Limited.\nAction: Finish[Oneida Limited]",
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                action_response=Response(
                                    input_text="",
                                    output_text="Search[silverware company Pierrepont Noyes]\nObservation 2: Could not find [silverware company Pierrepont Noyes]. Similar: []\nThought: I should try searching for Pierrepont Noyes to see if I can find any information about the company he was associated with.\nAction: Search[Pierrepont Noyes]\nObservation 3: Pierrepont Noyes (1868–1933) was an American industrialist, socialist, and founder of the Oneida Community.\nThought: Pierrepont Noyes was the founder of the Oneida Community. I should search for the history of the Oneida Community to see if it is the silverware company in question.\nAction: Search[Oneida Community silverware company]\nObservation 4: The Oneida Community was a religiously-centered communal society founded by John Humphrey Noyes in 1848 in Oneida, New York. The Oneida Community practiced communalism, complex marriage, male continence, mutual criticism, and asceticism.\nThought: The Oneida Community is the religious Utopian group mentioned in the question. I should now search for the connection between the Oneida Community and the silverware company.\nAction: Search[Oneida Community silverware company connection]\nObservation 5: Could not find [Oneida Community silverware company connection]. Similar: []\nThought: Since I couldn't find a direct connection, I should try searching for the history of the Oneida Community to see if it eventually led to the creation of a silverware company.\nAction: Search[Oneida Community history]\nObservation 6: The Oneida Community dissolved in 1881, and some of its members formed the Oneida Community Limited to continue the manufacture of silverware.\nThought: The Oneida Community eventually led to the formation of the Oneida Community Limited, a silverware company. The answer to the question is Oneida Community Limited.\nAction: Finish[Oneida Community Limited]",
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ),
                            ReflexionReActReActStepOutput(
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
                                thought_response=Response(
                                    input_text="",
                                    output_text="Pierrepont Noyes was the head of Oneida Limited, a silverware company. I need to confirm if Oneida Limited was indeed started as a religious Utopian group.\nAction: Search[Oneida Limited history]\nObservation 3: Oneida Limited was founded by John Humphfrey Noyes in the mid-19th century as a religious commune known as the Oneida Community.\nThought: Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.\nAction: Finish[Oneida Limited]",
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                action_response=Response(
                                    input_text="",
                                    output_text="Search[Oneida Limited religious Utopian group]\nObservation 3: Could not find [Oneida Limited religious Utopian group]. Similar: ['Oneida, New York', 'Oneida Community', 'Oneida Silverware']\nThought: I can search for the history of Oneida Limited to confirm if it was started as a religious Utopian group.\nAction: Search[Oneida Limited history]\nObservation 4: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silverplated cutlery and tableware for the consumer and foodservice industries.\nThought: The search results do not provide information about the religious Utopian origins of Oneida Limited. I should look up the history of the Oneida Community to find the connection.\nAction: Lookup[Oneida Community]\nObservation 5: (Result 1 / 1) The Oneida Community was a religious communal society founded by John Humphrey Noyes in 1848 in Oneida, New York.\nThought: The Oneida Community was a religious communal society founded by John Humphrey Noyes in 1848. Pierrepont Noyes, who later ran Oneida Limited, was brought up in the Oneida Community. Therefore, the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes is Oneida Limited.\nAction: Finish[Oneida Limited]",
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ),
                            ReflexionReActReActStepOutput(
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
                                thought_response=Response(
                                    input_text="",
                                    output_text="Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.\nAction: Finish[Oneida Limited]",
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                action_response=Response(
                                    input_text="",
                                    output_text="Finish[Oneida Limited]",
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ),
                        ],
                        reflections=[],
                        reflection_response=None,
                    )
                ],
            ),
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
    reflexion_react_agent = ReflexionReActAgent(
        llm=llm, benchmark="hotpotqa", testing=True
    )
    strategy = ExpeLGeneralStrategy(
        llm=llm, reflexion_react_agent=reflexion_react_agent, testing=True
    )
    strategy.reflexion_react_agent.strategy.docstore.search = lambda x: "Search result"
    strategy.reflexion_react_agent.strategy.docstore.lookup = lambda x: "Lookup result"
    new_experiences = strategy.gather_experience(
        questions=hotpotqa.question.values[-1:],
        keys=hotpotqa.answer.values[-1:],
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=EXPEL_REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        reflect_examples=HOTPOTQA_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
        reflect_prompt=EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        reflect_strategy="reflexion",
        additional_keys=[{"insights": ""}],
        reflect_additional_keys=[{}],
        patience=1,
    )
    assert new_experiences == gt_experience
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
    strategy = ExpeLGeneralStrategy(
        llm=llm, reflexion_react_agent=reflexion_react_agent
    )

    compares_response, successes_response = strategy.extract_insights(experiences)
    assert strategy.insight_memory.insights == gt_insights
    assert compares_response == [
        Response(
            input_text="",
            output_text="ADD 11: Always try multiple variations of search terms when looking for specific information.\nADD 12: If unable to find relevant information through initial searches, consider looking for official announcements or press releases from the company.\nREMOVE 3: Always use the exact search term provided in the question, do not try variations.\nEDIT 7: Make sure to exhaust all possible search options before concluding that the information is unavailable.",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        )
    ]

    assert successes_response == []


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


def test_reset() -> None:
    """Test reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    reflexion_react_agent = ReflexionReActAgent(llm=llm, benchmark="hotpotqa")
    strategy = ExpeLGeneralStrategy(llm=llm, reflexion_react_agent=reflexion_react_agent)

    strategy.experience_memory.experiences = "dog"
    strategy.insight_memory.insights = ["turtle"]
    strategy.reset()
    assert strategy.experience_memory.experiences == []
    assert strategy.insight_memory.insights == []