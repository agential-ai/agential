"""Unit tests for LATS QA strategies."""

from langchain_community.docstore.wikipedia import Wikipedia

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.lats.node import Node
from agential.cog.lats.output import (
    LATSEvaluateResponse,
    LATSGenerateResponse,
    LATSReActStepOutput,
    LATSSimulationOutput,
    LATSSimulationResponse,
    LATSSimulationStepResponse,
    LATSStepOutput,
)
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
)
from agential.llm.llm import MockLLM, Response
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


def test_generate() -> None:
    """Test the generate method."""
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

    gt_additional_info = ""

    gt_value_cache = {
        "\nThought 1: I need to search for VIVA Media AG and find out its new acronym after changing its name in 2004.\nAction 1: Search[VIVA Media AG]\nObservation 1: Badr Hari is the best kick boxer in the world.::": "I need to search for VIVA Media AG to find out what their new acronym stands for after changing their name in 2004.\nAction 1: Search[VIVA Media AG]\nObservation 1: VIVA Media AG was a German media company that operated several television channels.\nThought 2: Since the search did not provide the information I need, I should look for the new acronym after their name change in 2004.\nAction 2: Lookup[new acronym'The trajectory is incorrect because the search query did not yield results for VIVA Media AG. This indicates that the initial search was not specific enough or possibly the entity has limited online presence. Future attempts should consider refining the search terms or looking for alternative sources of information.\nCorrectness score: 2",
        "\nThought 1: I need to search for VIVA Media AG to find out what their new acronym stands for after changing their name in 2004.\nAction 1: Search[VIVA Media AG]\nObservation 1: Badr Hari is the best kick boxer in the world.::": "This trajectory is incorrect because the search did not yield results for VIVA Media AG. The action taken was appropriate, but the lack of relevant information hindered progress towards finding the acronym. In the future, it would be beneficial to explore alternative sources or search for related entities that might provide the necessary information.\nCorrectness score: 2",
    }

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

    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = LATSQAStrategy(
        llm=llm,
        n_samples=2,
        max_reflections=4,
        depth_limit=5,
        max_unique=5,
        cache_values=True,
        testing=True,
    )
    strategy.docstore.search = (
        lambda x: "Badr Hari is the best kick boxer in the world."
    )

    out = strategy.generate(
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

    assert out.answer.to_dict() == gt_terminal_node_state
    assert out.total_completion_cost == 0.0012
    assert out.total_completion_tokens == 600
    assert out.total_prompt_cost == 0.00045000000000000015
    assert out.total_prompt_tokens == 300
    assert out.total_tokens == 900
    assert out.total_cost == 0.0016500000000000002
    assert out.total_prompt_time == 15.0
    assert out.total_time == 0.5
    print(repr(out.additional_info))
    assert out.additional_info == gt_additional_info

    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == []
    assert strategy.value_cache == gt_value_cache
    assert strategy.root.to_dict() == {
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

    gt_thought_model_responses = [
        "I need to search for the name of the kick boxer who was once considered the best but has been involved in controversies and crimes",
        "I need to search for the best kickboxer who has been involved in controversies and crimes of violence",
        "I need to search for the name of the kick boxer who was once considered the best in the world and has been involved in controversies",
        "I need to search for the best kick boxer who has been involved in controversies relating to unsportsmanlike conduct and crimes of violence outside the ring",
        "I need to search for the kickboxer who was once considered the best in the world but has been involved in controversies",
    ]

    gt_action_model_responses = [
        "Search[best kick boxer controversies crimes]",
        "Search[best kick boxer controversies crimes]\nObservation 0: No exact matches found",
        "Search[best kick boxer controversies]\nObservation 0: Could not find [best kick boxer controversies]",
        "Search[best kick boxer controversies violence]\nObservation 0: Could not find [best kick boxer controversies violence]",
        "Search[best kickboxer controversies]\nObservation 0: The search results show multiple kickboxers who have been involved in controversies",
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
    strategy.docstore.search = (
        lambda x: "Badr Hari is the best kick boxer in the world."
    )

    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'
    key = "Badr Hari"

    root = strategy.initialize()

    (children_nodes, generate_metrics) = strategy.generate_children_nodes(
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
    for gt_state, node, t, a, r, gt_t, gt_a in zip(
        gt_states,
        children_nodes,
        generate_metrics.thoughts_metrics,
        generate_metrics.actions_metrics,
        generate_metrics.reflections_metrics,
        gt_thought_model_responses,
        gt_action_model_responses,
    ):
        assert node.state == gt_state
        assert node.depth == 1
        assert node.reward == 0
        assert node.value == 0
        assert node.is_terminal is False
        assert node.visits == 0
        assert t == gt_t
        assert a == gt_a

    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == []
    assert strategy.value_cache == {}
    assert strategy.root.to_dict() == {
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
    }

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

    gt_thought_model_responses = [
        "I need to search for the best kick boxer in the world who has been involved in controversies related to unsportsmanlike conduct and crimes of violence outside the ring",
        "I need to search for the best kick boxer in the world and then look into his controversies related to unsportsmanlike conduct and crimes of violence",
        "I need to search for the best kick boxer in the world who has been involved in controversies related to unsportsmanlike conduct and violence outside of the ring",
        "I need to search for the best kickboxer in the world who has been involved in controversies regarding unsportsmanlike conduct and crimes of violence outside the ring",
        "I need to search for the best kick boxer in the world and his controversies regarding unsportsmanlike conducts and crimes of violence",
    ]

    gt_action_model_responses = [
        "Search[best kickboxer controversies violence]\nObservation 1: Could not find [best kickboxer controversies violence]",
        "Search[best kick boxer in the world]\nObservation 1: There have been several renowned kickboxers throughout history, such as Buakaw Banchamek, Ernesto Hoost, and Ramon Dekkers",
        "Search[best kick boxer in the world controversies]\nObservation 1: Could not find [best kick boxer in the world controversies]",
        "Search[best kickboxer controversies]\nObservation 1: Could not find [best kickboxer controversies]",
        "Search[best kick boxer in the world controversies]\nObservation 1: Could not find [best kick boxer in the world controversies]",
    ]

    gt_failed_trajectories = [
        {"trajectory": "Failed trajectory 1", "final_answer": "Incorrect answer 1"},
        {"trajectory": "Failed trajectory 2", "final_answer": "Incorrect answer 2"},
        {"trajectory": "Failed trajectory 1", "final_answer": "Incorrect answer 1"},
    ]
    gt_reflection_map = [
        {
            "trajectory": "Failed trajectory 1",
            "reflection": "My reasoning for this question failed because I did not narrow down the search to focus on kick boxers and instead ended up with unrelated information",
        },
        {
            "trajectory": "Failed trajectory 2",
            "reflection": "My reasoning failed because I did not focus on gathering specific information related to the individual's kickboxing career and controversies, leading to an incorrect answer",
        },
    ]

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
    (children_nodes, generate_metrics) = strategy.generate_children_nodes(
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
    for gt_state, node, t, a, r, gt_t, gt_a in zip(
        gt_states,
        children_nodes,
        generate_metrics.thoughts_metrics,
        generate_metrics.actions_metrics,
        generate_metrics.reflections_metrics,
        gt_thought_model_responses,
        gt_action_model_responses,
    ):
        assert node.state == gt_state
        assert node.depth == 1
        assert node.reward == 0
        assert node.value == 0
        assert node.is_terminal is False
        assert node.visits == 0
        assert t == Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        )
        assert a == Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        )
    assert strategy.failed_trajectories == gt_failed_trajectories
    assert strategy.reflection_map == gt_reflection_map
    assert strategy.value_cache == {}
    assert strategy.root.to_dict() == {
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
    }

    # Test case with a terminal child node (reward 0)
    responses = [
        "I think the answer is Mike Tyson.",
        "Finish[Mike Tyson]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = LATSQAStrategy(llm=llm, n_samples=1)

    root = strategy.initialize()
    (children_nodes, generate_metrics) = strategy.generate_children_nodes(
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
    assert len(children_nodes) == 1
    assert children_nodes[0].state.thought == "I think the answer is Mike Tyson."
    assert children_nodes[0].state.action_type == "Finish"
    assert children_nodes[0].state.query == "Mike Tyson"
    assert children_nodes[0].is_terminal
    assert children_nodes[0].reward == 0

    assert len(generate_metrics.thoughts_metrics) == 1

    assert generate_metrics.thoughts_metrics[0] == Response(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )
    assert generate_metrics.actions_metrics[0] == Response(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )
    assert len(generate_metrics.actions_metrics) == 1
    assert len(generate_metrics.reflections_metrics) == 0
    assert strategy.failed_trajectories == [
        {
            "trajectory": "\nThought 1: I think the answer is Mike Tyson.\nAction 1: Finish[Mike Tyson]\nObservation 1: Answer is INCORRECT",
            "final_answer": "mike tyson",
        }
    ]
    assert strategy.reflection_map == []
    assert strategy.value_cache == {}
    assert strategy.root == root


def test_generate_action() -> None:
    """Test the generate_action method."""
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

    trajectory, action_type, query, action_response = strategy.generate_action(
        question,
        examples,
        trajectory,
        reflections,
        depth,
        prompt,
        additional_keys,
    )
    assert (
        trajectory
        == "Thought 2: I should search for information about the capital of France.\nAction 2: Search[capital of France]"
    )
    assert action_type == "Search"
    assert query == "capital of France"
    assert action_response == Response(input_text='', output_text='Search[capital of France]', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)

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

    values, values_evaluation_response = strategy.evaluate_node(
        root, question, examples, prompt, {}
    )

    assert len(values) == 2
    assert values == [
        {"explanation": "Good trajectory.", "value": 0.8},
        {"explanation": "", "value": -10000000000.0},
    ]

    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == [
        {
            "trajectory": "Failed trajectory",
            "reflection": "This trajectory failed because...",
        }
    ]
    assert strategy.value_cache == {
        "\nThought 1: Child 1::Question: What is the capital of France?\nFailed trajectory\n\nExplanation: This trajectory is incorrect as This trajectory failed because...\nCorrectness score: 1": "Explanation: Good trajectory. Correctness score: 8"
    }
    assert strategy.root == root

    assert child1.value == 0.8
    assert child2.value == 0  # Terminal node, value not updated.

    expected_value_response = [
        Response(input_text='', output_text='Explanation: Good trajectory. Correctness score: 8', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5),
        None,
    ]

    for i, value_met in zip(
        values_evaluation_response.values_response, expected_value_response
    ):
        assert i == value_met

    # Test caching.
    strategy.cache_values = True
    cached_values, values_evaluation_response = strategy.evaluate_node(
        root, question, examples, prompt, {}
    )
    assert cached_values == values
    assert values_evaluation_response.values_response == [None, None]

    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == [
        {
            "trajectory": "Failed trajectory",
            "reflection": "This trajectory failed because...",
        }
    ]
    assert strategy.value_cache == {
        "\nThought 1: Child 1::Question: What is the capital of France?\nFailed trajectory\n\nExplanation: This trajectory is incorrect as This trajectory failed because...\nCorrectness score: 1": "Explanation: Good trajectory. Correctness score: 8"
    }
    assert strategy.root == root

    # Test with empty reflection_map.
    strategy.reflection_map = []
    empty_reflection_values, values_evaluation_response = strategy.evaluate_node(
        root, question, examples, prompt, {}
    )
    assert values_evaluation_response.values_response == [Response(input_text='', output_text='Explanation: Good trajectory. Correctness score: 8', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5), None]

    assert empty_reflection_values == values

    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == []
    assert strategy.value_cache == {
        "\nThought 1: Child 1::Question: What is the capital of France?\nFailed trajectory\n\nExplanation: This trajectory is incorrect as This trajectory failed because...\nCorrectness score: 1": "Explanation: Good trajectory. Correctness score: 8",
        "\nThought 1: Child 1::": "Explanation: Good trajectory. Correctness score: 8",
    }
    assert strategy.root == root


def test_simulate_node() -> None:
    """Test the simulate_node method."""
    expected_current_nodes = [
        {
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
        {
            "state": LATSReActStepOutput(
                thought="I need to search for the capital of France",
                action_type="Search",
                query="capital of France",
                observation="Badr Hari is the best kick boxer in the world.",
                answer="",
                external_tool_info={
                    "search_result": "Badr Hari is the best kick boxer in the world.",
                    "lookup_result": "",
                },
            ),
            "visits": 0,
            "value": 0,
            "depth": 1,
            "is_terminal": False,
            "reward": 0,
        },
    ]

    expected_simulation_children_nodes = [
        [
            {
                "state": LATSReActStepOutput(
                    thought="I need to search for the capital of France",
                    action_type="Search",
                    query="capital of France",
                    observation="Badr Hari is the best kick boxer in the world.",
                    answer="",
                    external_tool_info={
                        "search_result": "Badr Hari is the best kick boxer in the world.",
                        "lookup_result": "",
                    },
                ),
                "visits": 0,
                "value": 0,
                "depth": 1,
                "is_terminal": False,
                "reward": 0,
            }
        ],
    ]

    expected_simulation_values = [
        [
            {"explanation": "", "value": -10000000000.0},
            {"explanation": "", "value": -10000000000.0},
        ],
        [
            {"explanation": "Explanation not found", "value": 0.0},
            {"explanation": "Explanation not found", "value": 0.0},
        ],
        [
            {"explanation": "Explanation not found", "value": 0.0},
            {"explanation": "Explanation not found", "value": 0.0},
        ],
    ]

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

    strategy = LATSQAStrategy(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), depth_limit=3, n_samples=2
    )
    strategy.docstore.search = (
        lambda x: "Badr Hari is the best kick boxer in the world."
    )
    root_node = strategy.initialize()

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

    (
        simulation_reward,
        simulation_terminal_node,
        simulation_current_nodes,
        simulation_children_nodes,
        simulation_values,
        simulation_response,
    ) = strategy.simulate_node(
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

    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == []
    assert strategy.value_cache == {}
    assert strategy.root == root_node

    assert simulation_reward == -1.0

    assert simulation_terminal_node.to_dict() == {
        "state": LATSReActStepOutput(
            thought="This trajectory is incorrect as it did not provide any relevant information regarding the capital of France",
            action_type="",
            query="",
            observation="Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].",
            answer="",
            external_tool_info={"search_result": "", "lookup_result": ""},
        ),
        "visits": 0,
        "value": 0,
        "depth": 3,
        "is_terminal": False,
        "reward": 0,
    }

    for expected_node, node in zip(expected_current_nodes, simulation_current_nodes):
        assert node.to_dict() == expected_node

    for node_list, expected_node_list in zip(
        simulation_children_nodes, expected_simulation_children_nodes
    ):
        for node, expected_node in zip(node_list, expected_node_list):
            assert node.to_dict() == expected_node

    # Flatten the list using itertools.chain
    for i in simulation_response.simulation_step_response:
        assert i.generate_response.thoughts_response == [
            Response(
                input_text="",
                output_text="",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                input_text="",
                output_text="",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ]
        assert i.generate_response.actions_response == [
            Response(
                input_text="",
                output_text="",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                input_text="",
                output_text="",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ]
        assert i.generate_response.reflections_response == []
        assert i.evaluate_response.values_response == [None, None] or [
            Response(
                input_text="",
                output_text="",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                input_text="",
                output_text="",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ]

    assert simulation_values == expected_simulation_values


def test_expand_node() -> None:
    """Test the expand_node method."""
    gt_thought_model_responses = [
        Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        ),
        Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        ),
        Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        ),
        Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        ),
        Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        ),
    ]
    gt_action_model_responses = [
        Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        ),
        Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        ),
        Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        ),
        Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        ),
        Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
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
    strategy.docstore.search = (
        lambda x: "Badr Hari is the best kick boxer in the world."
    )
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'
    key = "Badr Hari"

    root = strategy.initialize()

    (children_nodes, generate_metrics) = strategy.expand_node(
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

    expected_nodes = [
        {
            "state": LATSReActStepOutput(
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
            "visits": 0,
            "value": 0,
            "depth": 1,
            "is_terminal": False,
            "reward": 0,
        }
    ]

    for expected_node, node in zip(expected_nodes, children_nodes):
        assert node.to_dict() == expected_node
    assert len(children_nodes) == 5

    assert generate_metrics.reflections_metrics == []
    for t, a, gt_t, gt_a in zip(
        generate_metrics.thoughts_metrics,
        generate_metrics.actions_metrics,
        gt_thought_model_responses,
        gt_action_model_responses,
    ):

        assert t == gt_t
        assert a == gt_a

    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == []
    assert strategy.value_cache == {}
    assert strategy.root == root


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
