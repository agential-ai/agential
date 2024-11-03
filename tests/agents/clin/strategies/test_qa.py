"""Test the CLIN QA strategy."""

import tiktoken

from agential.agents.clin.output import CLINOutput, CLINReActStepOutput, CLINStepOutput
from agential.agents.clin.prompts import (
    CLIN_ADAPT_META_SUMMARY_SYSTEM,
    CLIN_ADAPT_SUMMARY_SYSTEM,
    CLIN_INSTRUCTION_HOTPOTQA,
    CLIN_META_SUMMARY_INSTRUCTION_HOTPOTQA,
    CLIN_SUMMARY_INSTRUCTION_FEVER,
    CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
)
from agential.agents.clin.strategies.qa import CLINFEVERStrategy, CLINQAStrategy
from agential.core.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.core.llm import MockLLM, Response
from agential.utils.docstore import DocstoreExplorer


def test_init() -> None:
    """Test CLIN QA strategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = CLINQAStrategy(llm=llm, memory=None)
    assert strategy.max_trials == 3
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 5000
    assert strategy.enc == tiktoken.encoding_for_model("gpt-3.5-turbo")
    assert strategy.testing is False
    assert isinstance(strategy.docstore, DocstoreExplorer)


def test_generate() -> None:
    """Test CLIN QA strategy generate."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    key = "Gesellschaft mit beschränkter Haftung"

    gt_out = CLINOutput(
        answer="Direct search on VIVA Media AG's official website or press releases",
        total_prompt_tokens=110,
        total_completion_tokens=220,
        total_tokens=330,
        total_prompt_cost=0.00016500000000000003,
        total_completion_cost=0.00043999999999999996,
        total_cost=0.000605,
        total_prompt_time=5.5,
        total_time=0.5,
        additional_info=[
            CLINStepOutput(
                steps=[
                    CLINReActStepOutput(
                        thought="I need to search VIVA Media AG, find out their new name after 2004, and then find what the acronym stands for.",
                        action_type="Search",
                        query="VIVA Media AG",
                        observation="Buakaw Banchamek has faced several controversies and legal issues.",
                        answer="",
                        external_tool_info={
                            "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                            "lookup_result": "",
                        },
                        is_correct=False,
                        thought_response=Response(
                            input_text="",
                            output_text="I need to search VIVA Media AG, find out their new name after 2004, and then find what the acronym stands for.\nAction 1: Search[VIVA Media AG]\nObservation 1: Could not find [VIVA Media AG]. Similar: ['Viva Media', 'Viva Media', 'Viva Media GmbH', 'Viva Media (record label)'].\nThought 2: I should search for Viva Media GmbH instead to find information about VIVA Media AG.\nAction 2: Search[Viva Media GmbH]\nObservation 2: Viva Media GmbH is a German video game company that publishes and distributes PC video games in Germany.\nThought 3: I should search for the name change of VIVA Media AG and find out the acronym.\nAction 3: Search[VIVA Media AG name change]\nObservation 3: Could not find the name change information for VIVA Media AG. Similar: ['List of asset management firms', 'List of management consulting firms', 'List of semiconductor fabrication plants', 'List of user research practices', 'List of university presses'].\nThought 4: It seems the information about the name change is not readily available. I should attempt to find the acronym through related searches.\nAction 4: Search[Viva Media GmbH acronym]\nObservation 4: Could not find the acronym for Viva Media GmbH. Similar: ['Viva Media GmbH revenue', 'Viva Media GmbH history', 'Viva Media GmbH founders', 'Viva Media GmbH products'].\nThought 5: I may not be able to find the acronym for VIVA Media AG through direct searches. I should try to find any connections between VIVA Media AG and its new name to infer the acronym.\nAction 5: Finish[unable to find the answer]",
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
                            output_text="Search[VIVA Media AG]\nObservation 1: VIVA Media AG was a German media company founded in 1993 and acquired by Viacom in 2004.\nThought 2: Viacom acquired VIVA Media AG in 2004, so I need to find out the new name of the company after the acquisition.\nAction 2: Search[VIVA Media AG name change 2004]\nObservation 2: Could not find [VIVA Media AG name change 2004]. Similar: ['Name change', 'Corporate name changes']\nThought 3: I should instead search for Viacom's acquisition of VIVA Media AG in 2004 to find out the new name.\nAction 3: Search[Viacom acquisition of VIVA Media AG 2004]\nObservation 3: Viacom acquired VIVA Media AG in 2004 and renamed it MTV Central and Eastern Europe.\nThought 4: The new acronym stands for MTV. I need to confirm this.\nAction 4: Lookup[new acronym stands for]\nObservation 4: The new acronym for VIVA Media AG after the acquisition was MTV, which stands for Music Television.\nThought 5: The new acronym for VIVA Media AG after the acquisition in 2004 was MTV, which stands for Music Television.\nAction 5: Finish[MTV - Music Television]",
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    CLINReActStepOutput(
                        thought="This is not the information I needed. I will search again for VIVA Media AG to find their new name after 2004.",
                        action_type="Search",
                        query="VIVA Media AG new name",
                        observation="Buakaw Banchamek has faced several controversies and legal issues.",
                        answer="",
                        external_tool_info={
                            "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                            "lookup_result": "",
                        },
                        is_correct=False,
                        thought_response=Response(
                            input_text="",
                            output_text='This is not the information I needed. I will search again for VIVA Media AG to find their new name after 2004.\nAction 2: Search[VIVA Media AG]\nObservation 2: VIVA Media AG was a German media company that rebranded as VIVACOM in 2004.\nThought 3: Now I need to find out what VIVACOM stands for, which is the new acronym for VIVA Media AG.\nAction 3: Search[VIVACOM acronym meaning]\nObservation 3: VIVACOM stands for "Viva Communications".\nThought 4: The new acronym for VIVA Media AG after 2004 is "Viva Communications".\nAction 4: Finish[Viva Communications]',
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
                            output_text="Search[VIVA Media AG new name]\nObservation 2: VIVA Media AG changed its name to VMS AG in 2004.\nThought 3: Now that I know their new name is VMS AG, I need to find out what the acronym stands for.\nAction 3: Search[VMS AG acronym]\nObservation 3: There is no specific information available directly related to the acronym of VMS AG. I should try looking up VMS AG on another platform or source.\nAction 4: Lookup[acronym VMS AG]\nObservation 4: The acronym VMS AG stands for Video Music Station Aktiengesellschaft.\nThought 5: The new acronym for VIVA Media AG, which changed its name to VMS AG in 2004, stands for Video Music Station Aktiengesellschaft.\nAction 5: Finish[Video Music Station Aktiengesellschaft]",
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    CLINReActStepOutput(
                        thought="I will search for VIVA Media AG's name change in 2004 to find what the new acronym stands for.",
                        action_type="Search",
                        query="VIVA Media AG name change 2004",
                        observation="Buakaw Banchamek has faced several controversies and legal issues.",
                        answer="",
                        external_tool_info={
                            "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                            "lookup_result": "",
                        },
                        is_correct=False,
                        thought_response=Response(
                            input_text="",
                            output_text="I will search for VIVA Media AG's name change in 2004 to find what the new acronym stands for.\nAction 3: Search[VIVA Media AG name change 2004]\nObservation 3: The new name of VIVA Media AG after 2004 is VIVA Entertainment.\nThought 4: Now I need to find out what the acronym VIVA stands for.\nAction 4: Search[VIVA Entertainment acronym]\nObservation 4: VIVA acronym stands for Videoverwertung in Augsburg. \nThought 5: VIVA acronym stands for Videoverwertung in Augsburg. So the answer is Videoverwertung in Augsburg.\nAction 5: Finish[Videoverwertung in Augsburg]",
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
                            output_text="Search[VIVA Media AG name change 2004]",
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    CLINReActStepOutput(
                        thought="I need to refine my search terms to find the relevant information about VIVA Media AG's name change in 2004 and the acronym it stands for.",
                        action_type="Search",
                        query="VIVA Media AG rebrand 2004 acronym",
                        observation="Buakaw Banchamek has faced several controversies and legal issues.",
                        answer="",
                        external_tool_info={
                            "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                            "lookup_result": "",
                        },
                        is_correct=False,
                        thought_response=Response(
                            input_text="",
                            output_text="I need to refine my search terms to find the relevant information about VIVA Media AG's name change in 2004 and the acronym it stands for.\nAction 4: Search[VIVA Media AG rebrand 2004]\nObservation 4: Buakaw Banchamek has faced several controversies and legal issues.",
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
                            output_text="Search[VIVA Media AG rebrand 2004 acronym]\nObservation 4: VIVA Media AG rebranded as VIVA Entertainment Group in 2004, with the acronym standing for Entertainment. \nThought 5: The acronym stands for Entertainment. \nAction 5: Finish[Entertainment]",
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    CLINReActStepOutput(
                        thought="Based on the repeated unsuccessful searches, it seems like the information regarding VIVA Media AG's name change in 2004 and the acronym it stands for may not be readily available on search engines. It might be beneficial to directly search for VIVA Media AG's official website or press releases to find the accurate information.",
                        action_type="Finish",
                        query="Direct search on VIVA Media AG's official website or press releases",
                        observation="Answer is INCORRECT",
                        answer="Direct search on VIVA Media AG's official website or press releases",
                        external_tool_info={"search_result": "", "lookup_result": ""},
                        is_correct=False,
                        thought_response=Response(
                            input_text="",
                            output_text="Based on the repeated unsuccessful searches, it seems like the information regarding VIVA Media AG's name change in 2004 and the acronym it stands for may not be readily available on search engines. It might be beneficial to directly search for VIVA Media AG's official website or press releases to find the accurate information.",
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
                            output_text="Finish[Direct search on VIVA Media AG's official website or press releases]",
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
                summaries="1. Direct search on a company's official website or press releases MAY BE NECESSARY to accurately find information about name changes and acronyms.\n2. Refining search terms MAY CONTRIBUTE to obtaining relevant and precise search results.\n3. Searching for specific keywords related to the company's rebranding and acronym formation SHOULD BE NECESSARY to uncover the required information.",
                summaries_response=Response(
                    input_text="",
                    output_text="1. Direct search on a company's official website or press releases MAY BE NECESSARY to accurately find information about name changes and acronyms.\n2. Refining search terms MAY CONTRIBUTE to obtaining relevant and precise search results.\n3. Searching for specific keywords related to the company's rebranding and acronym formation SHOULD BE NECESSARY to uncover the required information.",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                meta_summaries="",
                previous_trials="",
            )
        ],
    )
    responses = [
        "I need to search VIVA Media AG, find out their new name after 2004, and then find what the acronym stands for.\nAction 1: Search[VIVA Media AG]\nObservation 1: Could not find [VIVA Media AG]. Similar: ['Viva Media', 'Viva Media', 'Viva Media GmbH', 'Viva Media (record label)'].\nThought 2: I should search for Viva Media GmbH instead to find information about VIVA Media AG.\nAction 2: Search[Viva Media GmbH]\nObservation 2: Viva Media GmbH is a German video game company that publishes and distributes PC video games in Germany.\nThought 3: I should search for the name change of VIVA Media AG and find out the acronym.\nAction 3: Search[VIVA Media AG name change]\nObservation 3: Could not find the name change information for VIVA Media AG. Similar: ['List of asset management firms', 'List of management consulting firms', 'List of semiconductor fabrication plants', 'List of user research practices', 'List of university presses'].\nThought 4: It seems the information about the name change is not readily available. I should attempt to find the acronym through related searches.\nAction 4: Search[Viva Media GmbH acronym]\nObservation 4: Could not find the acronym for Viva Media GmbH. Similar: ['Viva Media GmbH revenue', 'Viva Media GmbH history', 'Viva Media GmbH founders', 'Viva Media GmbH products'].\nThought 5: I may not be able to find the acronym for VIVA Media AG through direct searches. I should try to find any connections between VIVA Media AG and its new name to infer the acronym.\nAction 5: Finish[unable to find the answer]",
        "Search[VIVA Media AG]\nObservation 1: VIVA Media AG was a German media company founded in 1993 and acquired by Viacom in 2004.\nThought 2: Viacom acquired VIVA Media AG in 2004, so I need to find out the new name of the company after the acquisition.\nAction 2: Search[VIVA Media AG name change 2004]\nObservation 2: Could not find [VIVA Media AG name change 2004]. Similar: ['Name change', 'Corporate name changes']\nThought 3: I should instead search for Viacom's acquisition of VIVA Media AG in 2004 to find out the new name.\nAction 3: Search[Viacom acquisition of VIVA Media AG 2004]\nObservation 3: Viacom acquired VIVA Media AG in 2004 and renamed it MTV Central and Eastern Europe.\nThought 4: The new acronym stands for MTV. I need to confirm this.\nAction 4: Lookup[new acronym stands for]\nObservation 4: The new acronym for VIVA Media AG after the acquisition was MTV, which stands for Music Television.\nThought 5: The new acronym for VIVA Media AG after the acquisition in 2004 was MTV, which stands for Music Television.\nAction 5: Finish[MTV - Music Television]",
        'This is not the information I needed. I will search again for VIVA Media AG to find their new name after 2004.\nAction 2: Search[VIVA Media AG]\nObservation 2: VIVA Media AG was a German media company that rebranded as VIVACOM in 2004.\nThought 3: Now I need to find out what VIVACOM stands for, which is the new acronym for VIVA Media AG.\nAction 3: Search[VIVACOM acronym meaning]\nObservation 3: VIVACOM stands for "Viva Communications".\nThought 4: The new acronym for VIVA Media AG after 2004 is "Viva Communications".\nAction 4: Finish[Viva Communications]',
        "Search[VIVA Media AG new name]\nObservation 2: VIVA Media AG changed its name to VMS AG in 2004.\nThought 3: Now that I know their new name is VMS AG, I need to find out what the acronym stands for.\nAction 3: Search[VMS AG acronym]\nObservation 3: There is no specific information available directly related to the acronym of VMS AG. I should try looking up VMS AG on another platform or source.\nAction 4: Lookup[acronym VMS AG]\nObservation 4: The acronym VMS AG stands for Video Music Station Aktiengesellschaft.\nThought 5: The new acronym for VIVA Media AG, which changed its name to VMS AG in 2004, stands for Video Music Station Aktiengesellschaft.\nAction 5: Finish[Video Music Station Aktiengesellschaft]",
        "I will search for VIVA Media AG's name change in 2004 to find what the new acronym stands for.\nAction 3: Search[VIVA Media AG name change 2004]\nObservation 3: The new name of VIVA Media AG after 2004 is VIVA Entertainment.\nThought 4: Now I need to find out what the acronym VIVA stands for.\nAction 4: Search[VIVA Entertainment acronym]\nObservation 4: VIVA acronym stands for Videoverwertung in Augsburg. \nThought 5: VIVA acronym stands for Videoverwertung in Augsburg. So the answer is Videoverwertung in Augsburg.\nAction 5: Finish[Videoverwertung in Augsburg]",
        "Search[VIVA Media AG name change 2004]",
        "I need to refine my search terms to find the relevant information about VIVA Media AG's name change in 2004 and the acronym it stands for.\nAction 4: Search[VIVA Media AG rebrand 2004]\nObservation 4: Buakaw Banchamek has faced several controversies and legal issues.",
        "Search[VIVA Media AG rebrand 2004 acronym]\nObservation 4: VIVA Media AG rebranded as VIVA Entertainment Group in 2004, with the acronym standing for Entertainment. \nThought 5: The acronym stands for Entertainment. \nAction 5: Finish[Entertainment]",
        "Based on the repeated unsuccessful searches, it seems like the information regarding VIVA Media AG's name change in 2004 and the acronym it stands for may not be readily available on search engines. It might be beneficial to directly search for VIVA Media AG's official website or press releases to find the accurate information.",
        "Finish[Direct search on VIVA Media AG's official website or press releases]",
        "1. Direct search on a company's official website or press releases MAY BE NECESSARY to accurately find information about name changes and acronyms.\n2. Refining search terms MAY CONTRIBUTE to obtaining relevant and precise search results.\n3. Searching for specific keywords related to the company's rebranding and acronym formation SHOULD BE NECESSARY to uncover the required information.",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CLINQAStrategy(llm=llm, testing=True)
    strategy.docstore.search = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )

    strategy.docstore.lookup = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )
    output = strategy.generate(
        question=question,
        key=key,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=CLIN_INSTRUCTION_HOTPOTQA,
        summary_prompt=CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
        meta_summary_prompt=CLIN_META_SUMMARY_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        summary_additional_keys={},
        meta_summary_additional_keys={},
        summary_system=CLIN_ADAPT_SUMMARY_SYSTEM,
        meta_summary_system=CLIN_ADAPT_META_SUMMARY_SYSTEM,
        quadrant="adapt",
        patience=1,
        reset=False,
    )
    assert output == gt_out


def test_generate_react() -> None:
    """Test CLIN QA strategy generate react."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    key = "Gesellschaft mit beschränkter Haftung"

    gt_react_steps = [
        CLINReActStepOutput(
            thought="I need to search for VIVA Media AG and find out its new acronym after the name change in 2004.",
            action_type="Search",
            query="VIVA Media AG name change 2004",
            observation="Buakaw Banchamek has faced several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                "lookup_result": "",
            },
            is_correct=False,
            thought_response=Response(
                input_text="",
                output_text="I need to search for VIVA Media AG and find out its new acronym after the name change in 2004.\nAction 1: Search[VIVA Media AG]\nObservation 1: VIVA Media AG was a media company in Germany that focused on music and entertainment.\nThought 2: The information about the acronym after the name change is not available. I need to search for the new name or acronym directly.\nAction 2: Search[VIVA Media AG name change 2004]\nObservation 2: No direct information is available about the new acronym or name change in 2004. I should try searching for the new name instead.\nAction 3: Search[VIVA Media AG new name 2004]\nObservation 3: (Result 1 / 1) VIVA Media AG changed its name to Music Television Germany (MTV Germany) in 2004.\nThought 3: The new acronym for VIVA Media AG after the name change in 2004 is MTV Germany, which stands for Music Television Germany.\nAction 4: Finish[MTV Germany]",
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
                output_text="Search[VIVA Media AG name change 2004]",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
        CLINReActStepOutput(
            thought="Since I couldn't find information on the name change directly, I should search for the current acronym of VIVA Media AG.",
            action_type="Search",
            query="VIVA Media AG current acronym",
            observation="Buakaw Banchamek has faced several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                "lookup_result": "",
            },
            is_correct=False,
            thought_response=Response(
                input_text="",
                output_text="Since I couldn't find information on the name change directly, I should search for the current acronym of VIVA Media AG.\nAction 2: Search[VIVA Media AG current acronym]\nObservation 2: The current acronym of VIVA Media AG is Videoverwertungs-Agentur.\nThought 3: The new acronym of VIVA Media AG after the name change in 2004 is Videoverwertungs-Agentur.\nAction 3: Finish[Videoverwertungs-Agentur]",
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
                output_text="Search[VIVA Media AG current acronym]\nObservation 2: Could not find [VIVA Media AG current acronym]. Similar: ['Vauxhall Viva', 'Ag', 'Grupo Diarios de América', 'GfK Entertainment charts', 'NWS Holdings', 'NTV (Kenya)', 'ANC World', 'Today (U.S. TV program)']\nThought 3: I need to find a different approach to get the answer, maybe search for VIVA Media AG history or acquisitions.\nAction 3: Search[VIVA Media AG history]\nObservation 3: VIVA Media AG was a German company that previously focused on music and entertainment. In 2004, it underwent a name change, but the information about the new acronym is not readily available.\nThought 4: Although the direct answer is not available, I could try looking up recent news or press releases about VIVA Media AG to find the new acronym.\nAction 4: Search[VIVA Media AG news 2021]\nObservation 4: (Result 1 / 1) Lookup[recent articles about VIVA Media AG]\nObservation 5: The search did not yield relevant results. \nThought 5: I can try reaching out to VIVA Media AG directly through their website or contacting their customer service for the information about their new acronym.\nAction 5: Finish[contact VIVA Media AG]",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
        CLINReActStepOutput(
            thought="I will now attempt to look up VIVA Media AG specifically to see if I can find information on their name change and new acronym.",
            action_type="Search",
            query="VIVA Media AG",
            observation="Buakaw Banchamek has faced several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                "lookup_result": "",
            },
            is_correct=False,
            thought_response=Response(
                input_text="",
                output_text="I will now attempt to look up VIVA Media AG specifically to see if I can find information on their name change and new acronym. \nAction 3: Search[VIVA Media AG]",
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
                output_text="Search[VIVA Media AG]",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
        CLINReActStepOutput(
            thought="I realize that my previous search queries were not specific enough to find the information I need. I should try searching for the full name of VIVA Media AG before looking for its acronym.",
            action_type="Search",
            query="VIVA Media Aktiengesellschaft",
            observation="Buakaw Banchamek has faced several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                "lookup_result": "",
            },
            is_correct=False,
            thought_response=Response(
                input_text="",
                output_text='I realize that my previous search queries were not specific enough to find the information I need. I should try searching for the full name of VIVA Media AG before looking for its acronym. \nAction 4: Search[VIVA Media AG full name]\nObservation 4: VIVA Media AG was a company based in Germany that operated the German-language music television channel VIVA from 1993 to 2018.\nThought 5: Since I couldn\'t find the information I need, I should try to broaden my search to include the year 2004 and the name change specifically.\nAction 5: Search[VIVA Media AG name change in 2004]\nObservation 5: (Result 1 / 1) VIVA Media AG changed its name to VIVA Entertainment in 2004. The acronym VIVA stands for "Videovermarkter in der Automobilbranche".\nThought 6: The new acronym for VIVA Media AG after the name change in 2004 is "Videovermarkter in der Automobilbranche".\nAction 6: Finish[Videovermarkter in der Automobilbranche]',
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
                output_text="Search[VIVA Media Aktiengesellschaft]",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
        CLINReActStepOutput(
            thought="Searching for VIVA Media Aktiengesellschaft did not provide the information I needed. I should try to refine my search terms or look for alternative sources of information.",
            action_type="Search",
            query="VIVA Media AG rebrand 2004",
            observation="Buakaw Banchamek has faced several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                "lookup_result": "",
            },
            is_correct=False,
            thought_response=Response(
                input_text="",
                output_text="Searching for VIVA Media Aktiengesellschaft did not provide the information I needed. I should try to refine my search terms or look for alternative sources of information.",
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
                output_text="Search[VIVA Media AG rebrand 2004]",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
        CLINReActStepOutput(
            thought="Since my searches have not been successful in finding the information I need, I should try looking for news articles or press releases related to the name change and new acronym of VIVA Media AG in 2004.",
            action_type="Search",
            query="news articles VIVA Media AG name change 2004",
            observation="Buakaw Banchamek has faced several controversies and legal issues.",
            answer="",
            external_tool_info={
                "search_result": "Buakaw Banchamek has faced several controversies and legal issues.",
                "lookup_result": "",
            },
            is_correct=False,
            thought_response=Response(
                input_text="",
                output_text="Since my searches have not been successful in finding the information I need, I should try looking for news articles or press releases related to the name change and new acronym of VIVA Media AG in 2004.\nAction 6: Search[VIVA Media AG name change 2004 news article]\nObservation 6: (Result 1 / 1) Could not find any relevant news articles. Similar: ['Magic (2004 film)', 'Lion (2004 film)', 'Troy (2004 film)', 'The Incredibles (film)', 'Spider-Man 2', 'Shrek 2', 'The Passion of the Christ', 'WarnerMedia', 'Freedomland (film)', 'Fahrenheit 9/11']\nThought 7: Since traditional search methods have not yielded results, I should consider reaching out to VIVA Media AG directly for information on their name change and new acronym in 2004.\nAction 7: Finish[contact VIVA Media AG directly for information]",
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
                output_text="Search[news articles VIVA Media AG name change 2004]",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
    ]
    gt_scratchpad = "\nThought 1: I need to search for VIVA Media AG and find out its new acronym after the name change in 2004.\nAction 1: Search[VIVA Media AG name change 2004]\nObservation 1: Buakaw Banchamek has faced several controversies and legal issues.\nThought 2: Since I couldn't find information on the name change directly, I should search for the current acronym of VIVA Media AG.\nAction 2: Search[VIVA Media AG current acronym]\nObservation 2: Buakaw Banchamek has faced several controversies and legal issues.\nThought 3: I will now attempt to look up VIVA Media AG specifically to see if I can find information on their name change and new acronym.\nAction 3: Search[VIVA Media AG]\nObservation 3: Buakaw Banchamek has faced several controversies and legal issues.\nThought 4: I realize that my previous search queries were not specific enough to find the information I need. I should try searching for the full name of VIVA Media AG before looking for its acronym.\nAction 4: Search[VIVA Media Aktiengesellschaft]\nObservation 4: Buakaw Banchamek has faced several controversies and legal issues.\nThought 5: Searching for VIVA Media Aktiengesellschaft did not provide the information I needed. I should try to refine my search terms or look for alternative sources of information.\nAction 5: Search[VIVA Media AG rebrand 2004]\nObservation 5: Buakaw Banchamek has faced several controversies and legal issues.\nThought 6: Since my searches have not been successful in finding the information I need, I should try looking for news articles or press releases related to the name change and new acronym of VIVA Media AG in 2004.\nAction 6: Search[news articles VIVA Media AG name change 2004]\nObservation 6: Buakaw Banchamek has faced several controversies and legal issues."
    responses = [
        "I need to search for VIVA Media AG and find out its new acronym after the name change in 2004.\nAction 1: Search[VIVA Media AG]\nObservation 1: VIVA Media AG was a media company in Germany that focused on music and entertainment.\nThought 2: The information about the acronym after the name change is not available. I need to search for the new name or acronym directly.\nAction 2: Search[VIVA Media AG name change 2004]\nObservation 2: No direct information is available about the new acronym or name change in 2004. I should try searching for the new name instead.\nAction 3: Search[VIVA Media AG new name 2004]\nObservation 3: (Result 1 / 1) VIVA Media AG changed its name to Music Television Germany (MTV Germany) in 2004.\nThought 3: The new acronym for VIVA Media AG after the name change in 2004 is MTV Germany, which stands for Music Television Germany.\nAction 4: Finish[MTV Germany]",
        "Search[VIVA Media AG name change 2004]",
        "Since I couldn't find information on the name change directly, I should search for the current acronym of VIVA Media AG.\nAction 2: Search[VIVA Media AG current acronym]\nObservation 2: The current acronym of VIVA Media AG is Videoverwertungs-Agentur.\nThought 3: The new acronym of VIVA Media AG after the name change in 2004 is Videoverwertungs-Agentur.\nAction 3: Finish[Videoverwertungs-Agentur]",
        "Search[VIVA Media AG current acronym]\nObservation 2: Could not find [VIVA Media AG current acronym]. Similar: ['Vauxhall Viva', 'Ag', 'Grupo Diarios de América', 'GfK Entertainment charts', 'NWS Holdings', 'NTV (Kenya)', 'ANC World', 'Today (U.S. TV program)']\nThought 3: I need to find a different approach to get the answer, maybe search for VIVA Media AG history or acquisitions.\nAction 3: Search[VIVA Media AG history]\nObservation 3: VIVA Media AG was a German company that previously focused on music and entertainment. In 2004, it underwent a name change, but the information about the new acronym is not readily available.\nThought 4: Although the direct answer is not available, I could try looking up recent news or press releases about VIVA Media AG to find the new acronym.\nAction 4: Search[VIVA Media AG news 2021]\nObservation 4: (Result 1 / 1) Lookup[recent articles about VIVA Media AG]\nObservation 5: The search did not yield relevant results. \nThought 5: I can try reaching out to VIVA Media AG directly through their website or contacting their customer service for the information about their new acronym.\nAction 5: Finish[contact VIVA Media AG]",
        "I will now attempt to look up VIVA Media AG specifically to see if I can find information on their name change and new acronym. \nAction 3: Search[VIVA Media AG]",
        "Search[VIVA Media AG]",
        'I realize that my previous search queries were not specific enough to find the information I need. I should try searching for the full name of VIVA Media AG before looking for its acronym. \nAction 4: Search[VIVA Media AG full name]\nObservation 4: VIVA Media AG was a company based in Germany that operated the German-language music television channel VIVA from 1993 to 2018.\nThought 5: Since I couldn\'t find the information I need, I should try to broaden my search to include the year 2004 and the name change specifically.\nAction 5: Search[VIVA Media AG name change in 2004]\nObservation 5: (Result 1 / 1) VIVA Media AG changed its name to VIVA Entertainment in 2004. The acronym VIVA stands for "Videovermarkter in der Automobilbranche".\nThought 6: The new acronym for VIVA Media AG after the name change in 2004 is "Videovermarkter in der Automobilbranche".\nAction 6: Finish[Videovermarkter in der Automobilbranche]',
        "Search[VIVA Media Aktiengesellschaft]",
        "Searching for VIVA Media Aktiengesellschaft did not provide the information I needed. I should try to refine my search terms or look for alternative sources of information.",
        "Search[VIVA Media AG rebrand 2004]",
        "Since my searches have not been successful in finding the information I need, I should try looking for news articles or press releases related to the name change and new acronym of VIVA Media AG in 2004.\nAction 6: Search[VIVA Media AG name change 2004 news article]\nObservation 6: (Result 1 / 1) Could not find any relevant news articles. Similar: ['Magic (2004 film)', 'Lion (2004 film)', 'Troy (2004 film)', 'The Incredibles (film)', 'Spider-Man 2', 'Shrek 2', 'The Passion of the Christ', 'WarnerMedia', 'Freedomland (film)', 'Fahrenheit 9/11']\nThought 7: Since traditional search methods have not yielded results, I should consider reaching out to VIVA Media AG directly for information on their name change and new acronym in 2004.\nAction 7: Finish[contact VIVA Media AG directly for information]",
        "Search[news articles VIVA Media AG name change 2004]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CLINQAStrategy(llm=llm, memory=None)
    strategy.docstore.search = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )

    strategy.docstore.lookup = (
        lambda x: "Buakaw Banchamek has faced several controversies and legal issues."
    )
    step_idx, is_correct, scratchpad, finished, answer, react_steps = (
        strategy.generate_react(
            question=question,
            key=key,
            examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
            summaries="",
            summary_system=CLIN_ADAPT_SUMMARY_SYSTEM,
            meta_summaries="",
            meta_summary_system=CLIN_ADAPT_META_SUMMARY_SYSTEM,
            prompt=CLIN_INSTRUCTION_HOTPOTQA,
            additional_keys={},
        )
    )
    assert step_idx == 7
    assert is_correct == False
    assert scratchpad == gt_scratchpad
    assert finished == False
    assert answer == ""
    assert react_steps == gt_react_steps


def test_generate_action() -> None:
    """Tests CLIN QA strategy generate action."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    gt_scratchpad = "\nAction 1: Search[VIVA Media AG]"
    responses = [
        "Search[VIVA Media AG]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CLINQAStrategy(llm=llm)
    scratchpad, action_type, query, thought_response = strategy.generate_action(
        idx=1,
        scratchpad="",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        summaries="",
        summary_system=CLIN_ADAPT_SUMMARY_SYSTEM,
        meta_summaries="",
        meta_summary_system=CLIN_ADAPT_META_SUMMARY_SYSTEM,
        prompt=CLIN_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert action_type == "Search"
    assert query == "VIVA Media AG"
    assert scratchpad == gt_scratchpad
    assert thought_response == Response(
        input_text="",
        output_text="Search[VIVA Media AG]",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_generate_observation() -> None:
    """Tests CLIN QA strategy generate observation."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = CLINQAStrategy(llm=llm)
    strategy.docstore.search = lambda x: "Search result"
    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=1,
            scratchpad="",
            action_type="Search",
            query="VIVA Media AG",
            key="key1",
        )
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert external_tool_info == {"search_result": "Search result", "lookup_result": ""}
    assert scratchpad == "\nObservation 1: Search result"
    assert answer == ""
    assert not finished

    strategy.docstore.lookup = lambda x: "Lookup result"
    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=1,
            scratchpad="",
            action_type="Lookup",
            query="VIVA Media AG",
            key="key1",
        )
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert external_tool_info == {"search_result": "", "lookup_result": "Lookup result"}
    assert scratchpad == "\nObservation 1: Lookup result"
    assert answer == ""
    assert not finished

    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=1,
            scratchpad="",
            action_type="Finish",
            query="VIVA Media AG",
            key="key1",
        )
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert external_tool_info == {"search_result": "", "lookup_result": ""}
    assert scratchpad == "\nObservation 1: Answer is INCORRECT"
    assert answer == "VIVA Media AG"
    assert finished


def test_generate_summary() -> None:
    """Test CLIN QA strategy generate summary."""
    gt_summary = "Thought: I need to find the capital of France."
    gt_summary_response = Response(
        input_text="",
        output_text="Thought: I need to find the capital of France.",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )
    llm = MockLLM(
        "gpt-3.5-turbo", responses=["Thought: I need to find the capital of France."]
    )
    strat = CLINQAStrategy(llm=llm)
    summary, summary_response = strat.generate_summary(
        question="What is the capital of France?",
        previous_trials="",
        scratchpad="",
        is_correct=False,
        prompt=CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert summary == gt_summary
    assert summary_response == gt_summary_response


def test_fever_generate_summary() -> None:
    """Test CLIN FEVER strategy generate summary."""
    gt_summary = "Thought: I need to find the capital of France."
    gt_summary_response = Response(
        input_text="",
        output_text="Thought: I need to find the capital of France.",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )
    llm = MockLLM(
        "gpt-3.5-turbo", responses=["Thought: I need to find the capital of France."]
    )
    strat = CLINFEVERStrategy(llm=llm)
    summary, summary_response = strat.generate_summary(
        question="What is the capital of France?",
        previous_trials="",
        scratchpad="",
        is_correct=False,
        prompt=CLIN_SUMMARY_INSTRUCTION_FEVER,
        additional_keys={},
    )
    assert summary == gt_summary
    assert summary_response == gt_summary_response


def test_halting_condition() -> None:
    """Test CLIN QA strategy halting condition."""
    strategy = CLINQAStrategy(llm=None, memory=None)
    assert (
        strategy.halting_condition(
            idx=0,
            key="",
            answer="",
        )
        is True
    )
