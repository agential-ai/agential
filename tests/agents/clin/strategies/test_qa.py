"""Test the CLIN QA strategy."""

import tiktoken

from agential.agents.clin.output import CLINReActStepOutput
from agential.agents.clin.prompts import (
    CLIN_ADAPT_META_SUMMARY_SYSTEM,
    CLIN_ADAPT_SUMMARY_SYSTEM,
    CLIN_INSTRUCTION_HOTPOTQA,
)
from agential.agents.clin.strategies.qa import CLINQAStrategy
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
