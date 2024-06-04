"""Unit tests for Reflexion QA strategies."""
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.modules.reflect.reflexion import ReflexionCoTReflector
from agential.cog.strategies.reflexion.qa import (
    parse_qa_action,
    ReflexionCoTQAStrategy,
)
from agential.cog.prompts.agent.reflexion import (
    REFLEXION_COT_INSTRUCTION_HOTPOTQA,
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA
)
from agential.cog.prompts.benchmark.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT
)


def test_parse_qa_action() -> None:
    """Test the parse_qa_action function."""
    assert parse_qa_action("QA[question]") == ("QA", "question")
    assert parse_qa_action("QA[]") == ("", "")
    assert parse_qa_action("QA") == ("", "")


def test_reflexion_cot_init() -> None:
    """Test ReflexionCoTQAStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate() -> None:
    """Tests ReflexionCoTQAStrategy generate."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    key = "Gesellschaft mit beschränkter Haftung"
    context = 'VIVA Media GmbH (until 2004 "VIVA Media AG") is a music television network originating from Germany. It was founded for broadcast of VIVA Germany as VIVA Media AG in 1993 and has been owned by their original concurrent Viacom, the parent company of MTV, since 2004. Viva channels exist in some European countries; the first spin-offs were launched in Poland and Switzerland in 2000.\n\nA Gesellschaft mit beschränkter Haftung (] , abbreviated GmbH ] and also GesmbH in Austria) is a type of legal entity very common in Germany, Austria, Switzerland (where it is equivalent to a S.à r.l.) and Liechtenstein. In the United States, the equivalent type of entity is the limited liability company (LLC). The name of the GmbH form emphasizes the fact that the owners ("Gesellschafter", also known as members) of the entity are not personally liable for the company\'s debts. "GmbH"s are considered legal persons under German and Austrian law. Other variations include mbH (used when the term "Gesellschaft" is part of the company name itself), and gGmbH ("gemeinnützige" GmbH) for non-profit companies.'

    # Incorrect.
    gt_scratchpad = '\nThought: The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.'
    gt_out = 'The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.'  
    responses=[
        'The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.',
        "Finish[Company with Limited Liability]",
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTQAStrategy(llm=llm)
    out = strategy.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT, 
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_HOTPOTQA, 
        additional_keys={}
    )
    assert out == gt_out
    assert strategy._scratchpad == gt_scratchpad


def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTQAStrategy generate_action."""


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTQAStrategy generate_observation."""


def test_reflexion_cot_create_output_dict() -> None:
    """Tests ReflexionCoTQAStrategy create_output_dict."""


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTQAStrategy halting_condition."""


def test_reflexion_cot_reset() -> None:
    """Tests ReflexionCoTQAStrategy reset."""


def test_reflexion_cot_reflect() -> None:
    """Tests ReflexionCoTQAStrategy reflect."""


def test_reflexion_cot_should_reflect() -> None:
    """Tests ReflexionCoTQAStrategy should_reflect."""
