"""Unit tests for Reflexion Math strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.strategies.reflexion.math import (
    ReflexionReActMathStrategy,
    ReflexionCoTGSM8KStrategy,
    ReflexionCoTSVAMPStrategy,
    ReflexionCoTTabMWPStrategy,
    ReflexionReActGSM8KStrategy,
    ReflexionReActSVAMPStrategy,
    ReflexionReActTabMWPStrategy,
    parse_math_action_cot,
    parse_math_action_react
)
from agential.cog.prompts.agent.reflexion import (
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_COT_INSTRUCTION_GSM8K,
    REFLEXION_COT_REFLECT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
)
from agential.cog.prompts.benchmark.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_COT,
    GSM8K_FEWSHOT_EXAMPLES_REACT,
)


def test_parse_math_action_cot() -> None:
    """Tests parse_math_action_cot."""
    action = "Finish the calculation```python\nresult = 5 + 3\n```"
    action_type, query = parse_math_action_cot(action)
    assert action_type == "Finish"
    assert query == "result = 5 + 3"

    action = "complete the task```python\nanswer = 10 * 2\n```"
    action_type, query = parse_math_action_cot(action)
    assert action_type == ""
    assert query == ""


def test_parse_math_action_react() -> None:
    """Tests parse_math_action_react."""
    action = "Calculate the sum```python\nsum = 4 + 6\n```"
    action_type, query = parse_math_action_react(action)
    assert action_type == "Calculate"
    assert query == "sum = 4 + 6"

    action = "Finish the operation```python\nresult = 7 - 2\n```"
    action_type, query = parse_math_action_react(action)
    assert action_type == "Finish"
    assert query == "result = 7 - 2"

    action = "complete the task```python\noutput = 10 / 2\n```"
    action_type, query = parse_math_action_react(action)
    assert action_type == ""
    assert query == ""


def test_reflexion_cot_init() -> None:
    """Tests ReflexionCoTQAStrategy init."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTGSM8KStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate() -> None:
    """Tests ReflexionCoTQAStrategy generate."""


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


def test_reflexion_cot_reflect_condition() -> None:
    """Tests ReflexionCoTQAStrategy reflect_condition."""


def test_reflexion_cot_instantiate_strategies() -> None:
    """Tests ReflexionCoTQAStrategy instantiate strategies."""
    llm = FakeListChatModel(responses=[])
    gsm8k_strategy = ReflexionCoTGSM8KStrategy(llm=llm)
    svamp_strategy = ReflexionCoTSVAMPStrategy(llm=llm)
    tabmwp_strategy = ReflexionCoTTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, ReflexionCoTGSM8KStrategy)
    assert isinstance(svamp_strategy, ReflexionCoTSVAMPStrategy)
    assert isinstance(tabmwp_strategy, ReflexionCoTTabMWPStrategy)


def test_reflexion_react_init() -> None:
    """Tests ReflexionReActQAStrategy init."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionReActGSM8KStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionReActReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_react_generate() -> None:
    """Tests ReflexionReActQAStrategy generate."""


def test_reflexion_react_generate_action() -> None:
    """Tests ReflexionReActQAStrategy generate_action."""


def test_reflexion_react_generate_observation() -> None:
    """Tests ReflexionReActQAStrategy generate_observation."""


def test_reflexion_react_create_output_dict() -> None:
    """Tests ReflexionReActQAStrategy create_output_dict."""


def test_reflexion_react_react_create_output_dict() -> None:
    """Tests ReflexionReActQAStrategy react_create_output_dict."""


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReActQAStrategy halting_condition."""


def test_reflexion_react_react_halting_condition() -> None:
    """Tests ReflexionReActQAStrategy react_halting_condition."""


def test_reflexion_react_reset() -> None:
    """Tests ReflexionReActQAStrategy reset."""


def test_reflexion_react_reflect() -> None:
    """Tests ReflexionReActQAStrategy reflect."""


def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReActQAStrategy reflect_condition."""


def test_reflexion_react_instantiate_strategies() -> None:
    """Tests ReflexionReActQAStrategy instantiate strategies."""
    llm = FakeListChatModel(responses=[])
    gsm8k_strategy = ReflexionReActGSM8KStrategy(llm=llm)
    svamp_strategy = ReflexionReActSVAMPStrategy(llm=llm)
    tabmwp_strategy = ReflexionReActTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, ReflexionReActGSM8KStrategy)
    assert isinstance(svamp_strategy, ReflexionReActSVAMPStrategy)
    assert isinstance(tabmwp_strategy, ReflexionReActTabMWPStrategy)
