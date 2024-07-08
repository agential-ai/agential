"""CRITIC prompts and fewshot examples selector."""

from typing import Dict

from agential.base.selector import BaseSelector
from agential.cog.critic.prompts import (
    AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
    CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
    CRITIC_CRITIQUE_INSTRUCTION_FEVER,
    CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_INSTRUCTION_MBPP,
    CRITIC_CRITIQUE_INSTRUCTION_SVAMP,
    CRITIC_CRITIQUE_INSTRUCTION_TABMWP,
    CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_SVAMP,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_TABMWP,
    CRITIC_INSTRUCTION_AMBIGNQ,
    # Instructions.
    CRITIC_INSTRUCTION_FEVER,
    CRITIC_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_TRIVIAQA,
    CRITIC_POT_INSTRUCTION_GSM8K,
    CRITIC_POT_INSTRUCTION_HUMANEVAL,
    CRITIC_POT_INSTRUCTION_MBPP,
    CRITIC_POT_INSTRUCTION_SVAMP,
    CRITIC_POT_INSTRUCTION_TABMWP,
    # Few-shots.
    FEVER_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    MBPP_FEWSHOT_EXAMPLES_CRITIC,
    MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    SVAMP_FEWSHOT_EXAMPLES_CRITIC,
    SVAMP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    TABMWP_FEWSHOT_EXAMPLES_CRITIC,
    TABMWP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
)
from agential.manager.constants import Benchmarks

CRITIC_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": CRITIC_INSTRUCTION_HOTPOTQA,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": CRITIC_INSTRUCTION_FEVER,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_FEVER,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": CRITIC_INSTRUCTION_TRIVIAQA,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": CRITIC_INSTRUCTION_AMBIGNQ,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": CRITIC_POT_INSTRUCTION_GSM8K,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": CRITIC_POT_INSTRUCTION_SVAMP,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_SVAMP,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": CRITIC_POT_INSTRUCTION_TABMWP,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_TABMWP,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": CRITIC_POT_INSTRUCTION_HUMANEVAL,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": CRITIC_POT_INSTRUCTION_MBPP,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_MBPP,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
    },
}

CRITIC_FEWSHOTS = {
    Benchmarks.HOTPOTQA: {
        "critique_examples": HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
    },
    Benchmarks.FEVER: {
        "critique_examples": FEVER_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": FEVER_FEWSHOT_EXAMPLES_CRITIC,
    },
    Benchmarks.TRIVIAQA: {
        "critique_examples": TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
    },
    Benchmarks.AMBIGNQ: {
        "critique_examples": AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
    },
    Benchmarks.GSM8K: {
        "critique_examples": GSM8K_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    },
    Benchmarks.SVAMP: {
        "critique_examples": SVAMP_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": SVAMP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    },
    Benchmarks.TABMWP: {
        "critique_examples": TABMWP_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": TABMWP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    },
    Benchmarks.HUMANEVAL: {
        "critique_examples": HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    },
    Benchmarks.MBPP: {
        "critique_examples": MBPP_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    },
}


class CriticSelector(BaseSelector):
    @staticmethod
    def get_fewshots(self, benchmark: str, **kwargs) -> Dict[str, str]:
        if benchmark not in CRITIC_FEWSHOTS:
            raise ValueError(f"Benchmark '{benchmark}' few-shots not found for CRITIC.")

        use_tool = kwargs.get("use_tool")
        if not use_tool:
            raise ValueError("`use_tool` not specified.")

        if use_tool:
            return {
                "critique_examples": CRITIC_FEWSHOTS[benchmark]["critique_examples"]
            }
        return {
            "critique_examples": CRITIC_FEWSHOTS[benchmark]["critique_examples_no_tool"]
        }

    @staticmethod
    def get_prompt(self, benchmark: str, **kwargs) -> str:
        if benchmark not in CRITIC_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for CRITIC.")

        use_tool = kwargs.get("use_tool")
        if not use_tool:
            raise ValueError("`use_tool` not specified.")

        if use_tool:
            return {
                "prompt": CRITIC_PROMPTS[benchmark]["prompt"],
                "critique_prompt": CRITIC_PROMPTS[benchmark]["critique_prompt"],
            }
        return {
            "prompt": CRITIC_PROMPTS[benchmark]["prompt"],
            "critique_prompt": CRITIC_PROMPTS[benchmark]["critique_prompt_no_tool"],
        }
