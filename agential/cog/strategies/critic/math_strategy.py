from typing import Dict
from agential.cog.functional.critic import _prompt_agent, _prompt_critique, safe_execute
from agential.cog.strategies.critic.base import CriticBaseStrategy
import re

class MathStrategy(CriticBaseStrategy):
    def __init__(self, llm):
        self.llm = llm

    def generate(
        self, 
        question: str, 
        examples: str, 
        prompt: str, 
        additional_keys: Dict[str, str]
    ) -> str:
        code = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            additional_keys=additional_keys,
            prompt=prompt,
        )
        try:
            matches = re.findall(r"`python\s+(.*?)\s+`", code, re.DOTALL)
            code = matches[0]
        except:
            pass

        return code

    def generate_critique(
        self, 
        idx: int, 
        question: str, 
        examples: str, 
        answer: str, 
        critique: str,
        prompt: str, 
        additional_keys: Dict[str, str], 
        use_interpreter_tool: bool, 
        use_search_tool: bool,
        max_interactions: int,
        **kwargs
    ):
        external_tool_info = {}
        if use_interpreter_tool:
            code_answer, execution_status = safe_execute(answer)
            external_tool_info = {
                "execution_status": execution_status,
                "code_answer": code_answer if code_answer is not None else "",
            }

        new_critique = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique="",
            additional_keys=external_tool_info if external_tool_info else additional_keys,
            prompt=prompt,
        ).split("Here's")[0]

        return new_critique, external_tool_info

    def create_output_dict(self, answer: str, critique: str, external_tool_info: Dict[str, str]) -> Dict[str, str]:
        output_dict = {"code": answer, "critique": critique, **external_tool_info}
        return output_dict

    def update_answer_based_on_critique(
        self, 
        question: str, 
        examples: str, 
        answer: str, 
        critique: str, 
        prompt: str, 
        additional_keys: Dict[str, str],
        external_tool_info: Dict[str, str],
        **kwargs
    ) -> str:
        return _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique=f"{critique}\n\nHere's a better solution:\n```python\n",
            additional_keys=external_tool_info if external_tool_info else additional_keys,
            prompt=prompt,
        ).split("```")[0]

    def halting_condition(self, critique: str) -> bool:
        pattern_correct = re.compile(r"\bis correct\.", re.IGNORECASE)
        pattern_incorrect = re.compile(r"\bincorrect\b", re.IGNORECASE)
        return bool(pattern_correct.search(critique)) and not bool(pattern_incorrect.search(critique))

    def reset(self) -> bool:
        pass