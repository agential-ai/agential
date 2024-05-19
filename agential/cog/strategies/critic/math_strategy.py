import re
from typing import Dict
from agential.cog.functional.critic import _prompt_agent, _prompt_critique, safe_execute
from agential.cog.strategies.critic.base import CriticBaseStrategy

class MathStrategy(CriticBaseStrategy):
    def generate(self, llm, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> str:
        code = _prompt_agent(
            llm=llm,
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

    def generate_critique(self, llm, question: str, examples: str, answer: str, prompt: str, additional_keys: Dict[str, str], critique_additional_keys: Dict[str, str], tests: str, use_interpreter_tool: bool):
        additional_keys_update = {}
        if use_interpreter_tool:
            code_answer, execution_status = safe_execute(answer)
            critique_additional_keys.update({
                "execution_status": execution_status,
                "code_answer": code_answer if code_answer else "",
            })

        critique = _prompt_critique(
            llm=llm,
            question=question,
            examples=examples,
            answer=answer,
            critique="",
            additional_keys=critique_additional_keys,
            prompt=prompt,
        ).split("Here's")[0]

        return critique, additional_keys_update

    def create_output_dict(self, answer: str, critique: str, additional_keys_update: Dict[str, str]) -> Dict[str, str]:
        output_dict = {"code": answer, "critique": critique}
        if "execution_status" in additional_keys_update:
            output_dict["execution_status"] = additional_keys_update["execution_status"]
        if "code_answer" in additional_keys_update:
            output_dict["code_answer"] = additional_keys_update["code_answer"]
        if "improved_code" in additional_keys_update:
            output_dict["improved_code"] = additional_keys_update["improved_code"]
        return output_dict

    def update_answer_based_on_critique(self, llm, question: str, answer: str, critique: str) -> str:
        return _prompt_critique(
            llm=llm,
            question=question,
            examples="",
            answer=answer,
            critique=critique + "\n\n" + "Here's a better solution:\n```python\n",
            additional_keys={},
            prompt="",
        ).split("```")[0]
