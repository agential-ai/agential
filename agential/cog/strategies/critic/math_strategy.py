from typing import Dict

from agential.cog.functional.critic import _prompt_agent, _prompt_critique, safe_execute
from agential.cog.strategies.critic.base import CriticBaseStrategy
from agential.utils.validation import validate_overlapping_keys


class CriticMathStrategy(CriticBaseStrategy):
    def __init__(self, llm, patience=2):
        self.llm = llm
        self.patience = patience
        self._answer_history = []
        self._prev_code_answer = None
        self.patience_counter = 0
        self._halt = False

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> str:
        answer = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            additional_keys=additional_keys,
            prompt=prompt,
        )
        answer = answer.split("```python")[-1].split("```")[0].strip()

        return answer

    def generate_critique(
        self,
        idx: int,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        use_tool: bool,
        max_interactions: int,
        **kwargs,
    ):
        external_tool_info = {}
        if use_tool:
            code_answer, execution_status = safe_execute(answer)
            external_tool_info = {
                "execution_status": execution_status,
                "code_answer": code_answer[0] if code_answer[0] is not None else "",
            }
            self._answer_history.append(
                {"answer": answer, "external_tool_info": external_tool_info}
            )

            if code_answer[0] == self._prev_code_answer:
                self.patience_counter += 1
                if self.patience_counter == self.patience:
                    self._halt = True
            else:
                self._prev_code_answer = code_answer[0]

            last_valid_idx = -1
            for i in range(len(self._answer_history) - 1, -1, -1):
                if (
                    self._answer_history[i]["external_tool_info"]["code_answer"]
                    is not None
                ):
                    last_valid_idx = i
                    break

            external_tool_info = self._answer_history[last_valid_idx][
                "external_tool_info"
            ]
            answer = self._answer_history[last_valid_idx]["answer"]

            validate_overlapping_keys(additional_keys, external_tool_info)

        additional_keys = additional_keys.copy()
        additional_keys.update(external_tool_info)

        new_critique = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique="",
            additional_keys=additional_keys,
            prompt=prompt,
        ).split("Here's")[0]

        return new_critique, external_tool_info

    def create_output_dict(
        self, answer: str, critique: str, external_tool_info: Dict[str, str]
    ) -> Dict[str, str]:
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
        **kwargs,
    ) -> str:
        validate_overlapping_keys(additional_keys, external_tool_info)
        additional_keys = additional_keys.copy()
        additional_keys.update(external_tool_info)

        new_answer = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique=f"{critique}\n\nHere's a better solution:\n```python\n",
            additional_keys=additional_keys,
            prompt=prompt,
        )
        new_answer = new_answer.split("```python")[-1].split("```")[0].strip()

        return new_answer

    def halting_condition(self, critique: str) -> bool:
        return self._halt

    def reset(self) -> bool:
        self._answer_history = []
        self._prev_code_answer = None
        self.patience_counter = 0
        self._halt = False


class CritGSM8KStrategy(CriticMathStrategy):
    pass


class CritSVAMPStrategy(CriticMathStrategy):
    pass


class CritTabMWPStrategy(CriticMathStrategy):
    pass
