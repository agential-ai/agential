from typing import Dict
from agential.cog.functional.critic import _prompt_agent, _prompt_critique, safe_execute
from agential.cog.strategies.critic.base import CriticBaseStrategy

def validate_overlapping_keys(dict_1: Dict[str, str], dict_2: Dict[str, str]) -> bool:
    overlapping_keys = dict_1.keys() & dict_2.keys()
    if overlapping_keys:
        raise ValueError(f"Overlapping keys detected: {overlapping_keys}")

class CodeStrategy(CriticBaseStrategy):
    def __init__(self, llm):
        self.llm = llm

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
        use_interpreter_tool: bool, 
        use_search_tool: bool,
        max_interactions: int,
        **kwargs
    ):
        external_tool_info = {}
        if use_interpreter_tool:
            if "tests" not in additional_keys:
                raise ValueError("The 'tests' parameter must be specified in the `additional_keys`.")
            tests = additional_keys["tests"]

            _, execution_status = safe_execute(f"{answer}\n\n{tests}")
            external_tool_info = {
                "execution_status": execution_status,
            }

            validate_overlapping_keys(additional_keys, external_tool_info)
            
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
        validate_overlapping_keys(additional_keys, external_tool_info)
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
        return "<CORRECT>" in critique.replace(" ", "").upper().strip() or ("correct" in critique and "incorrect" not in critique)

    def reset(self) -> bool:
        self._answer_history = []