import re
from typing import Dict
from agential.cog.functional.critic import _prompt_agent, _prompt_critique
from agential.cog.strategies.critic.base import CriticBaseStrategy

class QAStrategy(CriticBaseStrategy):
    def generate(self, llm, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> str:
        return _prompt_agent(
            llm=llm,
            question=question,
            examples=examples,
            additional_keys=additional_keys,
            prompt=prompt,
        )

    def generate_critique(self, llm, question: str, examples: str, answer: str, prompt: str, additional_keys: Dict[str, str], critique_additional_keys: Dict[str, str], tests: str, use_interpreter_tool: bool):
        additional_keys_update = {}
        critique = _prompt_critique(
            llm=llm,
            question=question,
            examples=examples,
            answer=answer,
            critique="",
            additional_keys=critique_additional_keys,
            prompt=prompt,
        ).split("> Evidence: ")[0]
        if "> Search Query: " in critique:
            _, search_query = critique.split("> Search Query:")[:2]
            search_query = search_query.split("\n")[0].strip()
            additional_keys_update["query"] = search_query
        elif "most possible answer: " in critique:
            _, revised_answer = critique.split("most possible answer: ")
            additional_keys_update["revised_answer"] = revised_answer.strip()
        return critique, additional_keys_update

    def create_output_dict(self, answer: str, critique: str, additional_keys_update: Dict[str, str]) -> Dict[str, str]:
        return {"answer": answer, "critique": critique}

    def update_answer_based_on_critique(self, llm, question: str, answer: str, critique: str) -> str:
        return answer
