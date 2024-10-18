"""CLIN general strategy."""


from typing import Dict, List, Tuple
from agential.agents.clin.strategies.base import CLINBaseStrategy
from agential.core.llm import BaseLLM, Response
from agential.agents.clin.output import CLINOutput

class CLINGeneralStrategy(CLINBaseStrategy):
    def __init__(self, llm: BaseLLM, testing: bool = False) -> None:
        super().__init__(llm, testing)

    def generate(self) -> CLINOutput:
        while not self.halting_condition():

            while not self.step_halting_condition():
                action, action_response = self.generate_action()
                obs, reward = self.generate_observation(action)

            summary = self.summarize()


    def generate_action(self, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[str, List[Response]]:
        return super().generate_action(question, examples, prompt, additional_keys)
    
    def summarize(self) -> Tuple[str | Response]:
        return super().summarize()
    
    def meta_summarize(self) -> Tuple[str | Response]:
        return super().meta_summarize()
    
    def reset(self) -> None:
        return super().reset()