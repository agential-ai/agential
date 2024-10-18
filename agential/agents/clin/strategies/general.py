"""CLIN general strategy."""


import time
from typing import Dict, List, Tuple, Union
from agential.agents.clin.strategies.base import CLINBaseStrategy
from agential.core.llm import BaseLLM, Response
from agential.agents.clin.output import CLINOutput, CLINStepOutput, CLINTrialStepOutput

class CLINGeneralStrategy(CLINBaseStrategy):
    def __init__(self, llm: BaseLLM, testing: bool = False) -> None:
        super().__init__(llm, testing)

    def generate(
        self,
        question: str,
        key: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        patience: int,
        reset: bool,
    ) -> CLINOutput:
        # while not self.halting_condition():

        #     while not self.step_halting_condition():
        #         action, action_response = self.generate_action()
        #         obs, reward = self.generate_observation(action)

        #     summary = self.summarize()

        start = time.time()

        # Reset.
        if reset:
            self.reset()

        scratchpad = ""
        summaries, meta_summaries = [], []
        answer = ""
        finished = False
        idx, step_idx, patience_cnt = 1, 1, 0
        steps: List[CLINStepOutput] = []
        while not self.halting_condition(idx=idx, key=key, answer=answer):
            
            step_idx, is_correct, scratchpad, finished, answer, react_steps = (
                self.generate_react(
                    question=question,
                    key=key,
                    examples=examples,
                    summaries=summaries,
                    meta_summaries=meta_summaries,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
            )

            steps.append(
                ReflexionReActStepOutput(
                    steps=react_steps,
                    reflections=reflections,
                    reflection_response=reflection_response,
                )
            )

            # Increment patience counter.
            if not is_correct:
                patience_cnt += 1
            if patience_cnt == patience:
                break

            idx += 1

    def generate_react(
        self,
        question: str,
        key: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[int, bool, str, bool, str, List[ReflexionReActReActStepOutput]]:
        """Generates a reaction based on the given question, key, examples, reflections, prompt, and additional keys.

        Args:
            question (str): The question to be answered.
            key (str): The key for the observation.
            examples (str): Examples to guide the reaction process.
            reflections (str): The reflections to guide the reaction process.
            prompt (str): The prompt or instruction to guide the reaction.
            additional_keys (Dict[str, str]): Additional keys for the reaction process.

        Returns:
            Tuple[int, bool, str, bool, str, List[ReflexionReActReActStepOutput]]: The reaction, whether the reaction is finished, the answer, whether the reaction is valid, the scratchpad, and the steps.
        """
        react_steps = []
        step_idx = 1
        scratchpad = ""
        finished = False
        answer = ""
        while not self.react_halting_condition(
            finished=finished,
            idx=step_idx,
            scratchpad=scratchpad,
            question=question,
            examples=examples,
            reflections=reflections,
            prompt=prompt,
            additional_keys=additional_keys,
        ):
            # Think.
            scratchpad, thought, thought_response = self.generate_thought(
                idx=step_idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                reflections=reflections,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Act.
            scratchpad, action_type, query, action_response = self.generate_action(
                idx=step_idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                reflections=reflections,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Observe.
            scratchpad, answer, finished, is_correct, obs, external_tool_info = (
                self.generate_observation(
                    idx=step_idx,
                    scratchpad=scratchpad,
                    action_type=action_type,
                    query=query,
                    key=key,
                )
            )

            react_steps.append(
                ReflexionReActReActStepOutput(
                    thought=thought,
                    action_type=action_type,
                    query=query,
                    observation=obs,
                    answer=answer,
                    external_tool_info=external_tool_info,
                    is_correct=is_correct,
                    thought_response=thought_response,
                    action_response=action_response,
                )
            )

            step_idx += 1

        return step_idx, is_correct, scratchpad, finished, answer, react_steps

    def generate_thought(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str]:
        """Generates a thought based on the given parameters."""
        pass

    def generate_action(self, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[str, List[Response]]:
        return super().generate_action(question, examples, prompt, additional_keys)
    
    def generate_observation(self, idx: int, scratchpad: str, action_type: str, query: str, key: str) -> Tuple[str, str, bool, bool, str, List[Response]]:
        return super().generate_observation(idx, scratchpad, action_type, query, key)

    def summarize(self) -> Tuple[str | Response]:
        return super().summarize()
    
    def meta_summarize(self) -> Tuple[str | Response]:
        return super().meta_summarize()
    
    def halting_condition(self, finished: bool) -> bool:
        return super().halting_condition(finished)

    def react_halting_condition(self, finished: bool, idx: int, scratchpad: str, question: str, examples: str, reflections: str, prompt: str, additional_keys: Dict[str, str]) -> bool:
        pass

    def reset(self) -> None:
        return super().reset()