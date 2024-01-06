"""Reflexion Agent implementation.

Original Paper: https://arxiv.org/abs/2303.11366
Paper Repositories: 
    - https://github.com/noahshinn/reflexion-draft
    - https://github.com/noahshinn/reflexion
"""
import re
import tiktoken
from tiktoken.core import Encoding
from typing import Any, List, Optional, Tuple

from langchain.prompts import PromptTemplate
from langchain_core.messages.human import (
    HumanMessage
)
from langchain_core.language_models.chat_models import BaseChatModel
from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.eval.reflexion import EM
from discussion_agents.cog.prompts.reflexion import (
    COT,
    COT_REFLECT,
    REFLECTION_HEADER,
    REFLECTION_AFTER_LAST_TRIAL_HEADER,
    LAST_TRIAL_HEADER,
    cot_reflect_prompt,
    cot_reflect_agent_prompt,
)

gpt3_5_turbo_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")  # https://openai.com/blog/gpt-4-api-general-availability

def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer: Encoding = gpt3_5_turbo_enc) -> str:
    # Split the scratchpad content into lines.
    lines = scratchpad.split('\n')
    # Filter out lines starting with 'Observation'.
    observations = filter(lambda x: x.startswith('Observation'), lines)
    # Sort observations by token count.
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    # Truncate observations if total token count exceeds limit.
    while len(tokenizer.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        # Replace the largest observation with a truncated message.
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    # Return formatted reflections if not empty.
    if reflections:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])
    else:
        return ""

def format_last_attempt(question: str, scratchpad: str, header: str = LAST_TRIAL_HEADER, tokenizer: Encoding = gpt3_5_turbo_enc) -> str:
    # Format the last attempt using the provided question and scratchpad.
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer=tokenizer).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

def format_step(step: str) -> str:
    # Remove leading/trailing newlines and spaces, and replace internal newlines with empty space.
    return step.strip('\n').strip().replace('\n', '')

def parse_action(string: str) -> Optional[Tuple[str, str]]:
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        return None

class ReflexionCoTAgent(BaseAgent):
    self_reflect_llm: BaseChatModel
    action_llm: BaseChatModel

    question: str
    context: str
    key: str
    agent_prompt: PromptTemplate = cot_reflect_agent_prompt
    reflect_prompt: PromptTemplate = cot_reflect_prompt
    cot_examples: str = COT
    reflect_examples: str = COT_REFLECT

    step_n: int = 0
    answer: str = ""
    reflections: List[str] = []
    reflections_str: str = ""
    scratchpad: str = ""
    finished: bool = False

    def step(self) -> None:
        # Think.
        self.scratchpad += f'\nThought:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act.
        action = self.prompt_agent()
        self.scratchpad += f'\nAction:'
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        print(self.scratchpad.split('\n')[-1])  

        self.scratchpad += f'\nObservation: '
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
        else:
            print('Invalid action type, please try again.')

    def run(self, reflexion_strategy: str = None) -> None:
        if self.step_n > 0 and not self.is_correct() and reflexion_strategy:
            self.reflect(reflexion_strategy)
        self.reset()
        self.step()
        self.step_n += 1

    def reflect(self, strategy: str) -> None:
        print('Running Reflexion strategy...')
        if strategy == "last_attempt":
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question, self.reflections[0])
        elif strategy == "reflexion":
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == "last_attempt_and_reflexion":
            self.reflections_str = format_last_attempt(self.question, self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += '\n'+ format_reflections(self.reflections, header = REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}.')
        print(self.reflections_str)

    def reset(self) -> None:
        self.scratchpad = ""
        self.finished = False

    def prompt_reflection(self) -> str:
        prompt = self.reflect_prompt.format(
            examples = self.reflect_examples,
            context = self.context,
            question = self.question,
            scratchpad = self.scratchpad
        )

        out = self.self_reflect_llm(
            [
                HumanMessage(
                    content=prompt,
                )
            ]
        ).content
        return format_step(out)

    def prompt_agent(self) -> str:
        prompt = self.agent_prompt.format(
            examples = self.cot_examples,
            reflections = self.reflections_str,
            context = self.context,
            question = self.question,
            scratchpad = self.scratchpad
        )

        out = self.action_llm(
            [
                HumanMessage(
                    content=prompt,
                )
            ]
        ).content
        return format_step(out)

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)
        
