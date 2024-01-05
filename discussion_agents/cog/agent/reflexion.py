"""Reflexion Agent."""
from typing import Any

from discussion_agents.cog.agent.base import BaseAgent

class ReflexionCoT(BaseAgent):
    self_reflect_llm: Any
    action_llm: Any
    question: str,
    context: str,
    key: str,
    agent_prompt: PromptTemplate = cot_reflect_agent_prompt,
    reflect_prompt: PromptTemplate = cot_reflect_prompt,
    cot_examples: str = COT,
    reflect_examples: str = COT_REFLECT,
    self_reflect_llm: AnyOpenAILLM = AnyOpenAILLM(
                            temperature=0,
                            max_tokens=250,
                            model_name="gpt-3.5-turbo",
                            model_kwargs={"stop": "\n"},
                            openai_api_key=os.environ['OPENAI_API_KEY']),
    action_llm: AnyOpenAILLM = AnyOpenAILLM(
                            temperature=0,
                            max_tokens=250,
                            model_name="gpt-3.5-turbo",
                            model_kwargs={"stop": "\n"},
                            openai_api_key=os.environ['OPENAI_API_KEY']),