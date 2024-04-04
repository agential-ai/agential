"""Functional module for CRITIC."""

from discussion_agents.cog.prompts.critic import CRITIC_INSTRUCTION_HOTPOTQA

def _build_agent_prompt(
    question: str,
    examples: str, 
    prompt: str = CRITIC_INSTRUCTION_HOTPOTQA
) -> str:
    pass

def _prompt_agent(
    question: str,
    examples: str,
    prompt: str = CRITIC_INSTRUCTION_HOTPOTQA
) -> str:
    pass