"""Prompts for ExpeL agent."""

system_template = """You are {ai_name}. {instruction}"""
EXISTING_RULES_AI_NAME = 'an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories'
NON_EXISTENT_RULES_AT_NAME = 'an advanced reasoning agent that can critique past task trajectories of youself'

SYSTEM_CRITIQUE_EXISTING_RULES_INSTRUCTION = """You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps."""
