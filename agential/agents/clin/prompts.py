"""CLIN prompts."""

CLIN_ADAPT_SUMMARY_SYSTEM = """Here is a summary of learnings based on your previous attempts on this task."""


CLIN_GEN_ENV_SUMMARY_SYSTEM = """Here is a summary of learnings based on your previous attempts to solve related tasks in some environments. However, your current environment can differ from previous environments in terms of presence of objects, starting location etc."""


CLIN_GEN_TASK_SUMMARY_SYSTEM = """Here is a summary of learnings based on your previous attempts to some tasks in your current environment."""


# ======================================================================== HOTPOTQA ======================================================================== #


CLIN_INSTRUCTION_HOTPOTQA = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

{summary_system}
These learnings capture important pre-conditions and mistakes: 
- X MAY BE NECESSARY to Y
- X SHOULD BE NECESSARY to Y
- X MAY NOT CONTRIBUTE to Y
- X DOES NOT CONTRIBUTE to Y

These can be useful for predicting your next action:
{summary}

Question: {question}{scratchpad}"""

