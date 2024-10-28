"""CLIN prompts."""

CLIN_ADAPT_SUMMARY_SYSTEM = (
    """Here is a summary of learnings based on your previous attempts on this task."""
)


CLIN_GEN_ENV_SUMMARY_SYSTEM = """Here is a summary of learnings based on your previous attempts to solve related tasks in some environments. However, your current environment can differ from previous environments in terms of presence of objects, starting location etc."""


CLIN_GEN_TASK_SUMMARY_SYSTEM = """Here is a summary of learnings based on your previous attempts to some tasks in your current environment."""


CLIN_ADAPT_META_SUMMARY_SYSTEM = (
    """Here is a summary of meta-learnings based on your previous attempts on this task."""
)


CLIN_GEN_ENV_META_SUMMARY_SYSTEM = """You are also provided with a set of META LEARNINGS that contains useful insights from agent's previous best attempts to solve the same type of tasks that you are currently solving in different environment configurations. Previous environment configurations may differ from the current one you are in, in terms of presence of objects, starting location etc."""


CLIN_GEN_TASK_META_SUMMARY_SYSTEM = """You are also provided with a set of META LEARNINGS that contains useful insights from agent's previous best attempts to solve DIFFERENT tasks in the SAME environment configuration that you are currently in. These learnings will contain related information about the environment such as presence of objects, starting location, navigational information, etc."""


# ======================================================================== HOTPOTQA ======================================================================== #


CLIN_INSTRUCTION_HOTPOTQA = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

{meta_summary_system}
META LEARNINGS:
{meta_summaries}

{summary_system}
These learnings capture important pre-conditions and mistakes: 
- X MAY BE NECESSARY to Y
- X SHOULD BE NECESSARY to Y
- X MAY NOT CONTRIBUTE to Y
- X DOES NOT CONTRIBUTE to Y

These can be useful for predicting your next action:
{summaries}

Question: {question}{scratchpad}"""


CLIN_SUMMARY_INSTRUCTION_HOTPOTQA = """Generate a summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
Each numbered item in the summary can ONLY be of the form:
- X MAY BE NECESSARY to Y.
- X SHOULD BE NECESSARY to Y.
- X MAY CONTRIBUTE to Y.
- X DOES NOT CONTRIBUTE to Y.

PREVIOUS LEARNINGS:
{previous_trials}

CURRENT TRIAL:
Question: {question}{scratchpad}

Summary of learnings as a numbered list:"""


CLIN_META_SUMMARY_INSTRUCTION_HOTPOTQA = """Generate a meta-summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
These summary of learnings should be general enough to be applicable other types of similar tasks and environments.
Each numbered item in the summary can ONLY be of the form:
- X MAY BE NECESSARY to Y.
- X SHOULD BE NECESSARY to Y.
- X MAY CONTRIBUTE to Y.
- X DOES NOT CONTRIBUTE to Y.

{meta_summary_system}
META LEARNINGS:
{meta_summaries}

PREVIOUS LEARNINGS:
{previous_trials}

CURRENT TRIAL:
Question: {question}{scratchpad}

Meta-summary of learnings as a numbered list:"""
