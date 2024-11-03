"""CLIN prompts."""

CLIN_ADAPT_SUMMARY_SYSTEM = (
    """Here is a summary of learnings based on your previous attempts on this task."""
)


CLIN_GEN_ENV_SUMMARY_SYSTEM = """Here is a summary of learnings based on your previous attempts to solve related tasks in some environments. However, your current environment can differ from previous environments in terms of presence of objects, starting location etc."""


CLIN_GEN_TASK_SUMMARY_SYSTEM = """Here is a summary of learnings based on your previous attempts to some tasks in your current environment."""


CLIN_ADAPT_META_SUMMARY_SYSTEM = """Here is a summary of meta-learnings based on your previous attempts on this task."""


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


# ======================================================================== FEVER ======================================================================== #


CLIN_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. Thought can reason about the current situation, and Action can be three types: 
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

Claim: {question}{scratchpad}"""


CLIN_SUMMARY_INSTRUCTION_FEVER = """Generate a summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
Each numbered item in the summary can ONLY be of the form:
- X MAY BE NECESSARY to Y.
- X SHOULD BE NECESSARY to Y.
- X MAY CONTRIBUTE to Y.
- X DOES NOT CONTRIBUTE to Y.

PREVIOUS LEARNINGS:
{previous_trials}

CURRENT TRIAL:
Claim: {question}{scratchpad}

Summary of learnings as a numbered list:"""


CLIN_META_SUMMARY_INSTRUCTION_FEVER = """Generate a meta-summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
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
Claim: {question}{scratchpad}

Meta-summary of learnings as a numbered list:"""


# ======================================================================== AMBIGNQ ======================================================================== #


CLIN_INSTRUCTION_AMBIGNQ = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
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


CLIN_SUMMARY_INSTRUCTION_AMBIGNQ = """Generate a summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
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


CLIN_META_SUMMARY_INSTRUCTION_AMBIGNQ = """Generate a meta-summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
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


# ======================================================================== TRIVIAQA ======================================================================== #


CLIN_INSTRUCTION_TRIVIAQA = """Solve a trivia question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
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


CLIN_SUMMARY_INSTRUCTION_TRIVIAQA = """Generate a summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
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


CLIN_META_SUMMARY_INSTRUCTION_TRIVIAQA = """Generate a meta-summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
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


# ======================================================================== GSM8K ======================================================================== #


CLIN_INSTRUCTION_GSM8K = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[\\n```python\\n<code>\\n```\\n], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[\\n```python\\n<code>\\n```\\n], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.
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


CLIN_SUMMARY_INSTRUCTION_GSM8K = """Generate a summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
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


CLIN_META_SUMMARY_INSTRUCTION_GSM8K = """Generate a meta-summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
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


# ======================================================================== SVAMP ======================================================================== #


CLIN_INSTRUCTION_SVAMP = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[\\n```python\\n<code>\\n```\\n], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[\\n```python\\n<code>\\n```\\n], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.
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


CLIN_SUMMARY_INSTRUCTION_SVAMP = """Generate a summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
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


CLIN_META_SUMMARY_INSTRUCTION_SVAMP = """Generate a meta-summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
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


# ======================================================================== TABMWP ======================================================================== #


CLIN_INSTRUCTION_TABMWP = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[\\n```python\\n<code>\\n```\\n], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[\\n```python\\n<code>\\n```\\n], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.
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

{question}
{scratchpad}"""


CLIN_SUMMARY_INSTRUCTION_TABMWP = """Generate a summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
Each numbered item in the summary can ONLY be of the form:
- X MAY BE NECESSARY to Y.
- X SHOULD BE NECESSARY to Y.
- X MAY CONTRIBUTE to Y.
- X DOES NOT CONTRIBUTE to Y.

PREVIOUS LEARNINGS:
{previous_trials}

CURRENT TRIAL:
{question}
{scratchpad}

Summary of learnings as a numbered list:"""


CLIN_META_SUMMARY_INSTRUCTION_TABMWP = """Generate a meta-summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
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
{question}
{scratchpad}

Meta-summary of learnings as a numbered list:"""


# ======================================================================== HUMANEVAL ======================================================================== #


CLIN_INSTRUCTION_HUMANEVAL = """Answer a coding question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Implement[\\n```python\\n<insert your code here>\\n```\\n], which implements the function to answer the question.
(2) Test[\\n```python\\n<insert your code here>\\n```\\n], which implements assert statement test cases to test the implemented code.
(3) Finish[\\n```python\\n<insert your answer here>\\n```\\n], which returns the code implementation and finishes the task.
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

```python
{question}
    pass
```

{scratchpad}"""


CLIN_SUMMARY_INSTRUCTION_HUMANEVAL = """Generate a summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
Each numbered item in the summary can ONLY be of the form:
- X MAY BE NECESSARY to Y.
- X SHOULD BE NECESSARY to Y.
- X MAY CONTRIBUTE to Y.
- X DOES NOT CONTRIBUTE to Y.

PREVIOUS LEARNINGS:
{previous_trials}

CURRENT TRIAL:
```python
{question}
    pass
```

{scratchpad}

Summary of learnings as a numbered list:"""


CLIN_META_SUMMARY_INSTRUCTION_HUMANEVAL = """Generate a meta-summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
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
```python
{question}
    pass
```

{scratchpad}

Meta-summary of learnings as a numbered list:"""


# ======================================================================== MBPP ======================================================================== #


CLIN_INSTRUCTION_MBPP = """Answer a coding question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Implement[\\n```python\\n<insert your code here>\\n```\\n], which implements the function to answer the question.
(2) Test[\\n```python\\n<insert your code here>\\n```\\n], which implements assert statement test cases to test the implemented code.
(3) Finish[\\n```python\\n<insert your answer here>\\n```\\n], which returns the code implementation and finishes the task.
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

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{scratchpad}"""


CLIN_SUMMARY_INSTRUCTION_MBPP = """Generate a summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
Each numbered item in the summary can ONLY be of the form:
- X MAY BE NECESSARY to Y.
- X SHOULD BE NECESSARY to Y.
- X MAY CONTRIBUTE to Y.
- X DOES NOT CONTRIBUTE to Y.

PREVIOUS LEARNINGS:
{previous_trials}

CURRENT TRIAL:
You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{scratchpad}

Summary of learnings as a numbered list:"""


CLIN_META_SUMMARY_INSTRUCTION_MBPP = """Generate a meta-summary of learnings, as a numbered list, that will help the agent to successfully accomplish the task.
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
You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{scratchpad}

Meta-summary of learnings as a numbered list:"""
