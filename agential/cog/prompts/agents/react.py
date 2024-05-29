"""ReAct prompts and fewshot examples."""

# ======================================================================== HOTPOTQA ======================================================================== #


REACT_INSTRUCTION_HOTPOTQA = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

You have a maximum of {max_steps} steps.
Question: {question}{scratchpad}"""


# ======================================================================== FEVER ======================================================================== #


REACT_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION.  and Action can be two types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(3) Finish[answer], which returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

You have a maximum of {max_steps} steps.
Claim: {question}{scratchpad}"""


# ======================================================================== AMBIGNQ ======================================================================== #


REACT_INSTRUCTION_AMBIGNQ = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be two types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(3) Finish[answer], which returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

You have a maximum of {max_steps} steps.
Question: {question}{scratchpad}"""


# ======================================================================== TRIVIAQA ======================================================================== #


REACT_INSTRUCTION_TRIVIAQA = """Answer a trivia question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

You have a maximum of {max_steps} steps.
Question: {question}{scratchpad}"""


# ======================================================================== MBPP ======================================================================== #


REACT_INSTRUCTION_MBPP = """Answer a coding question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Implement[code], which implements the function to answer the question.
(2) Test[keyword], which implements assert statement test cases to test the implemented code.
(3) Finish[answer], which returns the code implementation and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

You have a maximum of {max_steps} steps.
You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{scratchpad}"""

