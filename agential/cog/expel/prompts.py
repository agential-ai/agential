"""Prompts for ExpeL agent."""

# Insight Extraction: Comparison system prefix prompts.
SYSTEM_TEMPLATE = """You are {ai_name}. {instruction}"""
EXISTING_INSIGHTS_AI_NAME = "an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories"
NON_EXISTENT_INSIGHTS_AT_NAME = (
    "an advanced reasoning agent that can critique past task trajectories of youself"
)

# Insight Extraction: extraction type specifications.
SYSTEM_CRITIQUE_EXISTING_INSIGHTS_INSTRUCTION = """You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps."""
SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_INSIGHTS_INSTRUCTION = """You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer."""

# Insight Extraction: Comparison formatting specification.
FORMAT_INSIGHTS_OPERATION_TEMPLATE = """<OPERATION> <RULE NUMBER>: <RULE>

The available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. """

# Insight Extraction: Comparison prompt.
HUMAN_CRITIQUE_EXISTING_INSIGHTS_TEMPLATE = (
    """
Here are the two previous trials to compare and critique:
TRIAL TASK:
{question}

SUCCESSFUL TRIAL:
{success_traj}

FAILED TRIAL:
{failed_traj}

Here are the EXISTING RULES:
{existing_insights}

By examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:

"""
    + FORMAT_INSIGHTS_OPERATION_TEMPLATE
)

# Insight Extraction: All success prompt.
HUMAN_CRITIQUE_EXISTING_INSIGHTS_ALL_SUCCESS_TEMPLATE = (
    """
Here are the trials:
{success_trajs}

Here are the EXISTING RULES:
{existing_insights}

By examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:

"""
    + FORMAT_INSIGHTS_OPERATION_TEMPLATE
)

# Insight Extraction: Suffix prompt depending on insight count limit (full/not full).
CRITIQUE_SUMMARY_SUFFIX_FULL = """Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"""
CRITIQUE_SUMMARY_SUFFIX_NOT_FULL = (
    """Below are the operations you do to the above list of EXISTING RULES:"""
)


# ======================================================================== HOTPOTQA ======================================================================== #


# Task Inference: Similar to REFLEXION_REACT_INSTRUCTION_HOTPOTQA, but formatted as the ExpeL Stage 3 task inference prompt.
EXPEL_REFLEXION_REACT_INSTRUCTION_HOTPOTQA = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

The following are some experience you gather on a similar task of question answering using Wikipedia API. Use these as references to help you perform this task:
{insights}

{reflections}

Question: {question}{scratchpad}"""


EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


# ======================================================================== FEVER ======================================================================== #


EXPEL_REFLEXION_REACT_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

The following are some experience you gather on a similar task of question answering using Wikipedia API. Use these as references to help you perform this task:
{insights}

{reflections}

Claim: {question}{scratchpad}"""


EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_FEVER = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Claim: {question}{scratchpad}

Reflection:"""


# ======================================================================== AMBIGNQ ======================================================================== #


EXPEL_REFLEXION_REACT_INSTRUCTION_AMBIGNQ = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

The following are some experience you gather on a similar task of question answering using Wikipedia API. Use these as references to help you perform this task:
{insights}

{reflections}

Question: {question}{scratchpad}"""


EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_AMBIGNQ = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


# ======================================================================== TRIVIAQA ======================================================================== #


EXPEL_REFLEXION_REACT_INSTRUCTION_TRIVIAQA = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

The following are some experience you gather on a similar task of question answering using Wikipedia API. Use these as references to help you perform this task:
{insights}

{reflections}

Question: {question}{scratchpad}"""


EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_TRIVIAQA = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


# ======================================================================== GSM8K ======================================================================== #


EXPEL_REFLEXION_REACT_INSTRUCTION_GSM8K = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[code], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[code], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

The following are some experience you gather on a similar task of math reasoning using Python. Use these as references to help you perform this task:
{insights}

{reflections}

Question: {question}{scratchpad}"""


EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


# ======================================================================== SVAMP ======================================================================== #


EXPEL_REFLEXION_REACT_INSTRUCTION_SVAMP = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[code], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[code], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

The following are some experience you gather on a similar task of math reasoning using Python. Use these as references to help you perform this task:
{insights}

{reflections}

Question: {question}{scratchpad}"""


EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_SVAMP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


# ======================================================================== TABMWP ======================================================================== #


EXPEL_REFLEXION_REACT_INSTRUCTION_TABMWP = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[code], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[code], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

The following are some experience you gather on a similar task of math reasoning using Python. Use these as references to help you perform this task:
{insights}

{reflections}

{question}
{scratchpad}"""


EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_TABMWP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
{question}
{scratchpad}

Reflection:"""


# ======================================================================== HUMANEVAL ======================================================================== #


EXPEL_REFLEXION_REACT_INSTRUCTION_HUMANEVAL = """Answer a coding question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Implement[<insert your code here>], which implements the function to answer the question.
(2) Test[<insert your code here>], which implements assert statement test cases to test the implemented code.
(3) Finish[<insert your answer here>], which returns the code implementation and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

```python
{question}
    pass
```

The following are some experience you gather on a similar programming task using Python. Use these as references to help you perform this task:
{insights}

{reflections}

{scratchpad}"""


EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_HUMANEVAL = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
```python
{question}
    pass
```

{scratchpad}

Reflection:"""


# ======================================================================== MBPP ======================================================================== #


EXPEL_REFLEXION_REACT_INSTRUCTION_MBPP = """Answer a coding question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Implement[code], which implements the function to answer the question.
(2) Test[code], which implements assert statement test cases to test the implemented code.
(3) Finish[answer], which returns the code implementation and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

The following are some experience you gather on a similar programming task using Python. Use these as references to help you perform this task:
{insights}

{reflections}

{scratchpad}"""


EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{scratchpad}

Reflection:"""