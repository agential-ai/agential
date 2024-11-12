"""Agent Optimizer prompts and fewshot examples."""

# ======================================================================== AGENTOPTIMIZER ======================================================================== #


OPT_PROMPT = """You are a function optimizer. Your task is to maintain a list of functions for the assistant according to the existing function list and conversation history that happens between the assistant and the user.
You can perform one of the following four actions to manipulate the function list using the functions you have:
1. Revise one existing function (using revise_function).
2. Remove one existing function (using remove_function).
3. Add one new function (using add_function).
4. Directly return "TERMINATE" to me if no more actions are needed for the current function list.

Below are the principles that you need to follow for taking these four actions.
(1) Revise one existing function:
1. Pay more attention to the failed tasks and corresponding error information, and optimize the function used in these tasks according to the conversation history if needed.
2. A failed function call can occur due to incorrect input arguments (missing arguments) or an incorrect function code implementation. You should focus more on the function code implementation and make it easy to get success function call.
3. Do not revise the function that you think works well and plays a critical role in solving the problems according to the conversation history. Only making revisions if needed.
4. Sometimes, a NameError may occur. To fix this error, you can either revise the name of the function in the code implementation or revise the name of the function call to make these two names consistent.
(2) Remove one existing function:
1. Only remove the function that you think is not needed anymore in future tasks.
(3) Add one new function:
1. The added function should be general enough to be used in future tasks. For instance, if you encounter a problem that this function can solve, or one step of it, you can use the generated function directly instead of starting from scratch
2. The added new function should solve a higher-level question that encompasses the original query and extend the code's functionality to make it more versatile and widely applicable.
3. Replace specific strings or variable names with general variables to enhance the tool's applicability to various queries. All names used inside the function should be passed in as arguments.
Below is an example of a function that potentially deserves to be adde in solving MATH problems, which can be used to solve a higher-level question:
{{
    \"name\": \"evaluate_expression\",
    \"description\": \"Evaluate arithmetic or mathematical expressions provided as strings.\",
    \"arguments\": {{
        \"expression\": {{
            \"type\": \"string\",
            \"description\": \"The mathematical expression to evaluate.\"
        }}
    }},
    \"packages\": \"sympy\",
    \"code\": \"from sympy import sympify, SympifyError\\n\\ndef evaluate_expression(expression):\\n    try:\\n        result = sympify(expression)\\n        if result.is_number:\\n            result = float(result)\\n        else:\\n            result = str(result)\\n        return result\\n    except SympifyError as e:\\n        return str(e)\"
}}
(4) Directly return "TERMINATE":
If you think there is no need to perform any other actions for the current function list since the current list is optimal more actions will harm the performance in future tasks. Please directly reply to me with "TERMINATE".

One function signature includes the following five elements:
1. Function name
2. Function description
3. JSON schema of arguments encoded as a string
4. A list of package names imported by the function packages
5. The code implementation

Below are the signatures of the current functions:
List A: {best_functions}.
The following list are the function signatures that you have after taking {actions_num} actions to manipulate List A:
List B: {incumbent_functions}.

{accumulated_experience}

Here are {best_conversations_num} conversation histories of solving {best_conversations_num} tasks using List A.
History:
{best_conversations_history}

{statistic_informations}

According to the information I provide, please take one of four actions to manipulate list B using the functions you know.
Instead of returning TERMINATE directly or taking no action, you should try your best to optimize the function list. Only take no action if you really think the current list is optimal, as more actions will harm performance in future tasks.
Even adding a general function that can substitute the assistant’s repeated suggestions of Python code with the same functionality could also be helpful.
"""



# ======================================================================== HOTPOTQA ======================================================================== #


REACT_INSTRUCTION_HOTPOTQA = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}{scratchpad}"""


# ======================================================================== FEVER ======================================================================== #


REACT_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. Thought can reason about the current situation, and Action can be two types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

Claim: {question}{scratchpad}"""


# ======================================================================== AMBIGNQ ======================================================================== #


REACT_INSTRUCTION_AMBIGNQ = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be two types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}{scratchpad}"""


# ======================================================================== TRIVIAQA ======================================================================== #


REACT_INSTRUCTION_TRIVIAQA = """Answer a trivia question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}{scratchpad}"""


# ======================================================================== GSM8K ======================================================================== #


REACT_INSTRUCTION_GSM8K = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[code], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[code], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}
{scratchpad}"""


# ======================================================================== SVAMP ======================================================================== #


REACT_INSTRUCTION_SVAMP = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[code], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[code], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}
{scratchpad}"""


# ======================================================================== TABMWP ======================================================================== #


REACT_INSTRUCTION_TABMWP = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[code], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[code], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

{question}
{scratchpad}"""


# ======================================================================== HUMANEVAL ======================================================================== #


REACT_INSTRUCTION_HUMANEVAL = """Answer a coding question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Implement[<insert your code here>], which implements the function to answer the question.
(2) Test[<insert your code here>], which implements assert statement test cases to test the implemented code.
(3) Finish[<insert your answer here>], which returns the code implementation and finishes the task.
You have a maximum of {max_steps} steps.

```python
{question}
    pass
```

{scratchpad}"""


# ======================================================================== MBPP ======================================================================== #


REACT_INSTRUCTION_MBPP = """Answer a coding question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
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

{scratchpad}"""
