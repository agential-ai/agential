"""Standard prompting prompts module."""

# ======================================================================== HOTPOTQA ======================================================================== #

STANDARD_INSTRUCTION_HOTPOTQA = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """

# ======================================================================== FEVER ======================================================================== #

STANDARD_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. 

{examples}
(END OF EXAMPLES)

Claim: {question}
A: """

# ======================================================================== TRIVIAQA ======================================================================== #

STANDARD_INSTRUCTION_TRIVIAQA = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """

# ======================================================================== AMBIGNQ ======================================================================== #

STANDARD_INSTRUCTION_AMBIGNQ = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """

# ======================================================================== GSM8K ======================================================================== #

STANDARD_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """

# ======================================================================== SVAMP ======================================================================== #

STANDARD_INSTRUCTION_SVAMP = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """

# ======================================================================== TABMWP ======================================================================== #

STANDARD_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

{question}
A: """

# ======================================================================== HUMANEVAL ======================================================================== #

STANDARD_INSTRUCTION_HUMANEVAL = """Implement the function below. Provide the entire function implementation.
{question}

```python"""

# ======================================================================== MBPP ======================================================================== #

STANDARD_INSTRUCTION_MBPP = """{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

```python"""
