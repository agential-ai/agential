"""Prompts for CoT."""

# ======================================================================== HOTPOTQA ======================================================================== #

COT_INSTRUCTION_HOTPOTQA = """{examples}
(END OF EXAMPLES)

Question: {question}
Thought: """

# ======================================================================== FEVER ======================================================================== #

COT_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. 

{examples}
(END OF EXAMPLES)

Claim: {question}
Thought: """

# ======================================================================== TRIVIAQA ======================================================================== #

COT_INSTRUCTION_TRIVIAQA = """{examples}
(END OF EXAMPLES)

Question: {question}
Thought: """

# ======================================================================== AMBIGNQ ======================================================================== #

COT_INSTRUCTION_AMBIGNQ = """{examples}
(END OF EXAMPLES)

Question: {question}
Thought: """

# ======================================================================== GSM8K ======================================================================== #

COT_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Question: {question}
Thought: """

# ======================================================================== SVAMP ======================================================================== #

COT_INSTRUCTION_SVAMP = """{examples}
(END OF EXAMPLES)

Question: {question}
Thought: """

# ======================================================================== TABMWP ======================================================================== #

COT_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

{question}
Thought: """

# ======================================================================== HUMANEVAL ======================================================================== #

COT_INSTRUCTION_HUMANEVAL = """Question: {question}
Answer: """

# ======================================================================== MBPP ======================================================================== #

COT_INSTRUCTION_MBPP = """{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

Thought: """
