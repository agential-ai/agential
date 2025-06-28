"""ReAct prompts and fewshot examples."""

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
(1) Implement[\n```python\n<insert your code here>\n```], which implements the function to answer the question.
(2) Test[\n```python\n<insert your code here>\n```], which implements assert statement test cases to test the implemented code.
(3) Finish[\n```python\n<insert your answer here>\n```], which returns the code implementation and finishes the task.
You have a maximum of {max_steps} steps.

```python
{question}
    pass
```

{scratchpad}"""


# ======================================================================== MBPP ======================================================================== #


REACT_INSTRUCTION_MBPP = """Answer a coding question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Implement[\n```python\n<insert your code here>\n```], which implements the function to answer the question.
(2) Test[\n```python\n<insert your code here>\n```], which implements assert statement test cases to test the implemented code.
(3) Finish[\n```python\n<insert your answer here>\n```], which returns the code implementation and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{scratchpad}"""


AMBIGNQ_FEWSHOT_EXAMPLES_REACT = """Question: When did men's figure skating become a summer Olympic sport?
Thought 1: Let's search the question in google
Action 1: Search[When did men's figure skating become a summer Olympic sport?]
Observation 1: [Figure skating at the Olympic Games - Wikipedia] Figure skating was first contested in the Olympic Games at the 1908 Summer Olympics . Since 1924, the sport has been a part of the Winter Olympic Games .
Thought 2: The evidence suggests Figure skating became an Olympic sport at the 1908 Summer Olympics, and has been a part of the Winter Olympic Games in 1924.
Action 2: Finish[1908]

---

Question: When was the last time the USA men's national soccer team missed the World Cup?
Thought 1: Let's search the question in google
Action 1: Search[When was the last time the USA men's national soccer team missed the World Cup? site: wikipedia.org]
Observation 1: [United States at the FIFA World Cup - Wikipedia] The United States participated in every World Cup from 1990 through 2014, but did not qualify in 2018, marking first time the team had missed a World Cup since 1986. They returned to the World Cup by qualifying for the 2022 edition .
Thought 2: The evidence suggests that the USA men's national soccer team did not qualify for the World Cup in 2018, but qualify for 2022, 2018 > 1986, so the last time the USA men's national soccer team missed the World Cup was in 2018.
Action 2: Finish[2018]

---

Question: What does barium do in a CT scan?
Thought 1: Let's search the question in Google.
Action 1: Search[What does barium do in a CT scan? site: wikipedia.org]
Observation 1: [Barium sulfate suspension - Wikipedia] Barium sulfate suspension, often simply called barium, is a contrast agent used during X-rays. Specifically, it is used to improve visualization of the gastrointestinal tract (esophagus, stomach, intestines) on plain X-ray or computed tomography.
Thought 2: The evidence suggests that barium is used to "improve visualization of the gastrointestinal tract."
Action 2: Search[Why barium is used in CT scans?]
Observation 2: [What To Expect Before Your CT Scan - St. Elizabeth's Medical Center] The barium helps to highlight body areas for the CT scan. If you are having a scan of a body part other than the abdomen, you should arrive 30 minutes before your appointed time.
Thought 3: The evidence suggests that barium helps "highlight body areas" for the CT scan.
Action 3: Search[Barium CT scan mechanism]
Observation 3: [Radiopaedia.org - Barium Studies] Barium sulfate, due to its radio-opacity, absorbs X-rays and appears white on the CT images, providing a clear outline of the gastrointestinal tract.
Thought 4: The evidence suggests that barium absorbs X-rays and appears white on CT images, outlining the gastrointestinal tract.
Action 4: Finish[to improve visualization of the gastrointestinal tract]

---

Question: Where was the fort located at which the first shot of the civil war was fired?
Thought 1: Let's search the question on Google.
Action 1: Search[Where was the fort located at which the first shot of the civil war was fired?]
Observation 1: [Civil War Begins - U.S. Senate] April 12, 1861 At 4:30 a.m. on April 12, 1861, Confederate troops fired on Fort Sumter in South Carolina’s Charleston Harbor. Less than 34 hours later, Union forces surrendered.
Thought 2: The evidence suggests that the first shot of the Civil War was fired at Fort Sumter. Fort Sumter is located in South Carolina’s Charleston Harbor. This step is to verify this information from another source to ensure accuracy.
Action 2: Search[Fort Sumter first shot of the Civil War location]
Observation 2: [History Channel] The first shots of the Civil War were fired at Fort Sumter, located in Charleston Harbor, South Carolina, on April 12, 1861, when Confederate forces opened fire on the fort.
Thought 3: The information is consistent across multiple sources, confirming that the first shot of the Civil War was fired at Fort Sumter in Charleston Harbor, South Carolina.
Action 3: Finish[Charleston Harbor, South Carolina]

---

Question: When did Nando's come to the UK?
Thought 1: Let's search the question in Google.
Action 1: Search[When did Nando's come to the UK? site: wikipedia.org]
Observation 1: [Nando's - Wikipedia] Nando's opened its first restaurants within the United Kingdom, in 1992, in the west London suburbs of Ealing and Earls Court, initially focusing on takeaway food.
Thought 2: The evidence suggests that Nando's first opened in the UK in 1992.
Action 2: Search[confirmation Nando's first opened in the UK in 1992]
Observation 2: [The Independent - Nando's History] Nando's made its debut in the UK in 1992, starting in west London.
Thought 3: Since multiple sources confirm that Nando's opened in the UK in 1992, it is likely accurate.
Action 3: Finish[1992]"""


FEVER_FEWSHOT_EXAMPLES_REACT = """Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought 1: I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993,[1] and had his breakthrough role in Denmark with the film Nightwatch (1994). He played Jaime Lannister in the HBO fantasy drama series Game of Thrones, for which he received two Primetime Emmy Award nominations for Outstanding Supporting Actor in a Drama Series.. Coster-Waldau has appeared in numerous films in his native Denmark and Scandinavia, including Headhunters (2011) and A Thousand Times Good Night (2013). In the U.S, his debut film role was in the war film Black Hawk Down (2001), playing Medal of Honor recipient Gary Gordon.[2] He then played a detective in the short-lived Fox television series New Amsterdam (2008), and appeared in the 2009 Fox television film Virtuality, originally intended as a pilot.
Thought 2: Because he "appeared in the 2009 Fox television film Virtuality", he should have worked with the Fox Broadcasting Company.
Action 2: Finish[SUPPORTS]

---

Claim: Stranger Things is set in Bloomington, Indiana.
Thought 1: I should search for Stranger Things, and see if it is set in Bloomington, Indiana.
Action 1: Search[Stranger Things]
Observation 1: Stranger Things is an American science fiction horror drama television series created by the Duffer Brothers. Set in the 1980s, primarily in the fictional town of Hawkins, Indiana, the series centers on a number of mysteries and supernatural events occurring around the town and their impact on an ensemble of child and adult characters. 
Thought 2: The observation says that it is set in a "fictional town of Hawkins, Indiana", so it is not set in Bloomington.
Action 2: Finish[REFUTES]

---

Claim: "Beautiful" by Christina Aguilera reached number two on the Billboard Hot 100 in 2003.
Thought 1: I need to search the song Beautiful and find if it reached number two on the Billboard Hot 100 in 2003.
Action 1: Search[Beautiful]
Observation 1: Could not find [Beautiful]. Similar: ['Beautiful', 'Beautiful, Beautiful', 'A Beautiful Mind (film)', 'Beautiful (Christina Aguilera song)', 'Life Is Beautiful'].
Thought 2: From suggestions, I should search "Beautiful (Christina Aguilera song)" to find the song.
Action 2: Search[Beautiful (Christina Aguilera song)]
Observation 2: "Beautiful" is a song recorded by American singer Christina Aguilera for her fourth studio album, Stripped (2002).
Thought 3: It does not mention Billboard, so I need to look up "Billboard Hot 100" to find if it reached number two on it in 2003.
Action 3: Lookup[Billboard Hot 100]
Observation 3: (Result 1 / 3) The song peaked at number two on the Billboard Hot 100 in the United States, where it was certified Gold for 500,000 units shipped.
Thought 4: It only says the song peaked at number two on the Billboard Hot 100, but not if it was in 2003. I am not sure if this claim is true or not.
Action 4: Finish[NOT ENOUGH INFO]"""


GSM8K_FEWSHOT_EXAMPLES_REACT = """Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Thought 1: First, I need to find out how many lollipops Jason gave to Denny.
Action 1: Calculate[
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
answer = jason_lollipops_after - jason_lollipops_initial
```
]
Observation 1: 
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
answer = jason_lollipops_after - jason_lollipops_initial
```
Execution Status: Done
Output: answer = -8
Thought 2: The answer should be positive because Jason gave lollipops to Denny. I need to reverse the subtraction.
Action 2: Calculate[
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
answer = jason_lollipops_initial - jason_lollipops_after
```
]
Observation 2: 
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
answer = jason_lollipops_initial - jason_lollipops_after
```
Execution Status: Done
Output: answer = 8
Thought 3: Jason gave 8 lollipops to Denny.
Action 3: Finish[
```python
answer = 8
```
]
Observation 3: 
```python
answer = 8
```

---

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Thought 1: I need to determine how many trees were planted.
Action 1: Calculate[
```python
trees_initial = 15
trees_after = 21
answer = trees_initial - trees_after
```
]
Observation 1: 
```python
trees_initial = 15
trees_after = 21
answer = trees_initial - trees_after
```
Execution Status: Done
Output: answer = -6
Thought 2: The answer should be positive because trees are being added, not removed. I need to reverse the subtraction.
Action 2: Calculate[
```python
trees_initial = 15
trees_after = 21
answer = trees_after - trees_initial
```
]
Observation 2: 
```python
trees_initial = 15
trees_after = 21
answer = trees_after - trees_initial
```
Execution Status: Done
Output: answer = 6
Thought 3: The workers planted 6 trees today.
Action 3: Finish[
```python
answer = 6
```
]
Observation 3: 
```python
answer = 6
```

---

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Thought 1: I need to calculate the total number of toys received.
Action 1: Calculate[
```pyhton 
toys_received_mom = 2
toys_received_dad = 2
toys_received = toys_received_mom + toys_received_dad
```
]
Observation 1:
```pyhton 
toys_received_mom = 2
toys_received_dad = 2
answer = toys_received_mom + toys_received_dad
```
Execution Status: Done
Output: answer = 4
Thought 2: I need to calculate the total number of toys Shawn has after receiving gifts from his parents.
Action 2: Calculate[
```python
toys_initial = 5
toys_received = 2 * 3
answer = toys_initial + toys_received
```
]
Observation 2: 
```python
toys_initial = 5
toys_received = 2 * 3
answer = toys_initial + toys_received
```
Execution Status: Done
Output: answer = 11
Thought 3: I incorrectly calculated the total toys received. Shawn got 2 toys each from his mom and dad, so the correct total received should be 2 + 2.
Action 3: Calculate[
```python
toys_initial = 5
toys_received = 2 + 2
answer = toys_initial + toys_received
```
]
Observation 3: 
```python
toys_initial = 5
toys_received = 2 + 2
answer = toys_initial + toys_received
```
Execution Status: Done
Output: answer = 9
Thought 4: Shawn now has 9 toys.
Action 4: Finish[
```python
answer = 9
```
]
Observation 4: 
```python
answer = 9
```

---

Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
Thought 1: I need to calculate the total number of computers installed from Monday to Thursday.
Action 1: Calculate[
```python
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between Monday and Thursday
computers_added = computers_per_day * num_days
answer = computers_initial + computers_added
```
]
Observation 1: 
```python
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between Monday and Thursday
computers_added = computers_per_day * num_days
answer = computers_initial + computers_added
```
Execution Status: Done
Output: answer = 29
Thought 2: There are now 29 computers in the server room.
Action 2: Finish[
```python
answer = 29
```
]
Observation 2:
```python
answer = 29
```

---

Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
Thought 1: I need to find out how many golf balls Michael has after losing some on Tuesday and Wednesday.
Action 1: Calculate[
```python
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
answer = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
```
]
Observation 1: 
```python
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
answer = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
```
Execution Status: Done
Output: answer = 33
Thought 2: Michael has 33 golf balls left at the end of Wednesday.
Action 2: Finish[
```python
answer = 33
```
]
Observation 2: 
```python
answer = 33
```

---

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Thought 1: I need to calculate the total number of cars in the parking lot after 2 more cars arrive.
Action 1: Calculate[
```python
cars_initial = 3
cars_arrived = 2
answer = cars_initial + cars_arrived
```
]
Observation 1: 
```python
cars_initial = 3
cars_arrived = 2
answer = cars_initial + cars_arrived
```
Execution Status: Done
Output: answer = 5
Thought 2: There are 5 cars in the parking lot now.
Action 2: Finish[
```python
answer = 5
```
]
Observation 2: 
```python
answer = 5
```

---

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Thought 1: I need to calculate how much money Olivia has left after buying bagels.
Action 1: Calculate[
```python
money_initial = 23
bagels = 5
bagel_cost = 3
money_spent = bagels * bagel_cost
answer = money_initial - money_spent
```
]
Observation 1: 
```python
money_initial = 23
bagels = 5
bagel_cost = 3
money_spent = bagels * bagel_cost
answer = money_initial - money_spent
```
Execution Status: Done
Output: answer = 8
Thought 2: Olivia has $8 left.
Action 2: Finish[
```python
answer = 8
```
]
Observation 2: 
```python
answer = 8
```

---

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Thought 1: I need to find out the total number of chocolates left after Leah and her sister ate some.
Action 1: Calculate[
```python
leah_chocolates = 32
sister_chocolates = 42
chocolates_eaten = 35
answer = leah_chocolates - chocolates_eaten
```
]
Observation 1:
```python
leah_chocolates = 32
sister_chocolates = 42
chocolates_eaten = 35
answer = leah_chocolates - chocolates_eaten
```
Execution Status: Done
Output: answer = -3
Thought 2: The calculation is incorrect because it only considers Leah's chocolates. I need to include her sister's chocolates as well.
Action 2: Calculate[
```python
leah_chocolates = 32
sister_chocolates = 42
chocolates_eaten = 35
answer = sister_chocolates - chocolates_eaten
```
]
Observation 2:
```python
leah_chocolates = 32
sister_chocolates = 42
chocolates_eaten = 35
answer = sister_chocolates - chocolates_eaten
```
Execution Status: Done
Output: answer = 7
Thought 3: The calculation is still incorrect because it only considers the sister's chocolates. I need to consider the total number of chocolates initially.
Action 3: Calculate[
```python
leah_chocolates = 32
sister_chocolates = 42
chocolates_eaten = 35
answer = (leah_chocolates + sister_chocolates) - chocolates_eaten - sister_chocolates
```
]
Observation 3:
```python
leah_chocolates = 32
sister_chocolates = 42
chocolates_eaten = 35
answer = (leah_chocolates + sister_chocolates) - chocolates_eaten - sister_chocolates
```
Execution Status: Done
Output: answer = -3
Thought 4: The calculation is still incorrect because it incorrectly subtracts the sister's chocolates again. I need to correct this error.
Action 4: Calculate[
```python
leah_chocolates = 32
sister_chocolates = 42
total_chocolates = leah_chocolates + sister_chocolates
chocolates_eaten = 35
answer = total_chocolates - chocolates_eaten
```
]
Observation 4:
```python
leah_chocolates = 32
sister_chocolates = 42
total_chocolates = leah_chocolates + sister_chocolates
chocolates_eaten = 35
answer = total_chocolates - chocolates_eaten
```
Execution Status: Done
Output: answer = 39
Thought 5: They have 39 chocolates left.
Action 5: Finish[
```python
answer = 39
```
]
Observation 5:
```python
answer = 39
```"""


HOTPOTQA_FEWSHOT_EXAMPLES_REACT = """Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]

---

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: I need to confirm who Matt Groening is and his connection to "The Simpsons".
Action 1: Search[Matt Groening]
Observation 1: Matt Groening is an American cartoonist, writer, producer, and animator, best known for creating the animated television series "The Simpsons".
Thought 2: Since Matt Groening created "The Simpsons" and named the character Milhouse, I need to find out whom he named Milhouse after.
Action 2: Search[Milhouse inspiration]
Observation 2: Matt Groening named the character Milhouse after Richard Milhous Nixon, the 37th President of the United States.
Thought 3: To ensure accuracy, I should verify the connection between Matt Groening naming Milhouse after Richard Nixon.
Action 3: Search[Milhouse named after Richard Nixon]
Observation 3: It is confirmed that Matt Groening named Milhouse after Richard Nixon, as stated in various sources discussing the origins of "The Simpsons" characters.
Thought 4: Allie Goertz's song about Milhouse connects back to Matt Groening's naming of the character after Richard Nixon.
Action 4: Finish[Richard Nixon]

---

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action 3: Finish[The Saimaa Gesture]

---

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]

---

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search for the founding dates of Arthur's Magazine and First for Women to determine which was started first.
Action 1: Search[founding date of Arthur's Magazine]
Observation 1: Arthur's Magazine was founded in 1844.
Thought 2: Now I need to find the founding date of First for Women.
Action 2: Search[founding date of First for Women]
Observation 2: First for Women was founded in 1989.
Thought 3: I need to verify these dates to ensure their accuracy.
Action 3: (Result 1 / 1) Lookup[Arthur's Magazine founding date, First for Women founding date]
Observation 3: Arthur's Magazine is founded in 1844 and First for Women is founded in 1989.
Thought 4: Since Arthur's Magazine was founded in 1844 and First for Women was founded in 1989, Arthur's Magazine was started first.
Action 4: Finish[Arthur's Magazine]

---

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action 3: Finish[yes]"""


HUMANEVAL_FEWSHOT_EXAMPLES_REACT = """"""


MBPP_FEWSHOT_EXAMPLES_REACT = """You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists.
Your code should pass these tests:

assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))

Thought 1: I need to write a function that accepts two tuples and finds the common elements between them. I will use a nested loop to check each element in the first tuple against each element in the second tuple.
Action 1: Implement[
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i, j in zip(test_tup1, test_tup2):
        if i == j:
            res.append(i)
    return tuple(res)
```
]
Observation 1: 
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i, j in zip(test_tup1, test_tup2):
        if i == j:
            res.append(i)
    return tuple(res)
```
Execution Status: Done
Thought 2: I need to test the function to ensure it works correctly with different test cases.
Action 2: Test[
```python
assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))
```
]
Observation 2: 
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i, j in zip(test_tup1, test_tup2):
        if i == j:
            res.append(i)
    return tuple(res)

assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))
```
Execution Status: AssertionError()
Thought 3: The function incorrectly uses `zip`, which only compares elements at the same positions in both tuples. I need to compare each element in the first tuple with all elements in the second tuple.
Action 3: Implement[
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i in test_tup1:
        if i in test_tup2:
            res.append(i)
    return tuple(res)
```
]
Observation 3: 
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i in test_tup1:
        if i in test_tup2:
            res.append(i)
    return tuple(res)
```
Thought 4: I need to test the updated function to ensure it works correctly with different test cases.
Action 4: Test[
```python
assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))
```
]
Observation 4: 
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i in test_tup1:
        if i in test_tup2:
            res.append(i)
    return tuple(res)

assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))
```
Execution Status: Done
Thought 5: The function now works correctly for the provided test cases.
Action 5: Finish[
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i in test_tup1:
        if i in test_tup2:
            res.append(i)
    return tuple(res)
```
]
Observation 5:
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i in test_tup1:
        if i in test_tup2:
            res.append(i)
    return tuple(res)
```

---

You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers.
Your code should pass these tests:

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False

Thought 1: I need to write a function that can determine if a number is not a prime by checking its divisibility by any number up to its square root.
Action 1: Implement[
```python
import math

def is_not_prime(n):
    result = True
    for i in range(1, int(math.sqrt(n))):
        if n % i == 1:
            result = False
            break
    return result
```
]
Observation 1: 
```python
import math

def is_not_prime(n):
    result = True
    for i in range(1, int(math.sqrt(n))):
        if n % i == 1:
            result = False
            break
    return result
```
Execution Status: Done
Thought 2: I need to test the function to ensure it works correctly with different test cases.
Action 2: Test[
```python
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False
```
]
Observation 2: 
```python
import math

def is_not_prime(n):
    result = True
    for i in range(1, int(math.sqrt(n))):
        if n % i == 1:
            result = False
            break
    return result

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False
```
Execution Status: AssertionError()
Thought 3: The function is incorrect because it uses the wrong modulus condition (`n % i == 1` instead of `n % i == 0`) and starts the loop from 1 instead of 2. I need to correct these errors.
Action 3: Implement[
```python
import math

def is_not_prime(n):
    if n <= 1:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False
```
]
Observation 3: 
```python
import math

def is_not_prime(n):
    if n <= 1:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False
```
Thought 4: I need to test the updated function to ensure it works correctly with different test cases.
Action 4: Test[
```python
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False
```
]
Observation 4: 
```python
import math

def is_not_prime(n):
    if n <= 1:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False
```
Execution Status: Done
Thought 5: The function now works correctly for the provided test cases.
Action 5: Finish[
```python
import math

def is_not_prime(n):
    if n <= 1:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False
```
]
Observation 5:
```python
import math

def is_not_prime(n):
    if n <= 1:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False
```

---

You are an expert Python programmer, and here is your task: Write a function to find the n largest integers from a given list of numbers, returned in descending order.
Your code should pass these tests:

assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]

Thought 1: I need to write a function that finds the n largest integers in a list and returns them in descending order.
Action 1: Implement[
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```
]
Observation 1: 
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```
Execution Status: Done
Thought 2: I need to test the function to ensure it works correctly with different test cases.
Action 2: Test[
```python
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]
```
]
Observation 2: 
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums

assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]
```
Execution Status: Done
Thought 3: The function works correctly for the provided test cases.
Action 3: Finish[
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```
]
Observation 3:
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```"""


SVAMP_FEWSHOT_EXAMPLES_REACT = """Question: James bought 93 red and 10 blue stickers, he used 31 red stickers on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
Thought 1: First, I need to find out how many red stickers James has left after using some on his fridge.
Action 1: Calculate[
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers + used_red_stickers
```
]
Observation 1: 
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers + used_red_stickers
```
Execution Status: Done
Output: answer = 124
Thought 2: The answer is incorrect. I need to subtract the used stickers from the original number of stickers to get the correct answer.
Action 2: Calculate[
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers - used_red_stickers
```
]
Observation 2: 
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers - used_red_stickers
```
Execution Status: Done
Output: answer = 62
Thought 3: James has 62 red stickers left.
Action 3: Finish[
```python
answer = 62
```
]
Observation 3: 
```python
answer = 62
```

---

Question: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy each egg?
Thought 1: I need to determine the final price of each egg after applying the discount.
Action 1: Calculate[
```python
original_egg_price_in_dollars = 80
discount_price = 29
answer = original_egg_price_in_dollars + discount_dollars
```
]
Observation 1: 
```python
original_egg_price_in_dollars = 80
discount_price = 29
answer = original_egg_price_in_dollars + discount_dollars
```
Execution Status: NameError("name 'discount_dollars' is not defined")
Output: answer = None
Thought 2: The above code causes the "NameError" because it use the variable `discount_dollars` before it is defined.
Action 2: Calculate[
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars + discount_dollars
```
]
Observation 2:
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars + discount_dollars
```
Execution Status: Done
Output: answer = 109
Thought 3: The answer is incorrect. I need to subtract the discount from the original price to get the correct answer.
Action 3: Calculate[
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars - discount_dollars
```
]
Observation 3: 
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars - discount_dollars
```
Execution Status: Done
Output: answer = 51
Thought 4: Each egg costs 51 dollars after the discount.
Action 4: Finish[
```python
answer = 51
```
]
Observation 4: 
```python
answer = 51
```

---

Question: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books. How many books did Danny have at first?
Thought 1: I need to determine the number of books Danny had initially by subtracting the number of books bought from the current total.
Action 1: Calculate[
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now - num_books_bought_at_store
```
]
Observation 1: 
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now - num_books_bought_at_store
```
Execution Status: Done
Output: answer = 20
Thought 2: Danny had 20 books initially.
Action 2: Finish[
```python
answer = 20
```
]
Observation 2: 
```python
answer = 20
```

---

Question: There were 108 chickens and 20 sheep at the farm, some of the chickens and sheep were sold. There are 87 chickens and 18 sheep left now. How many chickens were sold?
Thought 1: I need to find out how many chickens were sold by subtracting the current number of chickens from the initial number.
Action 1: Calculate[
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before - num_chicken_now
```
]
Observation 1: 
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before - num_chicken_now
```
Execution Status: Done
Output: answer = 21
Thought 2: 21 chickens were sold.
Action 2: Finish[
```python
answer = 21
```
]
Observation 2: 
```python
answer = 21
```

---

Question: Katty scored 2 goals on Monday, 8 goals on Tuesday and 9 goals on Wednesday. How many did Katty score on Monday and Wednesday?
Thought 1: I need to find out the total number of goals scored on Monday and Wednesday.
Action 1: Calculate[
```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday + num_goals_on_wednesday
```
]
Observation 1: 
```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday + num_goals_on_wednesday
```
Execution Status: Done
Output: answer = 11
Thought 2: Katty scored 11 goals on Monday and Wednesday combined.
Action 2: Finish[
```python
answer = 11
```
]
Observation 2: 
```python
answer = 11
```

---

Question: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined. How many more girls than boys are in the Masquerade?
Thought 1: I need to calculate the total number of girls in the Masquerade.
Action 1: Calculate[
```python
num_girls_before = 5
num_girls_joined = 12
answer = num_girls_before + num_girls_joined
```
]
Observation 1: 
```python
num_girls_before = 5
num_girls_joined = 12
answer = num_girls_before + num_girls_joined
```
Execution Status: Done
Output: answer = 17
Thought 2: I need to calculate the total number of boys in the Masquerade.
Action 2: Calculate[
```python
num_boys_before = 4
num_boys_joined = 7
answer = num_boys_before + num_boys_joined
```
]
Observation 2:
```python
num_boys_before = 4
num_boys_joined = 7
answer = num_boys_before + num_boys_joined
```
Execution Status: Done
Output: answer = 11
Thought 3: I need to calculate the total number of girls and boys in the Masquerade and find the difference between them.
Action 3: Calculate[
```python
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
answer = total_girls - total_boys
```
]
Observation 3: 
```python
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
answer = total_girls - total_boys
```
Execution Status: Done
Output: answer = 6
Thought 4: There are 6 more girls than boys in the Masquerade.
Action 4: Finish[
```python
answer = 6
```
]
Observation 4: 
```python
answer = 6
```

---

Question: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creams, and he has 2 ice creams left now. How many ice creams did Getty purchase?
Thought 1: I need to calculate the number of ice creams Joseph had initially and subtract this from the total to find out how many Getty purchased.
Action 1: Calculate[
```python
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
answer = total_ice_creams - num_ice_creams_bought_by_joseph
```
]
Observation 1: 
```python
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
answer = total_ice_creams - num_ice_creams_bought_by_joseph
```
Execution Status: Done
Output: answer = 22
Thought 2: Getty purchased 22 ice creams.
Action 2: Finish[
```python
answer = 22
```
]
Observation 2: 
```python
answer = 22
```"""

TABMWP_FEWSHOT_EXAMPLES_REACT = """Read the following table regarding "Coin collections" and then write Python code to answer a question:

Name | Number of coins
Braden | 76
Camilla | 94
Rick | 86
Mary | 84
Hector | 80
Devin | 83
Emily | 82
Avery | 87

Question: Some friends discussed the sizes of their coin collections. What is the mean of the numbers?
Thought 1: I need to calculate the mean of the coin collection numbers.
Action 1: Calculate[
```python
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
answer = sum(number_of_coins_for_different_person) * len(number_of_coins_for_different_person)
```
]
Observation 1: 
```python
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
answer = sum(number_of_coins_for_different_person) * len(number_of_coins_for_different_person)
```
Execution Status: Done
Output: answer = 5376
Thought 2: The output is incorrect. I need to divide the sum of the numbers by the count of the numbers to get the correct mean.
Action 2: Calculate[
```python
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
answer = sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person)
```
]
Observation 2: 
```python
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
answer = sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person)
```
Execution Status: Done
Output: answer = 84
Thought 3: The mean number of coins in the collections is 84.
Action 3: Finish[
```python
answer = 84
```
]
Observation 3: 
```python
answer = 84
```

---

Read the following table regarding "" and then write Python code to answer a question:

Price | Quantity demanded | Quantity supplied
$155 | 22,600 | 5,800
$275 | 20,500 | 9,400
$395 | 18,400 | 13,000
$515 | 16,300 | 16,600
$635 | 14,200 | 20,200

Question: Look at the table. Then answer the question. At a price of $155, is there a shortage or a surplus? Choose from the the options: [shortage, surplus]
Thought 1: I need to compare the quantity demanded and the quantity supplied at a price of $155.
Action 1: Calculate[
```python
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_price_155 > quantity_supplied_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'
```
]
Observation 1: 
```python
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_price_155 > quantity_supplied_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'
```
Execution Status: Done
Output: answer = 'shortage'
Thought 2: At a price of $155, there is a shortage.
Action 2: Finish[
```python
answer = 'shortage'
```
]
Observation 2: 
```python
answer = 'shortage'
```

---

Read the following table regarding "Cans of food collected" and then write Python code to answer a question:

Samir | 7
Kristen | 4
Dakota | 7
Jamie | 8
Maggie | 9

Question: Samir's class recorded how many cans of food each student collected for their canned food drive. What is the median of the numbers?
Thought 1: I need to calculate the median of the number of cans collected.
Action 1: Calculate[
```python
cans = [7, 4, 7, 8, 9]
cans = sorted(cans)
middle1 = (len(cans) - 1) // 2
middle2 = len(cans) // 2
answer = (cans[middle1] + cans[middle2]) / 2
```
]
Observation 1: 
```python
cans = [7, 4, 7, 8, 9]
cans = sorted(cans)
middle1 = (len(cans) - 1) // 2
middle2 = len(cans) // 2
answer = (cans[middle1] + cans[middle2]) / 2
```
Execution Status: Done
Output: answer = 7
Thought 2: The median number of cans collected is 7.
Action 2: Finish[
```python
answer = 7
```
]
Observation 2: 
```python
answer = 7
```

---

Read the following table regarding "" and then write Python code to answer a question:

toy boat | $5.54
toy guitar | $8.23
set of juggling balls | $5.01
trivia game | $8.18
jigsaw puzzle | $5.30
toy dinosaur | $3.00

Question: Lorenzo has $13.50. Does he have enough to buy a toy guitar and a set of juggling balls? Choose from the the options: ['yes', 'no']
Thought 1: I need to determine if Lorenzo has enough money to buy both the toy guitar and the set of juggling balls.
Action 1: Calculate[
```python
guitar_price = 8.23
juggling_balls = 5.01
total_money = 13.5
if total_money > juggling_balls + guitar_price:
    answer = "yes"
else:
    answer = "no"
```
]
Observation 1: 
```python
guitar_price = 8.23
juggling_balls = 5.01
total_money = 13.5
if total_money > juggling_balls + guitar_price:
    answer = "yes"
else:
    answer = "no"
```
Execution Status: Done
Output: answer = "yes"
Thought 2: Lorenzo does have enough money to buy both the toy guitar and the set of juggling balls.
Action 2: Finish[
```python
answer = "yes"
```
]
Observation 2: 
```python
answer = "yes"
```"""


TRIVIAQA_FEWSHOT_EXAMPLES_REACT = """Question: Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842?
Thought 1: Let's search the question in google
Action 1: Search[Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842? site:wikipedia.org]
Observation 1: [Wedding March (Mendelssohn) - Wikipedia] Mendelssohn's 'Wedding March' was originally written as incidental music for A Midsummer Night's Dream in 1842.
Thought 2: The evidence suggests that Mendelssohn's 'Wedding March' was originally written as incidental music for A Midsummer Night's Dream in 1842. Let's verify this with another source.
Action 2: Search[Mendelssohn's 'Wedding March' A Midsummer Night's Dream 1842]
Observation 2: [Britannica - Wedding March] Felix Mendelssohn composed his 'Wedding March' as part of his incidental music for William Shakespeare's play A Midsummer Night's Dream in 1842.
Thought 3: Since multiple sources confirm that Mendelssohn's 'Wedding March' was originally written as incidental music for A Midsummer Night's Dream in 1842, it is likely accurate.
Action 3: Finish[A Midsummer Night's Dream]

---

Question: "Christ in the House of his Parents" is one of the best known paintings of which artist?
Thought 1: Let's search the question in google
Action 1: Search["Christ in the House of his Parents" is one of the best known paintings of which artist? site:wikipedia.org]
Observation 1: [Christ in the House of His Parents - Wikipedia] "Christ in the House of His Parents" is a painting by John Everett Millais.
Thought 2: The evidence suggests that "Christ in the House of His Parents" is a painting by John Everett Millais.
Action 2: Finish[John Everett Millais]

---

Question: Who designed the National Theatre building on the South Bank in London?
Thought 1: Let's search the question in google
Action 1: Search[Who designed the National Theatre building on the South Bank in London? site:wikipedia.org]
Observation 1: [National Theatre (London) - Wikipedia] The National Theatre building on the South Bank in London was designed by Denys Lasdun.
Thought 2: The evidence suggests that the National Theatre building on the South Bank in London was designed by Denys Lasdun.
Action 2: Finish[Denys Lasdun]

---

Question: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?
Thought 1: Let's search the question in google
Action 1: Search[Which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics? site:wikipedia.org]
Observation 1: [Bode Miller - Wikipedia] The only American skier who won the gold medal in the Men's Combined at the 2010 Winter Olympics is Bode Miller.
Thought 2: The evidence suggests that Bode Miller is the only American skier who won the gold medal in the Men's Combined at the 2010 Winter Olympics.
Action 2: Finish[Bode Miller]"""
