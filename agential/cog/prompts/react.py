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

HOTPOTQA_FEWSHOT_EXAMPLES = """Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
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
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]
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
Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]
Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]
Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action 3: Finish[yes]"""


# ======================================================================== FEVER ======================================================================== #


REACT_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION.  and Action can be two types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}{scratchpad}"""

FEVER_FEWSHOT_EXAMPLES = """Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought 1: I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993,[1] and had his breakthrough role in Denmark with the film Nightwatch (1994). He played Jaime Lannister in the HBO fantasy drama series Game of Thrones, for which he received two Primetime Emmy Award nominations for Outstanding Supporting Actor in a Drama Series.. Coster-Waldau has appeared in numerous films in his native Denmark and Scandinavia, including Headhunters (2011) and A Thousand Times Good Night (2013). In the U.S, his debut film role was in the war film Black Hawk Down (2001), playing Medal of Honor recipient Gary Gordon.[2] He then played a detective in the short-lived Fox television series New Amsterdam (2008), and appeared in the 2009 Fox television film Virtuality, originally intended as a pilot.
Thought 2: Because he "appeared in the 2009 Fox television film Virtuality", he should have worked with the Fox Broadcasting Company.
Action 2: Finish[SUPPORTS]

Claim: Stranger Things is set in Bloomington, Indiana.
Thought 1: I should search for Stranger Things, and see if it is set in Bloomington, Indiana.
Action 1: Search[Stranger Things]
Observation 1: Stranger Things is an American science fiction horror drama television series created by the Duffer Brothers. Set in the 1980s, primarily in the fictional town of Hawkins, Indiana, the series centers on a number of mysteries and supernatural events occurring around the town and their impact on an ensemble of child and adult characters. 
Thought 2: The observation says that it is set in a "fictional town of Hawkins, Indiana", so it is not set in Bloomington.
Action 2: Finish[REFUTES]

Claim: Beautiful reached number two on the Billboard Hot 100 in 2003.?
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


# ======================================================================== SVAMP ======================================================================== #

#svamp for react instructions

REACT_SVAMP_INSTRUCTIONS = """{examples}
(END OF EXAMPLES)

Question: {question}
Answer: {answer}
"""


REACT_SVAMP_EXAMPLES_DIRECT = """
Question: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
Answer: 62
Question: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy for each egg?
Answer: 51
Question: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books. How many books did danny have at first?
Answer: 20
Question: There were 108 chickens and 20 sheeps at the farm, some of chickens and sheeps were sold. There are 87 chickens and 18 sheeps left now. How many chickens were sold?
Answer: 21
Question: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday. How many did Katty score on monday and wednesday?
Answer: 11
Question: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined.  How many more girls than boys are in the Masquerade?
Answer: 6
Question: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now.  How much ice creasm did Getty purchase?
Answer: 22"""


REACT_SVAMP_INSTRUCTIONS_CRITIC = """{examples}
(END OF EXAMPLES)
Question: {question}
Execution: Done
Output: {answer}
What's the problem with the above code?
1. 
{critique}
"""

#svamp for react critic CRITIQUE questions

REACT_SVAMP_EXAMPLES_CRITIQUE   = """
---
Question: Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. How much did Marco's strawberries weigh?
```python
num_strawberries_dad_picked = 11
num_strawberries_marco_picked = 30
answer = num_strawberries_dad_picked + num_strawberries_marco_picked
```
Execution: Done
Output: answer = 41.0
What's the problem with the above code?
1. The weight of straberries should be positive, 41.0 > 0, it's reasonable.
2. Let's check the code:
> answer = num_strawberries_dad_picked + num_strawberries_marco_picked
The above code calculates the total weight of strawberries picked by both Marco and his dad, instead of finding the weight of strawberries picked by Marco alone.
According to the question, the total weight of strawberries picked by both Marco and his dad is 30 pounds, and Marco's dad's strawberries weighed 11 pounds. To find the weight of Marco's strawberries, we need to subtract the weight of his dad's strawberries from the total weight.
Here's a better solution:
```python
total_weight = 30
dad_weight = 11
# Calculate the weight of Marco's strawberries
marco_weight = total_weight - dad_weight
answer = marco_weight
```
---
Question: Zachary did 46 push-ups and 58 crunches in gym class today. David did 38 more push-ups but 62 less crunches than zachary. How many more crunches than push-ups did Zachary do?
```python
zachary_push_ups = 46
zachary_crunches = 58
zachary_diff = zachary_crunches - zachary_push_ups
answer = zachary_diff
```
Execution: Done
Output: answer = 12
What's the problem with the above code?
1. The answer, 12, is a reasonable result.
2. Let's check the code:
> zachary_diff = zachary_crunches - zachary_push_ups
The code calculates the difference between the number of crunches and push-ups that Zachary did, it is correct.
The information about David is not relevant to the question and does not need to be considered in the calculation. There is no problem with the above code. 
Here's a better solution:
```python
zachary_push_ups = 46
zachary_crunches = 58
# Calculate the difference between Zachary's crunches and push-ups
zachary_diff = zachary_crunches - zachary_push_ups
answer = zachary_diff
```
---
Question: Faye had 35 packs of pencils each one having 4 pencils. She was placing her pencils into rows with 2 pencils in each row. How many rows could she make?
```python
num_packs_of_pencils = 35
num_pencils_in_each_pack = 4
answer = num_packs_of_pencils * num_pencils_in_each_pack
```
Execution: Done
Output: answer = 140.0
What's the problem with the above code?
1. The answer rows 140 > 0, it's reasonable.
2. Let's check the code:
> answer = num_packs_of_pencils * num_pencils_in_each_pack
The code calculates the total number of pencils Faye has, but it does not calculate the number of rows she can make. It needs to be modified to take into account the fact that she is placing 2 pencils in each row.
Here's a better solution:
```python
num_packs_of_pencils = 35
num_pencils_in_each_pack = 4
# Calculate the total number of pencils
total_pencils = num_packs_of_pencils * num_pencils_in_each_pack
# Calculate the number of rows with 2 pencils in each row
num_pencils_in_each_row = 2
num_rows = total_pencils // num_pencils_in_each_row
answer = num_rows
```
---
Question: Jack received 4 emails and sent 2 emails in the morning. He then received 6 emails and sent 8 emails in the afternoon. How many emails did Jack send in the day?
```python
num_emails_received_in_morning = 4
num_emails_sent_in_morning = 2
num_emails_received_in_afternoon = 6
num_emails_sent_in_afternoon = 8
answer = num_emails_sent_in_morning + num_emails_sent_in_afternoon
```
Execution: Done
Output: answer = 10.0
What's the problem with the above code?
1. The answer, 10, is a reasonable result.
2. Let's check the code:
> answer = num_emails_sent_in_morning + num_emails_sent_in_afternoon
The code calculates the total number of emails Jack sent during the day by adding the number of emails he sent in the morning to the number of emails he sent in the afternoon. There is no problem with the above code. 
Here's a better solution:
```python
num_emails_received_in_morning = 4
num_emails_sent_in_morning = 2
num_emails_received_in_afternoon = 6
num_emails_sent_in_afternoon = 8
answer = num_emails_sent_in_morning + num_emails_sent_in_afternoon
```
---
"""

#svamp for react pot questions

REACT_SVAMP_EXAMPLES_POT = """
Read the following passages to answer questions with Python code, store the result as a 'answer' variable:
Question: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
# Python code, return answer
``` python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers - used_red_stickers
```
Answer: answer
Question: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy for each egg?
# Python code, return answer
``` python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars - discount_dollars
```
Answer: answer
Question: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books. How many books did danny have at first?
# Python code, return answer
``` python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now - num_books_bought_at_store
```
Answer: answer
Question: There were 108 chickens and 20 sheeps at the farm, some of chickens and sheeps were sold. There are 87 chickens and 18 sheeps left now. How many chickens were sold?
# Python code, return answer
``` python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before - num_chicken_now
```
Answer: answer
Question: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday. How many did Katty score on monday and wednesday?
# Python code, return answer
``` python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday + num_goals_on_wednesday
```
Answer: answer
Question: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined.  How many more girls than boys are in the Masquerade?
# Python code, return answer
``` python
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
answer = total_girls - total_boys
```
Answer: answer
Question: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now.  How much ice creasm did Getty purchase?
# Python code, return answer
``` python
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
answer = total_ice_creams - num_ice_creams_bought_by_joseph
```
Answer: answer
"""
