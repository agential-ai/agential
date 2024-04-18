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





# ======================================================================== TABMWP ======================================================================== #


REACT_INSTRUCTION_TABMWP = """Solve a math word problem by reasoning step-by-step. You can use the given table to help answer the question. Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity in the problem and returns the most relevant information.
(2) Lookup[keyword], which looks up a specific keyword in the results returned by the last Search.
(3) Finish[answer], which returns the final answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}

Table:
{table}
{scratchpad}"""

TABMWP_FEWSHOT_EXAMPLES = """Question: A school is going on a field trip to the zoo. If each van can hold 5 people and there are 2 students and 3 adults going, how many vans will they need?
Thought 1: Okay, let's break this down step-by-step. First, I'll find the total number of people going on the trip. The problem says there are 2 students and 3 adults going.
Action 1: Finish[2 + 3 = 5 people total]
Observation 1: The total number of people going on the trip is 5.
Thought 2: Great, now I know the total number of people is 5. Next, I need to find how many people can fit in each van.
Action 2: Search[each van can hold]
Observation 2: Each van can hold 5 people.
Thought 3: Since each van holds 5 people and there are 5 people total, they will need 1 van.
Action 3: Finish[5 people / 5 per van = 1 van needed]

Question: Carrie needs to buy shirts for her school's band. Small shirts cost $10 each, and large shirts cost $13 each. She buys 4 small shirts and 8 large shirts. How much does Carrie spend altogether?
Thought 1: Okay, let's organize the given information:
- Small shirts cost $10 each
- Large shirts cost $13 each  
- Carrie buys 4 small shirts and 8 large shirts
To find the total amount spent, I first need to calculate the cost of the small shirts and large shirts separately.
Action 1: Lookup[Small shirts cost]
Observation 1: Small shirts cost $10 each.
Thought 2: Carrie bought 4 small shirts at $10 each. So the total cost for small shirts is 4 * $10 = $40.  
Action 2: Lookup[Large shirts cost]
Observation 2: Large shirts cost $13 each.
Thought 3: Carrie bought 8 large shirts at $13 each. So the total cost for large shirts is 8 * $13 = $104.
Action 3: Lookup[Carrie buys]
Observation 3: Carrie buys 4 small shirts and 8 large shirts.
Thought 4: To get the total amount spent, I need to add the costs for small and large shirts together.
Small shirts cost = $40
Large shirts cost = $104
Total spent = $40 + $104 = $144
Action 4: Finish[$144]

Question: Jamie bought some boxes of cookies to share with his friends. Each box has 8 bags of cookies. If Jamie bought 5 boxes total, how many bags of cookies does he have?
Table: 
Box | Contains      
1   | 8 bags
2   | 16 bags
3   | 24 bags
4   | 32 bags 
5   | 40 bags

Thought 1: The question is asking how many total bags of cookies Jamie has. The table shows the number of bags in each box. 
Action 1: Search[Jamie bought]  
Observation 1: Jamie bought 5 boxes total.
Thought 2: The table says that 5 boxes contain 40 bags. So the total number of bags Jamie has is 40.
Action 2: Finish[40 bags]
    
Question: Leah made 38 cards by hand and bought the rest from a store. She bought 4 packs with each pack having 32 cards. How many cards does Leah have total?
Table:
Cards           | Amount
Handmade        | 38
Store bought    | 4 packs
Cards per pack  | 32

Thought 1: To find the total number of cards Leah has, I need to add the handmade cards and the store bought cards.
Action 1: Search[Handmade]
Observation 1: Leah made 38 cards by hand.
Thought 2: Leah has 38 handmade cards. Now I need to find how many store bought cards she has. The table shows she bought 4 packs with 32 cards per pack.
Action 2: Lookup[4 packs]
Observation 2: Leah bought 4 packs of cards from the store. Each pack has 32 cards.
Thought 3: To get the total store bought cards, I multiply the number of packs by the cards per pack.
4 packs * 32 cards/pack = 128 store bought cards
Now I can add that to the handmade cards.
38 handmade + 128 store bought = 166 total cards  
Action 3: Finish[166]

Question: {question}
Table: 
{table}

Thought 1:"""

REACT_INSTRUCTION_TABMWP_NO_TABLE = """Solve a math word problem by reasoning step-by-step. Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity in the problem and returns the most relevant information.
(2) Lookup[keyword], which looks up a specific keyword in the results returned by the last Search.
(3) Finish[answer], which returns the final answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}{scratchpad}"""

TABMWP_FEWSHOT_EXAMPLES_NO_TABLE = """Question: A school is going on a field trip to the zoo. If each van can hold 5 people and there are 2 students and 3 adults going, how many vans will they need?
Thought 1: Okay, let's break this down step-by-step. First, I'll find the total number of people going on the trip. The problem says there are 2 students and 3 adults going.
Action 1: Finish[2 + 3 = 5 people total]
Observation 1: The total number of people going on the trip is 5.
Thought 2: Great, now I know the total number of people is 5. Next, I need to find how many people can fit in each van.
Action 2: Search[each van can hold]
Observation 2: Each van can hold 5 people.
Thought 3: Since each van holds 5 people and there are 5 people total, they will need 1 van.
Action 3: Finish[5 people / 5 per van = 1 van needed]

Question: Carrie needs to buy shirts for her school's band. Small shirts cost $10 each, and large shirts cost $13 each. She buys 4 small shirts and 8 large shirts. How much does Carrie spend altogether?
Thought 1: Okay, let's organize the given information:
- Small shirts cost $10 each
- Large shirts cost $13 each  
- Carrie buys 4 small shirts and 8 large shirts
To find the total amount spent, I first need to calculate the cost of the small shirts and large shirts separately.
Action 1: Lookup[Small shirts cost]
Observation 1: Small shirts cost $10 each.
Thought 2: Carrie bought 4 small shirts at $10 each. So the total cost for small shirts is 4 * $10 = $40.  
Action 2: Lookup[Large shirts cost]
Observation 2: Large shirts cost $13 each.
Thought 3: Carrie bought 8 large shirts at $13 each. So the total cost for large shirts is 8 * $13 = $104.
Action 3: Lookup[Carrie buys]
Observation 3: Carrie buys 4 small shirts and 8 large shirts.
Thought 4: To get the total amount spent, I need to add the costs for small and large shirts together.
Small shirts cost = $40
Large shirts cost = $104
Total spent = $40 + $104 = $144
Action 4: Finish[$144]

Question: A pizza shop has 12 pizzas in stock. During the lunch rush, they sold 3/4 of their pizzas. How many pizzas were sold?
Thought 1: I need to find how many pizzas were sold. First, I'll find out how many pizzas the shop had originally.
Action 1: Search[pizza shop has]
Observation 1: The pizza shop has 12 pizzas in stock.
Thought 2: The shop originally had 12 pizzas. Next, I need to find what fraction of the pizzas were sold.
Action 2: Search[they sold]
Observation 2: During the lunch rush, they sold 3/4 of their pizzas.
Thought 3: To calculate how many pizzas were sold, I need to take 3/4 of the original 12 pizzas.
3/4 of 12 = (3/4) * 12 = 9
So 9 pizzas were sold.
Action 3: Finish[9 pizzas]  

Question: Kim is training for a triathlon. Last weekend she biked a total of 132 kilometers. The weekend before she biked 35 fewer kilometers than that. How many kilometers did Kim bike the weekend before last?
Thought 1: To find how many kilometers Kim biked the weekend before last, I first need to know how many kilometers she biked last weekend.
Action 1: Search[Last weekend she biked]
Observation 1: Last weekend Kim biked a total of 132 kilometers. 
Thought 2: Kim biked 132 km last weekend. The problem says the weekend before she biked 35 fewer kilometers than that. So I need to subtract 35 from 132.
132 - 35 = 97
Therefore, the weekend before last Kim biked 97 kilometers.
Action 2: Finish[97 kilometers]

Question: {question}

Thought 1:"""