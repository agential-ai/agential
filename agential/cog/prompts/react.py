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


# ======================================================================== AMBIGQA ======================================================================== #


REACT_INSTRUCTION_AMBIGQA = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be two types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}{scratchpad}"""

REACT_AMBIGQA_EXAMPLES = """Question: When did men's figure skating become a summer Olympic sport? 
Thought 1: Let's search the question in google 
Action 1: Search[When did men's figure skating become a summer Olympic sport?] 
Observation 1: [Figure skating at the Olympic Games - Wikipedia] Figure skating was first contested in the Olympic Games at the 1908 Summer Olympics . Since 1924, the sport has been a part of the Winter Olympic Games . 
Thought 2: The evidence suggests Figure skating became an Olympic sport at the 1908 Summer Olympics, and has been a part of the Winter Olympic Games in 1924. 
Action 2: Finish[1908]

Question: When was the last time the USA men's national soccer team missed the World Cup? 
Thought1: Let's search the question in google Action 1: Search[When was the last time the USA men's national soccer team missed the World Cup? site: wikipedia.org] 
Observation 1: [United States at the FIFA World Cup - Wikipedia] The United States participated in every World Cup from 1990 through 2014, but did not qualify in 2018, marking first time the team had missed a World Cup since 1986. They returned to the World Cup by qualifying for the 2022 edition . 
Thought 2: The evidence suggests that the USA men's national soccer team did not qualify for the World Cup in 2018, but qualify for 2022, 2018 > 1986, so the last time the USA men's national soccer team missed the World Cup was in 2018. 
Action 2: Finish[2018]

Question: What does barium do in a ct scan? 
Thought 1: Let's search the question in google Action 1: Search[What does barium do in a ct scan? site: wikipedia.org] 
Observation 1: [Barium sulfate suspension - Wikipedia] Barium sulfate suspension, often simply called barium, is a contrast agent used during X-rays. Specifically it is used to improve visualization of the gastrointestinal tract (esophagus, stomach, intestines) on plain X-ray or computed tomography. Thougth 2: The evidence suggests that barium is used to "improve visualization of the gastrointestinal tract". Action 2: Search[Why barium used in CT scans?] Observation 2: [What To Expect Before Your CT Scan - St. Elizabeth's Medical Center] The barium helps to highlight body areas for the CT scan. If you are having a scan of a body part other than the abdomen you should arrive 30 minutes before your appointed time. Thougth 3: The evidence suggests that barium helps "highlight body areas" for the CT scan. Action 3: Finish[to improve visualization of the gastrointestinal tract]

Question: Where was the fort located at which the first shot of the civil war was fired? 
Thought 1: Let's search the question in google 
Action 1: Search[Where was the fort located at which the first shot of the civil war was fired?] 
Observation 1: [Civil War Begins - U.S. Senate] April 12, 1861 At 4:30 a.m. on April 12, 1861, Confederate troops fired on Fort Sumter in South Carolina’s Charleston Harbor. Less than 34 hours later, Union forces surrendered. 
Thought 2: The evidence suggests that the first shot of the Civil War was fired at Fort Sumter. Fort Sumter is located in South Carolina’s Charleston Harbor. 
Action 2: Finish[Charleston Harbor, South Carolina]

Question: When did nando's come to the uk? 
Thought 1: Let's search the question in google 
Action 1: Search[When did nando's come to the uk? site: wikipedia.org] 
Observation 1: [Nando's - Wikipedia] Nando's opened its first restaurants within the United Kingdom, in 1992, in the west London suburbs of Ealing and Earls Court, initially focusing on takeaway food. 
Thougth 2: The evidence suggests that Nando's first opened in the UK in 1992. 
Action 2: Finish[1992]

Question: Who plays wilmas mom in the flintstones movie? 
Thought 1: Let's search the question in google 
Action 1: Search[Who played Wilma's mom in the flintstones movie? site: wikipedia.org] 
Observation 1: [Elizabeth Perkins - Wikipedia] Elizabeth Ann Perkins (born November 18, 1960 ) is an American actress. 
Thought 2: The evidence does not mention Wilma's mom in the flintstones movie. 
Action 2: Search[Who played Wilma's mom in the flintstones movie?] 
Observation 2: [] The Flintstones / Wilma Flintstone / Mother / Played by Elizabeth Taylor 
Thought 3: The evidence shows that Elizabeth Taylor played the role of Wilma's mother. 
Action 3: Finish[Elizabeth Taylor]"""

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

TRIVIAQA_FEWSHOT_EXAMPLES = """Question: Which innovation for the car was developed by Prince Henry of Prussia in 1911? 
Thought 1: Let's search the question in google
Action 1: Search[Which innovation for the car was developed by Prince Henry of Prussia in 1911? site: wikipedia.org]
Observation 1: [Prince Henry of Prussia (1862–1929) - Wikipedia] Henry was interested in motor cars as well and supposedly invented a windshield wiper and, according to other sources, the car horn.
Thought 2: The evidence suggests that Prince Henry of Prussia invented a windshield wiper and the car horn. 
Action 2: Search[When did Prince Henry of Prussia invented a windshield wiper and the car horn?]
Observation 2: [110 years ago: windscreen wiper patent for Prince Henry of Prussia] Quite apart from a member of the German aristocracy: it was Prince Henry of Prussia (1862-1929) who was granted the first German patent about the windscreen wiper on 24. March 1908. 
Thought 3: According to the evidence, Prince Henry of Prussia was granted the first German patent for the windshield wiper.
Action 3: Finish[windshield wiper]

Question: When did the founder of Jehovah's Witnesses say the world would end?
Thought 1: Let's search the question in google
Action 1: Search[When did the founder of Jehovah's Witnesses say the world would end? site: wikipedia.org]
Observation 1: [Eschatology of Jehovah's Witnesses - Wikipedia] In 1911, Russell wrote that October 1914 would witness the "full end" of Babylon, or nominal Christianity, "utterly destroyed as a system". At first, the hopes ... 
Thought 2: The evidence shows that the founder of Jehovah's Witnesses predicted the world would end in October 1914
Action 2: Finish[1914]

Question: Three people have featured on the reverse of the £5 note, the two most recent are George Stephenson and Elizabeth Fry, who was the first?
Thought 1: Let's search the question in google
Action 1: Search[Three people have featured on the reverse of the £5 note, the two most recent are George Stephenson and Elizabeth Fry, who was the first?]
Observation 1: [Bank of England £5 note - Wikipedia] The Bank of England £5 note, also known as a fiver, is a sterling banknote . It is the smallest denomination of banknote currently issued by the Bank of England.
Thought 2: The evidence is about £5 note, it has no information about the featured people we want to find out. 
Action 2: Search[Who was the first person to feature on the reverse of the £5 note? site: wikipedia.org] 
Observation 2: [Bank of England £5 note - Wikipedia] The old paper note, first issued in 2002 and bearing the image of prison reformer Elizabeth Fry on the reverse, was phased out and ceased to be legal tender after 5 May 2017. 
Thought 3: The evidence only mentions Elizabeth Fry, and from the question, we know that she was not the first person to feature on the reverse of the £5 note. 
Action 3: Search[People have featured on the reverse of the £5 note: George Stephenson and Elizabeth Fry and?]
Observation 3: [Historical British figures featured on banknotes (1970-2017)] £5 – The Duke of Wellington, George Stephenson, Elizabeth Fry, Sir Winston Churchill £10 – Florence Nightingale, Charles Dickens, Charles Darwin £20 – William Shakespeare, ...
Thought 4: The evidence shows that the Duke of Wellington and Sir Winston Churchill were also featured on £5.
Action 4: Finish[Duke of Wellington]

Question: What state had its bi-centenary of joining the Union a year after North Carolina? 
Thought 1: Let's search the question in google 
Action 1: Search[What state had its bi-centenary of joining the Union a year after North Carolina? site: wikipedia.org] 
Observation 1: [List of U.S. states by date of admission to the Union - Wikipedia] 24. Missouri, August 10, 1821 · (admitted) ; 25. Arkansas, June 15, 1836 · (admitted) ... 
Thought 2: The evidence does not provide information about the state had its bi-centenary of joining the Union a year after North Carolina. To answer the question, we need to find the state joining the Union a year after North Carolina. 
Action 2: Search[Which state joined the Union a year after North Carolina? site: wikipedia.org] 
Observation 2: [List of U.S. states by date of admission to the Union - Wikipedia...] ^ This list does not account for the secession of 11 states (Virginia, North Carolina, South Carolina, Georgia, Florida, Alabama, Mississippi, Tennessee, ... 
Thought 3: Not enough evidence, we need further search. 
Action 3: Search[What state joined the Union a year after North Carolina?] 
Observation 3: [States by Order of Entry into Union - Infoplease] Joining the Union | State | Entered Union | Year Settled | | North Carolina | Nov. 21, 1789 | 1660 | | Rhode Island | May 29, 1790 | 1636 | | Vermont | Mar. 4, 1791 | 1724 | 
Thought 4: The evidence shows North Carolina entered Union in 1789, and Rhode Island entered Union in 1790, which is a year after North Carolina. 
Action 4: Finish[Rhode Island]"""

REACT_INSTRUCTION_TABMWP = """Read the following table and question to solve the problem,providing Observation that SUPPORTS the answer, or indicating if there is NOT ENOUGH INFORMATION. Action can be two types:
(1) python-generator[Thought], generates an executable python python code using the Sympy library using the Context. The first line of the code should be 'from sympy import *'
(2) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.
Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}{scratchpad}"""

TABMWP_FEWSHOT_EXAMPLES = """Table:
Price | Quantity demanded | Quantity supplied
$895 | 21,000 | 3,400
$945 | 17,200 | 7,400
$995 | 13,400 | 11,400
$1,045 | 9,600 | 15,400
$1,095 | 5,800 | 19,400

Questions: Look at the table. Then answer the question. At a price of $995, is there a shortage or
a surplus? Please select from the following options: ['shortage', 'surplus'].
Thought 1: To determine whether there is a shortage or a surplus at a price of $995, I need to compare the quantity demanded to the quantity supplied at that price point.
Action 1 : python-generator('Calculate the difference between quantity demanded and quantity supplied at a price of $995 to determine if it results in a positive (surplus) or negative (shortage) value.')
Observation 1: [code python] # Given data for the price of $995 quantity_demanded = 13400 quantity_supplied = 11400 # Calculate difference to determine market condition difference = quantity_demanded - quantity_supplied difference
Thought 2: If the difference is positive, there is a surplus; if it is negative, there is a shortage. A zero difference indicates equilibrium.
Action 2: Finish['shortage']


Table:
toy guitar | $32.42
toy piano | $10.55
model railroad set | $65.51
toy rocket | $80.93
chess board | $15.76

Question: Cody has $49.50. Does he have enough to buy a toy guitar and a chess board? Please select from the following options: ['yes', 'no'].
Thought 1: To determine if Cody has enough money to buy both a toy guitar and a chess board, we need to add the prices of these two items and compare the total with Cody's available funds of $49.50.
Action 1 : python-generator('Calculate the combined cost of the toy guitar and the chess board and compare it to Cody's budget to see if he has sufficient funds.')
Observation 1: [code python] # Prices of items price_toy_guitar = 32.42 price_chess_board = 15.76 # Calculate total cost of the toy guitar and chess board total_cost = price_toy_guitar + price_chess_board # Compare total cost with Cody's budget cody_budget = 49.50 is_enough_money = total_cost <= cody_budget total_cost, is_enough_money
Thought 2: If the total cost is less than or equal to Cody's budget, then he has enough money to purchase both items. Otherwise, he does not.
Action 2 : Finish['Yes']


Table:
Name | Number of coins
Shelby | 81
Oliver | 84
Jamal | 78
Vince | 81
Abby | 79
Farid | 77
Tara | 85
Krysta | 83

Question: Some friends discussed the sizes of their coin collections. What is the mean of the numbers?
Thought 1:To find the mean of the numbers representing the sizes of the coin collections, we need to sum all the coin counts and divide by the total number of entries.
Action 1 : python-generator('Compute the mean by summing all the values in the coin count list and dividing by the number of entries, which will give us the average number of coins per person.')
Observation 1: [code python] # Given data of the number of coins coin_counts = [81, 84, 78, 81, 79, 77, 85, 83] # Calculate the mean of the numbers mean_coins = sum(coin_counts) / len(coin_counts) mean_coins
Thought 2:The mean value calculated will provide the average number of coins held by each person in the group, which is the answer to the question.
Action 2 : Finish[81]


Table:
square beads | $2.97 per kilogram
oval beads | $3.41 per kilogram
flower-shaped beads | $2.18 per kilogram
star-shaped beads | $1.95 per kilogram
heart-shaped beads | $1.52 per kilogram
spherical beads | $3.42 per kilogram
rectangular beads | $1.97 per kilogram

Question: If Tracy buys 5 kilograms of spherical beads, 4 kilograms of star-shaped beads, and 3 kilograms of flower-shaped beads, how much will she spend? (unit: $)
Thought 1: To determine the total amount Tracy will spend, the cost per kilogram of each type of bead she buys must be multiplied by the respective quantities she purchases. We then sum these amounts to get the total cost.
Action 1 : python-generator('Calculate the total cost for Tracy by multiplying the price per kilogram of spherical beads, star-shaped beads, and flower-shaped beads by the respective quantities purchased and then summing these values.')
Observation 1: [code python] # Recalculating the total cost for Tracy with provided prices and quantities to ensure accuracy price_spherical_beads = 3.42 price_star_shaped_beads = 1.95 price_flower_shaped_beads = 2.18 quantity_spherical = 5 quantity_star_shaped = 4 quantity_flower_shaped = 3 # Calculate total cost again total_cost = (price_spherical_beads * quantity_spherical) + \ (price_star_shaped_beads * quantity_star_shaped) + \ (price_flower_shaped_beads * quantity_flower_shaped) total_cost
Thought 2:The calculation provides a straightforward method to find the total spending based on the quantities and prices of the beads.
Action 2 : Finish[$31.44]
"""