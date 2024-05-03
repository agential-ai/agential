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


# ======================================================================== GSM8k ======================================================================== #


REACT_INSTRUCTION_GSM8k = """Determine the solution to a math problem, providing Observation that SUPPORTS the answer, or indicating if there is NOT ENOUGH INFORMATION. Action can be two types:
(1) python_interpreter[Thought], executes given python code.
(2) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}{scratchpad} """

GSM8k_FEWSHOT_EXAMPLES = """Question: Olivia has $23. She bought five bagels for $3 each. Calculate how much money Olivia has left after buying the bagels.
Thought 1: o = 23 #olivia has $23;spent = 3 * 5 #she buys 5 bagels for $3 each;total = o - spent #the total is o-spent
Action 1: python_interpreter[Thought 1]
Observation 1: \{ o:23, spent:15, total: 8 \}
Thought 2: Olivia initially has $23. Each bagel costs $3, so five bagels cost $3 x 5 = $15. Subtracting $15 from $23, Olivia has $23 - $15 = $8 left.
Action 2: Finish[$8]

Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. Calculate how many golf balls Michael had at the end of Wednesday.
Thought 1: m = 58 #michael has 58 golf balls;lost = 23 #he first lost 23 golf balls;lost += 2 #he then lost 2 more;total = m-lost #the total is m-lost
Action 1: python_interpreter[Thought 1]
Observation 1: \{ m:58, spent:25, total: 33 \}
Thought 2: Michael initially had 58 golf balls. He lost 23 on Tuesday and 2 more on Wednesday. Subtracting these losses from the initial count, Michael had 58 - 23 - 2 = 33 golf balls at the end of Wednesday.
Action 2: Finish[33 golf balls]

Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. Calculate how many computers are now in the server room.
Thought 1: computers = 9 #there were initially 9 computers;new_installed = (5*4) #5 more were added each day for 4 days;total = computers + new_installed #the total is computers + new_intalled
Action 1: python_interpreter[Thought 1]
Observation 1: \{ computers:9, new_installed:20, total: 29 \}
Thought 2: Initially, there were 9 computers in the server room. Five more were installed each day for four days. So, the total number of computers now in the server room is 9 + (5 x 4) = 29.
Action 2: Finish[29 computers]

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. Calculate how many toys Shawn has now.
Thought 1: s=5 #shawn has 5 toys;given = 2+2 #two parents each give him two toys;total = s+given #the total is s + given
Action 1: python_interpreter[Thought 1]
Observation 1: \{ s:5, given:5, total: 9 \}
Thought 2: Shawn initially had 5 toys. He received 2 toys each from his mom and dad for Christmas, so he got 2 + 2 = 4 toys in total. Adding these to his initial count, Shawn now has 5 + 4 = 9 toys.
Action 2: Finish[9 toys]

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. Calculate how many lollipops Jason gave to Denny.
Thought 1: j=20 #jason had 20 lollipops;j_after = 12 #he has 12 after giving some to denny;total=j-j_after #the total is j-j_after
Action 1: python_interpreter[Thought 1]
Observation 1: \{ j:20, j_after:12, total: 8 \}
Thought 2: Jason initially had 20 lollipops. After giving some to Denny, he has 12 left. So, Jason gave away 20 - 12 = 8 lollipops to Denny.
Action 2: Finish[8 lollipops]

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, calculate how many pieces they have left in total.
Thought 1: l=32 #leah had 32 chocolates;s=42 #her sister had 42;ate=35 #they ate 35;total=l+s-ate #total is l+s-ate
Action 1: python_interpreter[Thought 1]
Observation 1: \{ l:32, s:42, ate:35, total: 39 \}
Thought 2: Leah and her sister initially had a total of 32 + 42 = 74 chocolates. After eating 35, they have 74 - 35 = 39 pieces left in total.
Action 2: Finish[39 pieces left]

Question: If there are 3 cars in the parking lot and 2 more cars arrive, calculate how many cars are in the parking lot now.
Thought 1: cars=3 #there are 3 cars;new_cars=2 #two new cars arrive;total=cars+new_cars #the total is cars + new_cars
Action 1: python_interpreter[Thought 1]
Observation 1: \{ cars:3, new_cars:2, total: 5 \}
Thought 2: Initially, there were 3 cars in the parking lot. With 2 more arriving, the total number of cars in the parking lot now is 3 + 2 = 5.
Action 2: Finish[5 cars]

Question: There are 15 trees in the grove. Grove workers will plant trees today, and after they are done, there will be 21 trees. Calculate how many trees the grove workers planted today.
Thought 1: trees=15 #there are 15 trees initially;end_trees=21 #there will be 21 trees;total=end_trees-trees #the total is end_trees-trees
Action 1: python_interpreter[Thought 1]
Observation 1: \{ trees:15, end_trees:21, total: 6 \}
Thought 2: Initially, there were 15 trees in the grove. After the workers planted trees, there were 21. So, the workers planted 21 - 15 = 6 trees today.
Action 2: Finish[6 trees]"""

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
