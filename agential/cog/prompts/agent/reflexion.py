"""Reflexion prompts and fewshot examples."""

# Header for formatting reflections when reflection strategy is "last_attempt_and_reflexion".
REFLECTION_AFTER_LAST_TRIAL_HEADER = "The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n"

# Default header for formatting reflections (_format_reflections).
REFLECTION_HEADER = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n"

# Default header for formatting last attempt (_format_last_attempt).
LAST_TRIAL_HEADER = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n"


# ======================================================================== HOTPOTQA ======================================================================== #


HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT = """Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: Let's think step by step. The eastern sector of Colorado orogeny extends into the Rocky Mountains. The Rocky Mountains rise in elevation from around 1,800 to 14,000 ft, so the answer is 1,800 to 14,000 ft.
Action: Finish[1,800 to 14,000 ft]

Reflection: My reasoning for the elevation range of the eastern sector of the Colorado orogeny failed because I incorrectly associated it with the Rocky Mountains instead of the High Plains. In the future, when attempting this question, I should focus on better understanding the geography and geological formations involved in the Colorado orogeny to avoid confusion.

---

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: Let's think step by step. Milhouse was named after a prominent cartoon character, Mickey Mouse, so the answer is Mickey Mouse.
Action: Finish[Mickey Mouse]

Reflection: My reasoning for the naming of the character Milhouse in "The Simpsons" failed because I incorrectly assumed that the character was named after a prominent cartoon character, Mickey Mouse. In the future, when attempting this question, I should focus on better researching the background of "The Simpsons" and Matt Groening's influences in naming characters to avoid making incorrect assumptions.

---

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought: Let's think step by step. Adam Clayton Powell was a significant figure, so the documentary about Finnish rock groups must be about him.
Action: Finish[Adam Clayton Powell]

Reflection: My reasoning failed because I associated Adam Clayton Powell with the topic due to his prominence rather than considering the specific subject matter of Finnish rock groups. In the future, I should focus on the context and subject of the documentary rather than the prominence of the individuals involved.

---

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: Let's think step by step. Nicholas Ray and Elia Kazan were both known for their political activism, so the answer must be political activist.
Action: Finish[political activist]

Reflection: My reasoning failed because I focused on their political involvement rather than their primary professions. In the future, I should consider the main professional roles they are renowned for, which in this case are director, screenwriter, and actor.

---

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought: Let's think step by step. First for Women sounds more established and older, so it must have been started first.
Action: Finish[First for Women]

Reflection: My reasoning failed because I relied on the perceived establishment of the magazine names rather than researching their actual start dates. In the future, I should look up the publication history of each magazine to determine which one was started first.

---

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: Let's think step by step. Pavel Urysohn and Leonid Levin worked in different fields; Urysohn in mathematics and Levin in computer science, so the answer must be No.
Action: Finish[No]

Reflection: My reasoning failed because I did not recognize the overlap in their fields of study. In the future, I should consider the broader context of their work and understand that both were known for contributions in theoretical and applied mathematics, making the correct answer Yes."""


REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Question: {question}{scratchpad}

Reflection:"""


REFLEXION_COT_INSTRUCTION_HOTPOTQA = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""


REFLEXION_REACT_INSTRUCTION_HOTPOTQA = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""


REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Question: {question}{scratchpad}

Reflection:"""

HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT = """Previous Trial:
Thought 1: The Colorado orogeny is a mountain-building event in the western United States, so the elevation range should be quite high.
Action 1: Search[eastern sector of the Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas. This took place from 1780 to 1650 million years ago (Mya), during the Paleoproterozoic (Statherian Period). It is recorded in the Colorado orogen, a >500-km-wide belt of oceanic arc rock that extends southward into New Mexico. The Colorado orogeny was likely part of the larger Yavapai orogeny.
Thought 2: The eastern sector of the Colorado orogeny likely extends into high elevation areas due to its mountain-building nature.
Action 2: Lookup[elevation]
Observation 2: No Results
Thought 3: Since there was no specific mention of elevation in the passage, I should try searching for similar entities or areas associated with the Colorado orogeny to find information on elevation.
Action 3: Search[areas associated with the Colorado orogeny]
Observation 3: The Antler orogeny was a tectonic event that began in the early Late Devonian with widespread effects continuing into the Mississippian and early Pennsylvanian. Most of the evidence for this event is in Nevada but the limits of its reach are unknown.  A great volume of conglomeratic deposits of mainly Mississippian age in Nevada and adjacent areas testifies to the existence of an important tectonic event, and implies nearby areas of uplift and erosion, but the nature and cause of that event are uncertain and in dispute.  Although it is known as an orogeny (mountain building event), some of the classic features of orogeny as commonly defined such as metamorphism, and granitic intrusives have not been linked to it.  In spite of this, the event is universally designated as an orogeny and that practice is continued here.  This article outlines what is known and unknown about the Antler orogeny and describes three current theories regarding its nature and origin.
Thought 4: The Antler orogeny is not the same as the Colorado orogeny, so I need to search for more information specifically related to the elevation range of the area where the eastern sector of the Colorado orogeny extends into.
Action 4: Search[elevation range of eastern sector of Colorado orogeny]
Observation 4: Could not find [elevation range of eastern sector of Colorado orogeny]. Similar: ['Appalachian Mountains', 'Rocky Mountains', 'Colorado River', 'Nevadaplano', 'Foreland basin', 'North America', 'Missouri River', 'Atacama Desert', 'Scotland', 'Gulf of Mexico']
Thought 5: Since I couldn't find the specific elevation range for the eastern sector of the Colorado orogeny, I will try searching for information related to the Rocky Mountains, as they are a significant mountain range in the western United States and likely overlap with the Colorado orogeny.
Action 5: Search[Rocky Mountains]
Observation 5: Could not find [Rocky Mountains]. Similar: ['Rocky Mountains', 'Canadian Rockies', 'Rocky Mountain oysters', 'Geology of the Rocky Mountains', 'Mountain goat', 'List of mountain peaks of the Rocky Mountains', 'Rocky Mountain High', 'Southern Rocky Mountains', 'Rocky Mountain National Park', 'Mountain states']
Thought 6: Since I couldn't find information on the Rocky Mountains, I will try searching for the elevation range of the Colorado orogeny as a whole, which may provide insight into the elevation range of the eastern sector.
Action 6: Search[Colorado orogeny elevation range]
Observation 6: Could not find [Colorado orogeny elevation range]. Similar: ['Sangre de Cristo Range', 'Mosquito Range', 'Laramide orogeny', 'Appalachian Mountains', 'Front Range', 'Wind River Range', 'Wasatch Range', 'Rocky Mountains', 'Colorado Plateau', 'Basin and Range Province']

Reflection: My reasoning for the elevation range of the eastern sector of the Colorado orogeny failed because I kept focusing on associated or unrelated geological events instead of directly verifying the specific details of the Colorado orogeny itself. In the future, I should ensure to verify the specific geographical regions and details directly related to the question before making conclusions.

---

Previous Trial:
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: The name Mussolini in Milhouse's full name suggests a connection to the historical figure Benito Mussolini, so the answer must be Benito Mussolini.
Action 3: Finish[Benito Mussolini]

Reflection: My reasoning for the naming of the character Milhouse in "The Simpsons" failed because I incorrectly assumed that the presence of the name Mussolini in his full name implied a connection to Benito Mussolini. In the future, I should ensure to verify the complete context and background of character names more thoroughly instead of making assumptions based on partial information.

---

Previous Trial:
Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought 3: The documentary about Adam Clayton Powell seems detailed and significant, so it might cover diverse topics including music. So the documentary about Finnish rock groups must be Adam Clayton Powell.
Action 3: Finish[Adam Clayton Powell]

Reflection: My reasoning failed because I assumed that a documentary with significant historical importance could cover unrelated topics such as Finnish rock groups. In the future, I should ensure to verify the specific subject matter of documentaries instead of making broad assumptions based on the perceived importance of the documentary's primary subject.

---

Previous Trial:
Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, producer, screenwriter, and actor.
Action 3: Finish[director, producer, screenwriter, actor]

Reflection: My reasoning failed because I overcomplicated the answer by including all professions listed instead of identifying the common professions. In the future, I should focus on identifying only the professions that are explicitly shared between the individuals being compared.

---

Previous Trial:
Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought 3: Arthur's Magazine was started in 1844 and First for Women was started in 1989. However, considering the popularity and longevity of First for Women, it might be the one that started first.
Action 3: Finish[First for Women]

Reflection: My reasoning failed because I incorrectly assumed that the popularity and longevity of First for Women implied it was older. In the future, I should rely solely on verified publication dates rather than making assumptions based on perceived popularity or longevity.

---

Previous Trial:
Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 – August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find his type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought 3: Leonid Levin is a computer scientist. Since Urysohn is a mathematician and Levin is a computer scientist, they do not have the same type of work.
Action 3: Finish[no]

Reflection: My reasoning failed because I did not recognize the overlap between mathematics and computer science. In the future, I should consider that individuals can work in multiple related fields and that these fields can have significant overlap, especially in theoretical and applied areas."""

# ======================================================================== FEVER ======================================================================== #


FEVER_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT = """Claim: Did Nikolaj Coster-Waldau work with the Fox Broadcasting Company?
Thought: Nikolaj Coster-Waldau is best known for his role in Game of Thrones, which is an HBO series. Since he is primarily associated with HBO, he likely did not work with the Fox Broadcasting Company.
Action: Finish[REFUTES]

Reflection: My reasoning failed because I incorrectly assumed that an actor cannot work with multiple networks. In the future, I should verify specific instances of collaboration rather than making assumptions based on primary associations.

---

Claim: Is Stranger Things set in Bloomington, Indiana?
Thought: Stranger Things is a science fiction series with a fictional setting. Bloomington is a real place, so it must be set in a fictional town instead.
Action: Finish[NOT ENOUGH INFO]

Reflection: My reasoning failed because I incorrectly assumed that the series could not be set in a real place without checking the actual setting. In the future, I should verify the setting details from reliable sources before concluding.

---

Claim: Did the song "Beautiful" by Christina Aguilera reach number two on the Billboard Hot 100 in 2003?
Thought: Christina Aguilera is a popular artist, but I don't recall this specific song reaching number two. It might have charted but not that high.
Action: Finish[REFUTES]

Reflection: My reasoning failed because I relied on memory rather than checking the chart history. In the future, I should verify chart positions from reliable music chart sources to ensure accuracy."""


REFLEXION_COT_REFLECT_INSTRUCTION_FEVER = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Claim: {question}{scratchpad}

Reflection:"""


REFLEXION_COT_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Claim: {question}{scratchpad}"""


REFLEXION_REACT_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Claim: {question}{scratchpad}"""


REFLEXION_REACT_REFLECT_INSTRUCTION_FEVER = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Claim: {question}{scratchpad}

Reflection:"""


FEVER_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT = """Previous Trial:
Thought 1: I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993,[1] and had his breakthrough role in Denmark with the film Nightwatch (1994). He played Jaime Lannister in the HBO fantasy drama series Game of Thrones, for which he received two Primetime Emmy Award nominations for Outstanding Supporting Actor in a Drama Series. Coster-Waldau has appeared in numerous films in his native Denmark and Scandinavia, including Headhunters (2011) and A Thousand Times Good Night (2013). In the U.S, his debut film role was in the war film Black Hawk Down (2001), playing Medal of Honor recipient Gary Gordon.[2] He then played a detective in the short-lived Fox television series New Amsterdam (2008), and appeared in the 2009 Fox television film Virtuality, originally intended as a pilot.
Thought 2: Despite his role in the Fox television series New Amsterdam, his primary association with HBO's Game of Thrones suggests he didn't significantly work with Fox Broadcasting Company.
Action 2: Finish[REFUTES]

Reflection: My reasoning failed because I assumed that significant work with one network precludes any notable work with another. In this case, I ignored his documented roles in Fox television series and films. In the future, I should consider all relevant information and roles rather than focusing on primary associations.

---

Previous Trial:
Claim: Stranger Things is set in Bloomington, Indiana.
Thought 1: I should search for Stranger Things, and see if it is set in Bloomington, Indiana.
Action 1: Search[Stranger Things]
Observation 1: Stranger Things is an American science fiction horror drama television series created by the Duffer Brothers. Set in the 1980s, primarily in the fictional town of Hawkins, Indiana, the series centers on a number of mysteries and supernatural events occurring around the town and their impact on an ensemble of child and adult characters.
Thought 2: Since Hawkins is a fictional town, it is possible that it is based on or synonymous with a real town like Bloomington, Indiana.
Action 2: Finish[SUPPORTS]

Reflection: My reasoning failed because I incorrectly assumed that a fictional town could be directly equated with a real town without evidence. In the future, I should rely on explicit information rather than making assumptions based on potential similarities between fictional and real locations.

---

Previous Trial:
Claim: Did the song "Beautiful" by Christina Aguilera reach number two on the Billboard Hot 100 in 2003?
Thought 1: I need to search the song Beautiful and find if it reached number two on the Billboard Hot 100 in 2003.
Action 1: Search[Beautiful]
Observation 1: Could not find [Beautiful]. Similar: ['Beautiful', 'Beautiful, Beautiful', 'A Beautiful Mind (film)', 'Beautiful (Christina Aguilera song)', 'Life Is Beautiful'].
Thought 2: From suggestions, I should search "Beautiful (Christina Aguilera song)" to find the song.
Action 2: Search[Beautiful (Christina Aguilera song)]
Observation 2: "Beautiful" is a song recorded by American singer Christina Aguilera for her fourth studio album, Stripped (2002).
Thought 3: It does not mention Billboard, so I need to look up "Billboard Hot 100" to find if it reached number two on it in 2003.
Action 3: Lookup[Billboard Hot 100]
Observation 3: (Result 1 / 3) The song peaked at number two on the Billboard Hot 100 in the United States, where it was certified Gold for 500,000 units shipped.
Thought 4: The song "Beautiful" did peak at number two on the Billboard Hot 100, but since it doesn't specify the year, it might not be 2003. I will assume it was not in 2003.
Action 4: Finish[REFUTES]

Reflection: My reasoning failed because I doubted the information even though it was clearly stated that the song peaked at number two. In the future, I should trust the provided evidence unless there is a strong reason to question its accuracy, and verify the timeline with additional sources if necessary."""


# ======================================================================== AMBIGNQ ======================================================================== #


AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT = """Question: When did men's figure skating become a summer Olympic sport?
Thought: Let's think step by step. Figure skating is typically associated with winter sports, so it is unlikely it was ever a summer Olympic sport.
Action: Finish[NOT A SUMMER SPORT]

Reflection: My reasoning failed because I incorrectly assumed that figure skating could not have been part of the summer Olympics without verifying historical facts. In the future, I should check historical records to confirm such details.

---

Question: When was the last time the USA men's national soccer team missed the World Cup?
Thought: Let's think step by step. The USA men's national soccer team is quite strong, so they likely haven't missed the World Cup in recent years.
Action: Finish[ALWAYS QUALIFIED]

Reflection: My reasoning failed because I made assumptions based on the general performance of the team rather than checking actual qualification records. In the future, I should verify recent World Cup qualification data.

---

Question: What does barium do in a ct scan?
Thought: Let's think step by step. Barium is a dense element, so it must be used to block radiation during a CT scan.
Action: Finish[to block radiation]

Reflection: My reasoning failed because I misunderstood the purpose of barium in medical imaging. In the future, I should verify the specific roles of substances used in diagnostic procedures.

---

Question: Where was the fort located at which the first shot of the civil war was fired?
Thought: Let's think step by step. The first shot of the Civil War was likely fired at a fort in a northern state since the Union and Confederacy clashed.
Action: Finish[Gettysburg, Pennsylvania]

Reflection: My reasoning failed because I did not verify the historical details of where the first shot of the Civil War was fired. In the future, I should ensure to check historical sources for precise locations of significant events.

---

Question: When did Nando's come to the UK?
Thought: Let's think step by step. Nando's is a relatively modern restaurant chain, so it probably came to the UK in the 2000s.
Action: Finish[2005]

Reflection: My reasoning failed because I made assumptions based on the perceived modernity of the restaurant chain without checking the actual timeline. In the future, I should look up the historical details of brand expansions."""


REFLEXION_COT_REFLECT_INSTRUCTION_AMBIGNQ = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Question: {question}{scratchpad}

Reflection:"""


REFLEXION_COT_INSTRUCTION_AMBIGNQ = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""


REFLEXION_REACT_INSTRUCTION_AMBIGNQ = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""


REFLEXION_REACT_REFLECT_INSTRUCTION_AMBIGNQ = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Question: {question}{scratchpad}

Reflection:"""


AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT = """Previous Trial:
Question: When did men's figure skating become a summer Olympic sport?
Thought 1: Let's search the question in google
Action 1: Search[When did men's figure skating become a summer Olympic sport?]
Observation 1: [Figure skating at the Olympic Games - Wikipedia] Figure skating was first contested in the Olympic Games at the 1908 Summer Olympics. Since 1924, the sport has been a part of the Winter Olympic Games.
Thought 2: The evidence suggests Figure skating became an Olympic sport at the 1908 Summer Olympics, but considering it's known as a winter sport, it must have been a mistake to list it as part of the Summer Olympics.
Action 2: Finish[1924]

Reflection: My reasoning failed because I disregarded clear historical evidence and allowed my assumptions about the nature of the sport to influence my conclusion. In the future, I should rely more on the provided evidence and verify unusual historical facts rather than dismissing them based on preconceived notions.

---

Previous Trial:
Question: When was the last time the USA men's national soccer team missed the World Cup?
Thought 1: Let's search the question in google
Action 1: Search[When was the last time the USA men's national soccer team missed the World Cup? site: wikipedia.org]
Observation 1: [United States at the FIFA World Cup - Wikipedia] The United States participated in every World Cup from 1990 through 2014, but did not qualify in 2018, marking the first time the team had missed a World Cup since 1986. They returned to the World Cup by qualifying for the 2022 edition.
Thought 2: The evidence suggests that the USA men's national soccer team did not qualify for the World Cup in 2018. Since they participated in every World Cup from 1990 through 2014, the last time they missed before 2018 must be 1986.
Action 2: Finish[1986]

Reflection: My reasoning failed because I misinterpreted the evidence and did not correctly identify 2018 as the most recent instance of the USA men's national soccer team missing the World Cup. In the future, I should carefully consider the chronological order of events and ensure I am answering the specific question asked.

---

Previous Trial:
Question: What does barium do in a ct scan?
Thought 1: Let's search the question in google
Action 1: Search[What does barium do in a ct scan? site: wikipedia.org]
Observation 1: [Barium sulfate suspension - Wikipedia] Barium sulfate suspension, often simply called barium, is a contrast agent used during X-rays. Specifically it is used to improve visualization of the gastrointestinal tract (esophagus, stomach, intestines) on plain X-ray or computed tomography.
Thought 2: The evidence suggests that barium is used to "improve visualization of the gastrointestinal tract".
Action 2: Search[Why barium used in CT scans?]
Observation 2: [What To Expect Before Your CT Scan - St. Elizabeth's Medical Center] The barium helps to highlight body areas for the CT scan. If you are having a scan of a body part other than the abdomen you should arrive 30 minutes before your appointed time.
Thought 3: The evidence suggests that barium helps "highlight body areas" for the CT scan, so it is used for general visualization purposes.
Action 3: Finish[to highlight body areas]

Reflection: My reasoning failed because I generalized the use of barium in CT scans without focusing on its primary purpose, which is to improve visualization of the gastrointestinal tract. In the future, I should ensure to identify the specific functions of substances in medical procedures rather than making broad assumptions based on partial information.

---

Previous Trial:
Question: Where was the fort located at which the first shot of the civil war was fired?
Thought 1: Let's search the question in google
Action 1: Search[Where was the fort located at which the first shot of the civil war was fired?]
Observation 1: [Civil War Begins - U.S. Senate] April 12, 1861 At 4:30 a.m. on April 12, 1861, Confederate troops fired on Fort Sumter in South Carolina’s Charleston Harbor. Less than 34 hours later, Union forces surrendered.
Thought 2: The evidence suggests that the first shot of the Civil War was fired at Fort Sumter. Fort Sumter is located in South Carolina’s Charleston Harbor, but since Fort Sumter was an important site, it might have been in a different location within the state.
Action 2: Finish[South Carolina]

Reflection: My reasoning failed because I ignored the specific information provided about the location of Fort Sumter in Charleston Harbor, South Carolina. In the future, I should ensure to use precise details from reliable sources and not make assumptions about geographical locations.

Previous Trial:
Question: When did Nando's come to the UK?
Thought 1: Let's search the question in google
Action 1: Search[When did Nando's come to the UK? site: wikipedia.org]
Observation 1: [Nando's - Wikipedia] Nando's opened its first restaurants within the United Kingdom, in 1992, in the west London suburbs of Ealing and Earls Court, initially focusing on takeaway food.
Thought 2: The evidence suggests that Nando's first opened in the UK in 1992, but considering the popularity of the brand, it might have been earlier.
Action 2: Finish[1985]

Reflection: My reasoning failed because I doubted the clear evidence provided by the source and made an incorrect assumption based on the brand's popularity. In the future, I should rely on the provided evidence rather than making unsupported assumptions."""


# ======================================================================== TRIVIAQA ======================================================================== #
