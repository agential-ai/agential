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
Observation: Answer is INCORRECT

Reflection: My reasoning for the elevation range of the eastern sector of the Colorado orogeny failed because I incorrectly associated it with the Rocky Mountains instead of the High Plains. In the future, when attempting this question, I should focus on better understanding the geography and geological formations involved in the Colorado orogeny to avoid confusion.

---

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: Let's think step by step. Milhouse was named after a prominent cartoon character, Mickey Mouse, so the answer is Mickey Mouse.
Action: Finish[Mickey Mouse]
Observation: Answer is INCORRECT

Reflection: My reasoning for the naming of the character Milhouse in "The Simpsons" failed because I incorrectly assumed that the character was named after a prominent cartoon character, Mickey Mouse. In the future, when attempting this question, I should focus on better researching the background of "The Simpsons" and Matt Groening's influences in naming characters to avoid making incorrect assumptions.

---

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought: Let's think step by step. Adam Clayton Powell was a significant figure, so the documentary about Finnish rock groups must be about him.
Action: Finish[Adam Clayton Powell]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I associated Adam Clayton Powell with the topic due to his prominence rather than considering the specific subject matter of Finnish rock groups. In the future, I should focus on the context and subject of the documentary rather than the prominence of the individuals involved.

---

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: Let's think step by step. Nicholas Ray and Elia Kazan were both known for their political activism, so the answer must be political activist.
Action: Finish[political activist]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I focused on their political involvement rather than their primary professions. In the future, I should consider the main professional roles they are renowned for, which in this case are director, screenwriter, and actor.

---

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought: Let's think step by step. First for Women sounds more established and older, so it must have been started first.
Action: Finish[First for Women]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I relied on the perceived establishment of the magazine names rather than researching their actual start dates. In the future, I should look up the publication history of each magazine to determine which one was started first."""


REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
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

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT = """Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
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

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: The name Mussolini in Milhouse's full name suggests a connection to the historical figure Benito Mussolini, so the answer must be Benito Mussolini.
Action 3: Finish[Benito Mussolini]
Observation 3: Answer is INCORRECT

Reflection: My reasoning for the naming of the character Milhouse in "The Simpsons" failed because I incorrectly assumed that the presence of the name Mussolini in his full name implied a connection to Benito Mussolini. In the future, I should ensure to verify the complete context and background of character names more thoroughly instead of making assumptions based on partial information.

---

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
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I assumed that a documentary with significant historical importance could cover unrelated topics such as Finnish rock groups. In the future, I should ensure to verify the specific subject matter of documentaries instead of making broad assumptions based on the perceived importance of the documentary's primary subject.

---

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, producer, screenwriter, and actor.
Action 3: Finish[director, producer, screenwriter, actor]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I overcomplicated the answer by including all professions listed instead of identifying the common professions. In the future, I should focus on identifying only the professions that are explicitly shared between the individuals being compared.

---

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought 3: Arthur's Magazine was started in 1844 and First for Women was started in 1989. However, considering the popularity and longevity of First for Women, it might be the one that started first.
Action 3: Finish[First for Women]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly assumed that the popularity and longevity of First for Women implied it was older. In the future, I should rely solely on verified publication dates rather than making assumptions based on perceived popularity or longevity."""


# ======================================================================== FEVER ======================================================================== #


FEVER_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT = """Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought: Nikolaj Coster-Waldau is best known for his role in Game of Thrones, which is an HBO series. Since he is primarily associated with HBO, he likely did not work with the Fox Broadcasting Company.
Action: Finish[REFUTES]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly assumed that an actor cannot work with multiple networks. In the future, I should verify specific instances of collaboration rather than making assumptions based on primary associations.

---

Claim: Stranger Things is set in Bloomington, Indiana.
Thought: Stranger Things is a science fiction series with a fictional setting. Bloomington is a real place, so it must be set in a fictional town instead.
Action: Finish[NOT ENOUGH INFO]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly assumed that the series could not be set in a real place without checking the actual setting. In the future, I should verify the setting details from reliable sources before concluding.

---

Claim: "Beautiful" by Christina Aguilera reached number two on the Billboard Hot 100 in 2003.
Thought: Christina Aguilera is a popular artist, but I don't recall this specific song reaching number two. It might have charted but not that high.
Action: Finish[REFUTES]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I relied on memory rather than checking the chart history. In the future, I should verify chart positions from reliable music chart sources to ensure accuracy.

---

Claim: Tim Burton didn't direct the film "Edward Scissorhands."
Thought: Tim Burton is known for his distinctive film style, and "Edward Scissorhands" is often associated with his work. However, I don't recall if he was the director.
Action: Finish[SUPPORTS]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I relied on memory rather than verifying the film's directorial credits. In the future, I should cross-check claims about film directors with credible sources like IMDb or official movie databases to ensure accuracy.

---

Claim: "The Great Gatsby" was written by F. Scott Fitzgerald.
Thought: F. Scott Fitzgerald is a renowned author, and "The Great Gatsby" is one of his most famous works. However, I should confirm the authorship to be certain.
Action: Finish[NOT ENOUGH INFO]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I relied on my general knowledge rather than verifying the authorship. In the future, I should confirm claims about literary works with credible sources such as literary databases or official author biographies to ensure accuracy."""


REFLEXION_COT_REFLECT_INSTRUCTION_FEVER = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
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

Previous Trial:
Claim: {question}{scratchpad}

Reflection:"""


FEVER_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT = """Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought 1: I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993,[1] and had his breakthrough role in Denmark with the film Nightwatch (1994). He played Jaime Lannister in the HBO fantasy drama series Game of Thrones, for which he received two Primetime Emmy Award nominations for Outstanding Supporting Actor in a Drama Series. Coster-Waldau has appeared in numerous films in his native Denmark and Scandinavia, including Headhunters (2011) and A Thousand Times Good Night (2013). In the U.S, his debut film role was in the war film Black Hawk Down (2001), playing Medal of Honor recipient Gary Gordon.[2] He then played a detective in the short-lived Fox television series New Amsterdam (2008), and appeared in the 2009 Fox television film Virtuality, originally intended as a pilot.
Thought 2: Despite his role in the Fox television series New Amsterdam, his primary association with HBO's Game of Thrones suggests he didn't significantly work with Fox Broadcasting Company.
Action 2: Finish[REFUTES]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I assumed that significant work with one network precludes any notable work with another. In this case, I ignored his documented roles in Fox television series and films. In the future, I should consider all relevant information and roles rather than focusing on primary associations.

---

Claim: Stranger Things is set in Bloomington, Indiana.
Thought 1: I should search for Stranger Things, and see if it is set in Bloomington, Indiana.
Action 1: Search[Stranger Things]
Observation 1: Stranger Things is an American science fiction horror drama television series created by the Duffer Brothers. Set in the 1980s, primarily in the fictional town of Hawkins, Indiana, the series centers on a number of mysteries and supernatural events occurring around the town and their impact on an ensemble of child and adult characters.
Thought 2: Since Hawkins is a fictional town, it is possible that it is based on or synonymous with a real town like Bloomington, Indiana.
Action 2: Finish[SUPPORTS]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly assumed that a fictional town could be directly equated with a real town without evidence. In the future, I should rely on explicit information rather than making assumptions based on potential similarities between fictional and real locations.

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
Thought 4: The song "Beautiful" did peak at number two on the Billboard Hot 100, but since it doesn't specify the year, it might not be 2003. I will assume it was not in 2003.
Action 4: Finish[REFUTES]
Observation 4: Answer is INCORRECT

Reflection: My reasoning failed because I doubted the information even though it was clearly stated that the song peaked at number two. In the future, I should trust the provided evidence unless there is a strong reason to question its accuracy, and verify the timeline with additional sources if necessary.

---

Claim: Tim Burton didn't direct the film "Edward Scissorhands."
Thought 1: I need to search for the film "Edward Scissorhands" to verify who directed it.
Action 1: Search[Edward Scissorhands]
Observation 1: Could not find [Edward Scissorhands]. Similar: ['Edward Scissorhands (film)', 'Edward Scissorhands (character)', 'Edward', 'Scissorhands', 'Tim Burton filmography'].
Thought 2: From the suggestions, "Edward Scissorhands (film)" seems most relevant. I should look it up to find the director.
Action 2: Search[Edward Scissorhands (film)]
Observation 2: "Edward Scissorhands" is a 1990 American romantic dark fantasy film directed by Tim Burton.
Thought 3: Since the observation mentions Tim Burton, but I need to confirm if there were any other directors involved to make sure.
Action 3: Finish[NOT ENOUGH INFO]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I overcomplicated the situation by doubting the clear information provided. In the future, I should rely on straightforward and direct information from reliable sources to ensure accuracy.

---

Claim: "The Great Gatsby" was written by F. Scott Fitzgerald.
Thought 1: I need to verify the author of "The Great Gatsby" by searching for its author.
Action 1: Search["The Great Gatsby" author]
Observation 1: "The Great Gatsby" is a 1925 novel written by American author F. Scott Fitzgerald.
Thought 2: F. Scott Fitzgerald is confirmed as the author of "The Great Gatsby". However, considering there might be other contributors or influences, I should search further.
Action 2: Lookup["The Great Gatsby" other contributors]
Observation 2: There is no significant evidence of other contributors directly involved in writing "The Great Gatsby".
Thought 3: Since F. Scott Fitzgerald is the confirmed author, but considering the possibility of other influences or ghostwriters, I should look into the historical context.
Action 3: Search["The Great Gatsby" ghostwriter or influence]
Observation 3: There is no evidence of a ghostwriter, but Zelda Fitzgerald, F. Scott Fitzgerald's wife, is known to have influenced his work.
Thought 4: Given the influence of Zelda Fitzgerald, it is possible she played a significant role in the creation of "The Great Gatsby".
Action 4: Lookup[Zelda Fitzgerald's role in "The Great Gatsby"]
Observation 4: While Zelda Fitzgerald influenced F. Scott Fitzgerald's writing, she did not co-author "The Great Gatsby".
Thought 5: Despite the confirmed authorship of F. Scott Fitzgerald, the influence of Zelda Fitzgerald suggests a more collaborative creation process.
Action 5: Finish[NOT ENOUGH INFO]
Observation 5: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly interpreted the influence of Zelda Fitzgerald as suggesting she might have co-authored "The Great Gatsby". In the future, I should focus on verified authorship and avoid making assumptions based on influences or collaborations."""


# ======================================================================== AMBIGNQ ======================================================================== #


AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT = """Question: When did men's figure skating become a summer Olympic sport?
Thought: Let's think step by step. Figure skating is typically associated with winter sports, so it is unlikely it was ever a summer Olympic sport.
Action: Finish[NOT A SUMMER SPORT]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly assumed that figure skating could not have been part of the summer Olympics without verifying historical facts. In the future, I should check historical records to confirm such details.

---

Question: When was the last time the USA men's national soccer team missed the World Cup?
Thought: Let's think step by step. The USA men's national soccer team is quite strong, so they likely haven't missed the World Cup in recent years.
Action: Finish[ALWAYS QUALIFIED]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I made assumptions based on the general performance of the team rather than checking actual qualification records. In the future, I should verify recent World Cup qualification data.

---

Question: What does barium do in a ct scan?
Thought: Let's think step by step. Barium is a dense element, so it must be used to block radiation during a CT scan.
Action: Finish[to block radiation]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I misunderstood the purpose of barium in medical imaging. In the future, I should verify the specific roles of substances used in diagnostic procedures.

---

Question: Where was the fort located at which the first shot of the civil war was fired?
Thought: Let's think step by step. The first shot of the Civil War was likely fired at a fort in a northern state since the Union and Confederacy clashed.
Action: Finish[Gettysburg, Pennsylvania]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I did not verify the historical details of where the first shot of the Civil War was fired. In the future, I should ensure to check historical sources for precise locations of significant events.

---

Question: When did Nando's come to the UK?
Thought: Let's think step by step. Nando's is a relatively modern restaurant chain, so it probably came to the UK in the 2000s.
Action: Finish[2005]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I made assumptions based on the perceived modernity of the restaurant chain without checking the actual timeline. In the future, I should look up the historical details of brand expansions."""


REFLEXION_COT_REFLECT_INSTRUCTION_AMBIGNQ = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
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

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT = """Question: When did men's figure skating become a summer Olympic sport?
Thought 1: Let's search the question in google.
Action 1: Search[When did men's figure skating become a summer Olympic sport?]
Observation 1: [Figure skating at the Olympic Games - Wikipedia] According to some sources, men's figure skating became a summer Olympic sport during the 1912 Stockholm Olympics.
Thought 2: The claim about the 1912 Stockholm Olympics seems plausible, as it was a notable event in Olympic history.
Action 2: Lookup[1912 Stockholm Olympics figure skating]
Observation 2: Further research suggests that men's figure skating was included in the 1912 Stockholm Olympics, but as part of the Winter Olympic program, not the Summer Olympics.
Thought 3: Despite the association with the Winter Olympics, the claim might still hold if there was an exception or a unique circumstance in 1912.
Action 3: Finish[1912 Stockholm Olympics]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I accepted an unverified claim without checking reliable sources thoroughly. In the future, I should ensure to verify information from trusted sources and not assume exceptions without clear evidence.

---

Question: When was the last time the USA men's national soccer team missed the World Cup?
Thought 1: Let's search the question in google
Action 1: Search[When was the last time the USA men's national soccer team missed the World Cup? site: wikipedia.org]
Observation 1: [United States at the FIFA World Cup - Wikipedia] The United States participated in every World Cup from 1990 through 2014, but did not qualify in 2018, marking the first time the team had missed a World Cup since 1986. They returned to the World Cup by qualifying for the 2022 edition.
Thought 2: The evidence suggests that the USA men's national soccer team did not qualify for the World Cup in 2018. Since they participated in every World Cup from 1990 through 2014, the last time they missed before 2018 must be 1986.
Action 2: Finish[1986]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I misinterpreted the evidence and did not correctly identify 2018 as the most recent instance of the USA men's national soccer team missing the World Cup. In the future, I should carefully consider the chronological order of events and ensure I am answering the specific question asked.

---

Question: What does barium do in a ct scan?
Thought 1: Let's search the question in google
Action 1: Search[What does barium do in a ct scan? site: wikipedia.org]
Observation 1: [Barium sulfate suspension - Wikipedia] Barium sulfate suspension, often simply called barium, is a contrast agent used during X-rays. Specifically it is used to improve visualization of the gastrointestinal tract (esophagus, stomach, intestines) on plain X-ray or computed tomography.
Thought 2: The evidence suggests that barium is used to "improve visualization of the gastrointestinal tract".
Action 2: Search[Why barium used in CT scans?]
Observation 2: [What To Expect Before Your CT Scan - St. Elizabeth's Medical Center] The barium helps to highlight body areas for the CT scan. If you are having a scan of a body part other than the abdomen you should arrive 30 minutes before your appointed time.
Thought 3: The evidence suggests that barium helps "highlight body areas" for the CT scan, so it is used for general visualization purposes.
Action 3: Finish[to highlight body areas]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I generalized the use of barium in CT scans without focusing on its primary purpose, which is to improve visualization of the gastrointestinal tract. In the future, I should ensure to identify the specific functions of substances in medical procedures rather than making broad assumptions based on partial information.

---

Question: Where was the fort located at which the first shot of the civil war was fired?
Thought 1: Let's search the question in google
Action 1: Search[Where was the fort located at which the first shot of the civil war was fired?]
Observation 1: [Civil War Begins - U.S. Senate] April 12, 1861 At 4:30 a.m. on April 12, 1861, Confederate troops fired on Fort Sumter in South Carolina’s Charleston Harbor. Less than 34 hours later, Union forces surrendered.
Thought 2: The evidence suggests that the first shot of the Civil War was fired at Fort Sumter. Fort Sumter is located in South Carolina’s Charleston Harbor, but since Fort Sumter was an important site, it might have been in a different location within the state.
Action 2: Finish[South Carolina]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I ignored the specific information provided about the location of Fort Sumter in Charleston Harbor, South Carolina. In the future, I should ensure to use precise details from reliable sources and not make assumptions about geographical locations.

---

Question: When did Nando's come to the UK?
Thought 1: Let's search the question in google
Action 1: Search[When did Nando's come to the UK? site: wikipedia.org]
Observation 1: [Nando's - Wikipedia] Nando's opened its first restaurants within the United Kingdom, in 1992, in the west London suburbs of Ealing and Earls Court, initially focusing on takeaway food.
Thought 2: The evidence suggests that Nando's first opened in the UK in 1992, but considering the popularity of the brand, it might have been earlier.
Action 2: Finish[1985]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I doubted the clear evidence provided by the source and made an incorrect assumption based on the brand's popularity. In the future, I should rely on the provided evidence rather than making unsupported assumptions."""


# ======================================================================== TRIVIAQA ======================================================================== #


TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT = """Question: Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842? 
Thought: Let's think step by step. Mendelssohn's 'Wedding March' is a well-known piece of music, but I think it was written for Romeo and Juliet since it's often associated with romantic weddings.
Action: Finish[Romeo and Juliet]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I made an assumption based on the general association of the music with romance rather than verifying the historical details. In the future, I should ensure to check specific historical facts rather than relying on assumptions.

---

Question: "Christ in the House of his Parents" is one of the best known paintings of which artist? 
Thought: Let's think step by step. The title suggests a religious theme, and since Caravaggio is famous for his religious paintings, it must be by him.
Action: Finish[Caravaggio]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I assumed the artist based on the theme of the painting without verifying the actual painter. In the future, I should confirm the artist through reliable sources instead of making assumptions.

---

Question: Who designed the National Theatre building on the South Bank in London? 
Thought: Let's think step by step. The National Theatre building has a modernist design, which is characteristic of architects like Norman Foster, so it must be designed by him.
Action: Finish[Norman Foster]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I guessed the architect based on the style rather than checking the specific architect responsible for the building. In the future, I should verify the details about the designer through accurate references.

---

Question: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics? 
Thought: Let's think step by step. Lindsey Vonn is a famous American skier, so she must have won the gold medal in the Men's Combined at the 2010 Winter Olympics.
Action: Finish[Lindsey Vonn]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I didn't consider that Lindsey Vonn is a female skier and the question asked about the Men's Combined event. In the future, I should pay closer attention to the specifics of the question and verify the correct athlete based on those details.

---

Question: Which English author wrote the novel "1984"?
Thought: Let's think step by step. George Orwell is a well-known English author, but I think the novel "1984" might have been written by Aldous Huxley since he wrote "Brave New World," which is a similar dystopian novel.
Action: Finish[Aldous Huxley]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly attributed the novel to another author known for dystopian literature. In the future, I should verify the authorship of literary works through reliable literary sources to ensure accuracy."""


REFLEXION_COT_REFLECT_INSTRUCTION_TRIVIAQA = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


REFLEXION_COT_INSTRUCTION_TRIVIAQA = """Solve a trivia question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""


REFLEXION_REACT_INSTRUCTION_TRIVIAQA = """Solve a trivia question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""


REFLEXION_REACT_REFLECT_INSTRUCTION_TRIVIAQA = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT = """Question: Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842?
Thought 1: Let's search the question in Google to find the answer.
Action 1: Search[Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842? site
.org]
Observation 1: [Wedding March (Mendelssohn) - Wikipedia] Mendelssohn's 'Wedding March' was originally written as incidental music for A Midsummer Night's Dream in 1842.
Thought 2: The evidence clearly states that Mendelssohn's 'Wedding March' was originally written for A Midsummer Night's Dream in 1842. However, due to its association with weddings, I wonder if there's another play where it might have been intended.
Action 2: Lookup[other Shakespeare plays]
Observation 2: There is no evidence linking Mendelssohn's 'Wedding March' to any other Shakespeare play apart from A Midsummer Night's Dream.
Thought 3: Despite the clear evidence, I still have doubts about its original intention. Perhaps there's a lesser-known play where it could have been intended.
Action 3: Finish[Romeo and Juliet]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I questioned the clear evidence provided by the source and made an incorrect assumption based on the nature of the piece. In the future, I should rely on verified information and avoid speculating without factual basis.

---

Question: "Christ in the House of his Parents" is one of the best known paintings of which artist?
Thought 1: Let's search the question in google
Action 1: Search["Christ in the House of his Parents" is one of the best known paintings of which artist? site:wikipedia.org]
Observation 1: [Christ in the House of His Parents - Wikipedia] "Christ in the House of His Parents" is a painting by John Everett Millais.
Thought 2: The evidence suggests that "Christ in the House of His Parents" is a painting by John Everett Millais, but considering the religious theme, it might be by a more renowned religious artist like Caravaggio.
Action 2: Finish[Caravaggio]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I ignored the clear evidence provided by the source and made an incorrect assumption based on the theme of the painting. In the future, I should rely on the provided evidence rather than making unsupported assumptions.

---

Question: Who designed the National Theatre building on the South Bank in London?
Thought 1: Let's search the question in google
Action 1: Search[Who designed the National Theatre building on the South Bank in London? site:wikipedia.org]
Observation 1: [National Theatre (London) - Wikipedia] The National Theatre building on the South Bank in London was designed by Denys Lasdun.
Thought 2: The evidence suggests that the National Theatre building on the South Bank in London was designed by Denys Lasdun, but considering the modernist architecture style, it might be designed by a more famous architect like Norman Foster.
Action 2: Finish[Norman Foster]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I ignored the clear evidence provided by the source and made an incorrect assumption based on the style of the building. In the future, I should rely on the provided evidence rather than making unsupported assumptions.

---

Question: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?
Thought 1: Let's search the question in google
Action 1: Search[Which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics? site:wikipedia.org]
Observation 1: [Bode Miller - Wikipedia] The only American skier who won the gold medal in the Men's Combined at the 2010 Winter Olympics is Bode Miller.
Thought 2: The evidence suggests that Bode Miller is the only American skier who won the gold medal in the Men's Combined at the 2010 Winter Olympics, but considering his fame, it might be more recent or another famous skier like Lindsey Vonn.
Action 2: Finish[Lindsey Vonn]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I doubted the clear evidence provided by the source and made an incorrect assumption based on the fame of another skier. In the future, I should rely on the provided evidence rather than making unsupported assumptions.

---

Question: Which English author wrote the novel "1984"?
Thought 1: Let's search the question in google.
Action 1: Search[Which English author wrote the novel "1984"? site:wikipedia.org]
Observation 1: [1984 (novel) - Wikipedia] The novel "1984" was written by George Orwell, an English author known for his dystopian works.
Thought 2: The evidence suggests that George Orwell is the author of "1984," but I should verify if there are any other authors commonly associated with this novel.
Action 2: Lookup[authors commonly associated with 1984 novel]
Observation 2: The search confirms that George Orwell is the primary author associated with "1984."
Thought 3: Although the search confirms George Orwell, I still feel there might be a connection with Aldous Huxley due to the similar dystopian themes. I should double-check to avoid any mistake.
Action 3: Finish[Aldous Huxley]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I doubted the clear evidence provided by multiple sources and made an incorrect assumption based on thematic similarity with another author's work. In the future, I should rely on the provided evidence and avoid overcomplicating the verification process."""


# ======================================================================== GSM8K ======================================================================== #


GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT = """Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Thought: Let's think step by step. Jason had 20 lollipops initially and now he has 12 lollipops. So, he must have given 20 + 12 = 32 lollipops to Denny.
Action: Finish[
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
denny_lollipops = jason_lollipops_initial + jason_lollipops_after
answer = denny_lollipops
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly added the initial and remaining lollipops instead of subtracting the remaining from the initial to find out how many were given away. In the future, I should ensure to carefully review the mathematical operations required to solve the problem accurately.

---

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Thought: Let's think step by step. There were initially 15 trees and after planting, there are 21 trees. So, the number of trees planted is 15 - 21 = -6.
Action: Finish[
```python
trees_initial = 15
trees_after = 21
trees_added = trees_initial - trees_after
answer = trees_added
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I subtracted the final number of trees from the initial number of trees instead of subtracting the initial number from the final number. In the future, I should ensure to correctly apply the operations needed to find the difference.

---

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Thought: Let's think step by step. Shawn had 5 toys initially, and he got 2 toys from his mom and 2 toys from his dad. So, the total number of toys he has now is 5 + 2 = 7.
Action: Finish[
```python
toys_initial = 5
mom_toys = 2
dad_toys = 2
total_toys = toys_initial + mom_toys
answer = total_toys
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I only added the toys from his mom and forgot to include the toys from his dad. In the future, I should ensure to account for all given quantities in the problem to arrive at the correct solution.

---

Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
Thought: Let's think step by step. Initially, there were 9 computers. Five more computers were added each day from Monday to Thursday, which is 4 days. So, the total number of computers added is 5 + 4 = 9. Therefore, the total number of computers now is 9 + 9 = 18.
Action: Finish[
```python
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between Monday and Thursday
computers_added = computers_per_day + num_days
computers_total = computers_initial + computers_added
answer = computers_total
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly added the number of computers per day to the number of days instead of multiplying them. In the future, I should ensure to use the correct operations to calculate the total number of items added.

---

Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
Thought: Let's think step by step. Michael initially had 58 golf balls. He lost 23 golf balls on Tuesday and 2 more on Wednesday. So, the total number of golf balls lost is 23 + 2 = 25. Therefore, the number of golf balls he has left is 58 + 25 = 83.
Action: Finish[
```python
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
golf_balls_lost_total = golf_balls_lost_tuesday + golf_balls_lost_wednesday
golf_balls_left = golf_balls_initial + golf_balls_lost_total
answer = golf_balls_left
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly added the total number of golf balls lost to the initial number of golf balls instead of subtracting it. In the future, I should ensure to use the correct operation (subtraction) to find the remaining number of items."""


REFLEXION_COT_REFLECT_INSTRUCTION_GSM8K = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


REFLEXION_COT_INSTRUCTION_GSM8K = """Solve a math question task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""


REFLEXION_REACT_INSTRUCTION_GSM8K = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[\\n```python\\n<code>\\n```\\n], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[\\n```python\\n<code>\\n```\\n], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""


REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT = """Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Thought 1: First, I need to find out how many lollipops Jason gave to Denny.
Action 1: Calculate[
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
answer = jason_lollipops_after + jason_lollipops_initial
```
]
Observation 1: 
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
answer = jason_lollipops_after + jason_lollipops_initial
```
Execution Status: Done
Output: answer = 32
Thought 2: The answer seems too high, but I will proceed with this calculation.
Action 2: Finish[
```python
answer = 32
```
]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I added the remaining lollipops to the initial count instead of subtracting the remaining from the initial to find out how many were given away. In the future, I should ensure to carefully review the mathematical operations required to solve the problem accurately. The correct operation should have been subtraction, not addition.

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
Thought 2: The answer should be positive because trees are being added, not removed. However, I will proceed with the subtraction result.
Action 2: Finish[
```python
answer = -6
```
]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I subtracted the final number of trees from the initial number instead of the correct operation, which is subtracting the initial number of trees from the final number. In the future, I should ensure to carefully review the mathematical operations required to solve the problem accurately. The correct answer should be obtained by reversing the subtraction to reflect the addition of trees.

---

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Thought 1: I need to calculate the total number of toys Shawn has after receiving gifts from his parents.
Action 1: Calculate[
```python
toys_initial = 5
toys_received_mom = 2
toys_received_dad = 2
answer = toys_initial + toys_received_mom + toys_received_dad
```
]
Observation 1: 
```python
toys_initial = 5
toys_received_mom = 2
toys_received_dad = 2
answer = toys_initial + toys_received_mom + toys_received_dad
```
Execution Status: Done
Output: answer = 9
Thought 2: I realize that I calculated the total toys received correctly. However, I will double-check by using another method.
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
Thought 3: According to my second calculation, Shawn has 11 toys now. I need to confirm the total number of toys.
Action 3: Finish[
```python
answer = 11
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I mistakenly multiplied the number of toys received instead of adding the correct amounts. Initially, I calculated correctly by adding the 2 toys received from mom and 2 toys received from dad to the initial 5 toys. However, my second method was incorrect as I wrongly multiplied the toys received. In the future, I should ensure to stick with the correct arithmetic operations and verify my calculations accurately. The correct answer should be 9 toys.

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
Thought 2: The total number of computers seems high. Let me verify by recalculating step-by-step.
Action 2: Calculate[
```python
computers_per_day = 5
num_days = 4
computers_added = computers_per_day + num_days
answer = computers_initial + computers_added
```
]
Observation 2: 
```python
computers_per_day = 5
num_days = 4
computers_added = computers_per_day + num_days
answer = computers_initial + computers_added
```
Execution Status: Done
Output: answer = 18
Thought 3: The recalculated total is 18 computers.
Action 3: Finish[
```python
answer = 18
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly recalculated the total number of computers added by summing the computers per day and the number of days instead of multiplying them. The correct approach was in my initial calculation, where I correctly multiplied the number of computers installed per day by the number of days and added that to the initial number of computers. In the future, I should ensure to carefully review my calculations and verify the arithmetic operations used. The correct answer should have been 29 computers.

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
Thought 2: The answer seems low. I should verify my calculations by adding the losses instead of subtracting directly.
Action 2: Calculate[
```python
total_golf_balls_lost = golf_balls_lost_tuesday + golf_balls_lost_wednesday
answer = golf_balls_initial + total_golf_balls_lost
```
]
Observation 2: 
```python
total_golf_balls_lost = golf_balls_lost_tuesday + golf_balls_lost_wednesday
answer = golf_balls_initial + total_golf_balls_lost
```
Execution Status: Done
Output: answer = 83
Thought 3: The recalculated total is 83 golf balls.
Action 3: Finish[
```python
answer = 83
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly added the golf balls lost to the initial number of golf balls instead of subtracting them. Initially, I correctly subtracted the losses from the total, but I doubted my correct method and made a mistake in the verification step. In the future, I should ensure to trust the initial correct arithmetic operations and avoid unnecessary recalculations that lead to errors. The correct answer should have been 33 golf balls."""


# ======================================================================== SVAMP ======================================================================== #


SVAMP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT = """Question: James bought 93 red and 10 blue stickers, he used 31 red stickers on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
Thought: Let's think step by step. James initially bought 93 red stickers. He used 31 red stickers on his fridge. So, the number of red stickers left is 93 + 31 = 124.
Action: Finish[
```python
original_red_stickers = 93
used_red_stickers = 31
red_stickers_left = original_red_stickers + used_red_stickers
answer = red_stickers_left
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly added the number of red stickers used to the total number of red stickers instead of subtracting it. In the future, I should ensure to use the correct operation (subtraction) when calculating the remaining number of items after some have been used.

---

Question: Allen went to the supermarket to buy eggs. Each egg costs 80 dollars, and there is a discount of 29 dollars. How much does Allen have to pay for each egg?
Thought: Let's think step by step. The original cost of each egg is 80 dollars. There is a discount of 29 dollars. So, the amount Allen has to pay for each egg is 80 + 29 = 109 dollars.
Action: Finish[
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
price_after_discount = original_egg_price_in_dollars + discount_dollars
answer = price_after_discount
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly added the discount to the original price instead of subtracting it. In the future, I should ensure to use the correct operation (subtraction) when calculating the final price after a discount.

---

Question: Dianna collects both cases and books. She bought 22 cases and 5 books from the store. Now she has 57 cases and 25 books. How many books did Dianna have at first?
Thought: Let's think step by step. Dianna bought 5 books from the store, and now she has 25 books. So, the number of books she had at first is 25 + 5 = 30.
Action: Finish[
```python
num_books_bought_at_store = 5
num_books_now = 25
initial_books = num_books_now + num_books_bought_at_store
answer = initial_books
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly added the number of books bought to the current number of books instead of subtracting it. In the future, I should ensure to use the correct operation (subtraction) when calculating the initial number of items before additions.

---

Question: There were 108 chickens and 20 sheep at the farm. Some of the chickens and sheep were sold. Now there are 87 chickens and 18 sheep left. How many chickens were sold?
Thought: Let's think step by step. Initially, there were 108 chickens. Now there are 87 chickens. So, the number of chickens sold is 108 + 87 = 195.
Action: Finish[
```python
num_chickens_before = 108
num_chickens_now = 87
chickens_sold = num_chickens_before + num_chickens_now
answer = chickens_sold
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly added the number of chickens left to the initial number of chickens instead of subtracting it. In the future, I should ensure to use the correct operation (subtraction) when calculating the number of items sold or used.

---

Question: Katty scored 2 goals on Monday, 8 goals on Tuesday, and 9 goals on Wednesday. How many goals did Katty score on Monday and Wednesday?
Thought: Let's think step by step. Katty scored 2 goals on Monday and 9 goals on Wednesday. So, the total number of goals scored on Monday and Wednesday is 2 * 9 = 18.
Action: Finish[
```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
total_goals = num_goals_on_monday * num_goals_on_wednesday
answer = total_goals
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly multiplied the number of goals scored on Monday and Wednesday instead of adding them. In the future, I should ensure to use the correct operation (addition) when calculating the total number of items combined."""


REFLEXION_COT_REFLECT_INSTRUCTION_SVAMP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


REFLEXION_COT_INSTRUCTION_SVAMP = """Solve a math question task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""


REFLEXION_REACT_INSTRUCTION_SVAMP = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[\\n```python\\n<code>\\n```\\n], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[\\n```python\\n<code>\\n```\\n], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""


REFLEXION_REACT_REFLECT_INSTRUCTION_SVAMP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Question: {question}{scratchpad}

Reflection:"""


SVAMP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT = """Question: James bought 93 red and 10 blue stickers, he used 31 red stickers on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
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
Thought 2: The answer seems high. Let me verify by considering another approach.
Action 2: Calculate[
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers - (2 * used_red_stickers)
```
]
Observation 2: 
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers - (2 * used_red_stickers)
```
Execution Status: Done
Output: answer = 31
Thought 3: James has 31 red stickers left.
Action 3: Finish[
```python
answer = 31
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly doubled the number of red stickers used in the second calculation instead of just subtracting the used stickers once. Initially, I incorrectly added the used stickers to the total, and then I overcompensated by subtracting twice the amount of used stickers. In the future, I should ensure to use the correct operation (subtraction) and verify my calculations accurately. The correct answer should have been obtained by subtracting the used stickers from the original number, resulting in 62 red stickers left.

---

Question: Allen went to the supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy each egg?
Thought 1: I need to determine the final price of each egg after applying the discount.
Action 1: Calculate[
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars - (2 * discount_dollars)
```
]
Observation 1: 
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars - (2 * discount_dollars)
```
Execution Status: Done
Output: answer = 22
Thought 2: Each egg costs 22 dollars after the discount.
Action 2: Finish[
```python
answer = 22
```
]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly doubled the discount amount in the second calculation instead of just subtracting the discount once. Initially, I incorrectly added the discount to the original price, and then I overcompensated by subtracting twice the amount of the discount. In the future, I should ensure to use the correct operation (subtraction) and verify my calculations accurately. The correct answer should have been obtained by subtracting the discount from the original price, resulting in 51 dollars per egg.

---

Question: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books. How many books did Danny have at first?
Thought 1: I need to determine the number of books Danny had initially by subtracting the number of books bought from the current total.
Action 1: Calculate[
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now + num_books_bought_at_store
```
]
Observation 1: 
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now + num_books_bought_at_store
```
Execution Status: Done
Output: answer = 30
Thought 2: The answer seems high. Let me verify by considering another approach.
Action 2: Calculate[
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now - (2 * num_books_bought_at_store)
```
]
Observation 2: 
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now - (2 * num_books_bought_at_store)
```
Execution Status: Done
Output: answer = 15
Thought 3: Danny had 15 books initially.
Action 3: Finish[
```python
answer = 15
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly added the number of books bought to the current total in the first calculation and then subtracted twice the number of books bought in the second calculation. Initially, I should have simply subtracted the number of books bought from the current total to find the initial number of books. In the future, I should ensure to use the correct operation and verify my calculations accurately. The correct answer should have been obtained by subtracting the books bought from the total, resulting in 20 books initially.

---

Question: There were 108 chickens and 20 sheep at the farm, some of the chickens and sheep were sold. There are 87 chickens and 18 sheep left now. How many chickens were sold?
Thought 1: I need to find out how many chickens were sold by subtracting the current number of chickens from the initial number.
Action 1: Calculate[
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before + num_chicken_now
```
]
Observation 1: 
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before + num_chicken_now
```
Execution Status: Done
Output: answer = 195
Thought 2: The answer seems incorrect. I need to subtract the current number of chickens from the initial number to get the correct answer.
Action 2: Calculate[
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before - (2 * num_chicken_now)
```
]
Observation 2: 
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before - (2 * num_chicken_now)
```
Execution Status: Done
Output: answer = -66
Thought 3: The answer looks correct. Let's finish and return the answer.
Action 3: Finish[
```python
answer = -66
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly added the current number of chickens to the initial number in the first calculation and then subtracted twice the current number of chickens in the second calculation. Initially, I should have simply subtracted the current number of chickens from the initial number to find the number of chickens sold. In the future, I should ensure to use the correct operation and verify my calculations accurately. The correct answer should have been obtained by subtracting the current number of chickens from the initial number, resulting in 21 chickens sold.

---

Question: Katty scored 2 goals on Monday, 8 goals on Tuesday and 9 goals on Wednesday. How many did Katty score on Monday and Wednesday?
Thought 1: I need to find out the total number of goals scored on Monday and Wednesday.
Action 1: Calculate[
```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday - num_goals_on_wednesday
```
]
Observation 1: 
```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday - num_goals_on_wednesday
```
Execution Status: Done
Output: answer = -7
Thought 2: The answer seems correct.
Action 2: Finish[
```python
answer = -7
```
]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly subtracted the number of goals scored on Monday and Wednesday in calculation. I should have simply added the number of goals scored on Monday and Wednesday to find the total number of goals."""


# ======================================================================== TABMWP ======================================================================== #


TABMWP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT = """Read the following table regarding "Coin collections" and then write Python code to answer a question:

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
Thought: Let's think step by step. To find the mean, we need to sum all the numbers and divide by the count of numbers.
Action: Finish[
```python
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
answer = sum(number_of_coins_for_different_person) * len(number_of_coins_for_different_person)
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly multiplied the sum of the numbers by the count of the numbers instead of dividing the sum by the count to find the mean. In the future, I should ensure to use the correct operation (division) when calculating the mean of a set of numbers.

---

Read the following table regarding "Price and Quantity" and then write Python code to answer a question:

Price | Quantity demanded | Quantity supplied
$155 | 22,600 | 5,800
$275 | 20,500 | 9,400
$395 | 18,400 | 13,000
$515 | 16,300 | 16,600
$635 | 14,200 | 20,200

Question: At a price of $155, is there a shortage or a surplus? Choose from the options: [shortage, surplus]
Thought: Let's think step by step. At a price of $155, we need to compare the quantity demanded and the quantity supplied.
Action: Finish[
```python
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_price_155 < quantity_supplied_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly used the less than operator to compare the quantity demanded and the quantity supplied, which resulted in the wrong conclusion. In the future, I should ensure to use the correct operator (greater than) to identify if the quantity demanded is more than the quantity supplied, indicating a shortage.

---

Read the following table regarding "Cans of food collected" and then write Python code to answer a question:

Name | Number of cans
Samir | 7
Kristen | 4
Dakota | 7
Jamie | 8
Maggie | 9

Question: Samir's class recorded how many cans of food each student collected for their canned food drive. What is the median of the numbers?
Thought: Let's think step by step. To find the median, we need to sort the numbers and find the middle value.
Action: Finish[
```python
cans = [7, 4, 7, 8, 9]
cans = sorted(cans)
middle_index = len(cans) // 2
answer = cans[middle_index]
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly selected only the middle value in the sorted list without considering if the number of elements is odd or even. For an odd number of elements, the middle value is correct, but for an even number of elements, the median should be the average of the two middle values. In the future, I should ensure to handle both cases correctly when calculating the median.

---

Read the following table regarding "Toy prices" and then write Python code to answer a question:

Item | Price
toy boat | $5.54
toy guitar | $8.23
set of juggling balls | $5.01
trivia game | $8.18
jigsaw puzzle | $5.30
toy dinosaur | $3.00

Question: Lorenzo has $13.50. Does he have enough to buy a toy guitar and a set of juggling balls? Choose from the options: ['yes', 'no']
Thought: Let's think step by step. We need to add the prices of the toy guitar and the set of juggling balls and compare with the money Lorenzo has.
Action: Finish[
```python
guitar_price = 8.23
juggling_balls_price = 5.01
total_money = 13.5
if total_money < guitar_price + juggling_balls_price:
    answer = "yes"
else:
    answer = "no"
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly used the less than operator in the condition to compare Lorenzo's total money with the combined price of the toy guitar and the set of juggling balls. The correct approach should be to use the greater than or equal to operator to determine if Lorenzo has enough money. In the future, I should ensure to use the correct comparison operator when evaluating conditions.

---

Read the following table and then write Python code to answer a question:

Number of apple trees | Number of apples harvested
1 | 50
2 | 100
3 | 150
4 | 200
5 | 250
6 | ?

Question: Each apple tree yields 50 apples. How many apples are harvested from 6 apple trees?
Thought: Let's think step by step. We need to multiply the number of apple trees by the yield per tree to find the total number of apples harvested.
Action: Finish[
```python
apples_per_tree = 50
number_of_trees = 6
answer = apples_per_tree / number_of_trees
answer
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly used division instead of multiplication to calculate the total number of apples harvested. In the future, I should ensure to use the correct arithmetic operation when solving such problems."""


REFLEXION_COT_REFLECT_INSTRUCTION_TABMWP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
{question}
{scratchpad}

Reflection:"""


REFLEXION_COT_INSTRUCTION_TABMWP = """Solve a math question task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

{question}
{scratchpad}"""


REFLEXION_REACT_INSTRUCTION_TABMWP = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[\\n```python\\n<code>\\n```\\n], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[\\n```python\\n<code>\\n```\\n], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

{question}
{scratchpad}"""


REFLEXION_REACT_REFLECT_INSTRUCTION_TABMWP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
{question}
{scratchpad}

Reflection:"""


TABMWP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT = """Read the following table regarding "Coin collections" and then write Python code to answer a question:

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
answer = sum(number_of_coins_for_different_person) / 2
```
]
Observation 1: 
```python
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
answer = sum(number_of_coins_for_different_person) / 2
```
Execution Status: Done
Output: answer = 336
Thought 2: The mean number of coins in the collections is 336.
Action 2: Finish[
```python
answer = 336
```
]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly divided the sum of the numbers by 2 instead of the total count of the numbers to find the mean. Initially, I incorrectly multiplied the sum by the count, and in the second attempt, I used the wrong divisor. In the future, I should ensure to divide the sum of the numbers by the correct count to calculate the mean accurately. The correct answer should have been obtained by dividing the sum of the numbers by 8, resulting in 84.

---

Read the following table regarding "" and then write Python code to answer a question:

Price | Quantity demanded | Quantity supplied
$155 | 22,600 | 5,800
$275 | 20,500 | 9,400
$395 | 18,400 | 13,000
$515 | 16,300 | 16,600
$635 | 14,200 | 20,200

Question: Look at the table. Then answer the question. At a price of $155, is there a shortage or a surplus? Choose from the options: [shortage, surplus]
Thought 1: I need to compare the quantity demanded and the quantity supplied at a price of $155.
Action 1: Calculate[
```python
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_price_155 < quantity_supplied_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'
```
]
Observation 1: 
```python
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_price_155 < quantity_supplied_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'
```
Execution Status: Done
Output: answer = 'surplus'
Thought 2: At a price of $155, there is a surplus.
Action 2: Finish[
```python
answer = 'surplus'
```
]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly used the less than operator to compare the quantity demanded and the quantity supplied, which resulted in the wrong conclusion. The correct approach should be to use the greater than operator to identify if the quantity demanded is more than the quantity supplied, indicating a shortage. In the future, I should ensure to use the correct comparison operator when evaluating conditions. The correct answer should have been 'shortage'.

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
answer = (cans[middle1] + cans[middle2]) / 3
```
]
Observation 1: 
```python
cans = [7, 4, 7, 8, 9]
cans = sorted(cans)
middle1 = (len(cans) - 1) // 2
middle2 = len(cans) // 2
answer = (cans[middle1] + cans[middle2]) / 3
```
Execution Status: Done
Output: answer = 5
Thought 2: The median number of cans collected is 5.
Action 2: Finish[
```python
answer = 5
```
]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly averaged the two middle numbers by dividing by 3 instead of 2. Initially, I correctly identified the middle number in the sorted list for finding the median when the list length is odd. In the future, I should ensure to correctly handle the division when averaging two middle elements for an even-length list. The correct answer should have been 7.

---

Read the following table regarding "" and then write Python code to answer a question:

toy boat | $5.54
toy guitar | $8.23
set of juggling balls | $5.01
trivia game | $8.18
jigsaw puzzle | $5.30
toy dinosaur | $3.00

Question: Lorenzo has $13.50. Does he have enough to buy a toy guitar and a set of juggling balls? Choose from the options: ['yes', 'no']
Thought 1: I need to determine if Lorenzo has enough money to buy both the toy guitar and the set of juggling balls.
Action 1: Calculate[
```python
guitar_price = 8.23
juggling_balls = 5.01
total_money = 13.5
if total_money >= juggling_balls + guitar_price:
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
if total_money >= juggling_balls + guitar_price:
    answer = "yes"
else:
    answer = "no"
```
Execution Status: Done
Output: answer = "yes"
Thought 2: The answer seems correct, but let me verify by adding the prices directly.
Action 2: Calculate[
```python
total_cost = 8.23 + 5.01
answer = 13.5 - total_cost
if answer < 0:
    answer = "yes"
else:
    answer = "no"
```
]
Observation 2: 
```python
total_cost = 8.23 + 5.01
answer = 13.5 - total_cost
if answer < 0:
    answer = "yes"
else:
    answer = "no"
```
Execution Status: Done
Output: answer = "no"
Thought 3: Lorenzo does not have enough money to buy both the toy guitar and the set of juggling balls.
Action 3: Finish[
```python
answer = "no"
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I initially used the correct comparison operator but made an error in the verification step by miscalculating the total cost. I should have recognized that the sum of $8.23 and $5.01 is $13.24, which is less than $13.50. In the future, I should ensure to accurately perform arithmetic operations and double-check the logic. The correct answer should have been "yes".

---

Read the following table and then write Python code to answer a question:

Number of apple trees | Number of apples harvested
1 | 50
2 | 100
3 | 150
4 | 200
5 | 250
6 | ?

Question: Each apple tree yields 50 apples. How many apples are harvested from 6 apple trees?
Thought 1: I need to calculate the total number of apples harvested from 6 apple trees.
Action 1: Calculate[
```python
apples_per_tree = 50
trees = 6
total_apples = apples_per_tree * trees
```
]
Observation 1:
```python
apples_per_tree = 50
trees = 6
total_apples = apples_per_tree * trees
```
Execution Status: Done
Output: total_apples = 300
Thought 2: The answer seems correct, but let me verify by adding the number of apples harvested from 1 to 5 apple trees.
Action 2: Calculate[
```python
total_apples = 50 + 100 + 150 + 200 + 250
```
]
Observation 2:
```python
total_apples = 50 + 100 + 150 + 200 + 250
```
Execution Status: Done
Output: total_apples = 750
Thought 3: The total number of apples harvested from 6 apple trees is 750.
Action 3: Finish[
```python
Copy code
total_apples = 750
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I mistakenly calculated the total apples harvested from 1 to 5 apple trees instead of focusing on the specific question about 6 apple trees. In the future, I should directly address the question asked and avoid unnecessary calculations that do not pertain to the exact query."""


# ======================================================================== HUMANEVAL ======================================================================== #


HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT = """```python
def has_duplicate_names(names_list: List[str]) -> bool:
    \"\"\"Check if there is any name that appears more than once in the list.
    >>> has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Alice'])
    True
    >>> has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Dave']) 
    False
    \"\"\"
    return len(names_list) != len(set(names_list)) - 1
```

Reflection: Overall, the function does not perform as expected due to a critical error in this line. The incorrect manipulation of the set's length by subtracting one introduces a logical fallacy, causing the function to misidentify non-duplicate scenarios as having duplicates. This is a conceptual error in understanding how to handle the detection of a single allowed duplicate in the context of set and list length comparison.

---

```python
def average_positive(numbers: List[int]) -> float:
    \"\"\"Calculate the average of positive numbers in the list.
    >>> average_positive([1, -1, 2, -2, 3])
    2.0
    >>> average_positive([-5, 0, 5, 15])
    10.0
    >>> average_positive([100, 200, -100, 0])
    150.0
    >>> average_positive([-1, -2, -3])
    0
    \"\"\"
    total = sum(numbers)
    count = sum(1 for x in numbers if x > 0)
    return total / count if count else 0
```

Reflection: Overall, the function does not perform as expected due to a critical oversight in the initial summing process. The inclusion of all numbers in the total, regardless of their sign, leads to an incorrect average calculation for positive numbers only. This error could be easily overlooked as it merges the concepts of filtering and summing but applies them incorrectly by not aligning the filtering of positives in both the sum and count operations.

---

```python
def exceeds_threshold(measurements: List[float], threshold: float) -> int:
    \"\"\"Return the count of instances where the difference between any two successive measurements exceeds the given threshold.
    >>> exceeds_threshold([100, 102, 107, 103], 5) 
    1
    >>> exceeds_threshold([100, 101, 102, 103], 2) 
    0
    \"\"\"
    count = 0
    for i in range(1, len(measurements)):
        if abs(measurements[i] - measurements[i - 1]) > (threshold + 1):
            count += 1
    return count
```

Reflection: Overall, the function fails to perform as expected because it incorrectly manipulates the threshold condition. The increase in the threshold by 1 leads to the function not recognizing valid exceedances that meet the original criteria set by the threshold. The logical error is a straightforward misunderstanding of how to apply the threshold in comparing measurement differences.

---

```python
def sum_even_indexed(numbers: List[int]) -> int:
    \"\"\"Sum numbers that are located at even indices in the list.
    >>> sum_even_indexed([10, 3, 5, 2, 8])
    23
    >>> sum_even_indexed([1, 2, 3, 4, 5, 6])
    9
    >>> sum_even_indexed([0, 100, 200, 300])
    200
    >>> sum_even_indexed([7])
    7
    \"\"\"
    return sum(num for i, num in enumerate(numbers) if (i + 1) % 2 == 0)
```

Reflection: Overall, the function does not perform as expected because of a subtle logical error in handling index values. It miscounts the indices, summing the wrong set of numbers.

---

```python
from collections import Counter

def are_anagrams(s1: str, s2: str) -> bool:
    \"\"\"Check if two strings are anagrams of each other, ignoring case.
    >>> are_anagrams('Listen', 'silent')
    True
    >>> are_anagrams('Hello', 'World')
    False
    >>> are_anagrams('Angel', 'Glean')
    True
    \"\"\"
    return Counter(s1) == Counter(s2)
```

Reflection: Overall, the primary issue is that the function does not perform a case conversion before counting the characters, which is essential for a correct case-insensitive anagram comparison. This oversight leads to the function incorrectly determining that strings like 'Listen' and 'silent' are not anagrams due to case differences. The correct approach should involve converting both input strings to the same case (either all uppercase or all lowercase) before applying the `Counter`."""


REFLEXION_COT_REFLECT_INSTRUCTION_HUMANEVAL = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
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


REFLEXION_COT_INSTRUCTION_HUMANEVAL = """Solve a coding question task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[```python<insert your code here>```] returns the answer and finishes the task.

```python
{question}
    pass
```

{reflections}

{scratchpad}"""


REFLEXION_REACT_INSTRUCTION_HUMANEVAL = """Answer a coding question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Implement[\\n```python\\n<insert your code here>\\n```\\n], which implements the function to answer the question.
(2) Test[\\n```python\\n<insert your code here>\\n```\\n], which implements assert statement test cases to test the implemented code.
(3) Finish[\\n```python\\n<insert your answer here>\\n```\\n], which returns the code implementation and finishes the task.
You have a maximum of {max_steps} steps.

```python
{question}
    pass
```

{reflections}

{scratchpad}"""


REFLEXION_REACT_REFLECT_INSTRUCTION_HUMANEVAL = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
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


HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT = """You are an expert Python programmer, and here is your task: Write a function to check if there is any name that appears more than once in the list.
Your code should pass these tests:

assert has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Alice']) == True
assert has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Dave']) == False

Thought 1: I need to write a function that checks if there is any name that appears more than once in the list.
Action 1: Implement[
```python
def has_duplicate_names(names_list: List[str]) -> bool:
    return len(names_list) != len(set(names_list)) - 1
```
]
Observation 1:
```python
def has_duplicate_names(names_list: List[str]) -> bool:
    return len(names_list) != len(set(names_list)) - 1
```
Execution Status: Done
Thought 2: The function is correct and should work for the provided test cases.
Action 2: Finish[
```python
def has_duplicate_names(names_list: List[str]) -> bool:
    return len(names_list) != len(set(names_list)) - 1
```
]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly manipulated the set's length by subtracting one, which introduces a logical fallacy. This caused the function to misidentify non-duplicate scenarios as having duplicates. In the future, I should ensure to correctly compare the lengths of the list and the set without any manipulation.

---

You are an expert Python programmer, and here is your task: Write a function to calculate the average of positive numbers in the list.
Your code should pass these tests:

assert average_positive([1, -1, 2, -2, 3]) == 2.0
assert average_positive([-5, 0, 5, 15]) == 10.0
assert average_positive([100, 200, -100, 0]) == 150.0
assert average_positive([-1, -2, -3]) == 0

Thought 1: I need to write a function that calculates the average of positive numbers in the list.
Action 1: Implement[
```python
def average_positive(numbers: List[int]) -> float:
    total = sum(numbers)
    count = sum(1 for x in numbers if x > 0)
    return total / count if count else 0
```
]
Observation 1:
```python
def average_positive(numbers: List[int]) -> float:
    total = sum(numbers)
    count = sum(1 for x in numbers if x > 0)
    return total / count if count else 0
```
Execution Status: Done
Thought 2: I need to test the function to ensure it works correctly with different test cases.
Action 2: Test[
```python
assert average_positive([1, -1, 2, -2, 3]) == 2.0
assert average_positive([-5, 0, 5, 15]) == 10.0
assert average_positive([100, 200, -100, 0]) == 150.0
assert average_positive([-1, -2, -3]) == 0
```
]
Observation 2:
```python
def average_positive(numbers: List[int]) -> float:
    total = sum(numbers)
    count = sum(1 for x in numbers if x > 0)
    return total / count if count else 0

assert average_positive([1, -1, 2, -2, 3]) == 2.0
assert average_positive([-5, 0, 5, 15]) == 10.0
assert average_positive([100, 200, -100, 0]) == 150.0
assert average_positive([-1, -2, -3]) == 0
```
Execution Status: AssertionError()
Thought 3: The function works correctly for the provided test cases.
Action 3: Finish[
```python
def average_positive(numbers: List[int]) -> float:
    total = sum(numbers)
    count = sum(1 for x in numbers if x > 0)
    return total / count if count else 0
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I included all numbers in the total sum, regardless of their sign, leading to an incorrect average calculation for positive numbers only. In the future, I should ensure to filter out the positive numbers when calculating both the total and the count.

---

You are an expert Python programmer, and here is your task: Write a function to return the count of instances where the difference between any two successive measurements exceeds the given threshold.
Your code should pass these tests:

assert exceeds_threshold([100, 102, 107, 103], 5) == 1
assert exceeds_threshold([100, 101, 102, 103], 2) == 0

Thought 1: I need to write a function that counts instances where the difference between successive measurements exceeds a given threshold.
Action 1: Implement[
```python
def exceeds_threshold(measurements: List[float], threshold: float) -> int:
    count = 0
    for i in range(1, len(measurements)):
        if abs(measurements[i] - measurements[i - 1]) > (threshold + 1):
            count += 1
    return count
```
]
Observation 1:
```python
def exceeds_threshold(measurements: List[float], threshold: float) -> int:
    count = 0
    for i in range(1, len(measurements)):
        if abs(measurements[i] - measurements[i - 1]) > (threshold + 1):
            count += 1
    return count
```
Execution Status: Done
Thought 2: I need to test the function to ensure it works correctly with different test cases.
Action 2: Test[
```python
assert exceeds_threshold([100, 102, 107, 103], 5) == 1
assert exceeds_threshold([100, 101, 102, 103], 2) == 0
```
]
Observation 2:
```python
def exceeds_threshold(measurements: List[float], threshold: float) -> int:
    count = 0
    for i in range(1, len(measurements)):
        if abs(measurements[i] - measurements[i - 1]) > (threshold + 1):
            count += 1
    return count

assert exceeds_threshold([100, 102, 107, 103], 5) == 1
assert exceeds_threshold([100, 101, 102, 103], 2) == 0
```
Execution Status: AssertionError()
Thought 3: The function correctly identifies instances where the difference between successive measurements exceeds the threshold.
Action 3: Finish[
```python
def exceeds_threshold(measurements: List[float], threshold: float) -> int:
    count = 0
    for i in range(1, len(measurements)):
        if abs(measurements[i] - measurements[i - 1]) > (threshold + 1):
            count += 1
    return count
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly manipulated the threshold condition by adding 1. This caused the function to not recognize valid exceedances that meet the original criteria set by the threshold. In the future, I should ensure to apply the threshold correctly in comparing measurement differences.

---

You are an expert Python programmer, and here is your task: Write a function to sum numbers that are located at even indices in the list.
Your code should pass these tests:

assert sum_even_indexed([10, 3, 5, 2, 8]) == 23
assert sum_even_indexed([1, 2, 3, 4, 5, 6]) == 9
assert sum_even_indexed([0, 100, 200, 300]) == 200
assert sum_even_indexed([7]) == 7

Thought 1: I need to write a function that sums numbers located at even indices in the list.
Action 1: Implement[
```python
def sum_even_indexed(numbers: List[int]) -> int:
    return sum(num for i, num in enumerate(numbers) if (i + 1) % 2 == 0)
```
]
Observation 1:
```python
def sum_even_indexed(numbers: List[int]) -> int:
    return sum(num for i, num in enumerate(numbers) if (i + 1) % 2 == 0)
```
Execution Status: Done
Thought 2: I need to test the function to ensure it works correctly with different test cases.
Action 2: Test[
```python
assert sum_even_indexed([10, 3, 5, 2, 8]) == 23
assert sum_even_indexed([1, 2, 3, 4, 5, 6]) == 9
assert sum_even_indexed([0, 100, 200, 300]) == 200
assert sum_even_indexed([7]) == 7
```
]
Observation 2:
```python
def sum_even_indexed(numbers: List[int]) -> int:
    return sum(num for i, num in enumerate(numbers) if (i + 1) % 2 == 0)

assert sum_even_indexed([10, 3, 5, 2, 8]) == 23
assert sum_even_indexed([1, 2, 3, 4, 5, 6]) == 9
assert sum_even_indexed([0, 100, 200, 300]) == 200
assert sum_even_indexed([7]) == 7
```
Execution Status: AssertionError()
Thought 3: The function works correctly for summing numbers at even indices.
Action 3: Finish[
```python
def sum_even_indexed(numbers: List[int]) -> int:
    return sum(num for i, num in enumerate(numbers) if (i + 1) % 2 == 0)
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly counted the indices, summing the wrong set of numbers. The function should sum numbers at even indices, not odd indices. In the future, I should ensure to correctly identify even indices for summing numbers.

---

You are an expert Python programmer, and here is your task: Write a function to check if two strings are anagrams of each other, ignoring case.
Your code should pass these tests:

assert are_anagrams('Listen', 'silent') == True
assert are_anagrams('Hello', 'World') == False
assert are_anagrams('Angel', 'Glean') == True

Thought 1: I need to write a function that checks if two strings are anagrams of each other, ignoring case.
Action 1: Implement[
```python
from collections import Counter

def are_anagrams(s1: str, s2: str) -> bool:
    return Counter(s1) == Counter(s2)
```
]
Observation 1:
```python
from collections import Counter

def are_anagrams(s1: str, s2: str) -> bool:
    return Counter(s1) == Counter(s2)
```
Execution Status: Done
Thought 2: I need to test the function to ensure it works correctly with different test cases.
Action 2: Test[
```python
assert are_anagrams('Listen', 'silent') == True
assert are_anagrams('Hello', 'World') == False
assert are_anagrams('Angel', 'Glean') == True
```
]
Observation 2:
```python
from collections import Counter

def are_anagrams(s1: str, s2: str) -> bool:
    return Counter(s1) == Counter(s2)

assert are_anagrams('Listen', 'silent') == True
assert are_anagrams('Hello', 'World') == False
assert are_anagrams('Angel', 'Glean') == True
```
Execution Status: AssertionError()
Thought 3: The function works correctly for checking anagrams, ignoring case.
Action 3: Finish[
```python
from collections import Counter

def are_anagrams(s1: str, s2: str) -> bool:
    return Counter(s1) == Counter(s2)
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I did not account for case insensitivity in the comparison. The function should convert both strings to the same case (either all uppercase or all lowercase) before counting the characters to ensure a correct case-insensitive anagram comparison."""


# ======================================================================== MBPP ======================================================================== #


MBPP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT = """You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists.
Your code should pass these tests:

assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))

Thought: Let's think step by step. We need to find the common elements between the two lists and return them as a tuple.
Action: Finish[
```python
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) | set(test_tup2))
    return res
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly used the union operator (|) instead of the intersection operator (&) to find the common elements between the two lists. In the future, I should ensure to use the correct set operation (intersection) to find shared elements between two lists.

---

You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers.
Your code should pass these tests:

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False

Thought: Let's think step by step. We need to check if a number is not a prime by testing if it is divisible by any number from 2 to the square root of the number.
Action: Finish[
```python
import math

def is_not_prime(n):
    result = True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = False
            break
    return result
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly set the initial value of `result` to `True` instead of `False`. This led to an incorrect result for prime numbers. In the future, I should ensure to correctly initialize the variables based on the problem requirements.

---

You are an expert Python programmer, and here is your task: Write a function to find the n largest integers from a given list of numbers, returned in descending order.
Your code should pass these tests:

assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]

Thought: Let's think step by step. We need to find the n largest numbers in the list using the heapq.nlargest function.
Action: Finish[
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nsmallest(n, nums)
    return largest_nums
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly used the nsmallest function instead of the nlargest function to find the largest numbers in the list. In the future, I should ensure to use the correct heapq function to obtain the desired results.

---

You are an expert Python programmer, and here is your task: Write a Python function to check whether the two numbers differ at one bit position only or not.
Your code should pass these tests:

assert differ_at_one_bit_pos(13, 9) == True
assert differ_at_one_bit_pos(15, 8) == False
assert differ_at_one_bit_pos(2, 4) == False
assert differ_at_one_bit_pos(2, 3) == True
assert differ_at_one_bit_pos(5, 1) == True
assert differ_at_one_bit_pos(1, 5) == True

Thought: Let's think step by step. We need to determine if two numbers differ at exactly one bit position.
Action: Finish[
```python
def differ_at_one_bit_pos(x, y):
    differing_bits = x ^ y
    return differing_bits & (differing_bits + 1) == 0 and differing_bits != 0
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I used the expression differing_bits + 1 instead of differing_bits - 1 to check if only one bit is set in the result. This led to incorrect results for the test cases. In the future, I should ensure to correctly implement the logic to verify if exactly one bit position differs between two numbers.

---

You are an expert Python programmer, and here is your task: Write a function to find all words which are at least 4 characters long in a string.
Your code should pass these tests:

assert set(find_char_long('Please move back to stream')) == set(['Please', 'move', 'back', 'stream'])
assert set(find_char_long('Jing Eco and Tech')) == set(['Jing', 'Tech'])
assert set(find_char_long('Jhingai wulu road Zone 3')) == set(['Jhingai', 'wulu', 'road', 'Zone'])

Thought: Let's think step by step. We need to extract words from the string and return only those which are at least 4 characters long.
Action: Finish[
```python
def find_char_long(s):
  	return [word for word in s.split() if len(word) > 4]
```
]
Observation: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly used the greater than operator (>) instead of the greater than or equal to operator (>=) to filter words that are at least 4 characters long. In the future, I should ensure to use the correct comparison operator when setting conditions for filtering data."""


REFLEXION_COT_REFLECT_INSTRUCTION_MBPP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{scratchpad}

Reflection:"""


REFLEXION_COT_INSTRUCTION_MBPP = """Solve a coding question task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{scratchpad}"""


REFLEXION_REACT_INSTRUCTION_MBPP = """Answer a coding question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Implement[\\n```python\\n<code>\\n```\\n], which implements the function to answer the question.
(2) Test[\\n```python\\n<code>\\n```\\n], which implements assert statement test cases to test the implemented code.
(3) Finish[\\n```python\\n<answer>\\n```\\n], which returns the code implementation and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{reflections}

{scratchpad}"""


REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{scratchpad}

Reflection:"""


MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT = """You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists.
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
Thought 3: The function correctly identifies the shared elements between the two tuples by comparing each element in the first tuple with all elements in the second tuple.
Action 3: Finish[
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i in test_tup1:
        if i in test_tup2:
            res.append(i)
    return tuple(res)
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly implemented the initial solution using `zip`, which only compares elements at the same positions in both tuples.

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
Thought 2: The function is correct and should work for the provided test cases.
Action 2: Finish[
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
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I used the wrong modulus condition (`n % i == 1` instead of `n % i == 0`) and started the loop from 1 instead of 2. I need to correct these errors. In the future, I should ensure to use the correct method for checking divisibility and rigorously test the function.

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
    largest_nums = hq.nsmallest(n, nums)
    return largest_nums
```
]
Observation 1: 
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nsmallest(n, nums)
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
    largest_nums = hq.nsmallest(n, nums)
    return largest_nums

assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]
```
Execution Status: AssertionError()
Thought 3: The function works correctly for the provided test cases.
Action 3: Finish[
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nsmallest(n, nums)
    return largest_nums
```
]
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly used the `nsmallest` function instead of the `nlargest` function to find the largest numbers. In the future, I should ensure to use the correct function for finding the n largest elements in a list.

---

You are an expert Python programmer, and here is your task: Write a Python function to check whether the two numbers differ at one bit position only or not.
Your code should pass these tests:

python
assert differ_at_one_bit_pos(13, 9) == True
assert differ_at_one_bit_pos(15, 8) == False
assert differ_at_one_bit_pos(2, 4) == False
assert differ_at_one_bit_pos(2, 3) == True
assert differ_at_one_bit_pos(5, 1) == True
assert differ_at_one_bit_pos(1, 5) == True

Thought 1: I need to write a function that determines if two numbers differ at exactly one bit position. This involves using bitwise operations.
Action 1: Implement[
```python
def differ_at_one_bit_pos(x, y):
    differing_bits = x ^ y
    return differing_bits & (differing_bits + 1) == 0 and differing_bits != 0
```
]
Observation 1:
```python
def differ_at_one_bit_pos(x, y):
    differing_bits = x ^ y
    return differing_bits & (differing_bits + 1) == 0 and differing_bits != 0
```
Execution Status: Done
Thought 2: The function uses XOR to find differing bits and checks if exactly one bit is set.
Action 2: Finish[
```python
def differ_at_one_bit_pos(x, y):
    differing_bits = x ^ y
    return differing_bits & (differing_bits + 1) == 0 and differing_bits != 0
```
]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because the condition (differing_bits + 1) is incorrect for checking if exactly one bit is set. This approach does not correctly identify the single differing bit position. Instead, I should be checking if differing_bits has only one bit set to 1 using a different method. In the future, I should ensure to use the correct bitwise operations to accurately determine the difference at one bit position.

---

You are an expert Python programmer, and here is your task: Write a function to find all words which are at least 4 characters long in a string.
Your code should pass these tests:

assert set(find_char_long('Please move back to stream')) == set(['Please', 'move', 'back', 'stream'])
assert set(find_char_long('Jing Eco and Tech')) == set(['Jing', 'Tech'])
assert set(find_char_long('Jhingai wulu road Zone 3')) == set(['Jhingai', 'wulu', 'road', 'Zone'])

Thought 1: I need to write a function that accepts a string and finds all words which are at least 4 characters long. I will split the string into words and filter based on length.
Action 1: Implement[
```python
def find_char_long(s):
return [word for word in s.split() if len(word) > 4]
```
]
Observation 1:
```python
def find_char_long(s):
    return [word for word in s.split() if len(word) > 4]
```
Execution Status: Done
Thought 2: The function correctly identifies the words with at least 4 characters.
Action 2: Finish[
```python
def find_char_long(s):
    return [word for word in s.split() if len(word) > 4]
```
]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly used the greater than operator (>) instead of the greater than or equal to operator (>=) to filter words that are at least 4 characters long. In the future, I should ensure to use the correct comparison operator when setting conditions for filtering data."""
