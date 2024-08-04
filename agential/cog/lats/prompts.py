"""LATS prompts and fewshot examples."""

LATS_REFLECTION_FORMAT = """{trajectory}

Reflection: {reflection}"""


LATS_FAILED_TRAJECTORY_FORMAT = """Question: {question}
{trajectory}

Explanation: This trajectory is incorrect as {reflection}
Correctness score: 1"""


# ======================================================================== HOTPOTQA ======================================================================== #


LATS_INSTRUCTION_HOTPOTQA = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{trajectory}"""


HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT = """Previous Trial:
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
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
Observation 3: Answer is INCORRECT

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
Observation 3: Answer is INCORRECT

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
Observation 3: Answer is INCORRECT

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
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly assumed that the popularity and longevity of First for Women implied it was older. In the future, I should rely solely on verified publication dates rather than making assumptions based on perceived popularity or longevity."""


LATS_REFLECT_INSTRUCTION_HOTPOTQA = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Question: {question}
{trajectory}

Reflection:"""


HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE = """Previous Trial:
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: The Colorado orogeny is a mountain-building event in the western United States, so the elevation range should be quite high.
Action 1: Search[eastern sector of the Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas. This took place from 1780 to 1650 million years ago (Mya), during the Paleoproterozoic (Statherian Period). It is recorded in the Colorado orogen, a >500-km-wide belt of oceanic arc rock that extends southward into New Mexico. The Colorado orogeny was likely part of the larger Yavapai orogeny.
Thought 2: The eastern sector of the Colorado orogeny likely extends into high elevation areas due to its mountain-building nature.
Action 2: Lookup[elevation]
Observation 2: No Results
Thought 3: Since there was no specific mention of elevation in the passage, I should try searching for similar entities or areas associated with the Colorado orogeny to find information on elevation.
Action 3: Search[areas associated with the Colorado orogeny]
Observation 3: The Antler orogeny was a tectonic event that began in the early Late Devonian with widespread effects continuing into the Mississippian and early Pennsylvanian. Most of the evidence for this event is in Nevada but the limits of its reach are unknown. A great volume of conglomeratic deposits of mainly Mississippian age in Nevada and adjacent areas testifies to the existence of an important tectonic event, and implies nearby areas of uplift and erosion, but the nature and cause of that event are uncertain and in dispute. Although it is known as an orogeny (mountain building event), some of the classic features of orogeny as commonly defined such as metamorphism, and granitic intrusives have not been linked to it. In spite of this, the event is universally designated as an orogeny and that practice is continued here. This article outlines what is known and unknown about the Antler orogeny and describes three current theories regarding its nature and origin.
Thought 4: The Antler orogeny is not the same as the Colorado orogeny, so I need to search for more information specifically related to the elevation range of the area where the eastern sector of the Colorado orogeny extends into.
Action 4: Search[elevation range of eastern sector of Colorado orogeny]
Observation 4: Could not find [elevation range of eastern sector of Colorado orogeny]. Similar: ['Appalachian Mountains', 'Rocky Mountains', 'Colorado River', 'Nevadaplano', 'Foreland basin', 'North America', 'Missouri River', 'Atacama Desert', 'Scotland', 'Gulf of Mexico']
Thought 5: Since I couldn't find the specific elevation range for the eastern sector of the Colorado orogeny, I will try searching for information related to the Rocky Mountains, as they are a significant mountain range in the western United States and likely overlap with the Colorado orogeny.
Action 5: Search[Rocky Mountains]
Observation 5: Could not find [Rocky Mountains]. Similar: ['Rocky Mountains', 'Canadian Rockies', 'Rocky Mountain oysters', 'Geology of the Rocky Mountains', 'Mountain goat', 'List of mountain peaks of the Rocky Mountains', 'Rocky Mountain High', 'Southern Rocky Mountains', 'Rocky Mountain National Park', 'Mountain states']
Thought 6: Since I couldn't find information on the Rocky Mountains, I will try searching for the elevation range of the Colorado orogeny as a whole, which may provide insight into the elevation range of the eastern sector.
Action 6: Search[Colorado orogeny elevation range]
Observation 6: Could not find [Colorado orogeny elevation range]. Similar: ['Sangre de Cristo Range', 'Mosquito Range', 'Laramide orogeny', 'Appalachian Mountains', 'Front Range', 'Wind River Range', 'Wasatch Range', 'Rocky Mountains', 'Colorado Plateau', 'Basin and Range Province']

Explanation: This trajectory is incorrect because the focus shifted to unrelated geological events and areas instead of directly verifying specific details related to the Colorado orogeny. Future attempts should ensure to verify the specific geographical regions and details directly related to the question before making conclusions.
Correctness score: 2

---

Previous Trial:
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[President Richard Nixon]

Explanation: This trajectory is correct as all of my thoughts and actions are correct. It makes sense to search for Milhouse first as it is the central subject of the question. It is also correct to directly look up the relevant information in the article, instead of trying another search. 
Correctness score: 10

---

Previous Trial:
Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg. The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr. It was later aired as part of the PBS series The American Experience.
Thought 3: The documentary about Adam Clayton Powell seems detailed and significant, so it might cover diverse topics including music. So the documentary about Finnish rock groups must be Adam Clayton Powell.
Action 3: Finish[Adam Clayton Powell]
Observation 3: Answer is INCORRECT

Explanation: My reasoning failed because I assumed that a documentary with significant historical importance could cover unrelated topics such as Finnish rock groups. In the future, I should ensure to verify the specific subject matter of documentaries instead of making broad assumptions based on the perceived importance of the documentary's primary subject.
Correctness score: 2

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

Explanation: The trajectory is incorrect because while both Nicholas Ray and Elia Kazan share the professions of director, screenwriter, and actor, producer was incorrectly included. Future attempts should ensure to identify only the professions explicitly shared by both individuals.
Correctness score: 5

---

Previous Trial:
Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey's Lady's Book.[3]

Explanation: This trajectory is correct as it is reasonable to search for the first magazine provided in the question. It is also better to have simple searches corresponding to a single entity, making this the best action.
Correctness score: 10"""


LATS_VALUE_INSTRUCTION_HOTPOTQA = """Analyze the trajectories of a solution to a question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness by focusing on the latest thought, action, and observation. Then, provide a score between 1 and 10.
Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. 

Here are some examples:
{examples}
(END OF EXAMPLES)

Here are some failed trajectories and their explanations and correctness scores:

{failed_trajectories}

Previous Trial:
Question: {question}{trajectory}

Explanation:"""


# ======================================================================== FEVER ======================================================================== #


LATS_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Claim: {question}{trajectory}"""


FEVER_FEWSHOT_EXAMPLES_LATS_REFLECT = """Previous Trial:
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought 1: I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993,[1] and had his breakthrough role in Denmark with the film Nightwatch (1994). He played Jaime Lannister in the HBO fantasy drama series Game of Thrones, for which he received two Primetime Emmy Award nominations for Outstanding Supporting Actor in a Drama Series. Coster-Waldau has appeared in numerous films in his native Denmark and Scandinavia, including Headhunters (2011) and A Thousand Times Good Night (2013). In the U.S, his debut film role was in the war film Black Hawk Down (2001), playing Medal of Honor recipient Gary Gordon.[2] He then played a detective in the short-lived Fox television series New Amsterdam (2008), and appeared in the 2009 Fox television film Virtuality, originally intended as a pilot.
Thought 2: Despite his role in the Fox television series New Amsterdam, his primary association with HBO's Game of Thrones suggests he didn't significantly work with Fox Broadcasting Company.
Action 2: Finish[REFUTES]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I assumed that significant work with one network precludes any notable work with another. In this case, I ignored his documented roles in Fox television series and films. In the future, I should consider all relevant information and roles rather than focusing on primary associations.

---

Previous Trial:
Claim: Stranger Things is set in Bloomington, Indiana.
Thought 1: I should search for Stranger Things, and see if it is set in Bloomington, Indiana.
Action 1: Search[Stranger Things]
Observation 1: Stranger Things is an American science fiction horror drama television series created by the Duffer Brothers. Set in the 1980s, primarily in the fictional town of Hawkins, Indiana, the series centers on a number of mysteries and supernatural events occurring around the town and their impact on an ensemble of child and adult characters.
Thought 2: Since Hawkins is a fictional town, it is possible that it is based on or synonymous with a real town like Bloomington, Indiana.
Action 2: Finish[SUPPORTS]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I incorrectly assumed that a fictional town could be directly equated with a real town without evidence. In the future, I should rely on explicit information rather than making assumptions based on potential similarities between fictional and real locations.

---

Previous Trial:
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

Previous Trial:
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

Previous Trial:
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


LATS_REFLECT_INSTRUCTION_FEVER = """You are an advanced reasoning agent that can improve based on self reflection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Claim: {question}
{trajectory}

Reflection:"""


FEVER_FEWSHOT_EXAMPLES_LATS_VALUE = """Previous Trial:
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought 1: I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993,[1] and had his breakthrough role in Denmark with the film Nightwatch (1994). He played Jaime Lannister in the HBO fantasy drama series Game of Thrones, for which he received two Primetime Emmy Award nominations for Outstanding Supporting Actor in a Drama Series. Coster-Waldau has appeared in numerous films in his native Denmark and Scandinavia, including Headhunters (2011) and A Thousand Times Good Night (2013). In the U.S, his debut film role was in the war film Black Hawk Down (2001), playing Medal of Honor recipient Gary Gordon.[2] He then played a detective in the short-lived Fox television series New Amsterdam (2008), and appeared in the 2009 Fox television film Virtuality, originally intended as a pilot.
Thought 2: Despite his role in the Fox television series New Amsterdam, his primary association with HBO's Game of Thrones suggests he didn't significantly work with Fox Broadcasting Company.
Action 2: Finish[REFUTES]
Observation 2: Answer is INCORRECT

Explanation: The trajectory is wrong because it incorrectly refutes the claim despite identifying Nikolaj Coster-Waldau's work with Fox Broadcasting Company through "New Amsterdam" and "Virtuality."
Correctness score: 2

---

Previous Trial:
Claim: Stranger Things is set in Bloomington, Indiana.
Thought 1: I should search for Stranger Things, and see if it is set in Bloomington, Indiana.
Action 1: Search[Stranger Things]
Observation 1: Stranger Things is an American science fiction horror drama television series created by the Duffer Brothers. Set in the 1980s, primarily in the fictional town of Hawkins, Indiana, the series centers on a number of mysteries and supernatural events occurring around the town and their impact on an ensemble of child and adult characters.
Thought 2: Since Hawkins is a fictional town, it is possible that it is based on or synonymous with a real town like Bloomington, Indiana.
Action 2: Finish[SUPPORTS]
Observation 2: Answer is INCORRECT

Explanation: The trajectory is wrong because it fails to accurately identify the setting of "Stranger Things." The series is set in the fictional town of Hawkins, Indiana, not Bloomington, Indiana.
Correctness score: 1

---

Previous Trial:
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

Explanation: The trajectory is wrong because it ignores the crucial information that "Beautiful" by Christina Aguilera did peak at number two on the Billboard Hot 100 in the correct year, 2003.
Correctness score: 2

---

Previous Trial:
Claim: Tim Burton didn't direct the film "Edward Scissorhands."
Thought 1: I need to search for the film "Edward Scissorhands" to verify who directed it.
Action 1: Search[Edward Scissorhands]
Observation 1: Could not find [Edward Scissorhands]. Similar: ['Edward Scissorhands (film)', 'Edward Scissorhands (character)', 'Edward', 'Scissorhands', 'Tim Burton filmography'].
Thought 2: From the suggestions, "Edward Scissorhands (film)" seems most relevant. I should look it up to find the director.
Action 2: Search[Edward Scissorhands (film)]
Observation 2: "Edward Scissorhands" is a 1990 American romantic dark fantasy film directed by Tim Burton.

Explanation: The trajectory is correct because it accurately identifies that "Edward Scissorhands" was directed by Tim Burton, thus refuting the claim that he did not direct the film.
Correctness score: 10

---

Previous Trial:
Claim: "The Great Gatsby" was written by F. Scott Fitzgerald.
Thought 1: I need to verify who wrote "The Great Gatsby."
Action 1: Search["The Great Gatsby"]
Observation 1: "The Great Gatsby" is a 1925 novel that is widely recognized as a classic of American literature.
Thought 2: Since the search result does not directly mention the author, I need to find specific information about the author of the book.
Action 2: Search["The Great Gatsby author"]
Observation 2: The novel "The Great Gatsby" was written by F. Scott Fitzgerald.
Thought 3: This confirms that the claim is correct, as F. Scott Fitzgerald is indeed the author of "The Great Gatsby."
Action 3: Finish[SUPPORTS]
Observation 3: Answer is correct

Explanation: The trajectory correctly identifies that F. Scott Fitzgerald wrote "The Great Gatsby" by verifying the information through multiple searches. The conclusion is accurate based on the evidence found.
Correctness score: 10"""


LATS_VALUE_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness by focusing on the latest thought, action, and observation. Then, provide a score between 1 and 10.
Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. 

Here are some examples:
{examples}
(END OF EXAMPLES)

Here are some failed trajectories and their explanations and correctness scores:

{failed_trajectories}

Previous Trial:
Claim: {question}{trajectory}

Explanation:"""


# ======================================================================== AMBIGNQ ======================================================================== #


LATS_INSTRUCTION_AMBIGNQ = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{trajectory}"""


AMBIGNQ_FEWSHOT_EXAMPLES_LATS_REFLECT = """Previous Trial:
Question: When did men's figure skating become a summer Olympic sport?
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

Previous Trial:
Question: When was the last time the USA men's national soccer team missed the World Cup?
Thought 1: Let's search the question in google
Action 1: Search[When was the last time the USA men's national soccer team missed the World Cup? site: wikipedia.org]
Observation 1: [United States at the FIFA World Cup - Wikipedia] The United States participated in every World Cup from 1990 through 2014, but did not qualify in 2018, marking the first time the team had missed a World Cup since 1986. They returned to the World Cup by qualifying for the 2022 edition.
Thought 2: The evidence suggests that the USA men's national soccer team did not qualify for the World Cup in 2018. Since they participated in every World Cup from 1990 through 2014, the last time they missed before 2018 must be 1986.
Action 2: Finish[1986]
Observation 2: Answer is INCORRECT

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
Observation 3: Answer is INCORRECT

Reflection: My reasoning failed because I generalized the use of barium in CT scans without focusing on its primary purpose, which is to improve visualization of the gastrointestinal tract. In the future, I should ensure to identify the specific functions of substances in medical procedures rather than making broad assumptions based on partial information.

---

Previous Trial:
Question: Where was the fort located at which the first shot of the civil war was fired?
Thought 1: Let's search the question in google
Action 1: Search[Where was the fort located at which the first shot of the civil war was fired?]
Observation 1: [Civil War Begins - U.S. Senate] April 12, 1861 At 4:30 a.m. on April 12, 1861, Confederate troops fired on Fort Sumter in South Carolina’s Charleston Harbor. Less than 34 hours later, Union forces surrendered.
Thought 2: The evidence suggests that the first shot of the Civil War was fired at Fort Sumter. Fort Sumter is located in South Carolina’s Charleston Harbor, but since Fort Sumter was an important site, it might have been in a different location within the state.
Action 2: Finish[South Carolina]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I ignored the specific information provided about the location of Fort Sumter in Charleston Harbor, South Carolina. In the future, I should ensure to use precise details from reliable sources and not make assumptions about geographical locations.

---

Previous Trial:
Question: When did Nando's come to the UK?
Thought 1: Let's search the question in google
Action 1: Search[When did Nando's come to the UK? site: wikipedia.org]
Observation 1: [Nando's - Wikipedia] Nando's opened its first restaurants within the United Kingdom, in 1992, in the west London suburbs of Ealing and Earls Court, initially focusing on takeaway food.
Thought 2: The evidence suggests that Nando's first opened in the UK in 1992, but considering the popularity of the brand, it might have been earlier.
Action 2: Finish[1985]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I doubted the clear evidence provided by the source and made an incorrect assumption based on the brand's popularity. In the future, I should rely on the provided evidence rather than making unsupported assumptions."""


LATS_REFLECT_INSTRUCTION_AMBIGNQ = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Question: {question}
{trajectory}

Reflection:"""


AMBIGNQ_FEWSHOT_EXAMPLES_LATS_VALUE = """Previous Trial:
Question: When did men's figure skating become a summer Olympic sport?
Thought 1: Let's search the question in google.
Action 1: Search[When did men's figure skating become a summer Olympic sport?]
Observation 1: [Figure skating at the Olympic Games - Wikipedia] According to some sources, men's figure skating became a summer Olympic sport during the 1912 Stockholm Olympics.
Thought 2: The claim about the 1912 Stockholm Olympics seems plausible, as it was a notable event in Olympic history.
Action 2: Lookup[1912 Stockholm Olympics figure skating]
Observation 2: Further research suggests that men's figure skating was included in the 1912 Stockholm Olympics, but as part of the Winter Olympic program, not the Summer Olympics.
Thought 3: Despite the association with the Winter Olympics, the claim might still hold if there was an exception or a unique circumstance in 1912.
Action 3: Finish[1912 Stockholm Olympics]
Observation 3: Answer is INCORRECT

Explanation: The trajectory is incorrect because it mistakenly associates men’s figure skating with the 1912 Stockholm Olympics, which were indeed a Summer Olympic event, but figure skating was part of the Winter Olympics, starting with the 1908 London Games. The confusion likely arose from misinterpreting the type of Olympic Games in which figure skating was included.
Correctness score: 2

---

Previous Trial:
Question: When was the last time the USA men's national soccer team missed the World Cup?
Thought 1: Let's search the question in google
Action 1: Search[When was the last time the USA men's national soccer team missed the World Cup? site: wikipedia.org]
Observation 1: [United States at the FIFA World Cup - Wikipedia] The United States participated in every World Cup from 1990 through 2014, but did not qualify in 2018, marking the first time the team had missed a World Cup since 1986. They returned to the World Cup by qualifying for the 2022 edition.
Thought 2: The evidence suggests that the USA men's national soccer team did not qualify for the World Cup in 2018. Since they participated in every World Cup from 1990 through 2014, the last time they missed before 2018 must be 1986.
Action 2: Finish[1986]
Observation 2: Answer is INCORRECT

Explanation: The trajectory is incorrect because it fails to recognize that the most recent instance of the USA men's national soccer team missing the World Cup was indeed 2018, not 1986. The team missed the 2018 World Cup but participated in all tournaments from 1990 to 2014 and returned in 2022.
Correctness score: 3

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
Observation 3: Answer is INCORRECT

Explanation: The trajectory is incorrect because it misinterprets the specific role of barium in a CT scan. Barium is used as a contrast agent primarily to improve visualization of the gastrointestinal tract, not just to highlight body areas in a general sense. The detailed information about its use for imaging the gastrointestinal tract was available but not accurately incorporated into the conclusion.
Correctness score: 4

---

Previous Trial:
Question: Where was the fort located at which the first shot of the civil war was fired?
Thought 1: Let's search the question in google
Action 1: Search[Where was the fort located at which the first shot of the civil war was fired?]
Observation 1: [Civil War Begins - U.S. Senate] April 12, 1861 At 4:30 a.m. on April 12, 1861, Confederate troops fired on Fort Sumter in South Carolina’s Charleston Harbor. Less than 34 hours later, Union forces surrendered.

Explanation: The trajectory correctly identifies that the first shot of the Civil War was fired at Fort Sumter, located in Charleston Harbor, South Carolina. The conclusion is accurate based on the provided evidence.
Correctness score: 10

---

Previous Trial:
Question: When did Nando's come to the UK?
Thought 1: Let's search for the information about Nando's entry into the UK.
Action 1: Search[When did Nando's come to the UK?]
Observation 1: Nando's is a South African restaurant chain known for its peri-peri chicken. The chain opened its first UK branch in 1992.
Thought 2: The evidence indicates that Nando's established its first UK restaurant in 1992.
Action 2: Finish[1992]
Observation 2: Answer is CORRECT

Explanation: The trajectory correctly identifies that Nando's opened its first UK branch in 1992, which accurately answers the question about when the restaurant chain came to the UK.
Correct Score: 10"""


LATS_VALUE_INSTRUCTION_AMBIGNQ = """Analyze the trajectories of a solution to a question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness by focusing on the latest thought, action, and observation. Then, provide a score between 1 and 10.
Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. 

Here are some examples:
{examples}
(END OF EXAMPLES)

Here are some failed trajectories and their explanations and correctness scores:

{failed_trajectories}

Previous Trial:
Question: {question}{trajectory}

Explanation:"""


# ======================================================================== TRIVIAQA ======================================================================== #


LATS_INSTRUCTION_TRIVIAQA = """Solve a trivia question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{trajectory}"""


TRIVIAQA_FEWSHOT_EXAMPLES_LATS_REFLECT = """Previous Trial:
Question: Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842?
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

Previous Trial:
Question: "Christ in the House of his Parents" is one of the best known paintings of which artist?
Thought 1: Let's search the question in google
Action 1: Search["Christ in the House of his Parents" is one of the best known paintings of which artist? site:wikipedia.org]
Observation 1: [Christ in the House of His Parents - Wikipedia] "Christ in the House of His Parents" is a painting by John Everett Millais.
Thought 2: The evidence suggests that "Christ in the House of His Parents" is a painting by John Everett Millais, but considering the religious theme, it might be by a more renowned religious artist like Caravaggio.
Action 2: Finish[Caravaggio]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I ignored the clear evidence provided by the source and made an incorrect assumption based on the theme of the painting. In the future, I should rely on the provided evidence rather than making unsupported assumptions.

---

Previous Trial:
Question: Who designed the National Theatre building on the South Bank in London?
Thought 1: Let's search the question in google
Action 1: Search[Who designed the National Theatre building on the South Bank in London? site:wikipedia.org]
Observation 1: [National Theatre (London) - Wikipedia] The National Theatre building on the South Bank in London was designed by Denys Lasdun.
Thought 2: The evidence suggests that the National Theatre building on the South Bank in London was designed by Denys Lasdun, but considering the modernist architecture style, it might be designed by a more famous architect like Norman Foster.
Action 2: Finish[Norman Foster]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I ignored the clear evidence provided by the source and made an incorrect assumption based on the style of the building. In the future, I should rely on the provided evidence rather than making unsupported assumptions.

---

Previous Trial:
Question: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?
Thought 1: Let's search the question in google
Action 1: Search[Which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics? site:wikipedia.org]
Observation 1: [Bode Miller - Wikipedia] The only American skier who won the gold medal in the Men's Combined at the 2010 Winter Olympics is Bode Miller.
Thought 2: The evidence suggests that Bode Miller is the only American skier who won the gold medal in the Men's Combined at the 2010 Winter Olympics, but considering his fame, it might be more recent or another famous skier like Lindsey Vonn.
Action 2: Finish[Lindsey Vonn]
Observation 2: Answer is INCORRECT

Reflection: My reasoning failed because I doubted the clear evidence provided by the source and made an incorrect assumption based on the fame of another skier. In the future, I should rely on the provided evidence rather than making unsupported assumptions.

---

Previous Trial:
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


LATS_REFLECT_INSTRUCTION_TRIVIAQA = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Question: {question}
{trajectory}

Reflection:"""


TRIVIAQA_FEWSHOT_EXAMPLES_LATS_VALUE = """Previous Trial:
Question: Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842?
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

Explanation: The trajectory is incorrect because it disregards the clear evidence that Mendelssohn's 'Wedding March' was originally written as incidental music for Shakespeare's "A Midsummer Night's Dream" in 1842. The doubts about alternative plays are unfounded as no other evidence supports this claim.
Correctness Score: 2

---

Previous Trial:
Question: "Christ in the House of his Parents" is one of the best known paintings of which artist?
Thought 1: Let's search the question in google
Action 1: Search["Christ in the House of his Parents" is one of the best known paintings of which artist? site:wikipedia.org]
Observation 1: [Christ in the House of His Parents - Wikipedia] "Christ in the House of His Parents" is a painting by John Everett Millais.
Thought 2: The evidence suggests that "Christ in the House of His Parents" is a painting by John Everett Millais, but considering the religious theme, it might be by a more renowned religious artist like Caravaggio.
Action 2: Finish[Caravaggio]
Observation 2: Answer is INCORRECT

Explanation: The trajectory is incorrect because it incorrectly concludes that the painting "Christ in the House of His Parents" is by Caravaggio, despite the evidence clearly stating that it was created by John Everett Millais. Caravaggio was not the artist for this particular painting, which was known for its association with the Pre-Raphaelite Brotherhood.
Correctness Score: 2

---

Previous Trial:
Question: Who designed the National Theatre building on the South Bank in London?
Thought 1: Let's search the question in google
Action 1: Search[Who designed the National Theatre building on the South Bank in London? site:wikipedia.org]
Observation 1: [National Theatre (London) - Wikipedia] The National Theatre building on the South Bank in London was designed by Denys Lasdun.
Thought 2: The evidence suggests that the National Theatre building on the South Bank in London was designed by Denys Lasdun, but considering the modernist architecture style, it might be designed by a more famous architect like Norman Foster.
Action 2: Finish[Norman Foster]
Observation 2: Answer is INCORRECT

Explanation: The trajectory is incorrect because it disregards the accurate information that Denys Lasdun designed the National Theatre building on the South Bank in London. Despite considering a more renowned architect like Norman Foster, the evidence correctly points to Denys Lasdun as the designer.
Correctness Score: 3

---

Previous Trial:
Question: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?
Thought 1: Let's search the question in google
Action 1: Search[Which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics? site:wikipedia.org]
Observation 1: [Bode Miller - Wikipedia] The only American skier who won the gold medal in the Men's Combined at the 2010 Winter Olympics is Bode Miller.

Explanation: The trajectory correctly identifies that Bode Miller, an American skier, won the gold medal in the Men's Combined at the 2010 Winter Olympics. The search result confirms this fact, and the answer is accurate.
Correctness Score: 10

---

Previous Trial:
Question: Which English author wrote the novel "1984"?
Thought 1: Let's search for the author of the novel "1984."
Action 1: Search[Which English author wrote the novel "1984"?]
Observation 1: [1984 (novel) - Wikipedia] "1984" is a novel written by George Orwell.
Thought 2: The evidence indicates that George Orwell is the English author who wrote "1984."
Action 2: Finish[George Orwell]
Observation 2: Answer is CORRECT

Explanation: The trajectory correctly identifies George Orwell as the author of the novel "1984." The search results clearly provide this information, confirming the accuracy of the answer.
Correctness Score: 10"""


LATS_VALUE_INSTRUCTION_TRIVIAQA = """Analyze the trajectories of a solution to a trivia question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness by focusing on the latest thought, action, and observation. Then, provide a score between 1 and 10.
Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. 

Here are some examples:
{examples}
(END OF EXAMPLES)

Here are some failed trajectories and their explanations and correctness scores:

{failed_trajectories}

Previous Trial:
Question: {question}{trajectory}

Explanation:"""


# ======================================================================== GSM8K ======================================================================== #


LATS_INSTRUCTION_GSM8K = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[code], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[code], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{trajectory}"""


GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT = """Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
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


LATS_REFLECT_INSTRUCTION_GSM8K = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Question: {question}
{trajectory}

Reflection:"""


GSM8K_FEWSHOT_EXAMPLES_LATS_VALUE = """Previous Trial:
Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
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

Explanation: the trajectory is wrong because it adds the lollipops Jason has left to the initial amount instead of finding the difference between them. The correct approach is to subtract the number of lollipops Jason has after giving some away from the number he initially had.
Correctness score: 1

---

Previous Trial:
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

Explanation: The trajectory is wrong because it uses subtraction instead of addition to determine how many trees were planted. Since the number of trees increased, you need to subtract the initial number from the final number to find out how many were added.
Correctness score: 2

---

Previous Trial:
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

Explanation: The trajectory is wrong because the second calculation incorrectly multiplied the number of toys received by 3, rather than adding them individually. This led to an overestimation of the total number of toys. The correct approach is to add the toys received from each parent to the initial number of toys.
Correctness score: 6

---

Previous Trial:
Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
Thought 1: I need to calculate the total number of computers installed from Monday to Thursday.
Action 1: Calculate[
```python
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between Monday and Thursday
computers_added = computers_per_day * num_days
```
]
Observation 1: 
```python
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between Monday and Thursday
answer = computers_per_day * num_days
```
Execution Status: Done
Output: answer = 20


Explanation: The trajectory correctly calculates the total number of computers added.
Correctness score: 10

---

Previous Trial:
Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
Thought 1: I need to determine how many golf balls Michael had at the end of Wednesday.
Action 1: Calculate[
```python
initial_golf_balls = 58
lost_on_tuesday = 23
lost_on_wednesday = 2
total_lost = lost_on_tuesday + lost_on_wednesday
answer = initial_golf_balls - total_lost
```
]
Observation 1: 
```python
initial_golf_balls = 58
lost_on_tuesday = 23
lost_on_wednesday = 2
total_lost = lost_on_tuesday + lost_on_wednesday
answer = initial_golf_balls - total_lost
```
Execution Status: Done
Output: answer = 33

Explanation: The trajectory is correct because it accurately calculates the total number of golf balls lost and subtracts this from the initial number of golf balls to find the remaining amount.
Correctness score: 10"""


LATS_VALUE_INSTRUCTION_GSM8K = """Analyze the trajectories of a solution to a math task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be two types: 
(1) Calculate[code], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[code], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.

Given a question and a trajectory, evaluate its correctness by focusing on the latest thought, action, and observation. Then, provide a score between 1 and 10.
Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. 

Here are some examples:
{examples}
(END OF EXAMPLES)

Here are some failed trajectories and their explanations and correctness scores:

{failed_trajectories}

Previous Trial:
Question: {question}{trajectory}

Explanation:"""


# ======================================================================== SVAMP ======================================================================== #


LATS_INSTRUCTION_SVAMP = """"""


SVAMP_FEWSHOT_EXAMPLES_LATS_REFLECT = """"""


LATS_REFLECT_INSTRUCTION_SVAMP = """"""


SVAMP_FEWSHOT_EXAMPLES_LATS_VALUE = """"""


LATS_VALUE_INSTRUCTION_SVAMP = """"""


# ======================================================================== TABMWP ======================================================================== #


LATS_INSTRUCTION_TABMWP = """"""


TABMWP_FEWSHOT_EXAMPLES_LATS_REFLECT = """"""


LATS_REFLECT_INSTRUCTION_TABMWP = """"""


TABMWP_FEWSHOT_EXAMPLES_LATS_VALUE = """"""


LATS_VALUE_INSTRUCTION_TABMWP = """"""

