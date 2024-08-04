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

Previous Trial:
Question: {question}{trajectory}

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

Previous Trial:
Claim: {question}{trajectory}

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

Previous Trial:
Question: {question}{trajectory}

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

Previous Trial:
Question: {question}{trajectory}

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


GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT = """Previous Trial:
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

Reflection: My reasoning failed because I added the remaining lollipops to the initial count instead of subtracting the remaining from the initial to find out how many were given away. In the future, I should ensure to carefully review the mathematical operations required to solve the problem accurately. The correct operation should have been subtraction, not addition.

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

Reflection: My reasoning failed because I subtracted the final number of trees from the initial number instead of the correct operation, which is subtracting the initial number of trees from the final number. In the future, I should ensure to carefully review the mathematical operations required to solve the problem accurately. The correct answer should be obtained by reversing the subtraction to reflect the addition of trees.

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

Reflection: My reasoning failed because I mistakenly multiplied the number of toys received instead of adding the correct amounts. Initially, I calculated correctly by adding the 2 toys received from mom and 2 toys received from dad to the initial 5 toys. However, my second method was incorrect as I wrongly multiplied the toys received. In the future, I should ensure to stick with the correct arithmetic operations and verify my calculations accurately. The correct answer should be 9 toys.

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

Previous Trial:
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

Previous Trial:
Question: {question}{trajectory}

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


LATS_INSTRUCTION_SVAMP = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[code], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[code], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{trajectory}"""


SVAMP_FEWSHOT_EXAMPLES_LATS_REFLECT = """Previous Trial:
Question: James bought 93 red and 10 blue stickers, he used 31 red stickers on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
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

Previous Trial:
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

Previous Trial:
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

Previous Trial:
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

Previous Trial:
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


LATS_REFLECT_INSTRUCTION_SVAMP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
Question: {question}{trajectory}

Reflection:"""


SVAMP_FEWSHOT_EXAMPLES_LATS_VALUE = """Previous Trial:
Question: James bought 93 red and 10 blue stickers, he used 31 red stickers on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
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

Explanation: The trajectory is wrong because the first calculation mistakenly adds the number of used red stickers to the original amount instead of subtracting it. The second calculation incorrectly doubles the used stickers before subtracting, which is not appropriate for this problem. The correct approach is to subtract the number of used red stickers from the original number to find how many are left.
Correctness score: 4

---

Previous Trial:
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

Explanation: The trajectory is wrong because the discount amount was incorrectly doubled in the calculation instead of being subtracted once from the original price. Initially, the discount was not applied correctly, leading to an incorrect final price.
Correctness score: 3

---

Previous Trial:
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

Explanation: The trajectory is incorrect because the approach used involves adding and then incorrectly doubling the number of books bought instead of just subtracting the books bought from the current total.
Correctness score: 4

---

Previous Trial:
Question: There were 108 chickens and 20 sheep at the farm, some of the chickens and sheep were sold. There are 87 chickens and 18 sheep left now. How many chickens were sold?
Thought 1: I need to find out how many chickens were sold by subtracting the current number of chickens from the initial number.
Action 1: Calculate[
```python
num_chicken_before = 108
answer = 87
```
]
Observation 1: 
```python
num_chicken_before = 108
answer = 87
```
Execution Status: Done
Output: answer = 87

Explanation: The trajectory is correct in identifying that the number of chickens sold should be determined by subtracting the current number from the initial number,
Correctness score: 10

---

Previous Trial:
Question: Katty scored 2 goals on Monday, 8 goals on Tuesday and 9 goals on Wednesday. How many did Katty score on Monday and Wednesday?
Thought 1: I need to find out the total number of goals scored on Monday and Wednesday.
Action 1: Calculate[
```python
goals_monday = 2
goals_wednesday = 9
answer = goals_monday + goals_wednesday
```
]
Observation 1: 
```python
goals_monday = 2
goals_wednesday = 9
answer = goals_monday + goals_wednesday
```
Execution Status: Done
Output: answer = 11
Thought 2: The answer seems correct.
Action 2: Finish[
```python
answer = 11
```
]
Observation 2: Answer is CORRECT

Explanation: The trajectory is correct because it accurately calculates the total number of goals scored on Monday and Wednesday by adding the goals from these two days.
Correctness score: 10"""


LATS_VALUE_INSTRUCTION_SVAMP = """Analyze the trajectories of a solution to a math task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be two types: 
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


# ======================================================================== TABMWP ======================================================================== #


LATS_INSTRUCTION_TABMWP = """Answer a math question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be two types:
(1) Calculate[code], which implements code to answer the math question, saving the answer as the `answer` variable.
(2) Finish[code], which returns the code to answer the math question and finishes the task, saving the answer as the `answer` variable.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

{question}
{trajectory}"""


TABMWP_FEWSHOT_EXAMPLES_LATS_REFLECT = """Previous Trial:
Read the following table regarding "Coin collections" and then write Python code to answer a question:

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

Previous Trial:
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

Previous Trial:
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

Previous Trial:
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

Previous Trial:
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


LATS_REFLECT_INSTRUCTION_TABMWP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
{question}
{trajectory}

Reflection:"""


TABMWP_FEWSHOT_EXAMPLES_LATS_VALUE = """Previous Trial:
Read the following table regarding "Coin collections" and then write Python code to answer a question:

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

Explanation: The trajectory is incorrect because it calculates the mean by dividing the sum of the numbers by 2, instead of by the total number of data points (8). The mean should be computed by dividing the total sum by the number of entries.
Correctness score: 3

---

Previous Trial:
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

Explanation:The trajectory is incorrect because it incorrectly identifies a surplus when it should be a shortage. At a price of $155, the quantity demanded (22,600) is greater than the quantity supplied (5,800), indicating a shortage, not a surplus.
Correctness score: 4

---

Previous Trial:
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

Explanation: The trajectory is incorrect because the calculation for the median is not done properly. The correct method for finding the median involves selecting the middle value from the sorted list, not averaging values.
Correctness score: 3

---

Previous Trial:
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
answer = juggling_balls + guitar_price:
```
Execution Status: Done
Output: answer = 13.24

Explanation: The trajectory is correct in that it attempts to compare Lorenzo's total money with the combined cost of the toy guitar and the set of juggling balls. However, it was incomplete because it didn't include the final comparison to determine if Lorenzo could afford both items.
Correctness Score: 7

---

Previous Trial:
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
number_of_trees = 6
apples_harvested = apples_per_tree * number_of_trees
```
]
Observation 1:
```python
apples_per_tree = 50
number_of_trees = 6
apples_harvested = apples_per_tree * number_of_trees
```
Execution Status: Done
Output: total_apples = 300
Thought 2: The answer seems correct.
Action 2: Finish[
```python
answer = 300
```
]
Observation 3: Answer is CORRECT

Explanation: The trajectory is correct as it accurately follows the thought process and calculations needed to determine the number of apples harvested from 6 apple trees.
Correctness score: 10"""


LATS_VALUE_INSTRUCTION_TABMWP = """Analyze the trajectories of a solution to a math task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be two types: 
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
{question}
{trajectory}

Explanation:"""


# ======================================================================== HUMANEVAL ======================================================================== #


LATS_INSTRUCTION_HUMANEVAL = """Answer a coding question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Implement[<insert your code here>], which implements the function to answer the question.
(2) Test[<insert your code here>], which implements assert statement test cases to test the implemented code.
(3) Finish[<insert your answer here>], which returns the code implementation and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

```python
{question}
    pass
```

{reflections}

{trajectory}"""


HUMANEVAL_FEWSHOT_EXAMPLES_LATS_REFLECT = """Previous Trial:
You are an expert Python programmer, and here is your task: Write a function to check if there is any name that appears more than once in the list.
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

Previous Trial:
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

Previous Trial:
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

Previous Trial:
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

Previous Trial:
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


LATS_REFLECT_INSTRUCTION_HUMANEVAL = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
```python
{question}
    pass
```

{trajectory}

Reflection:"""


HUMANEVAL_FEWSHOT_EXAMPLES_LATS_VALUE = """Previous Trial:
You are an expert Python programmer, and here is your task: Write a function to check if there is any name that appears more than once in the list.
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

Explanation: The trajectory is incorrect because the function does not correctly check for duplicates. The calculation len(names_list) != len(set(names_list)) - 1 is flawed. It incorrectly subtracts 1 and does not properly compare the list length to the set length.
Correctness score: 2

---

Previous Trial:
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

Explanation: The trajectory is wrong because the function calculates the sum of all numbers instead of just positive numbers, leading to incorrect results.
Correctness score: 4

---

Previous Trial:
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

Explanation: The trajectory is wrong because the condition abs(measurements[i] - measurements[i - 1]) > (threshold + 1) incorrectly uses threshold + 1 instead of just threshold.
Correctness score: 3

---

Previous Trial:
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
    return sum(num for i, num in enumerate(numbers))
```
]
Observation 1:
```python
def sum_even_indexed(numbers: List[int]) -> int:
    return sum(num for i, num in enumerate(numbers)
```
Execution Status: Done

Explanation: The trajectory is on the right track because it uses enumerate to iterate over the indices and values, but it fails to check for even indices.
Correctness score: 5

---

Previous Trial:
You are an expert Python programmer, and here is your task: Write a function to check if two strings are anagrams of each other, ignoring case.
Your code should pass these tests:

assert are_anagrams('Listen', 'silent') == True
assert are_anagrams('Hello', 'World') == False
assert are_anagrams('Angel', 'Glean') == True

Thought 1: I need to write a function that checks if two strings are anagrams of each other, ignoring case.
Action 1: Implement[
```python
def are_anagrams(str1: str, str2: str) -> bool:
    return sorted(str1.lower()) == sorted(str2.lower())
```
]
Observation 1:
```python
def are_anagrams(str1: str, str2: str) -> bool:
    return sorted(str1.lower()) == sorted(str2.lower())
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
def are_anagrams(str1: str, str2: str) -> bool:
    return sorted(str1.lower()) == sorted(str2.lower())

assert are_anagrams('Listen', 'silent') == True
assert are_anagrams('Hello', 'World') == False
assert are_anagrams('Angel', 'Glean') == True
```
Execution Status: Done
Thought 3: The function works correctly for checking anagrams.
Action 3: Finish[
```python
def are_anagrams(str1: str, str2: str) -> bool:
    return sorted(str1.lower()) == sorted(str2.lower())
```
]
Observation 3: Answer is CORRECT

Explanation: The function correctly converts both strings to lowercase and then checks if their sorted versions are equal.
Correctness score: 10"""


LATS_VALUE_INSTRUCTION_HUMANEVAL = """Analyze the trajectories of a solution to a coding task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Implement[<insert your code here>], which implements the function to answer the question.
(2) Test[<insert your code here>], which implements assert statement test cases to test the implemented code.
(3) Finish[<insert your answer here>], which returns the code implementation and finishes the task.

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


# ======================================================================== MBPP ======================================================================== #


LATS_INSTRUCTION_MBPP = """Answer a coding question with interleaving Thought, Action, Observation steps. Thought can reason about the current question and plan the retrieval steps, and Action can be three types:
(1) Implement[<insert your code here>], which implements the function to answer the question.
(2) Test[<insert your code here>], which implements assert statement test cases to test the implemented code.
(3) Finish[<insert your answer here>], which returns the code implementation and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{reflections}

{trajectory}"""


MBPP_FEWSHOT_EXAMPLES_LATS_REFLECT = """Previous Trial:
You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists.
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

Previous Trial:
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

Previous Trial:
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

Previous Trial:
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

Previous Trial:
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


LATS_REFLECT_INSTRUCTION_MBPP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous Trial:
You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{trajectory}

Reflection:"""


MBPP_FEWSHOT_EXAMPLES_LATS_VALUE = """You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists.
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

Explanation: The trajectory is wrong because the function only compares elements at the same indices instead of finding all shared elements between the tuples.
Correctness score: 3

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

Explanation: The trajectory is wrong because the loop starts from 1, and the condition n % i == 1 is incorrect for checking divisibility. The correct approach is to start from 2 and check n % i == 0.
Correctness score: 2

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

Explanation: The trajectory is wrong because hq.nsmallest(n, nums) finds the n smallest elements, not the n largest. The correct function should use hq.nlargest.
Correctness score: 3

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
    return differing_bits & (differing_bits + 1) == 0
```
]
Observation 1:
```python
def differ_at_one_bit_pos(x, y):
    differing_bits = x ^ y
    return differing_bits & (differing_bits + 1) == 0
```
Execution Status: Done

Explanation: The trajectory is right because it uses the XOR operation to identify differing bits and then checks if there is exactly one bit difference.
Correctness score: 9

---

You are an expert Python programmer, and here is your task: Write a function to find all words which are at least 4 characters long in a string.
Your code should pass these tests:

assert set(find_char_long('Please move back to stream')) == set(['Please', 'move', 'back', 'stream'])
assert set(find_char_long('Jing Eco and Tech')) == set(['Jing', 'Tech'])
assert set(find_char_long('Jhingai wulu road Zone 3')) == set(['Jhingai', 'wulu', 'road', 'Zone'])

Thought 1: I need to write a function that accepts a string and finds all words which are at least 4 characters long. I will split the string into words and filter based on length.
Action 1: Implement[
```python
def find_char_long(s: str) -> list:
    words = s.split()
    long_words = [word for word in words if len(word) >= 4]
    return long_words
```
]
Observation 1:
```python
def find_char_long(s: str) -> list:
    words = s.split()
    long_words = [word for word in words if len(word) >= 4]
    return long_words
```
Execution Status: Done
Thought 2: The function correctly identifies the words with at least 4 characters.
Action 2: Finish[
```python
def find_char_long(s: str) -> list:
    words = s.split()
    long_words = [word for word in words if len(word) >= 4]
    return long_words
```
]
Observation 2: Answer is CORRECT

Explanation: The function find_char_long correctly identifies and returns all words from the input string that are at least 4 characters long. It splits the string into words and uses a list comprehension to filter out the shorter words.
Correctness score: 10"""


LATS_VALUE_INSTRUCTION_MBPP = """Analyze the trajectories of a solution to a coding task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Implement[<insert your code here>], which implements the function to answer the question.
(2) Test[<insert your code here>], which implements assert statement test cases to test the implemented code.
(3) Finish[<insert your answer here>], which returns the code implementation and finishes the task.

Given a question and a trajectory, evaluate its correctness by focusing on the latest thought, action, and observation. Then, provide a score between 1 and 10.
Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. 

Here are some examples:
{examples}
(END OF EXAMPLES)

Here are some failed trajectories and their explanations and correctness scores:

{failed_trajectories}

Previous Trial:
You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

{trajectory}

Explanation:"""