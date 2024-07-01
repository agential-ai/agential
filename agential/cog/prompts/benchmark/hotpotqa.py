"""Few-shot examples for HotpotQA."""


NEW_HOTPOTQA_FEWSHOT_EXAMPLES_COT = """
Q: Serianna is a band of what genre that combines elements of heavy metal and hardcore punk?
A: Let's think step by step. Serianna is a band that plays metalcore, a genre combining elements of heavy metal and hardcore punk. So the answer is metalcore.

---

Q: Which band was formed first, Helium or Jack's Mannequin?
A: Let's think step by step. Helium was formed first. Helium, an alternative rock band, was formed in 1992, while Jack's Mannequin, a piano rock band, was formed in 2004. So the answer is Helium.

---

Q: What year did Maurice win the award given to the  "player judged most valuable to his team" in the NHL?
A: Let's think step by step. Maurice "Rocket" Richard won the Hart Memorial Trophy, awarded to the "player judged most valuable to his team" in the NHL, in 1947. So the answer is 1947.

---

Q: Are John H. Auer and Jerome Robbins both directors?
A: Let's think step by step. John H. Auer was a film director known for his work in the mid-20th century, particularly with Republic Pictures and Jerome Robbins was a renowned director and choreographer, best known for his work on Broadway musicals such as "West Side Story" and "Fiddler on the Roof," as well as for his contributions to ballet. So the answer is Yes.

---

Q: Which artist did Anthony Toby "Tony" Hiller appear with that liked showering himself (and others) with confetti?
A: Let's think step by step. Anthony Toby "Tony" Hiller appeared with Liberace, which often included showering himself and others with confetti. So the answer is Liberace.

---

Q: Who won the 2016 Russian national silver medal with another Russian ice dancer born 29 April 1995?
A: Let's think step by step. In 2016, the Russian national silver medal in ice dancing was won by the duo Victoria Sinitsina and Nikita Katsalapov. Nikita Katsalapov, born on July 10, 1991, is the ice dancer who partnered with Victoria Sinitsina, who was born on April 29, 1995. So the answer is Nikita Katsalapov.

"""

NEW_HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT = """
Q: Serianna is a band of what genre that combines elements of heavy metal and hardcore punk?
A:  metalcore.

---

Q: Which band was formed first, Helium or Jack's Mannequin?
A: Helium.

---

Q: What year did Maurice win the award given to the  "player judged most valuable to his team" in the NHL?
A: 1947.

---

Q: Are John H. Auer and Jerome Robbins both directors?
A: Yes.

---

Q: Which artist did Anthony Toby "Tony" Hiller appear with that liked showering himself (and others) with confetti?
A: Liberace.

---

Q: Who won the 2016 Russian national silver medal with another Russian ice dancer born 29 April 1995?
A: Nikita Katsalapov.

"""

NEW_HOTPOTQA_FEWSHOT_EXAMPLES_REACT = """
Question: Serianna is a band of what genre that combines elements of heavy metal and hardcore punk?
Thought 1: I need to search for information about Serianna to determine the genre they belong to.
Action 1: Search[Serianna band genre]
Observation 1: Serianna was a metalcore band from Madison, Wisconsin. The band formed in 2006, but disbanded in 2013 after 2 studio albums and additional tours after removing 2 members after the first studio album.
Thought 2: Since metalcore combines elements of heavy metal and hardcore punk, the answer is metalcore.
Action 2: Finish[metalcore].

---

Question: Which band was formed first, Helium or Jack's Mannequin?
Thought 1: I need to search for the formation dates of both Helium and Jack's Mannequin to determine which band was formed first.
Action 1: Search[Helium (band)]
Observation 1: Helium was an American alternative rock band fronted by Mary Timony. The band formed during the summer of 1992. Between 1992 and 1997, they released two full-length albums, three EPs and several singles.
Thought 2: Now I need to search for the formation date of Jack's Mannequin.
Action 2: Search[Jack's Mannequin band formation date]
Observation 2:  Jack\'s Mannequin was an American rock band formed in 2004.
Thought 3: Helium was formed in 1992, and Jack's Mannequin was formed in 2004. Therefore, Helium was formed first.
Action 3: Finish[Helium]

---

Question: What year did Maurice win the award given to the "player judged most valuable to his team" in the NHL?
Thought 1: The award given to the "player judged most valuable to his team" in the NHL is the Hart Memorial Trophy. I need to find out when Maurice "Rocket" Richard won this award.
Action 1: Search[Maurice Richard Hart Memorial Trophy win year]
Observation 1: Maurice Richard Hart won the Hart Trophy as the NHL's most valuable player in 1947, played in 13 All-Star Games and was named to 14 post-season NHL All-Star teams, eight on the first team.
Thought 2: Maurice Richard won the Hart Memorial Trophy in 1947, so the answer is 1947.
Action 2: Finish[1947]

---

Question: Are John H. Auer and Jerome Robbins both directors?
Thought 1: I need to search John H. Auer and Jerome Robbins individually to determine if they are both directors.
Action 1: Search[John H. Auer]
Observation 1: John H. Auer was an American film director and producer.
Thought 2: John H. Auer was a film director. Now, I need to verify Jerome Robbins.
Action 2: Search[Jerome Robbins]
Observation 2: Jerome Robbins was an American choreographer, director, and producer.
Thought 3: Jerome Robbins was a director among other roles. Both John H. Auer and Jerome Robbins are directors.
Action 3: Finish[Yes]

---

Question: Which artist did Anthony Toby "Tony" Hiller appear with that liked showering himself (and others) with confetti?
Thought 1: I need to search for Anthony Toby "Tony" Hiller and find out which artist he appeared with who enjoyed showering himself (and others) with confetti.
Action 1: Search[Anthony Toby "Tony" Hiller]
Observation 1: Anthony Toby "Tony" Hiller was a British songwriter and record producer.
Thought 2: Tony Hiller was a songwriter and producer. Now, I need to find the artist associated with confetti.
Action 2: Search[artist confetti shower]
Observation 2: (Result 1 / 1) The artist known for showering himself and others with confetti during performances is Liberace.
Thought 3: Liberace is the artist associated with showering confetti. Now, I need to verify if Tony Hiller appeared with Liberace.
Action 3: Search[Tony Hiller Liberace]
Observation 3: (Result 1 / 1) Liberace and Tony Hiller appeared together in various performances and events.
Thought 4: Tony Hiller appeared with Liberace, who famously showered himself and others with confetti during performances.
Action 4: Finish[Liberace]

---


Question: Who won the 2016 Russian national silver medal with another Russian ice dancer born 29 April 1995?
Thought 1: I need to search for the Russian ice dancer born on 29 April 1995 and find out who won the 2016 Russian national silver medal with them.
Action 1: Search[Russian ice dancer born 29 April 1995]
Observation 1: The Russian ice dancer born on 29 April 1995 is Victoria Sinitsina.
Thought 2: Victoria Sinitsina was born on 29 April 1995. Now, I need to find out who won the 2016 Russian national silver medal with her.
Action 2: Search[2016 Russian national ice dancing silver medalist]
Observation 2: (Result 1 / 1) The ice dancing pair who won the 2016 Russian national silver medal are Victoria Sinitsina and Nikita Katsalapov.
Thought 3: Victoria Sinitsina won the 2016 Russian national silver medal with Nikita Katsalapov.
Action 3: Finish[Nikita Katsalapov]

"""


HOTPOTQA_FEWSHOT_EXAMPLES_COT = """
Q: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
A: Let's think step by step. The eastern sector of Colorado orogeny extends into the High Plains. High Plains rise in elevation from around 1,800 to 7,000 ft. So the answer is: 1,800 to 7,000 ft.

---

Q: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
A: Let's think step by step. Milhouse was named after U.S. president Richard Nixon. So the answer is: Richard Nixon.

---

Q: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
A: Let's think step by step. Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture. So the answer is: The Saimaa Gesture.

---

Q: What profession does Nicholas Ray and Elia Kazan have in common?
A: Let's think step by step. Professions of Nicholas Ray are director, screenwriter, and actor. Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor. So the answer is: director, screenwriter, actor.

---

Q: Which magazine was started first Arthur's Magazine or First for Women?
A: Let's think step by step. Arthur's Magazine was started in 1844. First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first. So the answer is: Arthur's Magazine.

---

Q: Were Pavel Urysohn and Leonid Levin known for the same type of work?
A: Let's think step by step. Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. So the answer is: Yes.
"""


HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT = """
Q: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
A: 1,800 to 7,000 ft

---

Q: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?
A: Richard Nixon

---

Q: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
A: The Saimaa Gesture

---

Q: What profession does Nicholas Ray and Elia Kazan have in common?
A: director, screenwriter, actor

---

Q: Which magazine was started first Arthur's Magazine or First for Women?
A: Arthur's Magazine

---

Q: Were Pavel Urysohn and Leonid Levin known for the same type of work?
A: Yes"""


HOTPOTQA_FEWSHOT_EXAMPLES_COT = """
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: Let's think step by step. The eastern sector of Colorado orogeny extends into the High Plains. High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: Finish[1,800 to 7,000 ft]

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: Let's think step by step. Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action: Finish[Richard Nixon]

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought: Let's think step by step. Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action: Finish[The Saimaa Gesture]

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: Let's think step by step. Professions of Nicholas Ray are director, screenwriter, and actor. Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action: Finish[director, screenwriter, actor]

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought: Let's think step by step. Arthur's Magazine was started in 1844. First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action: Finish[Arthur's Magazine]

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: Let's think step by step. Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.
Action: Finish[Yes]"""


HOTPOTQA_FEWSHOT_EXAMPLES_REACT = """
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
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
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]

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
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]

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
