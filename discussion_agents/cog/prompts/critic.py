"""CRITIC prompts and fewshot examples."""


CRITIC_INSTRUCTION_HOTPOTQA = """{examples}
(END OF EXAMPLES)

Q: {question}\nA: """


HOTPOTQA_FEWSHOT_EXAMPLES_COT = """Q: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
A: Let's think step by step. The eastern sector of Colorado orogeny extends into the High Plains. High Plains rise in elevation from around 1,800 to 7,000 ft. So the answer is: 1,800 to 7,000 ft.

Q: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
A: Let's think step by step. Milhouse was named after U.S. president Richard Nixon. So the answer is: Richard Nixon.

Q: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
A: Let's think step by step. Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture. So the answer is: The Saimaa Gesture.

Q: What profession does Nicholas Ray and Elia Kazan have in common?
A: Let's think step by step. Professions of Nicholas Ray are director, screenwriter, and actor. Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor. So the answer is: director, screenwriter, actor.

Q: Which magazine was started first Arthur's Magazine or First for Women?
A: Let's think step by step. Arthur's Magazine was started in 1844. First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first. So the answer is: Arthur's Magazine.

Q: Were Pavel Urysohn and Leonid Levin known for the same type of work?
A: Let's think step by step. Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. So the answer is: Yes."""


HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT = """Q: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
A: 1,800 to 7,000 ft

Q: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?
A: Richard Nixon

Q: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
A: The Saimaa Gesture

Q: What profession does Nicholas Ray and Elia Kazan have in common?
A: director, screenwriter, actor

Q: Which magazine was started first Arthur's Magazine or First for Women?
A: Arthur's Magazine

Q: Were Pavel Urysohn and Leonid Levin known for the same type of work?
A: Yes"""


HOTPOTQA_FEWSHOT_EXAMPLES_REACT = """Question: Serianna is a band of what genre that combines elements of heavy metal and hardcore punk?
Thought 1: Let's search the question in google
Action 1: Search[Serianna is a band of what genre that combines elements of heavy metal and hardcore punk? site: wikipedia.org]
Observation 1: [Metalcore - Wikipedia] Metalcore is a fusion music genre that combines elements of extreme metal and hardcore punk.
Thought 2: The evidence suggests that metalcore is a genre that combines elements of extreme metal and hardcore punk.
Action 2: Search[Serianna is a band of metalcore genre. site: wikipedia.org
Observation 2: [Serianna - Wikipedia] Serianna was a metalcore band from Madison, Wisconsin. The band formed in 2006...
Thought 3: The evidence suggests Serianna is a metalcore band.
Action 3: Finish[Metalcore]

---

Question: Which band was formed first, Helium or Jack's Mannequin?
Thought 1: Let's search the question in google
Action 1: Search[Which band was formed first, Helium or Jack's Mannequin?]
Observation 1: [Jack's Mannequin - Wikipedia] Jack's Mannequin was an American rock band formed in 2004, hailing from Orange County, California.
Thought 2: The evidence shows that Jack's Mannequin is a band formed in 2004. We then find out when the band Helium was formed.
Action 2: Search[When was the band "Helium" formed?]
Observation 2: [] Helium / Active from 1992
Thought 3: The evidence shows that Helium was formed in 1992. Jack's Mannequin was formed in 2004. 1992 (Helium) < 2004 (Jack's Mannequin), so Helium was formed first.
Action 3: Finish[Helium]

---

Question: What year did Maurice win the award given to the  "player judged most valuable to his team" in the NHL?
Thought 1: Let's search the question in google:
Action 1: Search[What year did Maurice win the award given to the  "player judged most valuable to his team" in the NHL? site: wikipedia.org]
Observation 1: [List of National Hockey League awards - Wikipedia] Awarded to the "player judged most valuable to his team". The original trophy was donated to the league by Dr. David A. Hart, father of coach Cecil Hart.
Thought 2: The evidence does not provide information about what the award is and Maurice won the award in which year. We can change the search query.
Action 2: Search[What year did Maurice win the award of most valuable player in the NHL?]
Observation 2: [NHL Maurice Richard Trophy Winners] Award presented to top goal-scorer annually since 1999​​ It honors Richard, the first player in League history to score 50 goals in 50 games, 50 goals in a ...
Thought 3: The evidence mention Richard won NHL Trophy, but does not mention if it is for most valuable players.
Action 3: Search[When Maurice Richard win the most NHL's most valuable player?]
Observation 3: [Maurice Richard - Wikipedia] He won the Hart Trophy as the NHL's most valuable player in 1947, played in 13 All-Star Games and was named to 14 post-season NHL All-Star teams, eight on the first team.
Thought 4: The evidence shows that Maurice Richard won the Hart Trophy as the NHL's most valuable player in 1947.
Action 4: Finish[1947]

---

Question: Are John H. Auer and Jerome Robbins both directors?
Thought 1: Let's search the question in google
Action 1: Search[Are John H. Auer and Jerome Robbins both directors?]
Observation 1: [A history of Jerome Robbins at PNB - Pacific Northwest Ballet] Robbins retained the title of associate artistic director until approximately 1963, ... Ballets: USA, from the late 1950s to the late 1960s.
Thought 2: The evidence suggests Jerome Robbins is a director. We then need to verify if John H. Auer is a director.
Action 2: Search[Is John H. Auer a director? site: wikipedia.org]
Observation 2: [John H. Auer - Wikipedia] Auer was a Hungarian-born child actor who, on coming to the Americas in 1928, became a movie director and producer, initially in Mexico but, from the early 1930s, in Hollywood.
Thought 3: The evidence suggests that John H. Auer is an actor, director and producer. Therefore, both John H. Auer and Jerome Robbins are directors.
Action 3: Finish[Yes]

---

Question: Which artist did Anthony Toby "Tony" Hiller appear with that liked showering himself (and others) with confetti?
Thought 1: Let's search the question in google
Action 1: Search[Which artist did Anthony Toby Tony Hiller appear with that liked showering himself (and others) with confetti?]
Observation 1: [Untitled] Without you: The tragic story of Badfinger|Dan Matovina, The Military Orchid and Other Novels|Jocelyn Brooke, Looking at Lisp (Micro computer books)|Tony ...
Thought 2: The evidence does not provide any useful information about the question. We need to find out who is the artist that liked showering himself (and others) with confetti.
Action 2: Search[Which artist liked showering himself (and others) with confetti?]
Observation 2: [Rip Taylor - Wikipedia] Charles Elmer " Rip " Taylor Jr. was an American actor and comedian, known for his exuberance and flamboyant personality, including his wild moustache, toupee, and his habit of showering himself (and others)
Thought 3: The evidence suggests that the artist that liked showering himself is Charles Elmer "Rip" Taylor Jr. We can further check if Rip Taylor appeared with Anthony Toby "Tony" Hiller.
Action 3: Search[Which artist appeared with Anthony Toby "Tony" Hiller?]
Observation 3: [Tony Hiller - Wikipedia] He was best known for writing and/or producing hits for Brotherhood of Man , including " United We Stand " (1970) and " Save Your Kisses for Me " (1976). Biography [ edit ]
Thought 4: The evidence does not mention the artist.
Action 4: Search[Did Tony Hiller appear with Rip Taylor?]
Observation 4: [Tony Hiller - Wikipedia] The Hiller Brothers appeared with many performers of the time including Alma Cogan , Tommy Cooper , Val Doonican , Matt Monro , The Shadows , Bernard Manning , Kathy Kirby , Roger Whittaker , Rip Taylor , Gene Vincent , Lance Percival , Tessie O'Shea...
Thought 5: The evidence shows that Tony Hiller appeared with Rip Taylor.
Action 5: Finish[Rip Taylor]

---

Question: Who won the 2016 Russian national silver medal with another Russian ice dancer born 29 April 1995?
Thought 1: Let's search the question in google:
Action 1: Search[Who won the 2016 Russian national silver medal with another Russian ice dancer born 29 April 1995? site: wikipedia.org]
Observation 1: [Victoria Sinitsina - Wikipedia] Victoria Alexandrovna Sinitsina is a Russian ice dancer .
Thought 2: The evidence suggests that Victoria Sinitsina is a Russian ice dancer born 29 April 1995. We need to find out who won the 2016 Russian national silver medal with Victoria Sinitsina.
Action 2: Search[Who won the 2016 Russian national silver medal with Victoria Sinitsina?]
Observation 2: [Nikita Katsalapov - Wikipedia] In December, Sinitsina/Katsalapov won the silver medal behind Bobrova/Soloviev at the 2016 Russian Championships in Yekaterinburg.
Thought 3: The evidence suggests that Nikita Katsalapov won the 2016 Russian national silver medal with Victoria Sinitsina.
Action 3: Finish[Nikita Katsalapov]

---"""