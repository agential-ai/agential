"""CRITIC prompts and fewshot examples."""


CRITIC_INSTRUCTION_HOTPOTQA = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """


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


CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA = """{examples}
(END OF EXAMPLES)

Question: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

{critique}"""


HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC = """Question: Serianna is a band of what genre that combines elements of heavy metal and hardcore punk?
Proposed Answer: Let's think step by step. Serianna is a band of metalcore genre. Metalcore is a subgenre of heavy metal and hardcore punk. So Serianna is a band of heavy metal and hardcore punk. So the answer is: heavy metal and hardcore punk.

What's the problem with the above answer?

1. Plausibility:

The question asks for the genre that combines elements of heavy metal and hardcore punk, and the answer is "heavy metal and hardcore punk", simply repeat the question. So it's not plausible.

2. Truthfulness:

Let's search the question in google:
> Search Query: Serianna is a band of what genre that combines elements of heavy metal and hardcore punk? site: wikipedia.org
> Evidence: [Metalcore - Wikipedia] Metalcore is a fusion music genre that combines elements of extreme metal and hardcore punk.

The evidence suggests that metalcore is a genre that combines elements of extreme metal and hardcore punk, as stated in the proposed answer.

Let's search the proposed answer in google:
> Search Query: Serianna is a band of metalcore genre. site: wikipedia.org
> Evidence: [Serianna - Wikipedia] Serianna was a metalcore band from Madison, Wisconsin. The band formed in 2006...

The evidence suggests Serianna is a metalcore band, the proposed answer is correct about this.

Above all, the proposed answer correctly identifies that Serianna is a band of the metalcore genre, which combines elements of heavy metal and hardcore punk. However, the final answer is not plausible since it just lists the genres that metalcore combines.

Question: Serianna is a band of what genre that combines elements of heavy metal and hardcore punk?
Here's the most possible answer: Let's think step by step. Serianna is a band of metalcore genre, which is a fusion genre that combines elements of heavy metal and hardcore punk. So the answer is: Metalcore.

---

Question: Which band was formed first, Helium or Jack's Mannequin?
Proposed Answer: Let's think step by step. Helium was formed in 1991. Jack's Mannequin was formed in 2004. 1991 (Helium) < 2004 (Jack's Mannequin), so Helium was formed first. So the answer is: Helium.

What's the problem with the above answer?

1. Plausibility:

The answer should be chosen between "Helium" and "Jack's Mannequin", and the answer is "Helium", so it seems good.

2. Truthfulness:

Let's search the question in google:

> Search Query: Which band was formed first, Helium or Jack's Mannequin?
> Evidence: [Jack's Mannequin - Wikipedia] Jack's Mannequin was an American rock band formed in 2004, hailing from Orange County, California.
 
The evidence shows that Jack's Mannequin is a band formed in 2004.

We then find out when the band Helium was formed.

> Search Query: When was the band "Helium" formed?
> Evidence: [] Helium / Active from 1992

The evidence shows that Helium was formed in 1992, not 1991 in the proposed answer.

The proposed answer gives the correct date of formation for Jack's Mannequin, but wrong about Helium, it needs revision. Notably, the final answer "Helium" is still correct since 1992 < 2004.

Question: Which band was formed first, Helium or Jack's Mannequin?
Here's the most possible answer: Let's think step by step. Helium was formed in 1992. Jack's Mannequin was formed in 2004. 1992 (Helium) < 2004 (Jack's Mannequin), so Helium was formed first. So the answer is: Helium.

---

Question: What year did Maurice win the award given to the  "player judged most valuable to his team" in the NHL?
Proposed Answer: Let's think step by step. Maurice Richard won the Hart Memorial Trophy in 1947. So the answer is: 1947.

What's the problem with the above answer?

1. Plausibility:

The question ask for a year, and the answer "1947" is a year. So it's plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: What year did Maurice win the award given to the  "player judged most valuable to his team" in the NHL? site: wikipedia.org
> Evidence: [List of National Hockey League awards - Wikipedia] Awarded to the "player judged most valuable to his team". The original trophy was donated to the league by Dr. David A. Hart, father of coach Cecil Hart.

The evidence does not provide information about what the award is and Maurice won the award in which year.

Let's search the proposed anwer in google:
> Search Query: Maurice Richard won the Hart Memorial Trophy in 1947.
> Evidence: [Maurice Richard - Wikipedia] He won the Hart Trophy as the NHL's most valuable player in 1947...

The evidence shows that Maurice Richard won the Hart Trophy as the NHL's most valuable player in 1947, which supports the answer.

Above all, the proposed answer is plausible, and the evidence support all its claims.

Question: What year did Maurice win the award given to the  "player judged most valuable to his team" in the NHL?
Here's the most possible answer: Let's think step by step. Maurice Richard won the Hart Memorial Trophy in 1947. So the answer is: 1947.

---

Question: Are John H. Auer and Jerome Robbins both directors?
Proposed Answer: Let's think step by step. John H. Auer is a director, producer, and screenwriter. Jerome Robbins is a director, producer, choreographer, and dancer.  So the answer is: Jerome Robbins.

What's the problem with the above answer?

1. Plausibility:

This is a yes/no question, the proposed answer is "Jerome Robbins", which does not clearly answer whether they are both directors.
The answer should be a clear "yes" or "no".

2. Truthfulness:

Let's search the question in google:

> Search Query: Are John H. Auer and Jerome Robbins both directors?
> Evidence: [A history of Jerome Robbins at PNB - Pacific Northwest Ballet] Robbins retained the title of associate artistic director until approximately 1963, ... Ballets: USA, from the late 1950s to the late 1960s.

The evidence suggests Jerome Robbins is a director.

We then need to verify if John H. Auer is a director.

> Search Query: Is John H. Auer a director? site: wikipedia.org
> Evidence: [John H. Auer - Wikipedia] Auer was a Hungarian-born child actor who, on coming to the Americas in 1928, became a movie director and producer, initially in Mexico but, from the early 1930s, in Hollywood.

The evidence suggests that John H. Auer is an actor, director and producer.

Therefore, both John H. Auer and Jerome Robbins are directors.

Question: Are John H. Auer and Jerome Robbins both directors?
Here's the most possible answer: Let's think step by step. John H. Auer is an actor, director and producer. Jerome Robbins is a director. Therefore, Both John H. Auer and Jerome Robbins are directors. So the answer is: Yes.

---

Question: Which artist did Anthony Toby "Tony" Hiller appear with that liked showering himself (and others) with confetti?
Proposed Answer: Let's think step by step. Anthony Toby "Tony" Hiller is a songwriter and record producer. He appeared with Brotherhood of Man. Brotherhood of Man liked showering himself (and others) with confetti. So the answer is: Brotherhood of Man.

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of an artist, and the answer "Brotherhood of Man" looks like a name.

2. Truthfulness:

Let's search the question in google:

> Search Query: Which artist did Anthony Toby Tony Hiller appear with that liked showering himself (and others) with confetti?
> Evidence: [Untitled] Without you: The tragic story of Badfinger|Dan Matovina, The Military Orchid and Other Novels|Jocelyn Brooke, Looking at Lisp (Micro computer books)|Tony ...

The evidence does not provide any useful information about the question.

Let's search the proposed answer in google:

> Search Query: Anthony Toby Tony Hiller appeared with Brotherhood of Man.
> Evidence: [Tony Hiller - Wikipedia] He was best known for writing and/or producing hits for Brotherhood of Man, including "United We Stand" (1970) and "Save Your Kisses for Me" (1976).

The evidence suggests Tony Hiller producing hits Brotherhood of Man.

> Search Query: Brotherhood of Man liked showering himself (and others) with confetti.
> Evidence: [New York showers confetti, love on U.S. women's soccer team] Discover Thomson Reuters By Matthew Lavietes... celebrating its World Cup triumph and hailing the players’ emergence as icons of women’s rights.

The evidence does not mention Brotherhood of Man. We need to check who is the correct artist that liked showering himself (and others) with confetti.

> Search Query: Which artist liked showering himself (and others) with confetti?
> Evidence: [Rip Taylor - Wikipedia] Charles Elmer " Rip " Taylor Jr. was an American actor and comedian, known for his exuberance and flamboyant personality, including his wild moustache, toupee, and his habit of showering himself (and others)

The evidence suggests that the artist that liked showering himself is Charles Elmer "Rip" Taylor Jr., not "Brotherhood of Man" as in the proposed answer.

We can further check if Rip Taylor appeared with Anthony Toby "Tony" Hiller.

> Search Query: Which artist appeared with Anthony Toby "Tony" Hiller?
> Evidence: [Tony Hiller - Wikipedia] He was best known for writing and/or producing hits for Brotherhood of Man , including " United We Stand " (1970) and " Save Your Kisses for Me " (1976). Biography [ edit ]

> Search Query: Did Tony Hiller appear with Rip Taylor?
> Evidence: [Tony Hiller - Wikipedia] The Hiller Brothers appeared with many performers of the time including Alma Cogan , Tommy Cooper , Val Doonican , Matt Monro , The Shadows , Bernard Manning , Kathy Kirby , Roger Whittaker , Rip Taylor , Gene Vincent , Lance Percival , Tessie O'Shea...

The evidence shows that Tony Hiller appeared with Rip Taylor.

The answer needs major revision.

Question: Which artist did Anthony Toby "Tony" Hiller appear with that liked showering himself (and others) with confetti?
Here's the most possible answer: Let's think step by step. The artist that liked showering himself with confetti is Rip Taylor. Anthony Toby "Tony" Hiller is a songwriter and record producer, he appeared with Rip Taylor. So the answer is: Rip Taylor.

---

Question: Who won the 2016 Russian national silver medal with another Russian ice dancer born 29 April 1995?
Proposed Answer: Let's think step by step. The 2016 Russian national silver medal was won by Alexandra Stepanova and Ivan Bukin. Alexandra Stepanova was born 29 April 1995. Ivan Bukin was born 10 April 1993. So the answer is: Ivan Bukin.

What's the problem with the above answer?

1. Plausibility:

The question asks for a name, and the answer "Ivan Bukin" is a name. So it's plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: Who won the 2016 Russian national silver medal with another Russian ice dancer born 29 April 1995? site: wikipedia.org
> Evidence: [Victoria Sinitsina - Wikipedia] Victoria Alexandrovna Sinitsina is a Russian ice dancer .

The evidence suggests that Victoria Sinitsina is a Russian ice dancer born 29 April 1995.

Let's search the proposed answer in google:

> Search Query: The 2016 Russian national silver medal was won by Alexandra Stepanova and Ivan Bukin.
> Evidence: [Alexandra Stepanova - Wikipedia] They won silver at the 2012 Russian Junior Championships. Stepanova/Bukin then competed at the 2012 World Junior Championships and won the silver medal.

From the evidence, Stepanova/Bukin won silver at the 2012 Russian Junior Championships, not 2016.

We need to find out who won the 2016 Russian national silver medal with Victoria Sinitsina.

> Search Query: Who won the 2016 Russian national silver medal with Victoria Sinitsina?
> Evidence: [Nikita Katsalapov - Wikipedia] In December, Sinitsina/Katsalapov won the silver medal behind Bobrova/Soloviev at the 2016 Russian Championships in Yekaterinburg.

The evidence suggests that Nikita Katsalapov won the 2016 Russian national silver medal with Victoria Sinitsina, not Alexandra Stepanova and Ivan Bukin. The answer provided in the proposed answer is incorrect.

Question: Who won the 2016 Russian national silver medal with another Russian ice dancer born 29 April 1995?
Here's the most possible answer: Let's think step by step. The 2016 Russian national silver medal in ice dancing was won by Victoria Sinitsina and Nikita Katsalapov. Victoria Sinitsina was born on April 29, 1995. So the answer is: Nikita Katsalapov.

---"""