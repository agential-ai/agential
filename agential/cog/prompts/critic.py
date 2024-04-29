"""CRITIC prompts and fewshot examples."""

# ======================================================================== AMBIGNQ ======================================================================== #

CRITIC_INSTRUCTION_AMBIGNQ = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """


AMBIGNQ_FEWSHOT_EXAMPLES_COT = """Q: What airport is closest to Palm Springs?
A: The nearest airport to Palm Springs is Indio/Palm Springs (PSP) Airport which is 2.1 miles away. So the answer is: Palm Springs International Airport

Q: What degree did Martin Luther King get?
A: Martin Luther King earned his Bachelor of Divinity degree from Crozer Theological Seminary, followed by a doctorate in Systematic Theology from Boston University. So the answer is: Bachelor of Divinity

Q: What countries does the Niger river flow through?
A: The Niger river runs in a crescent through Libya, Mali, Niger, on the border with Benin and then through Nigeria. So the answer is: Libya

Q: What type of currency is used in Puerto Rico?
A: Puerto Rico is a territory of the United States and uses the U.S. dollar. So the answer is: United States dollar

Q: Who played kitt in knight rider?
A: kitt was voice most often by William Daniels. So the answer is: William Daniels"""


AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT = """Q: When did men's figure skating become a summer Olympic sport?
A: 1908

Q: When did the all india workers and peasants party came in to existence?
A: November 1925

Q: Flight that went down in the hudson river?
A: US Airways Flight 1549

Q: Where are most of the world's earthquakes located?
A: Rim of Fire

Q: Csi when do grissom and sara reunite?
A: series finale

Q: Who did rizzo go to the dance with?
A: Kenickie Murdoch

Q: What country won the womens curling winter olympics 2018?
A: Sweden

Q: Who plays barnaby's wife in midsomer murders series 1-13?
A: Jane Wymark

Q: Who plays 7-9 year old Nelson Mandela in the movie Mandela: Long Walk to Freedom?
A: Siza Pini

Q: When did the movie coyote ugly come out?
A: August 4, 2000"""


AMBIGNQ_FEWSHOT_EXAMPLES_REACT = """Question: When did men's figure skating become a summer Olympic sport?
Thought 1: Let's search the question in google
Action 1: Search[When did men's figure skating become a summer Olympic sport?]
Observation 1: [Figure skating at the Olympic Games - Wikipedia] Figure skating was first contested in the Olympic Games at the 1908 Summer Olympics . Since 1924, the sport has been a part of the Winter Olympic Games .
Thought 2: The evidence suggests Figure skating became an Olympic sport at the 1908 Summer Olympics, and has been a part of the Winter Olympic Games in 1924.
Action 2: Finish[1908]

---

Question: When was the last time the USA men's national soccer team missed the World Cup?
Thought1: Let's search the question in google
Action 1: Search[When was the last time the USA men's national soccer team missed the World Cup? site: wikipedia.org]
Observation 1: [United States at the FIFA World Cup - Wikipedia] The United States participated in every World Cup from 1990 through 2014, but did not qualify in 2018, marking first time the team had missed a World Cup since 1986. They returned to the World Cup by qualifying for the 2022 edition .
Thought 2: The evidence suggests that the USA men's national soccer team did not qualify for the World Cup in 2018, but qualify for 2022, 2018 > 1986, so the last time the USA men's national soccer team missed the World Cup was in 2018.
Action 2: Finish[2018]

---

Question: What does barium do in a ct scan?
Thought 1: Let's search the question in google
Action 1: Search[What does barium do in a ct scan? site: wikipedia.org]
Observation 1: [Barium sulfate suspension - Wikipedia] Barium sulfate suspension, often simply called barium, is a contrast agent used during X-rays. Specifically it is used to improve visualization of the gastrointestinal tract (esophagus, stomach, intestines) on plain X-ray or computed tomography.
Thougth 2: The evidence suggests that barium is used to "improve visualization of the gastrointestinal tract".
Action 2: Search[Why barium used in CT scans?]
Observation 2: [What To Expect Before Your CT Scan - St. Elizabeth's Medical Center] The barium helps to highlight body areas for the CT scan. If you are having a scan of a body part other than the abdomen you should arrive 30 minutes before your appointed time.
Thougth 3: The evidence suggests that barium helps "highlight body areas" for the CT scan.
Action 3: Finish[to improve visualization of the gastrointestinal tract]

---

Question: Where was the fort located at which the first shot of the civil war was fired?
Thought 1: Let's search the question in google
Action 1: Search[Where was the fort located at which the first shot of the civil war was fired?]
Observation 1: [Civil War Begins - U.S. Senate] April 12, 1861 At 4:30 a.m. on April 12, 1861, Confederate troops fired on Fort Sumter in South Carolina’s Charleston Harbor. Less than 34 hours later, Union forces surrendered.
Thought 2: The evidence suggests that the first shot of the Civil War was fired at Fort Sumter. Fort Sumter is located in South Carolina’s Charleston Harbor.
Action 2: Finish[Charleston Harbor, South Carolina]

---

Question: When did nando's come to the uk?
Thought 1: Let's search the question in google
Action 1: Search[When did nando's come to the uk? site: wikipedia.org]
Observation 1: [Nando's - Wikipedia] Nando's opened its first restaurants within the United Kingdom, in 1992, in the west London suburbs of Ealing and Earls Court, initially focusing on takeaway food.
Thougth 2: The evidence suggests that Nando's first opened in the UK in 1992.
Action 2: Finish[1992]

---

Question: Who plays wilmas mom in the flintstones movie?
Thought 1: Let's search the question in google
Action 1: Search[Who played Wilma's mom in the flintstones movie? site: wikipedia.org]
Observation 1: [Elizabeth Perkins - Wikipedia] Elizabeth Ann Perkins (born November 18, 1960 ) is an American actress.
Thought 2: The evidence does not mention Wilma's mom in the flintstones movie.
Action 2: Search[Who played Wilma's mom in the flintstones movie?]
Observation 2: [] The Flintstones / Wilma Flintstone / Mother / Played by Elizabeth Taylor
Thought 3: The evidence shows that Elizabeth Taylor played the role of Wilma's mother.
Action 3: Finish[Elizabeth Taylor]

---"""


CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ = """{examples}
(END OF EXAMPLES)

Question: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

{critique}"""


AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC = """Question: When did men's figure skating become a summer Olympic sport?
Proposed Answer: Men's figure skating has never been a summer Olympic sport. It has been a part of the Winter Olympics since the first Winter Olympics in 1924. So the answer is: never

What's the problem with the above answer?

1. Plausibility:

The question asks for the time men's figure skating become a summer Olympic sport, and the answer "never" does not provide a time. So it's not plausible. The answer should be a time, like year or date.

2. Truthfulness:

Let's search the question in google:

> Search Query: When did men's figure skating become a summer Olympic sport?
> Evidence: [Figure skating at the Olympic Games - Wikipedia] Figure skating was first contested in the Olympic Games at the 1908 Summer Olympics . Since 1924, the sport has been a part of the Winter Olympic Games .

The evidence suggests Figure skating became an Olympic sport at the 1908 Summer Olympics, and has been a part of the Winter Olympic Games in 1924.

The answer is wrong by saying that men's figure skating has never been a part of the Summer Olympics.

Question: When did men's figure skating become a summer Olympic sport?
Here's the most possible answer: Men's figure skating became a part of the Olympic Games at the 1908 Summer Olympics, and it has been a part of the Winter Olympic Games since its inception in 1924. So the answer is: 1908.

---

Question: When is the date the last time washington won stanley cup?
Proposed Answer: The Washington Capitals won the Stanley Cup in 2018. So the answer is: 2018

What's the problem with the above answer?

1. Plausibility:

The question asks for a date, and the answer is "2018", which is a year but not a specific date. So it's not plausible. The answer should be a specific date.

Question: When is the date the last time washington won stanley cup?
Here's the most possible answer: The Washington Capitals won the Stanley Cup on June 7, 2018. So the answer is: June 7, 2018.

---

Question: When was the last time the USA men's national soccer team missed the World Cup?
Proposed Answer: The last time the USA men's national soccer team missed the World Cup was in 1986. So the answer is: 1986

What's the problem with the above answer?

1. Plausibility:

The question asks for a year, and the answer is "1986", which is a year. So it's plausible.

2. Truthfulness:

Let's search the question and proposed answer in google:

> Search Query: When was the last time the USA men's national soccer team missed the World Cup? site: wikipedia.org
> Evidence: [United States at the FIFA World Cup - Wikipedia] The United States participated in every World Cup from 1990 through 2014, but did not qualify in 2018, marking first time the team had missed a World Cup since 1986. They returned to the World Cup by qualifying for the 2022 edition .

> Search Query: The last time the USA men's national soccer team missed the World Cup was in 1986
> Evidence: [United States at the FIFA World Cup - Wikipedia] The United States participated in every World Cup from 1990 through 2014, but did not qualify in 2018, marking first time the team had missed a World Cup since 1986.

The evidence suggests that the USA men's national soccer team did not qualify for the World Cup in 2018, but qualify for 2022, 2018 > 1986, so the last time the USA men's national soccer team missed the World Cup was in 2018, not in 1986, which contradicts the proposed answer.

Above all, the USA men's national soccer team did miss the World Cup in 1986, but not the last time, the last time was in 2018.

Question: When was the last time the USA men's national soccer team missed the World Cup?
Here's the most possible answer: The last time the USA men's national soccer team missed the World Cup was in 2018. So the answer is: 2018.

---

Question: Where was tin pan alley in new york city?
Proposed Answer: Tin Pan Alley was a nickname for the collection of New York City music publishers and songwriters who dominated the popular music of the United States in the late 19th century and early 20th century. So the answer is: New York City

What's the problem with the above answer?

1. Plausibility:

The question asks for the location of Tin Pan Alley in new york specifically, and the answer is "New York City", which is too broad and does not provide a specific location. So it's not plausible. The answer should be a specific location in new york.

Here's the most possible answer: Tin Pan Alley was located on West 28th Street between Fifth and Sixth Avenues in the Flower District of Manhattan, New York City. So the answer is: West 28th Street between Fifth and Sixth Avenues in Manhattan.

---

Question: What does barium do in a ct scan?
Proposed Answer: Barium is used in CT scans to help highlight the digestive system. So the answer is: highlight the digestive system

What's the problem with the above answer?

1. Plausibility:

The question asks for the function of barium in a CT scan, and the answer is "highlight the digestive system", which is a function. So it's plausible.

2. Truthfulness:

Let's search the proposed answer:

> Search Query: Barium is used in CT scans to help highlight the digestive system
> Evidence: [Barium Sulfate (Enhancer)] Barium sulfate works by coating the inside of your esophagus, stomach, or intestines which allows them to be seen more clearly on a CT scan or other radiologic (x-ray) examination.

According to the evidence, the proposed answer is not completely accurate.

Let's search the question in google:

> Search Query: What does barium do in a ct scan? site: wikipedia.org
> Evidence: [Barium sulfate suspension - Wikipedia] Barium sulfate suspension, often simply called barium, is a contrast agent used during X-rays. Specifically it is used to improve visualization of the gastrointestinal tract (esophagus, stomach, intestines) on plain X-ray or computed tomography.

The evidence suggests that barium is used to "improve visualization of the gastrointestinal tract", which includes the digestive system. Therefore, the proposed answer should be more specific by staing "improve visualization".

> Search Query: Why barium used in CT scans?
> Evidence: [What To Expect Before Your CT Scan - St. Elizabeth's Medical Center] The barium helps to highlight body areas for the CT scan. If you are having a scan of a body part other than the abdomen you should arrive 30 minutes before your appointed time.

The evidence suggests that barium helps "highlight body areas" for the CT scan, not specifically the digestive system. However, it is possible that barium is used to highlight the digestive system in many cases, so the proposed answer is not complete.

Conclusion: While the answer "highlight the digestive system" is a common usage of barium in CT scans, it is not a complete description of barium's function. A more complete answer would be "to improve visualization of the gastrointestinal tract."

Question: What does barium do in a ct scan?
Here's the most possible answer: Barium is used in CT scans to improve visualization of the gastrointestinal tract, including the digestive system. So the answer is: to improve visualization of the gastrointestinal tract.

---

Question: Where was the fort located at which the first shot of the civil war was fired?
Proposed Answer: The first shot of the Civil War was fired at Fort Sumter in Charleston, South Carolina. So the answer is: Fort Sumter

What's the problem with the above answer?

1. Plausibility:

The question asks for the location of the fort, and the answer is "Fort Sumter", which may be a location. So it's plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: Where was the fort located at which the first shot of the civil war was fired?
> Evidence: [Civil War Begins - U.S. Senate] April 12, 1861 At 4:30 a.m. on April 12, 1861, Confederate troops fired on Fort Sumter in South Carolina’s Charleston Harbor. Less than 34 hours later, Union forces surrendered.

The evidence suggests that the first shot of the Civil War was fired at Fort Sumter, which is consistent with the proposed answer.

We then need to check whether Fort Sumter is located in Charleston, South Carolina.

> Search Query: Where is Fort Sumter located?
> Evidence: [Fort Sumter and Fort Moultrie National Historical Park (U.S. National ...] Fort Sumter is located in the middle of Charleston Harbor, and is only accessible by ferry rides through Fort Sumter Tours.

Conclusion: From the above evidence we know that the first shot of the Civil War was fired at Fort Sumter, and Fort Sumter is located in Charleston, the reasoning process is correct. However, the proposed answer should specify the location "Charleston" rather than only state "Fort Sumter".

Question: Where was the fort located at which the first shot of the civil war was fired?
Here's the most possible answer: The first shot of the Civil War was fired at Fort Sumter, which is located in Charleston, South Carolina. So the answer is: Charleston, South Carolina.

---

Question: When did nando's come to the uk?
Proposed Answer: Nando's first opened in the UK in 1992. So the answer is: 1992

What's the problem with the above answer?

1. Plausibility:

The question asks for a time, and the answer is "1992", which is a year. So it's plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: When did nando's come to the uk? site: wikipedia.org
> Evidence: [Nando's - Wikipedia] Nando's opened its first restaurants within the United Kingdom, in 1992, in the west London suburbs of Ealing and Earls Court, initially focusing on takeaway food.

The evidence suggests that Nando's first opened in the UK in 1992, which is consistent with the proposed answer. We can provide more detailed information in the answer.

Question: When did nando's come to the uk?
Here's the most possible answer: Nando's first opened its restaurants within the United Kingdom in 1992, specifically in the west London suburbs of Ealing and Earls Court. So the answer is: 1992

---

Question: Who plays wilmas mom in the flintstones movie?
Proposed Answer: Wilma's mom is played by Elizabeth Perkins in the 1994 live-action film The Flintstones. So the answer is: Elizabeth Perkins

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the actor who plays wilmas mom, and the answer is "Elizabeth Perkins", which is a name. So it's plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: Who played Wilma's mom in the flintstones movie? site: wikipedia.org
> Evidence: [Elizabeth Perkins - Wikipedia] Elizabeth Ann Perkins (born November 18, 1960 ) is an American actress.

The evidence does not mention Wilma's mom in the flintstones movie.

Let's search the proposed answer:

> Search Query: Wilma's mom is played by Elizabeth Perkins in the 1994 live-action film The Flintstones.
> Evidence: [The Flintstones (film) - Wikipedia] The film stars John Goodman as Fred Flintstone, Rick Moranis as Barney Rubble, Elizabeth Perkins as Wilma Flintstone, and Rosie O'Donnell as Betty Rubble, along with Kyle MacLachlan as Cliff Vandercave, a villainous executive-vice president of Fred's company, Halle Berry as Sharon Stone, his seductive secretary, and Elizabeth Taylor (in her final theatrical film appearance), as Pearl Slaghoople, Wilma's mother.

The evidence shows that Elizabeth Perkins did appear in The Flintstones movie as Wilma Flintstone, but not as Wilma's mother. And Elizabeth Taylor played as Pearl Slaghoople, the role of Wilma's mother in The Flintstones movie.

> Search Query: Who played Wilma's mom in the flintstones movie?
> Evidence: [] The Flintstones / Wilma Flintstone / Mother / Played by Elizabeth Taylor

The evidence shows that Elizabeth Taylor played the role of Wilma's mother, which contradicts the "Elizabeth Perkins" in the proposed answer.

Considering all above evidence, we need to correct the answer.

Question: Who plays wilmas mom in the flintstones movie?
Here's the most possible answer: Elizabeth Taylor played the role of Wilma's mother (ie., Pearl Slaghoople) in the 1994 live-action film The Flintstones. So the answer is: Elizabeth Taylor.

---"""


# ======================================================================== HOTPOTQA ======================================================================== #


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


# ======================================================================== TRIVIAQA ======================================================================== #


CRITIC_INSTRUCTION_TRIVIAQA = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """


TRIVIAQA_FEWSHOT_EXAMPLES_COT = """Q: Mendelssohn's 'Wedding March' was. originally written as incidental music for which Shakespeare play in 1842? 
A: Mendelssohn's 'Wedding March' was originally written as incidental music for A Midsummer Night's Dream in 1842. So the answer is: A Midsummer Night's Dream.

Q: Christ in the House of his Parents"" is one of the best known paintings of which artist?" 
A: "Christ in the House of his Parents" is a painting by John Everett Millais. So the answer is: John Everett Millais.

Q: Who designed the National Theatre building on the South Bank in London ? 
A: The National Theatre building on the South Bank in London was designed by Denys Lasdun. So the answer is: Denys Lasdun.

Q: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics? 
A: The only American skier who won the gold medal in the Men's Combined at the 2010 Winter Olympics is Bode Miller. So the answer is: Bode Miller.

Q: Famous composer, Handel, originally studied what? 
A: George Frideric Handel initially studied law at the University of Halle. So the answer is: Law.

Q: Which great philosopher corresponded with Queen Christina of Sweden in his final years and died in 1650 in Stockholm where he had been invited as a teacher for her? 
A: René Descartes is a great philosopher who corresponded with Queen Christina of Sweden in his final years and died in 1650 in Stockholm where he had been invited as a teacher for her. So the answer is: René Descartes."""


TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT = """Q: Mendelssohn's 'Wedding March' was. originally written as incidental music for which Shakespeare play in 1842? 
A: A Midsummer Night's Dream

Q: Christ in the House of his Parents"" is one of the best known paintings of which artist?" 
A: John Millais

Q: Who designed the National Theatre building on the South Bank in London ? 
A: Sir Denys Lasdun

Q: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?
A: Bodie Miller

Q: Famous composer, Handel, originally studied what? 
A: Law

Q: Which great philosopher corresponded with Queen Christina of Sweden in his final years and died in 1650 in Stockholm where he had been invited as a teacher for her? A: René Decartes

Q: Who was the female member of Britain's gang of four? 
A: Baroness Williams

Q: An icosahedron has how many faces? 
A: twenty

Q: What chemical makes hot peppers hot? 
A: Capscaicin

Q: What term applies to an engine comprising two banks of four cylinders inclined towards each other, with a common crankshaft? 
A: V8 engine"""

TRIVIAQA_FEWSHOT_EXAMPLES_REACT = """Question: Which innovation for the car was developed by Prince Henry of Prussia in 1911? 
Thought 1: Let's search the question in google
Action 1: Search[Which innovation for the car was developed by Prince Henry of Prussia in 1911? site: wikipedia.org]
Observation 1: [Prince Henry of Prussia (1862–1929) - Wikipedia] Henry was interested in motor cars as well and supposedly invented a windshield wiper and, according to other sources, the car horn.
Thought 2: The evidence suggests that Prince Henry of Prussia invented a windshield wiper and the car horn. 
Action 2: Search[When did Prince Henry of Prussia invented a windshield wiper and the car horn?]
Observation 2: [110 years ago: windscreen wiper patent for Prince Henry of Prussia] Quite apart from a member of the German aristocracy: it was Prince Henry of Prussia (1862-1929) who was granted the first German patent about the windscreen wiper on 24. March 1908. 
Thought 3: According to the evidence, Prince Henry of Prussia was granted the first German patent for the windshield wiper.
Action 3: Finish[windshield wiper]

-----
Question: When did the founder of Jehovah's Witnesses say the world would end?
Thought 1: Let's search the question in google
Action 1: Search[When did the founder of Jehovah's Witnesses say the world would end? site: wikipedia.org]
Observation 1: [Eschatology of Jehovah's Witnesses - Wikipedia] In 1911, Russell wrote that October 1914 would witness the "full end" of Babylon, or nominal Christianity, "utterly destroyed as a system". At first, the hopes ... 
Thought 2: The evidence shows that the founder of Jehovah's Witnesses predicted the world would end in October 1914
Action 2: Finish[1914]

------
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

------
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

CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA = """{examples}
(END OF EXAMPLES)

Question: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

{critique}"""


TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC = """Question: Which innovation for the car was developed by Prince Henry of Prussia in 1911? 

Proposed Answer: Prince Henry of Prussia developed the innovation for the car called the spark plug in 1911. So the answer is: Spark plug.

What's the problem with the above answer?

1.Plausibility:

The question asks for the name of the innovation, and the answer is "Spark plug", which is a name. So it's plausible.

2.Truthfulness:

Let's search the question in google:

> Search Query: Which innovation for the car was developed by Prince Henry of Prussia in 1911? site: wikipedia.org 
> Evidence: [Prince Henry of Prussia (1862–1929) - Wikipedia] Henry was interested in motor cars as well and supposedly invented a windshield wiper and, according to other sources, the car horn.

The evidence suggests that Prince Henry of Prussia invented a windshield wiper and the car horn.

> Search Query: When did Prince Henry of Prussia invented a windshield wiper and the car horn? 
> Evidence: [110 years ago: windscreen wiper patent for Prince Henry of Prussia] Quite apart from a member of the German aristocracy: it was Prince Henry of Prussia (1862-1929) who was granted the first German patent about the windscreen wiper on 24. March 1908.

According to the evidence, Prince Henry of Prussia was granted the first German patent for the windshield wiper.

Let's check the proposed answer:

> Search Query: Prince Henry of Prussia developed the innovation for the car called the spark plug in 1911.
> Evidence: [Inventors of the Spark Plug - ThoughtCo] Who invented the spark plug? Several notable inventors had a hand in developing igniters for the internal combustion engine. Missing: Prince Henry Prussia innovation 1911.

The evidence does not provide information about Prince Henry Prussia's innovation, let's search more evidence to verify the answer:

> Search Query: What's Prince Henry of Prussia's innovation for the car in 1911?
> Evidence: [110 years ago: windscreen wiper patent for Prince Henry of Prussia] To be exact, what Henry's invention consisted of was “that a window cleaner is attached to the side of the freestanding window and can be moved up and down by ...

According to the evidence previously collected, the most likely correct answer to the question is windshield wiper, let's revise the answer.

Question: Which innovation for the car was developed by Prince Henry of Prussia in 1911? 

Here's the most possible answer: Prince Henry of Prussia invented a windshield wiper. So the answer is: windshield wiper.

------

Question: When did the founder of Jehovah's Witnesses say the world would end? 
Proposed Answer: The founder of Jehovah's Witnesses said the world would end in 1914. So the answer is: 1914.

What's the problem with the above answer?

1.Plausibility:

The question asks for a time, and the answer is "1914", which is a year. So it's plausible.

2.Truthfulness:

Let's search the question in google:

> Search Query: When did the founder of Jehovah's Witnesses say the world would end? site: wikipedia.org
> Evidence: [Eschatology of Jehovah's Witnesses - Wikipedia] In 1911, Russell wrote that October 1914 would witness the "full end" of Babylon, or nominal Christianity, "utterly destroyed as a system". At first, the hopes ...

The evidence shows that the founder of Jehovah's Witnesses predicted the world would end in October 1914, which support the "1914" in the proposed answer.

Question: When did the founder of Jehovah's Witnesses say the world would end? 
Here's the most possible answer: The founder of Jehovah's Witnesses said the world would end in 1914. So the answer is: 1914.

Question: Three people have featured on the reverse of the £5 note, the two most recent are George Stephenson and Elizabeth Fry, who was the first? 
Proposed Answer: The first person to feature on the reverse of the £5 note was Sir Isaac Newton. So the answer is: Sir Isaac Newton.

What's the problem with the above answer?

1.Plausibility:

The question asks for the name of the first person to feature on the reverse of the £5 note, and the answer is "Sir Isaac Newton", which is a person's name. So it's plausible.

2.Truthfulness:

Let's search the question in google:
> Search Query: Three people have featured on the reverse of the £5 note, the two most recent are George Stephenson and Elizabeth Fry, who was the first? 
>  Evidence: [Bank of England £5 note - Wikipedia] The Bank of England £5 note, also known as a fiver, is a sterling banknote . It is the smallest denomination of banknote currently issued by the Bank of England.

The evidence is about £5 note, it has no information about the featured people we want to find out.
> Search Query: Who was the first person to feature on the reverse of the £5 note? site: wikipedia.org
> Evidence: [Bank of England £5 note - Wikipedia] The old paper note, first issued in 2002 and bearing the image of prison reformer Elizabeth Fry on the reverse, was phased out and ceased to be legal tender after 5 May 2017.

The evidence only mentions Elizabeth Fry, and from the question, we know that she was not the first person to feature on the reverse of the £5 note.

Let's check the proposed answer:

> Search Query: The first person to feature on the reverse of the £5 note was Sir Isaac Newton
> Evidence: [Featured Note of the Month: May 2019 | PMG - Paper Money Guaranty...] What makes it special? This note was one of the first of Britain’s “Series D Pictorial” banknotes, which were designed by Harry Eccleston and issued by the Bank of England from 1970 to the early 1990s....Though the obverse of each note features a portrait of Queen Elizabeth II (standard of all modern British currency), the reverse sides show various major figures in British history, such as William Shakespeare, Sir Walter Raleigh and Sir Isaac Newton.

The evidence shows that Sir Isaac Newton appeared on the reverse of the note, but it doesn't mention if he was the first person to feature on the reverse of the £5 note, let's search to find out this.

> Search Query: Did Isaac Newton appear on the reverse of the 5 pound note? 
> Evidence: [History of the use of the single crossbar pound sign on Bank of ...] The single crossbar on the £1 note was introduced in 1978 with the 'D' Series note depicting Isaac Newton on the reverse (the 'C' series did not have a pound sign)

The evidence shows that Sir Isaac Newton appeared on the reverse of the £1 note, instead of the £5 note.

The proposed "Sir Isaac Newton" is probably wrong, we need further search to find the correct person.

> Search Query: People have featured on the reverse of the £5 note: George Stephenson and Elizabeth Fry and? 
> Evidence: [Historical British figures featured on banknotes (1970-2017)] £5 – The Duke of Wellington, George Stephenson, Elizabeth Fry, Sir Winston Churchill £10 – Florence Nightingale, Charles Dickens, Charles Darwin £20 – William Shakespeare, ...

The evidence shows that the Duke of Wellington and Sir Winston Churchill were also featured on £5.

In summary, our collected evidence suggest that Sir Isaac Newton was not the first person to feature on the reverse of the £5 note, we need to give the most possible answer.

Question: Three people have featured on the reverse of the £5 note, the two most recent are George Stephenson and Elizabeth Fry, who was the first? 
Here's the most possible answer: The first person to feature on the reverse of the £5 note was Duke of Wellington. So the answer is: Duke of Wellington
-------

Question: What state had its bi-centenary of joining the Union a year after North Carolina? 
Proposed Answer: Rhode Island had its bi-centenary of joining the Union a year after North Carolina. So the answer is: Rhode Island.

What's the problem with the above answer?

Plausibility:
The question asks for the name of a state, and the answer is "Rhode Island", which is a state. So it's plausible.

Truthfulness:
Let's search the question in google:

> Search Query: What state had its bi-centenary of joining the Union a year after North Carolina? site: wikipedia.org
> Evidence: [List of U.S. states by date of admission to the Union - Wikipedia] 24. Missouri, August 10, 1821 · (admitted) ; 25. Arkansas, June 15, 1836 · (admitted)

The evidence does not provide information about the state had its bi-centenary of joining the Union a year after North Carolina.

Let's check the proposed answer:
Search Query: Rhode Island had its bi-centenary of joining the Union a year after North Carolina.
Evidence: [Rhode Island's Contributions to the Founding of the United States ...] On May 29, 1790, Rhode Island ratified the Constitution by a vote of 34 to 32, the narrowest margin of any state.

The evidence shows that Rhode Island ratified the Constitution in 1790.

To answer the question, we need to find the state joining the Union a year after North Carolina.

> Search Query: Which state joined the Union a year after North Carolina? site: wikipedia.org 
> Evidence: [List of U.S. states by date of admission to the Union - Wikipedia...] ^ This list does not account for the secession of 11 states (Virginia, North Carolina, South Carolina, Georgia, Florida, Alabama, Mississippi, Tennessee, ...

Not enough evidence, we need further search.

> Search Query: What state joined the Union a year after North Carolina? 
> Evidence: [States by Order of Entry into Union - Infoplease] Joining the Union | State | Entered Union | Year Settled | | North Carolina | Nov. 21, 1789 | 1660 | | Rhode Island | May 29, 1790 | 1636 | | Vermont | Mar. 4, 1791 | 1724 |

The evidence shows North Carolina entered Union in 1789, and Rhode Island entered Union in 1790, which is a year after North Carolina.
This support the proposed answer.

Question: What state had its bi-centenary of joining the Union a year after North Carolina?
Here's the most possible answer: Rhode Island had its bi-centenary of joining the Union a year after North Carolina. So the answer is: Rhode Island."""


# ======================================================================== SVAMP ======================================================================== #


CRITIC_INSTRUCTION_SVAMP = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """


SVAMP_FEWSHOT_EXAMPLES_DIRECT = """Q: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
A: 62

Q: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy for each egg?
A: 51

Q: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books. How many books did danny have at first?
A: 20

Q: There were 108 chickens and 20 sheeps at the farm, some of chickens and sheeps were sold. There are 87 chickens and 18 sheeps left now. How many chickens were sold?
A: 21

Q: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday. How many did Katty score on monday and wednesday?
A: 11

Q: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined. How many more girls than boys are in the Masquerade?
A: 6

Q: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now. How much ice creasm did Getty purchase?
A: 22"""


SVAMP_FEWSHOT_EXAMPLES_POT = """Q: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
Python code, return answer
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers - used_red_stickers
```
A: answer

Q: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy for each egg?
Python code, return answer
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars - discount_dollars
```
A: answer

Q: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books. How many books did danny have at first?
Python code, return answer
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now - num_books_bought_at_store
```
A: answer

Question: There were 108 chickens and 20 sheeps at the farm, some of chickens and sheeps were sold. There are 87 chickens and 18 sheeps left now. How many chickens were sold?
Python code, return answer
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before - num_chicken_now
```
A: answer

Question: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday. How many did Katty score on monday and wednesday?
Python code, return answer
```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday + num_goals_on_wednesday
```
A: answer

Question: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined.  How many more girls than boys are in the Masquerade?
Python code, return answer
```python
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
answer = total_girls - total_boys
```
A: answer

Question: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now.  How much ice creasm did Getty purchase?
Python code, return answer
```python
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
answer = total_ice_creams - num_ice_creams_bought_by_joseph
```
A: answer"""


CRITIC_CRITIQUE_INSTRUCTION_SVAMP = """{examples}
(END OF EXAMPLES)

Question: {question}
Execution: Done
Output: {answer}

What's the problem with the above code?

1. 

{critique}"""


SVAMP_FEWSHOT_EXAMPLES_CRITIC = """Question: Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. How much did Marco's strawberries weigh?
```python
num_strawberries_dad_picked = 11
num_strawberries_marco_picked = 30
answer = num_strawberries_dad_picked + num_strawberries_marco_picked
```
Execution: Done 
Output: answer = 41.0

What's the problem with the above code?

1. The weight of straberries should be positive, 41.0 > 0, it's reasonable.
2. Let's check the code:

> answer = num_strawberries_dad_picked + num_strawberries_marco_picked

The above code calculates the total weight of strawberries picked by both Marco and his dad, instead of finding the weight of strawberries picked by Marco alone.

According to the question, the total weight of strawberries picked by both Marco and his dad is 30 pounds, and Marco's dad's strawberries weighed 11 pounds. To find the weight of Marco's strawberries, we need to subtract the weight of his dad's strawberries from the total weight.

Here's a better solution:

```python
total_weight = 30
dad_weight = 11

# Calculate the weight of Marco's strawberries
marco_weight = total_weight - dad_weight
answer = marco_weight
```

---

Question: Zachary did 46 push-ups and 58 crunches in gym class today. David did 38 more push-ups but 62 less crunches than zachary. How many more crunches than push-ups did Zachary do?
```python
zachary_push_ups = 46
zachary_crunches = 58
zachary_diff = zachary_crunches - zachary_push_ups
answer = zachary_diff
```
Execution: Done 
Output: answer = 12

What's the problem with the above code?

1. The answer, 12, is a reasonable result.
2. Let's check the code:

>zachary_diff = zachary_crunches - zachary_push_ups

The code calculates the difference between the number of crunches and push-ups that Zachary did, it is correct.

The information about David is not relevant to the question and does not need to be considered in the calculation. There is no problem with the above code.

Here's a better solution:

```python
zachary_push_ups = 46
zachary_crunches = 58

# Calculate the difference between Zachary's crunches and push-ups
zachary_diff = zachary_crunches - zachary_push_ups
answer = zachary_diff
```

---

Question: Faye had 35 packs of pencils each one having 4 pencils. She was placing her pencils into rows with 2 pencils in each row. How many rows could she make?
```python
num_packs_of_pencils = 35
num_pencils_in_each_pack = 4
answer = num_packs_of_pencils * num_pencils_in_each_pack
```
Execution: Done 
Output: answer = 140.0

What's the problem with the above code?

1. The answer rows 140 > 0, it's reasonable.
2. Let's check the code:

>answer = num_packs_of_pencils * num_pencils_in_each_pack

The code calculates the total number of pencils Faye has, but it does not calculate the number of rows she can make. It needs to be modified to take into account the fact that she is placing 2 pencils in each row.

Here's a better solution:

```python
num_packs_of_pencils = 35
num_pencils_in_each_pack = 4

# Calculate the total number of pencils
total_pencils = num_packs_of_pencils * num_pencils_in_each_pack

# Calculate the number of rows with 2 pencils in each row
num_pencils_in_each_row = 2
num_rows = total_pencils // num_pencils_in_each_row
answer = num_rows
```

---

Question: Jack received 4 emails and sent 2 emails in the morning. He then received 6 emails and sent 8 emails in the afternoon. How many emails did Jack send in the day?
```python
num_emails_received_in_morning = 4
num_emails_sent_in_morning = 2
num_emails_received_in_afternoon = 6
num_emails_sent_in_afternoon = 8
answer = num_emails_sent_in_morning + num_emails_sent_in_afternoon
```
Execution: Done 
Output: answer = 10.0

What's the problem with the above code?

1. The answer, 10, is a reasonable result.
2. Let's check the code:

> answer = num_emails_sent_in_morning + num_emails_sent_in_afternoon

The code calculates the total number of emails Jack sent during the day by adding the number of emails he sent in the morning to the number of emails he sent in the afternoon. There is no problem with the above code.

Here's a better solution:

```python
num_emails_received_in_morning = 4
num_emails_sent_in_morning = 2
num_emails_received_in_afternoon = 6
num_emails_sent_in_afternoon = 8
answer = num_emails_sent_in_morning + num_emails_sent_in_afternoon
```

---"""
