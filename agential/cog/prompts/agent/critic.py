"""CRITIC prompts and fewshot examples."""

# ======================================================================== FEVER ======================================================================== #


CRITIC_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. 

{examples}
(END OF EXAMPLES)

Claim: {question}
A: """


CRITIC_CRITIQUE_INSTRUCTION_FEVER = """{examples}
(END OF EXAMPLES)

Claim: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

{critique}"""


FEVER_FEWSHOT_EXAMPLES_CRITIC = """Claim: Did Nikolaj Coster-Waldau work with the Fox Broadcasting Company?
Proposed Answer: REFUTES

What's the problem with the above answer?

1. Plausibility:

The answer "REFUTES" incorrectly negates the claim without supporting details.

2. Truthfulness:

> Search Query: Did Nikolaj Coster-Waldau work with Fox Broadcasting?
> Evidence: [Nikolaj Coster-Waldau - IMDb] Nikolaj Coster-Waldau appeared in the 2009 Fox television film Virtuality.

The evidence contradicts the proposed answer, confirming he did work with Fox in the television film Virtuality.

Claim: Did Nikolaj Coster-Waldau work with the Fox Broadcasting Company?
Here's the most possible answer: Yes, Nikolaj Coster-Waldau worked with the Fox Broadcasting Company as he appeared in the 2009 Fox television film Virtuality. So the answer is: SUPPORTS.

---

Claim: Is Stranger Things set in Bloomington, Indiana?
Proposed Answer: No

What's the problem with the above answer?

1. Plausibility:

The answer is correct but lacks specific details that verify the claim.

2. Truthfulness:

> Search Query: Setting of Stranger Things
> Evidence: Stranger Things is set in the fictional town of Hawkins, Indiana, not Bloomington.

Although the proposed answer is correct, it could be more informative by mentioning the specific setting.

Claim: Is Stranger Things set in Bloomington, Indiana?
Here's the most possible answer: No, Stranger Things is set in the fictional town of Hawkins, Indiana. So the answer is: REFUTES.

---

Claim: Did the song "Beautiful" by Christina Aguilera reach number two on the Billboard Hot 100 in 2003?
Proposed Answer: NOT ENOUGH INFO

What's the problem with the above answer?

1. Plausibility:

The answer "NOT ENOUGH INFO" is appropriate as it reflects the uncertainty due to incomplete information about the timing of the chart position.

2. Truthfulness:

> Search Query: Billboard Hot 100 position of "Beautiful" by Christina Aguilera in 2003
> Evidence: The song peaked at number two on the Billboard Hot 100, but the specific year it achieved this ranking was not directly specified in the sources found.

Given that the year 2003 is not verified in the available evidence, the proposed answer correctly reflects the uncertainty regarding the exact year of the chart position.

Claim: Did the song "Beautiful" by Christina Aguilera reach number two on the Billboard Hot 100 in 2003?
Here's the most possible answer: The song "Beautiful" by Christina Aguilera peaked at number two on the Billboard Hot 100, but there is no specific evidence confirming this occurred in 2003. So the answer is: NOT ENOUGH INFO."""


# ======================================================================== AMBIGNQ ======================================================================== #

CRITIC_INSTRUCTION_AMBIGNQ = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """


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
Here's the most possible answer: Elizabeth Taylor played the role of Wilma's mother (ie., Pearl Slaghoople) in the 1994 live-action film The Flintstones. So the answer is: Elizabeth Taylor."""


# ======================================================================== HOTPOTQA ======================================================================== #


CRITIC_INSTRUCTION_HOTPOTQA = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """


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
Here's the most possible answer: Let's think step by step. The 2016 Russian national silver medal in ice dancing was won by Victoria Sinitsina and Nikita Katsalapov. Victoria Sinitsina was born on April 29, 1995. So the answer is: Nikita Katsalapov."""


# ======================================================================== TRIVIAQA ======================================================================== #


CRITIC_INSTRUCTION_TRIVIAQA = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """


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

---

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

---

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


# ======================================================================== GSM8k ======================================================================== #


CRITIC_POT_INSTRUCTION_GSM8K = """# Write Python Code to solve the following questions. Store your result as a variable named 'answer'.

{examples}
(END OF EXAMPLES)

Question: {question}
# Python code, return answer"""


CRITIC_CRITIQUE_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Question: {question}
```python
{answer}
```
Execution: {execution_status} 
Output: answer = {code_answer}

What's the problem with the above code?

{critique}"""


GSM8K_FEWSHOT_EXAMPLES_CRITIC = """Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
# Python code
jason_lollipops_initial = 20
jason_lollipops_after = 12
denny_lollipops =  jason_lollipops_after - jason_lollipops_initial
answer = denny_lollipops

Execution: Done
Output: answer = -8

What's the problem with the above code?

 1. The answer should be positive because Jason gave lollipops to Denny

 2. Let's check the code:

> jason_lollipops_initial = 20
> jason_lollipops_after = 12
> denny_lollipops =  jason_lollipops_after - jason_lollipops_initial

The variable denny_lollipops is calculated incorrectly.
The formula used is jason_lollipops_after - jason_lollipops_initial, which results in a negative value because Jason gave away some lollipops

so The calculation should be jason_lollipops_initial - jason_lollipops_after.

Here's a better solution:

# Jason initially had 20 lollipops
jason_lollipops_initial = 20

# After giving some to Denny, Jason has 12 lollipops left
jason_lollipops_after = 12

# The number of lollipops Jason gave to Denny is the initial amount minus the remaining amount
denny_lollipops = jason_lollipops_initial - jason_lollipops_after

# The answer is the number of lollipops Jason gave to Denny
answer = denny_lollipops

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
# Python code
trees_initial = 15
trees_after = 21
answer = trees_initial - trees_after

Execution: Done
Output: answer = -6

What's the problem with the above code?

 1. The answer should be positive because trees are being added, not removed. I need to reverse the subtraction.

 2. Let's check the code:

> trees_initial = 15
> trees_after = 21
> answer = trees_initial - trees_after

The variable trees_planted is calculated incorrectly.
The formula used is  trees_initial - trees_after, which results in a negative value because Jason gave away some lollipops

so The calculation should be trees_after - trees_initial.

Here's a better solution:
# Initial number of trees in the grove
trees_initial = 15

# Number of trees after planting
trees_after = 21

# Calculate the number of trees planted
trees_planted = trees_after - trees_initial

# The answer is the number of trees planted
answer = trees_planted

---

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
toys_initial = 5
toys_received = 2 * 3
answer = toys_initial + toys_received
```
Execution Status: Done
Output: answer = 11

What's the problem with the above code?

 1. I incorrectly calculated the total toys received. Shawn got 2 toys each from his mom and dad, so the correct total received should be 2 + 2.

 2. Let's check the code:

> toys_initial = 5
> toys_received = 2 * 3 # Incorrect calculation here
> answer = toys_initial + toys_received

toys_received = 2 * 3 calculates the total number of toys received as 6, which assumes Shawn received two sets of toys, not two toys each from his mom and dad
so The calculation should be toys_initial + toys_received.

Here's a better solution:

# Initial number of toys Shawn has
toys_initial = 5

# Number of toys received from mom and dad
toys_received = 2 + 2  # Two toys from mom + two toys from dad

# Calculate the total number of toys Shawn has now
total_toys = toys_initial + toys_received

# The answer is the total number of toys Shawn has now
answer = total_toys

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
    Thought 2: There are now 29 computers in the server room.
    Action 2: Finish[
    ```python
    answer = 29
    ```
]
Action 2:Finish[
    ```python
    answer = 29
    ```
]
Observation 2:
    ```python
    answer = 29
```

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
    Thought 2: Michael has 33 golf balls left at the end of Wednesday.

Action 2: Finish[
    ```python
    answer = 33
    ```
]
Observation 2: 
    ```python
    answer = 33
```

----

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Thought 1: I need to calculate the total number of cars in the parking lot after 2 more cars arrive.
Action 1: Calculate[
    ```python
    cars_initial = 3
    cars_arrived = 2
    answer = cars_initial + cars_arrived
    ```
]
Observation 1: 
    ```python
    cars_initial = 3
    cars_arrived = 2
    answer = cars_initial + cars_arrived
    ```
    Execution Status: Done
    Output: answer = 5
    Thought 2: There are 5 cars in the parking lot now.
Action 2: Finish[
    ```python
    answer = 5
    ```
]
Observation 2: 
    ```python
    answer = 5
    ```
---

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

#python
money_initial = 23
bagels = 5
bagel_cost = 3
money_spent = bagels * bagel_cost
money_left =   money_spent - money_initial
answer = money_left


Execution: Done
Output: answer = -8

What's the problem with the above code?

1. the money_left should be positive number because need to determine how much money Olivia has left after buying the bagels

2. Let's check the code:

# Initial Money: Olivia starts with $23 (money_initial = 23).
> money_initial = 23
# Bagel Purchase: She buys 5 bagels, and each bagel costs $3 (bagels = 5, bagel_cost = 3).
> bagels = 5
> bagel_cost = 3
# Total Spent: The total money spent on bagels is calculated as money_spent = bagels * bagel_cost.
> money_spent = bagels * bagel_cost
# money_initial from money_spent, which is incorrect for determining how much money Olivia has left after buying the bagels.
> money_left =  money_spent - money_initial

answer = money_spent - money_initial calculates the difference in the wrong direction. It subtracts the initial amount from the spent amount, which gives a negative value (-8), indicating an incorrect result.

Here's a better solution:
```python
# Initial amount of money Olivia has
money_initial = 23

# Number of bagels bought
bagels = 5

# Cost per bagel
bagel_cost = 3

# Calculate the total amount spent on bagels
money_spent = bagels * bagel_cost

# Calculate how much money Olivia has left
money_left = money_initial - money_spent

# The answer is the amount of money Olivia has left
answer = money_left

---

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

#python
leah_chocolates = 32
sister_chocolates = 42
total_chocolates = leah_chocolates + sister_chocolates
chocolates_eaten = 35
chocolates_left = total_chocolates - chocolates_eaten
answer = chocolates_left

Execution: Done
Output: answer = 23

What's the problem with the above code?

1. The calculation subtracted the total chocolates eaten from the combined initial total of Leah's and her sister's chocolates, rather than from their individual initial amounts, resulting in an incorrect remaining chocolate count.

2. Let's check the code:

# Initial Chocolates: Leah initially has 32 chocolates and her sister has 42 chocolates
> leah_chocolates = 32
> sister_chocolates = 42
# The total number of chocolates they have together is calculated as 
> total_chocolates = leah_chocolates + sister_chocolates # correct formula
# They ate a total of 35 chocolates
> chocolates_eaten = 35
# To find out how many chocolates they have left in total, we subtract
> chocolates_left = total_chocolates - chocolates_eaten

In summary, the code makes errors in calculating the chocolates_left.
We need to find out how many chocolates they have left after eating 35 chocolates in total.

Here's a better solution:
```python
# Define initial number of chocolates
leah_chocolates = 32
sister_chocolates = 42

# Calculate total number of chocolates
total_chocolates = leah_chocolates + sister_chocolates

# Number of chocolates eaten
chocolates_eaten = 35

# Calculate number of chocolates left
chocolates_left = total_chocolates - chocolates_eaten

# Print the answer
print("Total chocolates left after eating:", chocolates_left)


```"""




CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Question: {question}
```python
{answer}
```

What's the problem with the above code?

{critique}"""


GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL = """Question: Janet hires six employees. Four of them are warehouse workers who make $15/hour, and the other two are managers who make $20/hour. Janet has to pay 10% of her workers' salaries in FICA taxes. If everyone works 25 days a month and 8 hours a day, how much does Janet owe total for their wages and taxes for one month?
```python
num_of_warehouse_workers = 4
num_of_managers = 2
wage_of_warehouse_workers = 15
wage_of_managers = 20
num_of_days = 25
num_of_hours = 8
total_hours = num_of_days * num_of_hours
total_wage = num_of_warehouse_workers * wage_of_warehouse_workers * total_hours + num_of_managers * wage_of_managers * total_hours
answer = total_wage * 1.1
```

What's the problem with the above code?

1. The total wages and taxes should be positive, 22000 > 0, it's reasonable.

2. Let's check the code:

> # Janet hires six employees. Four of them are warehouse workers who make $15/hour, and the other two are managers who make $20/hour.
> num_of_warehouse_workers = 4
> num_of_managers = 2
> wage_of_warehouse_workers = 15
> wage_of_managers = 20

This defines the number of warehouse workers and managers, and their wages, this is correct.

> num_of_days = 25
> num_of_hours = 8
> total_hours = num_of_days * num_of_hours

It calculates the total hours of working 25 days a month and 8 hours a day, that's right.

> total_wage = num_of_warehouse_workers * wage_of_warehouse_workers * total_hours + num_of_managers * wage_of_managers * total_hours

It calculates the total wage, this is correct.

> answer = total_wage * 1.1

Janet has to pay 10% taxes, 1 + 0.1 = 1.1, that code calculates the total wage and taxes for one month, that's correct.

Overall, the code understood the meaning of the question correctly and did not miss any information, and the answer 22000 is reasonable. However, there are some improvements that can be made to make the code more concise and easier to read.

Here's a better solution:
```python
# Define the wages for the workers
warehouse_wage = 15
manager_wage = 20

# Define the number of employees in each category
num_warehouse_workers = 4
num_managers = 2

# Calculate the total number of hours worked in a month
num_days = 25
hours_per_day = 8
total_hours = num_days * hours_per_day

# Calculate the total wages for the workers
warehouse_wages = num_warehouse_workers * warehouse_wage * total_hours
manager_wages = num_managers * manager_wage * total_hours
total_wages = warehouse_wages + manager_wages

# Calculate the total FICA taxes owed
fica_rate = 0.10
total_fica_tax = total_wages * fica_rate

# Calculate the total amount owed
total_owed = total_wages + total_fica_tax
answer = total_owed
```

---

Question: A farmer has twice as many pigs as cows, and 4 more cows than goats.  If the farmer has 56 animals total, how many pigs does he have?
```python
num_of_pigs = 2 * num_of_cows
num_of_cows = number_of_goats + 4
num_of_animals = 56
number_of_goats = (num_of_animals - 4) / 3
num_of_cows = (number_of_goats + 4) * 2
answer = num_of_cows
```

What's the problem with the above code?

1. The above code causes the "NameError" because it use the variable `num_of_cows` before it is defined.

2. The variable names in the code are a little bit confusing, becase both `num_of_pigs` and "number_of_goats" are used.

Let's analysis the problem, we know that the total number of animals are 56, but we don't konw the number of pigs, cows or goats, and we can't get any of them directly, so we can build equations to solve for the number of pigs.

Here's a better solution:
```python
# let's define the number of goats is x
# then the number of cows is 4+x
# then the number of pigs is 2(4+x) 
# so, total animals is x + (4+x) + 2(4+x) = 56
# Now we can solve for "x":
# x + 4 + x + 2x + 8 = 56
# 4x + 12 = 56
# 4x = 44
# x = 11
# So the farmer has 11 goats.
num_of_goats = 11
num_of_cows = num_of_goats + 4
# calculate the answer given the number of goats
num_of_pigs = 2 * num_of_cows
answer = num_of_pigs
```

---

Question: Ann, Bill, Cate, and Dale each buy personal pan pizzas cut into 4 pieces. If Bill and Dale eat 50% of their pizzas and Ann and Cate eat 75% of the pizzas, how many pizza pieces are left uneaten?
```python
num_of_pieces_per_pizza = 4
num_of_pizza = 4
num_of_pieces = num_of_pieces_per_pizza * num_of_pizza
num_of_pieces_eaten_by_bill_and_dale = num_of_pieces * 0.5
num_of_pieces_eaten_by_ann_and_cate = num_of_pieces * 0.75
answer = num_of_pieces - num_of_pieces_eaten_by_bill_and_dale - num_of_pieces_eaten_by_ann_and_cate
```

What's the problem with the above code?

1. The number of pizza pieces left uneaten should be a positive integer, -4.0 < 0, so the answer is not reasonable. 

2. Let's check the code:

> # Ann, Bill, Cate, and Dale each buy personal pan pizzas cut into 4 pieces.
> num_of_pieces_per_pizza = 4
> num_of_pizza = 4
> num_of_pieces = num_of_pieces_per_pizza * num_of_pizza

It calculates the total number of pieces `num_of_pieces`, that's correct.

> # Bill and Dale eat 50% of their pizzas and Ann and Cate eat 75% of the pizzas
> num_of_pieces_eaten_by_bill_and_dale = num_of_pieces * 0.5
> num_of_pieces_eaten_by_ann_and_cate = num_of_pieces * 0.75

According to the question, each person only eats their own personal pan pizza, `num_of_pieces * 0.5` means 50% of the total pieces, this is wrong.

Here's a better solution:
```python
pizza_pieces = 4  # each person buys a personal pan pizza cut into 4 pieces
ann_pieces = 4 * 0.75  # Ann eats 75% of her pizza
bill_pieces = 4 * 0.5  # Bill eats 50% of his pizza
cate_pieces = 4 * 0.75  # Cate eats 75% of her pizza
dale_pieces = 4 * 0.5  # Dale eats 50% of his pizza

total_pieces_eaten = ann_pieces + bill_pieces + cate_pieces + dale_pieces

total_pieces = pizza_pieces * 4  # there are 4 people
pieces_left = total_pieces - total_pieces_eaten

answer = pieces_left
```

---

Question: Tommy is making 12 loaves of bread. He needs 4 pounds of flour per loaf. A 10-pound bag of flour costs $10 and a 12-pound bag costs $13. When he is done making his bread, he has no use for flour and so he will throw away whatever is left. How much does he spend on flour if he buys the cheapest flour to get enough?
```python
num_of_loaves = 12
pounds_of_flour_per_loaf = 4
pounds_of_flour = num_of_loaves * pounds_of_flour_per_loaf
pounds_per_bag = 10
cost_of_10_pounds_bag = 10
cost_of_12_pounds_bag = 13
num_of_10_pounds_bag = pounds_of_flour / pounds_per_bag
num_of_12_pounds_bag = pounds_of_flour / pounds_per_bag
answer = min(num_of_10_pounds_bag * cost_of_10_pounds_bag, num_of_12_pounds_bag * cost_of_12_pounds_bag)
```

What's the problem with the above code?

1. The cost of flour should be a positive number, 48 > 0, it's reasonable.

2. Let's check the code:

> num_of_loaves = 12
> pounds_of_flour_per_loaf = 4
> pounds_of_flour = num_of_loaves * pounds_of_flour_per_loaf

It calculates the total pounds of flour needed, that's correct.

> # A 10-pound bag of flour costs $10 and a 12-pound bag costs $13
> pounds_per_bag = 10  # `pounds_per_bag` is ambiguous since there're two kinds of bags
> cost_of_10_pounds_bag = 10
> cost_of_12_pounds_bag = 13
> num_of_10_pounds_bag = pounds_of_flour / pounds_per_bag
> num_of_12_pounds_bag = pounds_of_flour / pounds_per_bag  # 12-pound bag has 12 pounds rather than 10, that's wrong

There's problems in calculating the number of bags needed. In addition, the number of bags should be integer, and to get enough flour we should round up.

> answer = min(num_of_10_pounds_bag * cost_of_10_pounds_bag, num_of_12_pounds_bag * cost_of_12_pounds_bag)

This piece code calculates the cheapest cost of flour, it's correct.

In summary, the code makes errors in calculating the cost.

To solve the problem, we first need to calculate how many pounds of flour Tommy needs in total. Then we need to compare the cost of buying a 10-pound bag of flour versus a 12-pound bag of flour and choose the cheaper option to get the required amount of flour.

Here's a better solution:
```python
import math
# Calculate how many pounds of flour Tommy needs
num_of_loaves = 12
pounds_of_flour_per_loaf = 4
total_pounds_of_flour = num_of_loaves * pounds_of_flour_per_loaf

cost_of_10_pounds_bag = 10
cost_of_12_pounds_bag = 13

# Calculate the number of bags needed
num_of_10_pounds_bag = math.ceil(total_pounds_of_flour / 10)
num_of_12_pounds_bag = math.ceil(total_pounds_of_flour / 12)

# Calculate the cost of flour
cost_of_10_pounds = num_of_10_pounds_bag * cost_of_10_pounds_bag
cost_of_12_pounds = num_of_12_pounds_bag * cost_of_12_pounds_bag

# Choose the cheapest option
total_cost = min(cost_of_10_pounds, cost_of_12_pounds)

answer = total_cost
```"""


# ======================================================================== SVAMP ======================================================================== #


CRITIC_POT_INSTRUCTION_SVAMP = """Read the following passages to answer questions with Python code, store the result as a 'answer' variable:

{examples}
(END OF EXAMPLES)

Question: {question}
# Python code, return answer"""


CRITIC_CRITIQUE_INSTRUCTION_SVAMP = """{examples}
(END OF EXAMPLES)

Question: {question}
```python
{answer}
```
Execution: {execution_status} 
Output: answer = {code_answer}

What's the problem with the above code?

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

> zachary_diff = zachary_crunches - zachary_push_ups

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

> answer = num_packs_of_pencils * num_pencils_in_each_pack

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



--------------------------------------------------Gao----------------------------------------------

Question: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
# Python code
red_stickers_bought = 93
blue_stickers_bought = 10
red_stickers_used_on_fridge = 31
blue_stickers_used_on_laptop = 7
answer = red_stickers_bought + red_stickers_used_on_fridge

Execution Status: Done
Output: answer = 124

What's the problem with the above code?

1. The answer red stickers 124 is greater than the number of stickers James initially had, which is incorrect.
2. Let's check the code:


> answer = red_stickers_bought + red_stickers_used_on_fridge

The code incorrectly adds the number of red stickers bought to the number of red stickers used, instead of subtracting the used stickers from the bought stickers.

Here's a better solution:
```python

red_stickers_bought = 93  # Total red stickers
red_stickers_used_on_fridge = 31  # Stickers used

# Calculate remaining stickers
remaining_red_stickers = red_stickers_bought - red_stickers_used_on_fridge  # Remaining stickers
answer = remaining_red_stickers  # Result

-------

Question: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy each egg?

cost_per_egg = 80
discount_per_egg = 29
answer = cost_per_egg + discount_per_egg

Execution: Done
Output: answer = 109

What's the problem with the above code?

1. The answer 109 dollars per egg is incorrect because it adds the discount instead of subtracting it.

2. Let's check the code:

> answer = cost_per_egg + discount_per_egg

The code incorrectly adds the discount to the cost of the egg instead of subtracting it.

Here's a better solution:
```python

cost_per_egg = 80  # Price per egg
discount_per_egg = 29  # Discount per egg

# Calculate final price
final_price_per_egg = cost_per_egg - discount_per_egg  # Final price per egg
answer = final_price_per_egg  # Result


--------


Question: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. 
# Python code
cases_bought = 22
books_bought = 5
total_cases_now = 57
total_books_now = 25
answer = total_books_now - books_bought

Execution: Done
Output: answer = 20

What's the problem with the above code?

1. The answer 20 books at first is reasonable.
2. Let's check the code:

> answer = total_books_now - books_bought

The code correctly calculates the number of books Danny had initially.

Here's a better solution:
```python

books_bought = 5  # Number of books Danny bought
total_books_now = 25  # Total books Danny has now

# Calculate initial number of books Danny had
initial_books = total_books_now - books_bought  # Initial number of books Danny had
answer = initial_books  # Result


-----

Question: There were 108 chickens and 20 sheep at the farm, some of the chickens and sheep were sold. There are 87 chickens and 18 sheep left now. How many chickens were sold?
# Python code
initial_chickens = 108
initial_sheep = 20
remaining_chickens = 87
remaining_sheep = 18

# Calculate the number of chickens sold
chickens_sold =  remaining_chickens - initial_chickens  
answer = chickens_sold

Execution: Done
Output: answer = -21

What's the problem with the above code?

1. The output answer = -21 indicates that more chickens were remaining than initially present, which is logically incorrect.

2. Let's check the code:

> chickens_sold = initial_chickens - remaining_chickens

The code incorrectly subtracts the remaining chickens from the initial chickens, which results in a negative value indicating additional chickens added instead of sold.

Here's a better solution:
```python

initial_chickens = 108  # Initial number of chickens
remaining_chickens = 87  # Remaining chickens

# Calculate the number of chickens sold
chickens_sold = initial_chickens - remaining_chickens   # Number of chickens sold
answer = chickens_sold  # Result



------

Question: Katty scored 2 goals on Monday, 8 goals on Tuesday and 9 goals on Wednesday. How many did Katty score on Monday and Wednesday?
# Python code
goals_on_monday = 2
goals_on_tuesday = 8
goals_on_wednesday = 9
answer = goals_on_monday + goals_on_wednesday

Execution: Done
Output: answer = 11

What's the problem with the above code?
1. The answer 11 goals scored on Monday and Wednesday is reasonable.
2. Let's check the code:

> answer = goals_on_monday + goals_on_wednesday

The code correctly calculates the total number of goals Katty scored on Monday and Wednesday.


Here's a better solution:
```python

goals_on_monday = 2  # Goals scored on Monday
goals_on_wednesday = 9  # Goals scored on Wednesday

# Calculate the total number of goals scored on Monday and Wednesday
total_goals = goals_on_monday + goals_on_wednesday  # Total goals scored
answer = total_goals  # Result


Question: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined. How many more girls than boys are in the Masquerade?
# Python code

num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls =  num_girls_joined - num_girls_before
total_boys = num_boys_before + num_boys_joined
answer = total_girls - total_boys

Execution: Done
Output: answer = 11

What's the problem with the initial output?

1. The initial output result of 11 is incorrect because This approach does not correctly calculate the total number of girls after the additions.
2. Let's check the corrected code:

> total_girls = num_girls_joined - num_girls_before

According to the question, we should should add the initial number of girls to the number of girls who joined to get the total number of girls after the additions.

Here's a better solution:
```python

num_girls_before = 5  # Number of girls before joining
num_girls_joined = 12  # Number of girls joined
num_boys_before = 4  # Number of boys before joining
num_boys_joined = 7  # Number of boys joined

# Calculate the total number of girls after joining
total_girls = num_girls_before + num_girls_joined

# Calculate the total number of boys after joining
total_boys = num_boys_before + num_boys_joined

# Calculate how many more girls than boys there are
answer = total_girls - total_boys  # Difference in number of girls and boys


--- 



```"""





CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_SVAMP = """{examples}
(END OF EXAMPLES)

Question: {question}
```python
{answer}
```

What's the problem with the above code?

{critique}"""


SVAMP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL = """Question: Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. How much did Marco's strawberries weigh?
```python
num_strawberries_dad_picked = 11
num_strawberries_marco_picked = 30
answer = num_strawberries_dad_picked + num_strawberries_marco_picked
```

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

What's the problem with the above code?

1. The answer, 12, is a reasonable result.
2. Let's check the code:

> zachary_diff = zachary_crunches - zachary_push_ups

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

What's the problem with the above code?

1. The answer rows 140 > 0, it's reasonable.
2. Let's check the code:

> answer = num_packs_of_pencils * num_pencils_in_each_pack

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
```"""


# ======================================================================== TABMWP ======================================================================== #


CRITIC_POT_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

{question}
# Python Code, return answer"""


CRITIC_CRITIQUE_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

{question}
```python
{answer}
```
Execution: {execution_status} 
Output: answer = {code_answer}

What's the problem with the above code?

{critique}"""


TABMWP_FEWSHOT_EXAMPLES_CRITIC = """Read the following table and then write Python code to answer a question:

llama | $2,860.00
macaw | $8,897.00
parrot | $1,586.00
boa constrictor | $1,382.00
kinkajou | $2,629.00

Question: How much money does Kendra need to buy a kinkajou and a parrot?
```python
kinkajou_price = 2629.00
parrot_price = 1586.00
total_money = kinkajou_price + parrot_price
```
Execution: Done
Output: answer = None

What's the problem with the above code?

The problem with the above code is that it does not assign the result of the calculation to the variable answer. The code correctly calculates the total money Kendra needs to buy a kinkajou and a parrot, but it does not store the result in the answer variable.

Here's a better solution:
```python
kinkajou_price = 2629.00  
parrot_price = 1586.00  
total_money = kinkajou_price + parrot_price  
answer = total_money  
```

---

Read the following table regarding "Words per page" and then write Python code to answer a question:

Stem | Leaf 
3 | 3
4 | 1, 7, 8
5 | 2, 5, 6
6 | 0, 0, 1, 3, 4
7 | 2, 2, 4, 5, 8
8 | 0, 1, 4, 8
9 | 0

Question: Cameron counted the number of words per page in his new book. How many pages have at least 70 words but fewer than 90 words?
```python
pages_with_70_words = 0
pages_with_80_words = 0
pages_with_90_words = 0
for stem, leaf in zip(stem, leaf):
    if stem == 4 and leaf >= 7 and leaf <= 8:
        pages_with_70_words += 1
    elif stem == 5 and leaf >= 2 and leaf <= 6:
        pages_with_70_words += 1
    elif stem == 6 and leaf >= 0 and leaf <= 4:
        pages_with_70_words += 1
    elif stem == 7 and leaf >= 2 and leaf <= 8:
        pages_with_70_words += 1
    elif stem == 8 and leaf >= 0 and leaf <= 4:
        pages_with_70_words += 1
    elif stem == 9 and leaf == 0:
        pages_with_70_words += 1
```
Execution: NameError("name 'stem' is not defined")
Output: answer = None

What's the problem with the above code?

1. The above code causes a "NameError" because it uses the variables `stem` and `leaf` without defining them. Moreover, the code tries to iterate over `stem` and `leaf` using zip, which is not correct because the stem and leaf are actually part of a single data structure, not two separate lists.
2. The variable names in the code are a little bit confusing, and the logic for counting pages with at least 70 words but fewer than 90 words is not correct.

To solve the problem, we should first properly represent the stem and leaf data structure and then count the pages with at least 70 words but fewer than 90 words.

Here's a better solution:
```python
# Define stem and leaf data structure as a list of tuples  
stem_and_leaf = [(3, [3]), (4, [1, 7, 8]), (5, [2, 5, 6]), (6, [0, 0, 1, 3, 4]), (7, [2, 2, 4, 5, 8]), (8, [0, 1, 4, 8]), (9, [0])]  
  
# Initialize counter for pages with at least 70 words but fewer than 90 words  
pages_with_70_to_89_words = 0  
  
# Iterate through the stem and leaf data structure and count relevant pages  
for stem, leaf_list in stem_and_leaf:  
    for leaf in leaf_list:  
        words_per_page = stem * 10 + leaf  
        if 70 <= words_per_page < 90:  
            pages_with_70_to_89_words += 1  
  
answer = pages_with_70_to_89_words  
```

---

Read the following table and then write Python code to answer a question:

Number of cow pastures | Number of sprinklers
1 | 10
2 | 20
3 | 30
4 | 40
5 | 50
6 | ?

Question: Each cow pasture has 10 sprinklers. How many sprinklers are in 6 cow pastures?
```python
number_of_sprinklers_for_1_pasture = 10
number_of_sprinklers_for_2_pastures = 20
number_of_sprinklers_for_3_pastures = 30
number_of_sprinklers_for_4_pastures = 40
number_of_sprinklers_for_5_pastures = 50
number_of_sprinklers_for_6_pastures = number_of_sprinklers_for_5_pastures + number_of_sprinklers_for_1_pasture
answer = number_of_sprinklers_for_6_pastures
```
Execution: Done
Output: answer = 60

What's the problem with the above code?

There is no problem with the above code. It correctly answers the question and calculates the number of sprinklers in 6 cow pastures. However, the code can be simplified, as there's a clear pattern in the relationship between the number of cow pastures and the number of sprinklers.

Here's a better solution:
```python
number_of_pastures = 6  
sprinklers_per_pasture = 10  
total_sprinklers = number_of_pastures * sprinklers_per_pasture  
answer = total_sprinklers  
```

---

Read the following table regarding "Company sizes" and then write Python code to answer a question:

Company | Number of employees
Madison Corporation | 454
Downtown Publishing | 442
Thompson Corporation | 452
Bronson Attorneys | 428

Question: The board of commerce recorded how many employees work at each company. Which company has the fewest employees? Please select from the following options: ['Madison Corporation', 'Downtown Publishing', 'Thompson Corporation', 'Bronson Attorneys']
```python
company_sizes = [454, 442, 452, 428]
company_sizes = sorted(company_sizes)
answer = company_sizes[0]
```
Execution: Done
Output: answer = 428.0

What's the problem with the above code?

The problem with the above code is that it returns the number of employees of the company with the fewest employees but not the name of the company itself, which is what the question asks for.

To fix this issue, we should use a dictionary to store the company names and their corresponding number of employees. Then, we can find the company with the fewest employees by iterating through the dictionary and comparing the values.

Here's a better solution:
```python
# Store the company names and their corresponding number of employees in a dictionary  
companies = {'Madison Corporation': 454,  
             'Downtown Publishing': 442,  
             'Thompson Corporation': 452,  
             'Bronson Attorneys': 428}  
  
# Find the company with the fewest employees  
min_employees = float('inf')  
min_company = None  
  
for company, num_employees in companies.items():  
    if num_employees < min_employees:  
        min_employees = num_employees  
        min_company = company  
  
answer = min_company  
```"""


CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

{question}
```python
{answer}
```

What's the problem with the above code?
{critique}"""


TABMWP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL = """Read the following table and then write Python code to answer a question:

llama | $2,860.00
macaw | $8,897.00
parrot | $1,586.00
boa constrictor | $1,382.00
kinkajou | $2,629.00

Question: How much money does Kendra need to buy a kinkajou and a parrot?
```python
kinkajou_price = 2629.00
parrot_price = 1586.00
total_money = kinkajou_price + parrot_price
```

What's the problem with the above code?

The problem with the above code is that it does not assign the result of the calculation to the variable answer. The code correctly calculates the total money Kendra needs to buy a kinkajou and a parrot, but it does not store the result in the answer variable.

Here's a better solution:
```python
kinkajou_price = 2629.00  
parrot_price = 1586.00  
total_money = kinkajou_price + parrot_price  
answer = total_money  
```

---

Read the following table regarding "Words per page" and then write Python code to answer a question:

Stem | Leaf 
3 | 3
4 | 1, 7, 8
5 | 2, 5, 6
6 | 0, 0, 1, 3, 4
7 | 2, 2, 4, 5, 8
8 | 0, 1, 4, 8
9 | 0

Question: Cameron counted the number of words per page in his new book. How many pages have at least 70 words but fewer than 90 words?
```python
pages_with_70_words = 0
pages_with_80_words = 0
pages_with_90_words = 0
for stem, leaf in zip(stem, leaf):
    if stem == 4 and leaf >= 7 and leaf <= 8:
        pages_with_70_words += 1
    elif stem == 5 and leaf >= 2 and leaf <= 6:
        pages_with_70_words += 1
    elif stem == 6 and leaf >= 0 and leaf <= 4:
        pages_with_70_words += 1
    elif stem == 7 and leaf >= 2 and leaf <= 8:
        pages_with_70_words += 1
    elif stem == 8 and leaf >= 0 and leaf <= 4:
        pages_with_70_words += 1
    elif stem == 9 and leaf == 0:
        pages_with_70_words += 1
```

What's the problem with the above code?

1. The above code causes a "NameError" because it uses the variables `stem` and `leaf` without defining them. Moreover, the code tries to iterate over `stem` and `leaf` using zip, which is not correct because the stem and leaf are actually part of a single data structure, not two separate lists.
2. The variable names in the code are a little bit confusing, and the logic for counting pages with at least 70 words but fewer than 90 words is not correct.

To solve the problem, we should first properly represent the stem and leaf data structure and then count the pages with at least 70 words but fewer than 90 words.

Here's a better solution:
```python
# Define stem and leaf data structure as a list of tuples  
stem_and_leaf = [(3, [3]), (4, [1, 7, 8]), (5, [2, 5, 6]), (6, [0, 0, 1, 3, 4]), (7, [2, 2, 4, 5, 8]), (8, [0, 1, 4, 8]), (9, [0])]  
  
# Initialize counter for pages with at least 70 words but fewer than 90 words  
pages_with_70_to_89_words = 0  
  
# Iterate through the stem and leaf data structure and count relevant pages  
for stem, leaf_list in stem_and_leaf:  
    for leaf in leaf_list:  
        words_per_page = stem * 10 + leaf  
        if 70 <= words_per_page < 90:  
            pages_with_70_to_89_words += 1  
  
answer = pages_with_70_to_89_words  
```

---

Read the following table and then write Python code to answer a question:

Number of cow pastures | Number of sprinklers
1 | 10
2 | 20
3 | 30
4 | 40
5 | 50
6 | ?

Question: Each cow pasture has 10 sprinklers. How many sprinklers are in 6 cow pastures?
```python
number_of_sprinklers_for_1_pasture = 10
number_of_sprinklers_for_2_pastures = 20
number_of_sprinklers_for_3_pastures = 30
number_of_sprinklers_for_4_pastures = 40
number_of_sprinklers_for_5_pastures = 50
number_of_sprinklers_for_6_pastures = number_of_sprinklers_for_5_pastures + number_of_sprinklers_for_1_pasture
answer = number_of_sprinklers_for_6_pastures
```

What's the problem with the above code?

There is no problem with the above code. It correctly answers the question and calculates the number of sprinklers in 6 cow pastures. However, the code can be simplified, as there's a clear pattern in the relationship between the number of cow pastures and the number of sprinklers.

Here's a better solution:
```python
number_of_pastures = 6  
sprinklers_per_pasture = 10  
total_sprinklers = number_of_pastures * sprinklers_per_pasture  
answer = total_sprinklers  
```

---

Read the following table regarding "Company sizes" and then write Python code to answer a question:

Company | Number of employees
Madison Corporation | 454
Downtown Publishing | 442
Thompson Corporation | 452
Bronson Attorneys | 428

Question: The board of commerce recorded how many employees work at each company. Which company has the fewest employees? Please select from the following options: ['Madison Corporation', 'Downtown Publishing', 'Thompson Corporation', 'Bronson Attorneys']
```python
company_sizes = [454, 442, 452, 428]
company_sizes = sorted(company_sizes)
answer = company_sizes[0]
```

What's the problem with the above code?

The problem with the above code is that it returns the number of employees of the company with the fewest employees but not the name of the company itself, which is what the question asks for.

To fix this issue, we should use a dictionary to store the company names and their corresponding number of employees. Then, we can find the company with the fewest employees by iterating through the dictionary and comparing the values.

Here's a better solution:
```python
# Store the company names and their corresponding number of employees in a dictionary  
companies = {'Madison Corporation': 454,  
             'Downtown Publishing': 442,  
             'Thompson Corporation': 452,  
             'Bronson Attorneys': 428}  
  
# Find the company with the fewest employees  
min_employees = float('inf')  
min_company = None  
  
for company, num_employees in companies.items():  
    if num_employees < min_employees:  
        min_employees = num_employees  
        min_company = company  
  
answer = min_company  
```"""


# ======================================================================== HUMANEVAL ======================================================================== #


CRITIC_POT_INSTRUCTION_HUMANEVAL = """You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. 

```python
{question}"""


CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL = """{examples}
(END OF EXAMPLES)

```python
{question}
{answer}

{tests}
```

Execution: {execution_status} 

What's the problem with the above code?

{critique}"""


HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC = """```python
def has_duplicate_names(names_list: List[str]) -> bool:
    \"\"\"Check if there is any name that appears more than once in the list.
    >>> has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Alice'])
    True
    >>> has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Dave']) 
    False
    \"\"\"
    return len(names_list) != len(set(names_list)) - 1

assert has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Alice']) == True
assert has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Dave']) == False
```

Execution: AssertionError()

What's the problem with the above code?

1. The function incorrectly returns True for a list without any duplicates. This indicates a logical error in the implementation, specifically due to an incorrect comparison operation.

2. Let's check the code:

> names_list = ['Alice', 'Bob', 'Charlie', 'Dave']

This defines a list of names without any deliberate duplication. However, the function still returns True.

> return len(names_list) != len(set(names_list)) - 1

This line is designed to check for duplicates by comparing the length of the list with the length of the set created from the list, which inherently removes any duplicate entries. The subtraction of one from the set’s length is intended to allow exactly one duplicated name in the list for the condition to hold true. However, this approach is logically flawed. The subtraction results in a condition that falsely identifies duplicates if there's even one unique item that makes the set’s size smaller than the list’s size. This error leads to a false positive for duplicates whenever the list has any unique items.

Overall, the function does not perform as expected due to a critical error in this line. The incorrect manipulation of the set's length by subtracting one introduces a logical fallacy, causing the function to misidentify non-duplicate scenarios as having duplicates. This is a conceptual error in understanding how to handle the detection of a single allowed duplicate in the context of set and list length comparison.

Here's a better solution (include only function implementation):
```python
def has_duplicate_names(names_list: List[str]) -> bool:
    \"\"\"Check if there is any name that appears more than once in the list.
    >>> has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Alice'])
    True
    >>> has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Dave']) 
    False
    \"\"\"
    return len(names_list) != len(set(names_list))
```

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

assert average_positive([1, -1, 2, -2, 3]) == 2.0
assert average_positive([-5, 0, 5, 15]) == 10.0
assert average_positive([100, 200, -100, 0]) == 150.0
assert average_positive([-1, -2, -3]) == 0
```

Execution: AssertionError()

What's the problem with the above code?

1. The function incorrectly returns an average of all numbers, not just the positives. This leads to incorrect calculations when negative numbers are present in the list.

2. Let's check the code:

> numbers = [1, -1, 2, -2, 3]

This list includes both positive and negative numbers, but the function's calculation should focus only on the positives.

> total = sum(numbers)

This line incorrectly calculates the total sum of all numbers in the list, not just the positive ones. This is the root of the error as it fails to exclude negative numbers and zeros, which should not contribute to the average of positive numbers. 

> count = sum(1 for x in numbers if x > 0)

While this line correctly counts the number of positive numbers, the previous calculation of the total sum includes all numbers, thus distorting the average calculation.

Overall, the function does not perform as expected due to a critical oversight in the initial summing process. The inclusion of all numbers in the total, regardless of their sign, leads to an incorrect average calculation for positive numbers only. This error could be easily overlooked as it merges the concepts of filtering and summing but applies them incorrectly by not aligning the filtering of positives in both the sum and count operations.

Here's a better solution (include only function implementation):
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
    positive_numbers = [num for num in numbers if num > 0]
    return sum(positive_numbers) / len(positive_numbers) if positive_numbers else 0
```

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

assert exceeds_threshold([100, 102, 107, 103], 5) == 1
assert exceeds_threshold([100, 101, 102, 103], 2) == 0
```

Execution: AssertionError()

What's the problem with the above code?

1. The function incorrectly fails to count differences that exactly meet the threshold because it mistakenly uses a threshold incremented by 1. This causes it to underreport the number of threshold exceedances.

2. Let's check the code:

> measurements = [100, 102, 107, 103]

This set of data points includes instances where the difference between successive measurements is equal to or greater than the threshold set.

> count = 0

Initializes a counter to zero. This line is correct as it sets up the counting variable which will be used to tally the number of exceedances.

> for i in range(1, len(measurements)):

Begins a loop starting from the second element (index 1) of the measurements list. This is correct, as it prepares to compare each element with its predecessor to check the difference against the threshold.

> if abs(measurements[i] - measurements[i - 1]) > (threshold + 1):

This line is the crux of the problem. It increases the threshold by 1, thereby raising the condition required to count an exceedance. The addition of 1 to the threshold misrepresents the intended logic of the function by requiring differences to exceed the original threshold by more than intended. This modification in the threshold results in undercounting actual exceedances, as it does not count differences that are exactly equal to the original threshold, or only slightly above it by less than 1.

Overall, the function fails to perform as expected because it incorrectly manipulates the threshold condition. The increase in the threshold by 1 leads to the function not recognizing valid exceedances that meet the original criteria set by the threshold. The logical error is a straightforward misunderstanding of how to apply the threshold in comparing measurement differences.

Here's a better solution (include only function implementation):
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
        if abs(measurements[i] - measurements[i - 1]) > threshold:
            count += 1
    return count
```

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

assert sum_even_indexed([10, 3, 5, 2, 8]) == 23
assert sum_even_indexed([1, 2, 3, 4, 5, 6]) == 9
assert sum_even_indexed([0, 100, 200, 300]) == 200
assert sum_even_indexed([7]) == 7
```

Execution: AssertionError()

What's the problem with the above code?

1. The function incorrectly sums up numbers at odd indices instead of even indices due to an off-by-one error. This error results from misinterpreting index positions because of adding 1 to the index before modulo operation.

2. Let's check the code:

> numbers = [10, 3, 5, 2, 8]

This defines a list of numbers where the correct function should sum the numbers at even indices (1, 3, 5) according to 0-based indexing.

> return sum(num for i, num in enumerate(numbers) if (i + 1) % 2 == 0)

This line contains the core functionality but introduces a logical error. The condition `(i + 1) % 2 == 0` is intended to sum numbers at even indices based on a zero-based index system. However, by adding 1 to the index, the function checks if the position is odd (1-based index), not even. This results in the function summing numbers at what are technically odd indices in a zero-based index system, like 1, 3, 5, etc., instead of 0, 2, 4.

Overall, the function does not perform as expected because of a subtle logical error in handling index values. It miscounts the indices, summing the wrong set of numbers.

Here's a better solution (include only function implementation):
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
    return sum(num for i, num in enumerate(numbers) if i % 2 == 0)
```

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

assert are_anagrams('Listen', 'silent') == True
assert are_anagrams('Hello', 'World') == False
assert are_anagrams('Angel', 'Glean') == True
```

Execution: AssertionError()

What's the problem with the above code?

1. The function fails to account for case sensitivity because it does not convert the strings to a uniform case before using the `Counter` to compare them, leading to incorrect results when strings differ only in case.

2. Let's check the code:

> s1 = 'Listen'; s2 = 'silent'

This defines two strings where the correct function should return True given they are case-insensitive anagrams.

> return Counter(s1) == Counter(s2)

The function returns `False` for `Counter('Listen') == Counter('silent')` because the `Counter` is case-sensitive, and thus counts 'L' and 'l' as different characters, resulting in unequal counters.

Overall, the primary issue is that the function does not perform a case conversion before counting the characters, which is essential for a correct case-insensitive anagram comparison. This oversight leads to the function incorrectly determining that strings like 'Listen' and 'silent' are not anagrams due to case differences. The correct approach should involve converting both input strings to the same case (either all uppercase or all lowercase) before applying the `Counter`.

Here's a better solution (include only function implementation):
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
    return Counter(s1.lower()) == Counter(s2.lower())
```"""


CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL = """{examples}
(END OF EXAMPLES)

```python
{question}
{answer}

{tests}
```

What's the problem with the above code?

{critique}"""


HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL = """```python
def has_duplicate_names(names_list: List[str]) -> bool:
    \"\"\"Check if there is any name that appears more than once in the list.
    >>> has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Alice'])
    True
    >>> has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Dave']) 
    False
    \"\"\"
    return len(names_list) != len(set(names_list)) - 1

assert has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Alice']) == True
assert has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Dave']) == False
```

What's the problem with the above code?

1. The function incorrectly returns True for a list without any duplicates. This indicates a logical error in the implementation, specifically due to an incorrect comparison operation.

2. Let's check the code:

> names_list = ['Alice', 'Bob', 'Charlie', 'Dave']

This defines a list of names without any deliberate duplication. However, the function still returns True.

> return len(names_list) != len(set(names_list)) - 1

This line is designed to check for duplicates by comparing the length of the list with the length of the set created from the list, which inherently removes any duplicate entries. The subtraction of one from the set’s length is intended to allow exactly one duplicated name in the list for the condition to hold true. However, this approach is logically flawed. The subtraction results in a condition that falsely identifies duplicates if there's even one unique item that makes the set’s size smaller than the list’s size. This error leads to a false positive for duplicates whenever the list has any unique items.

Overall, the function does not perform as expected due to a critical error in this line. The incorrect manipulation of the set's length by subtracting one introduces a logical fallacy, causing the function to misidentify non-duplicate scenarios as having duplicates. This is a conceptual error in understanding how to handle the detection of a single allowed duplicate in the context of set and list length comparison.

Here's a better solution (include only function implementation):
```python
def has_duplicate_names(names_list: List[str]) -> bool:
    \"\"\"Check if there is any name that appears more than once in the list.
    >>> has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Alice'])
    True
    >>> has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Dave']) 
    False
    \"\"\"
    return len(names_list) != len(set(names_list))
```

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

assert average_positive([1, -1, 2, -2, 3]) == 2.0
assert average_positive([-5, 0, 5, 15]) == 10.0
assert average_positive([100, 200, -100, 0]) == 150.0
assert average_positive([-1, -2, -3]) == 0
```

What's the problem with the above code?

1. The function incorrectly returns an average of all numbers, not just the positives. This leads to incorrect calculations when negative numbers are present in the list.

2. Let's check the code:

> numbers = [1, -1, 2, -2, 3]

This list includes both positive and negative numbers, but the function's calculation should focus only on the positives.

> total = sum(numbers)

This line incorrectly calculates the total sum of all numbers in the list, not just the positive ones. This is the root of the error as it fails to exclude negative numbers and zeros, which should not contribute to the average of positive numbers. 

> count = sum(1 for x in numbers if x > 0)

While this line correctly counts the number of positive numbers, the previous calculation of the total sum includes all numbers, thus distorting the average calculation.

Overall, the function does not perform as expected due to a critical oversight in the initial summing process. The inclusion of all numbers in the total, regardless of their sign, leads to an incorrect average calculation for positive numbers only. This error could be easily overlooked as it merges the concepts of filtering and summing but applies them incorrectly by not aligning the filtering of positives in both the sum and count operations.

Here's a better solution (include only function implementation):
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
    positive_numbers = [num for num in numbers if num > 0]
    return sum(positive_numbers) / len(positive_numbers) if positive_numbers else 0
```

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

assert exceeds_threshold([100, 102, 107, 103], 5) == 1
assert exceeds_threshold([100, 101, 102, 103], 2) == 0
```

What's the problem with the above code?

1. The function incorrectly fails to count differences that exactly meet the threshold because it mistakenly uses a threshold incremented by 1. This causes it to underreport the number of threshold exceedances.

2. Let's check the code:

> measurements = [100, 102, 107, 103]

This set of data points includes instances where the difference between successive measurements is equal to or greater than the threshold set.

> count = 0

Initializes a counter to zero. This line is correct as it sets up the counting variable which will be used to tally the number of exceedances.

> for i in range(1, len(measurements)):

Begins a loop starting from the second element (index 1) of the measurements list. This is correct, as it prepares to compare each element with its predecessor to check the difference against the threshold.

> if abs(measurements[i] - measurements[i - 1]) > (threshold + 1):

This line is the crux of the problem. It increases the threshold by 1, thereby raising the condition required to count an exceedance. The addition of 1 to the threshold misrepresents the intended logic of the function by requiring differences to exceed the original threshold by more than intended. This modification in the threshold results in undercounting actual exceedances, as it does not count differences that are exactly equal to the original threshold, or only slightly above it by less than 1.

Overall, the function fails to perform as expected because it incorrectly manipulates the threshold condition. The increase in the threshold by 1 leads to the function not recognizing valid exceedances that meet the original criteria set by the threshold. The logical error is a straightforward misunderstanding of how to apply the threshold in comparing measurement differences.

Here's a better solution (include only function implementation):
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
        if abs(measurements[i] - measurements[i - 1]) > threshold:
            count += 1
    return count
```

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

assert sum_even_indexed([10, 3, 5, 2, 8]) == 23
assert sum_even_indexed([1, 2, 3, 4, 5, 6]) == 9
assert sum_even_indexed([0, 100, 200, 300]) == 200
assert sum_even_indexed([7]) == 7
```

What's the problem with the above code?

1. The function incorrectly sums up numbers at odd indices instead of even indices due to an off-by-one error. This error results from misinterpreting index positions because of adding 1 to the index before modulo operation.

2. Let's check the code:

> numbers = [10, 3, 5, 2, 8]

This defines a list of numbers where the correct function should sum the numbers at even indices (1, 3, 5) according to 0-based indexing.

> return sum(num for i, num in enumerate(numbers) if (i + 1) % 2 == 0)

This line contains the core functionality but introduces a logical error. The condition `(i + 1) % 2 == 0` is intended to sum numbers at even indices based on a zero-based index system. However, by adding 1 to the index, the function checks if the position is odd (1-based index), not even. This results in the function summing numbers at what are technically odd indices in a zero-based index system, like 1, 3, 5, etc., instead of 0, 2, 4.

Overall, the function does not perform as expected because of a subtle logical error in handling index values. It miscounts the indices, summing the wrong set of numbers.

Here's a better solution (include only function implementation):
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
    return sum(num for i, num in enumerate(numbers) if i % 2 == 0)
```

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

assert are_anagrams('Listen', 'silent') == True
assert are_anagrams('Hello', 'World') == False
assert are_anagrams('Angel', 'Glean') == True
```

What's the problem with the above code?

1. The function fails to account for case sensitivity because it does not convert the strings to a uniform case before using the `Counter` to compare them, leading to incorrect results when strings differ only in case.

2. Let's check the code:

> s1 = 'Listen'; s2 = 'silent'

This defines two strings where the correct function should return True given they are case-insensitive anagrams.

> return Counter(s1) == Counter(s2)

The function returns `False` for `Counter('Listen') == Counter('silent')` because the `Counter` is case-sensitive, and thus counts 'L' and 'l' as different characters, resulting in unequal counters.

Overall, the primary issue is that the function does not perform a case conversion before counting the characters, which is essential for a correct case-insensitive anagram comparison. This oversight leads to the function incorrectly determining that strings like 'Listen' and 'silent' are not anagrams due to case differences. The correct approach should involve converting both input strings to the same case (either all uppercase or all lowercase) before applying the `Counter`.

Here's a better solution (include only function implementation):
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
    return Counter(s1.lower()) == Counter(s2.lower())
```"""


# ======================================================================== MBPP ======================================================================== #


# Ref: https://arxiv.org/pdf/2405.04434.
CRITIC_POT_INSTRUCTION_MBPP = """{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

```python"""


CRITIC_CRITIQUE_INSTRUCTION_MBPP = """{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests.

```python
{answer}

{tests}
```

Execution: {execution_status}

What's the problem with the above code?

{critique}"""


MBPP_FEWSHOT_EXAMPLES_CRITIC = """You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists.
Your code should pass these tests.

```python
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) | set(test_tup2))

assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))
```

Execution: AssertionError()

What's the problem with the above code?

1. The function incorrectly returns a set of all unique elements from both lists rather than the intersection. This indicates a logical error in the implementation due to incorrect set operation.

2. Let's check the code:

> test_tup1 = (3, 4, 5, 6)
> test_tup2 = (5, 7, 4, 10)

This should define two lists of numbers where only 4 and 5 are common in both.

> return tuple(set(test_tup1) | set(test_tup2))

This line erroneously uses the set union operator (`|`) which combines all elements from both sets, instead of the set intersection operator (`&`) which would correctly identify elements present in both sets. This error results in incorrect function output when no duplicates should be reported.

Overall, the function fails to perform as expected due to a critical error in using the wrong set operation. The incorrect manipulation of set operators introduces a fundamental flaw, mistaking union for intersection, thus misidentifying the intended behavior of finding common elements.

Here's a better solution:
```python
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) & set(test_tup2))
```

---

You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers.
Your code should pass these tests.

```python
import math

def is_not_prime(n):
    if n == 2:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n %% i == 0:
            return True
    return False

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False
```

Execution: AssertionError()

What's the problem with the above code?

> n = 2

This should identify the number 2, which is a prime number and should return False.

> if n == 2:
> return True

This line erroneously returns True for the input 2, indicating it as not prime, which is incorrect. The condition should be modified to correctly handle prime identification.

Here's a better solution:
```python
def is_not_prime(n):
    result = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n %% i == 0:
            result = True
            break
    return result
```

---

You are an expert Python programmer, and here is your task: Write a function to find the n largest integers from a given list of numbers, returned in descending order.
Your code should pass these tests.

```python
def heap_queue_largest(nums, n):
    return sorted(nums)[-n:]

assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]
```

Execution: AssertionError()

What's the problem with the above code?

1. The function incorrectly returns the sorted list of all elements, taking the last n elements. This indicates a logical error in the implementation due to incorrect sorting and slicing.

2. Let's check the code:

> nums = [25, 35, 22, 85, 14, 65, 75, 22, 58]
> n = 3

This should identify the three largest numbers in the list, which are 85, 75, and 65, in descending order.

> return sorted(nums)[-n:]

This line erroneously sorts the entire list and takes the last n elements, which may not be the n largest elements in descending order. This error results in incorrect function output when the largest elements should be reported.

Overall, the function fails to perform as expected due to a critical error in using the wrong sorting and slicing approach. The incorrect manipulation of list sorting and slicing introduces a fundamental flaw, misidentifying the intended behavior of finding the n largest elements.

Here's a better solution:
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```

---


You are an expert Python programmer, and here is your task: Write a python function to check whether the two numbers differ at one bit position only or not.
Your code should pass these tests.

```python
def differ_at_one_bit_pos(a, b):
    return a ^ b == 1

assert differ_at_one_bit_pos(13, 9) == True
assert differ_at_one_bit_pos(15, 8) == False
assert differ_at_one_bit_pos(2, 4) == False
assert differ_at_one_bit_pos(2, 3) == True
assert differ_at_one_bit_pos(5, 1) == True
assert differ_at_one_bit_pos(1, 5) == True
```

Execution: AssertionError()

What's the problem with the above code?

1. The function incorrectly returns True for inputs that do not differ at exactly one bit position. This indicates a logical error in the implementation due to incorrect comparison of the XOR result.

2. Let's check the code:

> a = 13
> b = 9

This should identify that 13 (1101 in binary) and 9 (1001 in binary) differ at more than one bit position and should return False.

> return a ^ b == 1

This line erroneously checks if the XOR result is exactly 1, which does not account for multiple differing bits. This error results in incorrect function output when more than one bit differs.

Overall, the function fails to perform as expected due to a critical error in comparing the XOR result. The incorrect manipulation of the XOR operation introduces a fundamental flaw, misidentifying the intended behavior of checking for exactly one differing bit.

Here's a better solution:
```python
def is_power_of_two(x):
    return x and (not(x & (x - 1)))

def differ_at_one_bit_pos(a, b):
    return is_power_of_two(a ^ b)
```

---

You are an expert Python programmer, and here is your task: Write a function to find all words which are at least 4 characters long in a string.
Your code should pass these tests.

```python
def find_char_long(text):
    return text.split()

assert set(find_char_long('Please move back to stream')) == set(['Please', 'move', 'back', 'stream'])
assert set(find_char_long('Jing Eco and Tech')) == set(['Jing', 'Tech'])
assert set(find_char_long('Jhingai wulu road Zone 3')) == set(['Jhingai', 'wulu', 'road', 'Zone'])
```

Execution: AssertionError()

What's the problem with the above code?

1. The function incorrectly returns all words in the string, rather than only those that are at least 4 characters long. This indicates a logical error in the implementation due to incorrect filtering of words.

2. Let's check the code:

> text = 'Please move back to stream'

This should identify all words which are at least 4 characters long, which are 'Please', 'move', 'back', and 'stream'.

> return text.split()

This line erroneously splits the string into all words, without filtering out words shorter than 4 characters. This error results in incorrect function output when only words with at least 4 characters should be reported.

Overall, the function fails to perform as expected due to a critical error in filtering the words. The incorrect manipulation of string splitting introduces a fundamental flaw, misidentifying the intended behavior of finding words with a minimum length.

Here's a better solution:
```python
import re

def find_char_long(text):
    return re.findall(r"\b\w{4,}\b", text)
```"""


CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP = """{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests.

```python
{answer}

{tests}
```

What's the problem with the above code?

{critique}"""


MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL = """You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists.
Your code should pass these tests.

```python
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) | set(test_tup2))

assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))
```

What's the problem with the above code?

1. The function incorrectly returns a set of all unique elements from both lists rather than the intersection. This indicates a logical error in the implementation due to incorrect set operation.

2. Let's check the code:

> test_tup1 = (3, 4, 5, 6)
> test_tup2 = (5, 7, 4, 10)

This should define two lists of numbers where only 4 and 5 are common in both.

> return tuple(set(test_tup1) | set(test_tup2))

This line erroneously uses the set union operator (`|`) which combines all elements from both sets, instead of the set intersection operator (`&`) which would correctly identify elements present in both sets. This error results in incorrect function output when no duplicates should be reported.

Overall, the function fails to perform as expected due to a critical error in using the wrong set operation. The incorrect manipulation of set operators introduces a fundamental flaw, mistaking union for intersection, thus misidentifying the intended behavior of finding common elements.

Here's a better solution:
```python
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) & set(test_tup2))
```

---

You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers.
Your code should pass these tests.

```python
import math

def is_not_prime(n):
    if n == 2:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False
```

What's the problem with the above code?

> n = 2

This should identify the number 2, which is a prime number and should return False.

> if n == 2:
> return True

This line erroneously returns True for the input 2, indicating it as not prime, which is incorrect. The condition should be modified to correctly handle prime identification.

Here's a better solution:
```python
def is_not_prime(n):
    result = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
            break
    return result
```

---

You are an expert Python programmer, and here is your task: Write a function to find the n largest integers from a given list of numbers, returned in descending order.
Your code should pass these tests.

```python
def heap_queue_largest(nums, n):
    return sorted(nums)[-n:]

assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]
```

What's the problem with the above code?

1. The function incorrectly returns the sorted list of all elements, taking the last n elements. This indicates a logical error in the implementation due to incorrect sorting and slicing.

2. Let's check the code:

> nums = [25, 35, 22, 85, 14, 65, 75, 22, 58]
> n = 3

This should identify the three largest numbers in the list, which are 85, 75, and 65, in descending order.

> return sorted(nums)[-n:]

This line erroneously sorts the entire list and takes the last n elements, which may not be the n largest elements in descending order. This error results in incorrect function output when the largest elements should be reported.

Overall, the function fails to perform as expected due to a critical error in using the wrong sorting and slicing approach. The incorrect manipulation of list sorting and slicing introduces a fundamental flaw, misidentifying the intended behavior of finding the n largest elements.

Here's a better solution:
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```

---


You are an expert Python programmer, and here is your task: Write a python function to check whether the two numbers differ at one bit position only or not.
Your code should pass these tests.

```python
def differ_at_one_bit_pos(a, b):
    return a ^ b == 1

assert differ_at_one_bit_pos(13, 9) == True
assert differ_at_one_bit_pos(15, 8) == False
assert differ_at_one_bit_pos(2, 4) == False
assert differ_at_one_bit_pos(2, 3) == True
assert differ_at_one_bit_pos(5, 1) == True
assert differ_at_one_bit_pos(1, 5) == True
```

What's the problem with the above code?

1. The function incorrectly returns True for inputs that do not differ at exactly one bit position. This indicates a logical error in the implementation due to incorrect comparison of the XOR result.

2. Let's check the code:

> a = 13
> b = 9

This should identify that 13 (1101 in binary) and 9 (1001 in binary) differ at more than one bit position and should return False.

> return a ^ b == 1

This line erroneously checks if the XOR result is exactly 1, which does not account for multiple differing bits. This error results in incorrect function output when more than one bit differs.

Overall, the function fails to perform as expected due to a critical error in comparing the XOR result. The incorrect manipulation of the XOR operation introduces a fundamental flaw, misidentifying the intended behavior of checking for exactly one differing bit.

Here's a better solution:
```python
def is_power_of_two(x):
    return x and (not(x & (x - 1)))

def differ_at_one_bit_pos(a, b):
    return is_power_of_two(a ^ b)
```

---

You are an expert Python programmer, and here is your task: Write a function to find all words which are at least 4 characters long in a string.
Your code should pass these tests.

```python
def find_char_long(text):
    return text.split()

assert set(find_char_long('Please move back to stream')) == set(['Please', 'move', 'back', 'stream'])
assert set(find_char_long('Jing Eco and Tech')) == set(['Jing', 'Tech'])
assert set(find_char_long('Jhingai wulu road Zone 3')) == set(['Jhingai', 'wulu', 'road', 'Zone'])
```

What's the problem with the above code?

1. The function incorrectly returns all words in the string, rather than only those that are at least 4 characters long. This indicates a logical error in the implementation due to incorrect filtering of words.

2. Let's check the code:

> text = 'Please move back to stream'

This should identify all words which are at least 4 characters long, which are 'Please', 'move', 'back', and 'stream'.

> return text.split()

This line erroneously splits the string into all words, without filtering out words shorter than 4 characters. This error results in incorrect function output when only words with at least 4 characters should be reported.

Overall, the function fails to perform as expected due to a critical error in filtering the words. The incorrect manipulation of string splitting introduces a fundamental flaw, misidentifying the intended behavior of finding words with a minimum length.

Here's a better solution:
```python
import re

def find_char_long(text):
    return re.findall(r"\b\w{4,}\b", text)
```"""


