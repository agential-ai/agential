"""Self-Refine prompts and fewshot examples."""

# ======================================================================== HOTPOTQA ======================================================================== #


SELF_REFINE_INSTRUCTION_HOTPOTQA = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """


HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES = """Q: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
A: 1,800 to 7,000 ft

What's the problem with the above answer?

1. Plausibility:

The question asks for the elevation range, and the answer provides "1,800 to 7,000 ft", which fits the expected format and seems plausible.

2. Truthfulness:

Let's search the question in google:
> Search Query: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into? site: wikipedia.org
> Evidence: [Colorado orogeny - Wikipedia] The eastern sector of the Colorado orogeny extends into regions that have elevations ranging from approximately 1,800 ft to 7,000 ft.

The evidence confirms that the eastern sector of the Colorado orogeny extends into regions with elevations ranging from 1,800 ft to 7,000 ft. Thus, the proposed answer is correct and truthful.

Let's search the proposed answer in google:
> Search Query: Elevation range for the area that the eastern sector of the Colorado orogeny extends into 1,800 to 7,000 ft site:wikipedia.org
> Evidence: [Colorado orogeny - Wikipedia] The eastern sector of the Colorado orogeny extends into regions that have elevations ranging from approximately 1,800 ft to 7,000 ft.

The evidence supports the proposed answer, confirming that it correctly identifies the elevation range.

Above all, the proposed answer correctly identifies the elevation range for the area that the eastern sector of the Colorado orogeny extends into. However, the explanation can be improved for clarity.

---

Q: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?
A: Richard Nixon

1. Plausibility:

The answer should provide the person after whom the character Milhouse was named. The proposed answer states "Richard Nixon," which fits the expected format and seems plausible.

2. Truthfulness:

Let's search the question in google:
> Search Query: "The Simpsons" character Milhouse named after who? site: wikipedia.org
> Evidence: [Milhouse Van Houten - Wikipedia] The character Milhouse was named after U.S. President Richard Nixon, whose middle name was Milhous.

The evidence confirms that Milhouse was indeed named after Richard Nixon, whose middle name was Milhous. Thus, the proposed answer is correct and truthful.

Let's search the proposed answer in google:
> Search Query: Milhouse character named after Richard Nixon site: wikipedia.org
> Evidence: [Milhouse Van Houten - Wikipedia] The character Milhouse was named after U.S. President Richard Nixon, whose middle name was Milhous.

The evidence supports the proposed answer, confirming that it correctly identifies the person after whom Milhouse was named.

Above all, the proposed answer correctly identifies that Milhouse was named after Richard Nixon. However, the explanation can be improved for clarity.

---

Q: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
A: The Saimaa Gesture

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of a documentary about Finnish rock groups, and the answer is "The Saimaa Gesture," which sounds plausible given the context.

2. Truthfulness:

Let's search for information about "The Saimaa Gesture":
> Search Query: The Saimaa Gesture documentary site: wikipedia.org
> Evidence: [The Saimaa Gesture - Wikipedia] The Saimaa Gesture is a 1981 documentary film about three Finnish rock groups on a tour around Lake Saimaa.

The evidence confirms that The Saimaa Gesture is indeed a documentary about Finnish rock groups, as stated in the proposed answer.

Let's search for information about "Adam Clayton Powell":
> Search Query: Adam Clayton Powell documentary site: wikipedia.org
> Evidence: [Adam Clayton Powell - Wikipedia] Adam Clayton Powell Jr. was an American Baptist pastor and politician who represented Harlem, New York City, in the United States House of Representatives.

The evidence shows that Adam Clayton Powell is not related to Finnish rock groups, supporting the correctness of the proposed answer.

Above all, the proposed answer correctly identifies that The Saimaa Gesture is the documentary about Finnish rock groups. However, the explanation can be improved for clarity.

---

Q: What profession does Nicholas Ray and Elia Kazan have in common?
A: director, screenwriter, actor

What's the problem with the above answer?

1. Plausibility:
The question asks for the profession that Nicholas Ray and Elia Kazan have in common. The proposed answer lists multiple professions without clarifying if they both held all those professions. So, it's not precise.

2. Truthfulness:
Let's search for the professions of Nicholas Ray and Elia Kazan.

> Search Query: Nicholas Ray profession site.org
> Evidence: [Nicholas Ray - Wikipedia] Nicholas Ray was an American film director and screenwriter, best known for the movie "Rebel Without a Cause".

The evidence suggests that Nicholas Ray was a director and screenwriter.

> Search Query: Elia Kazan profession site.org
> Evidence: [Elia Kazan - Wikipedia] Elia Kazan was a Greek-American director, producer, writer, and actor. He is noted for his work on Broadway and in Hollywood.

The evidence suggests that Elia Kazan was a director, writer, and actor.

The proposed answer correctly identifies the professions of both individuals but does not specify which profession they have in common.

---

Q: Which magazine was started first Arthur's Magazine or First for Women?
A: Arthur's Magazine

What's the problem with the above answer?

1. Plausibility:

The answer should be chosen between "Arthur's Magazine" and "First for Women," and the answer is "First for Women," so it seems plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: Arthur's Magazine first publication date
> Evidence: [Arthur's Magazine - Wikipedia] Arthur's Magazine was first published in 1844.

The evidence shows that Arthur's Magazine was first published in 1844.

> Search Query: First for Women first publication date
> Evidence: [First for Women - Wikipedia] First for Women was first published in 1989.

The evidence shows that First for Women was indeed first published in 1989.

The proposed answer gives the wrong dates of publication for Arthur's Magazine. The final answer "Arthur's Magazine" is correct since 1844 < 1989."""


SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA = """{examples}
(END OF EXAMPLES)

Question: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

"""


HOTPOTQA_REFINE_FEWSHOT_EXAMPLES = """Q: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
A: 1,800 to 7,000 ft

What's the problem with the above answer?

1. Plausibility:

The question asks for the elevation range, and the answer provides "1,800 to 7,000 ft", which fits the expected format and seems plausible.

2. Truthfulness:

Let's search the question in google:
> Search Query: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into? site: wikipedia.org
> Evidence: [Colorado orogeny - Wikipedia] The eastern sector of the Colorado orogeny extends into regions that have elevations ranging from approximately 1,800 ft to 7,000 ft.

The evidence confirms that the eastern sector of the Colorado orogeny extends into regions with elevations ranging from 1,800 ft to 7,000 ft. Thus, the proposed answer is correct and truthful.

Let's search the proposed answer in google:
> Search Query: Elevation range for the area that the eastern sector of the Colorado orogeny extends into 1,800 to 7,000 ft site:wikipedia.org
> Evidence: [Colorado orogeny - Wikipedia] The eastern sector of the Colorado orogeny extends into regions that have elevations ranging from approximately 1,800 ft to 7,000 ft.

The evidence supports the proposed answer, confirming that it correctly identifies the elevation range.

Above all, the proposed answer correctly identifies the elevation range for the area that the eastern sector of the Colorado orogeny extends into. However, the explanation can be improved for clarity.

Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Here's the most possible answer: Let's think step by step. The Colorado orogeny refers to a series of mountain-building events that shaped the region. The eastern sector of this orogeny extends into an area with an elevation range from approximately 1,800 ft to 7,000 ft. So the answer is: 1,800 to 7,000 ft.

---

Q: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?
A: Richard Nixon

1. Plausibility:

The answer should provide the person after whom the character Milhouse was named. The proposed answer states "Richard Nixon," which fits the expected format and seems plausible.

2. Truthfulness:

Let's search the question in google:
> Search Query: "The Simpsons" character Milhouse named after who? site: wikipedia.org
> Evidence: [Milhouse Van Houten - Wikipedia] The character Milhouse was named after U.S. President Richard Nixon, whose middle name was Milhous.

The evidence confirms that Milhouse was indeed named after Richard Nixon, whose middle name was Milhous. Thus, the proposed answer is correct and truthful.

Let's search the proposed answer in google:
> Search Query: Milhouse character named after Richard Nixon site: wikipedia.org
> Evidence: [Milhouse Van Houten - Wikipedia] The character Milhouse was named after U.S. President Richard Nixon, whose middle name was Milhous.

The evidence supports the proposed answer, confirming that it correctly identifies the person after whom Milhouse was named.

Above all, the proposed answer correctly identifies that Milhouse was named after Richard Nixon. However, the explanation can be improved for clarity.

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Here's the most possible answer: Let's think step by step. Allie Goertz wrote a song about Milhouse, a character from "The Simpsons" created by Matt Groening. Matt Groening named Milhouse after U.S. President Richard Nixon, whose middle name was Milhous. So the answer is: Richard Nixon.

---

Q: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
A: The Saimaa Gesture

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of a documentary about Finnish rock groups, and the answer is "The Saimaa Gesture," which sounds plausible given the context.

2. Truthfulness:

Let's search for information about "The Saimaa Gesture":
> Search Query: The Saimaa Gesture documentary site: wikipedia.org
> Evidence: [The Saimaa Gesture - Wikipedia] The Saimaa Gesture is a 1981 documentary film about three Finnish rock groups on a tour around Lake Saimaa.

The evidence confirms that The Saimaa Gesture is indeed a documentary about Finnish rock groups, as stated in the proposed answer.

Let's search for information about "Adam Clayton Powell":
> Search Query: Adam Clayton Powell documentary site: wikipedia.org
> Evidence: [Adam Clayton Powell - Wikipedia] Adam Clayton Powell Jr. was an American Baptist pastor and politician who represented Harlem, New York City, in the United States House of Representatives.

The evidence shows that Adam Clayton Powell is not related to Finnish rock groups, supporting the correctness of the proposed answer.

Above all, the proposed answer correctly identifies that The Saimaa Gesture is the documentary about Finnish rock groups. However, the explanation can be improved for clarity.

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Here's the most possible answer: Let's think step by step. Adam Clayton Powell is a name associated with an American politician and civil rights leader. The Saimaa Gesture is a documentary about three Finnish rock groups on a tour around Lake Saimaa. Therefore, The Saimaa Gesture is about Finnish rock groups. So the answer is: The Saimaa Gesture.

---

Q: What profession does Nicholas Ray and Elia Kazan have in common?
A: director, screenwriter, actor

What's the problem with the above answer?

1. Plausibility:
The question asks for the profession that Nicholas Ray and Elia Kazan have in common. The proposed answer lists multiple professions without clarifying if they both held all those professions. So, it's not precise.

2. Truthfulness:
Let's search for the professions of Nicholas Ray and Elia Kazan.

> Search Query: Nicholas Ray profession site.org
> Evidence: [Nicholas Ray - Wikipedia] Nicholas Ray was an American film director and screenwriter, best known for the movie "Rebel Without a Cause".

The evidence suggests that Nicholas Ray was a director and screenwriter.

> Search Query: Elia Kazan profession site.org
> Evidence: [Elia Kazan - Wikipedia] Elia Kazan was a Greek-American director, producer, writer, and actor. He is noted for his work on Broadway and in Hollywood.

The evidence suggests that Elia Kazan was a director, writer, and actor.

The proposed answer correctly identifies the professions of both individuals but does not specify which profession they have in common.

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Here's the most possible answer: Let's think step by step. Nicholas Ray was a director and screenwriter. Elia Kazan was a director, writer, and actor. Both Nicholas Ray and Elia Kazan worked as directors and screenwriters. So the answer is: director and screenwriter.

---

Q: Which magazine was started first Arthur's Magazine or First for Women?
A: Arthur's Magazine

What's the problem with the above answer?

1. Plausibility:

The answer should be chosen between "Arthur's Magazine" and "First for Women," and the answer is "First for Women," so it seems plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: Arthur's Magazine first publication date
> Evidence: [Arthur's Magazine - Wikipedia] Arthur's Magazine was first published in 1844.

The evidence shows that Arthur's Magazine was first published in 1844.

> Search Query: First for Women first publication date
> Evidence: [First for Women - Wikipedia] First for Women was first published in 1989.

The evidence shows that First for Women was indeed first published in 1989.

The proposed answer gives the wrong dates of publication for Arthur's Magazine. The final answer "Arthur's Magazine" is correct since 1844 < 1989.

Question: Which magazine was started first, Arthur's Magazine or First for Women?
Here's the most possible answer: Let's think step by step. Arthur's Magazine was first published in 1844. First for Women was first published in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first. So the answer is: Arthur's Magazine."""


SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA = """{examples}
(END OF EXAMPLES)

Question: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

{critique}

Question: {question}
Here's the most possible answer: """


# ======================================================================== TRIVIAQA ======================================================================== #


SELF_REFINE_INSTRUCTION_TRIVIAQA = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """


TRIVIAQA_CRITIQUE_FEWSHOT_EXAMPLES = """Q: Mendelssohn's 'Wedding March' was. originally written as incidental music for which Shakespeare play in 1842?
A: A Midsummer Night's Dream

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the Shakespeare play, and the answer is "Hamlet," which is a name of a Shakespeare play. So it's plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842? site: wikipedia.org
> Evidence: [Wedding March (Mendelssohn) - Wikipedia] The "Wedding March" in C major, written in 1842, is one of the most famous pieces by Mendelssohn. It was written as incidental music for William Shakespeare's play "A Midsummer Night's Dream."

The evidence suggests that Mendelssohn's 'Wedding March' was written as incidental music for "A Midsummer Night's Dream," not "Hamlet."

Let's check the proposed answer:

> Search Query: Mendelssohn's 'Wedding March' was originally written as incidental music for the play "Hamlet" in 1842.
> Evidence: [Hamlet - Wikipedia] "Hamlet" is a tragedy written by William Shakespeare at an uncertain date between 1599 and 1602. Mendelssohn did not write incidental music for "Hamlet."

The evidence shows that Mendelssohn did not write incidental music for "Hamlet," contradicting the proposed answer.

Above all, the proposed answer is incorrect because Mendelssohn's 'Wedding March' was not written for "Hamlet." It was actually written for "A Midsummer Night's Dream."

---

Q: \"\"\"Christ in the House of his Parents\"\" is one of the best known paintings of which artist?"
A: John Millais

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the artist, and the answer is "John Millais," which is a name of an artist. So it's plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: "Christ in the House of his Parents" is one of the best known paintings of which artist? site: wikipedia.org
> Evidence: [Christ in the House of His Parents - Wikipedia] "Christ in the House of His Parents" is a painting by John Everett Millais, which depicts the Holy Family in Saint Joseph's carpentry workshop.

The evidence shows that "Christ in the House of his Parents" is indeed a painting by John Everett Millais, which supports the proposed answer.

Let's check the proposed answer:

> Search Query: "Christ in the House of his Parents" painting by John Millais.
> Evidence: [John Everett Millais - Wikipedia] John Everett Millais was a British painter and one of the founders of the Pre-Raphaelite Brotherhood. One of his most famous works is "Christ in the House of His Parents."

The evidence confirms that John Everett Millais is the artist of "Christ in the House of his Parents."

Above all, the proposed answer is correct because "Christ in the House of his Parents" is indeed a painting by John Millais.

---

Q: Who designed the National Theatre building on the South Bank in London ?
A: Sir Denys Lasdun

What's the problem with the above answer?

1. Plausibility:
The question asks for the name of the designer of the National Theatre building on the South Bank in London, and the answer "Sir Norman Foster" is a plausible name of an architect. However, it might not be accurate.

2. Truthfulness:
Let's search the question in google:

> Search Query: Who designed the National Theatre building on the South Bank in London?
> Evidence: [National Theatre - Wikipedia] The National Theatre building on the South Bank in London was designed by Sir Denys Lasdun and opened in 1976.

The evidence shows that the National Theatre building was designed by Sir Denys Lasdun, not Sir Norman Foster.

Let's verify further to be sure.

> Search Query: Sir Norman Foster National Theatre building
> Evidence: [Sir Norman Foster - Wikipedia] Sir Norman Foster is a notable British architect known for various buildings, but there is no mention of him designing the National Theatre building.

The evidence confirms that Sir Norman Foster did not design the National Theatre building.

---

Q: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?
A: Bodie Miller

What's the problem with the above answer?

1. Plausibility:
The question asks for the name of the skier, and the answer is "Bodie Miller," which is a name. So it's plausible.

2. Truthfulness:
Let's search the question in google:

> Search Query: American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics site.org
> Evidence: [Bode Miller - Wikipedia] At the 2010 Winter Olympics, Miller won the gold medal in the super combined.

The evidence confirms that Bode Miller won the gold medal in the super combined at the 2010 Winter Olympics.

---

Q: Which English author wrote the novel "1984"?
A: George Orwell wrote the novel "1984". So the answer is: George Orwell.

What's the problem with the above answer?

1. Plausibility:
The question asks for the name of the author, and the answer is "George Orwell," which is a name. So it's plausible.

2. Truthfulness:
Let's search the question in google:

> Search Query: Which English author wrote the novel "1984"? site:wikipedia.org
> Evidence: [1984 (novel) - Wikipedia] "Nineteen Eighty-Four: A Novel, often published as '1984', is a dystopian social science fiction novel and cautionary tale, written by the English writer George Orwell.

The evidence confirms that George Orwell is indeed the author of the novel "1984"."""


SELF_REFINE_CRITIQUE_INSTRUCTION_TRIVIAQA = """{examples}
(END OF EXAMPLES)

Question: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

"""


TRIVIAQA_REFINE_FEWSHOT_EXAMPLES = """Q: Mendelssohn's 'Wedding March' was. originally written as incidental music for which Shakespeare play in 1842?
A: A Midsummer Night's Dream

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the Shakespeare play, and the answer is "Hamlet," which is a name of a Shakespeare play. So it's plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842? site: wikipedia.org
> Evidence: [Wedding March (Mendelssohn) - Wikipedia] The "Wedding March" in C major, written in 1842, is one of the most famous pieces by Mendelssohn. It was written as incidental music for William Shakespeare's play "A Midsummer Night's Dream."

The evidence suggests that Mendelssohn's 'Wedding March' was written as incidental music for "A Midsummer Night's Dream," not "Hamlet."

Let's check the proposed answer:

> Search Query: Mendelssohn's 'Wedding March' was originally written as incidental music for the play "Hamlet" in 1842.
> Evidence: [Hamlet - Wikipedia] "Hamlet" is a tragedy written by William Shakespeare at an uncertain date between 1599 and 1602. Mendelssohn did not write incidental music for "Hamlet."

The evidence shows that Mendelssohn did not write incidental music for "Hamlet," contradicting the proposed answer.

Above all, the proposed answer is incorrect because Mendelssohn's 'Wedding March' was not written for "Hamlet." It was actually written for "A Midsummer Night's Dream."

Question: Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842?
Here's the most possible answer: Let's think step by step. Mendelssohn's 'Wedding March' was written as incidental music for William Shakespeare's play "A Midsummer Night's Dream" in 1842. So the answer is: A Midsummer Night's Dream.

---

Q: \"\"\"Christ in the House of his Parents\"\" is one of the best known paintings of which artist?"
A: John Millais

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the artist, and the answer is "John Millais," which is a name of an artist. So it's plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: "Christ in the House of his Parents" is one of the best known paintings of which artist? site: wikipedia.org
> Evidence: [Christ in the House of His Parents - Wikipedia] "Christ in the House of His Parents" is a painting by John Everett Millais, which depicts the Holy Family in Saint Joseph's carpentry workshop.

The evidence shows that "Christ in the House of his Parents" is indeed a painting by John Everett Millais, which supports the proposed answer.

Let's check the proposed answer:

> Search Query: "Christ in the House of his Parents" painting by John Millais.
> Evidence: [John Everett Millais - Wikipedia] John Everett Millais was a British painter and one of the founders of the Pre-Raphaelite Brotherhood. One of his most famous works is "Christ in the House of His Parents."

The evidence confirms that John Everett Millais is the artist of "Christ in the House of his Parents."

Above all, the proposed answer is correct because "Christ in the House of his Parents" is indeed a painting by John Millais.

Question: "Christ in the House of his Parents" is one of the best known paintings of which artist?
Here's the most possible answer: "Christ in the House of his Parents" is one of the best known paintings of John Millais. So the answer is: John Millais.

---

Q: Who designed the National Theatre building on the South Bank in London ?
A: Sir Denys Lasdun

What's the problem with the above answer?

1. Plausibility:
The question asks for the name of the designer of the National Theatre building on the South Bank in London, and the answer "Sir Norman Foster" is a plausible name of an architect. However, it might not be accurate.

2. Truthfulness:
Let's search the question in google:

> Search Query: Who designed the National Theatre building on the South Bank in London?
> Evidence: [National Theatre - Wikipedia] The National Theatre building on the South Bank in London was designed by Sir Denys Lasdun and opened in 1976.

The evidence shows that the National Theatre building was designed by Sir Denys Lasdun, not Sir Norman Foster.

Let's verify further to be sure.

> Search Query: Sir Norman Foster National Theatre building
> Evidence: [Sir Norman Foster - Wikipedia] Sir Norman Foster is a notable British architect known for various buildings, but there is no mention of him designing the National Theatre building.

The evidence confirms that Sir Norman Foster did not design the National Theatre building.

Question: Who designed the National Theatre building on the South Bank in London?
Here's the most possible answer: The National Theatre building on the South Bank in London was designed by Sir Denys Lasdun. So the answer is: Sir Denys Lasdun.

---

Q: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?
A: Bodie Miller

What's the problem with the above answer?

1. Plausibility:
The question asks for the name of the skier, and the answer is "Bodie Miller," which is a name. So it's plausible.

2. Truthfulness:
Let's search the question in google:

> Search Query: American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics site.org
> Evidence: [Bode Miller - Wikipedia] At the 2010 Winter Olympics, Miller won the gold medal in the super combined.

The evidence confirms that Bode Miller won the gold medal in the super combined at the 2010 Winter Olympics.

Question: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?
Here's the most possible answer: Bodie Miller won the gold medal in the Men's Combined at the 2010 Winter Olympics. So the answer is: Bodie Miller

---

Q: Which English author wrote the novel "1984"?
A: George Orwell wrote the novel "1984". So the answer is: George Orwell.

What's the problem with the above answer?

1. Plausibility:
The question asks for the name of the author, and the answer is "George Orwell," which is a name. So it's plausible.

2. Truthfulness:
Let's search the question in google:

> Search Query: Which English author wrote the novel "1984"? site:wikipedia.org
> Evidence: [1984 (novel) - Wikipedia] "Nineteen Eighty-Four: A Novel, often published as '1984', is a dystopian social science fiction novel and cautionary tale, written by the English writer George Orwell.

The evidence confirms that George Orwell is indeed the author of the novel "1984".

Question: Which English author wrote the novel "1984"?
Here's the most possible answer: George Orwell wrote the novel "1984". So the answer is: George Orwell."""


SELF_REFINE_REFINE_INSTRUCTION_TRIVIAQA = """{examples}
(END OF EXAMPLES)

Question: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

{critique}

Question: {question}
Here's the most possible answer: """


# ======================================================================== AMBIGNQ ======================================================================== #


SELF_REFINE_INSTRUCTION_AMBIGNQ = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """


AMBIGNQ_CRITIQUE_FEWSHOT_EXAMPLES = """Q: When did men's figure skating become a summer Olympic sport?
A: 1908

What's the problem with the above answer?

1. Plausibility:

The question asks for the time men's figure skating become a summer Olympic sport, and the answer "never" does not provide a time. So it's not plausible. The answer should be a time, like year or date.

2. Truthfulness:

Let's search the question in google:

> Search Query: When did men's figure skating become a summer Olympic sport?
> Evidence: [Figure skating at the Olympic Games - Wikipedia] Figure skating was first contested in the Olympic Games at the 1908 Summer Olympics . Since 1924, the sport has been a part of the Winter Olympic Games .

The evidence suggests Figure skating became an Olympic sport at the 1908 Summer Olympics, and has been a part of the Winter Olympic Games in 1924.

The answer is wrong by saying that men's figure skating has never been a part of the Summer Olympics.

---

Q: When was the last time the USA men's national soccer team missed the World Cup?
A: 2018

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

---

Q: What does barium do in a ct scan?
A: to improve visualization of the gastrointestinal tract

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

---

Q: Where was the fort located at which the first shot of the civil war was fired?
A: Charleston Harbor, South Carolina

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

---

Q: When did nando's come to the uk?
A: 1992

What's the problem with the above answer?

1. Plausibility:

The question asks for a time, and the answer is "1992", which is a year. So it's plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: When did nando's come to the uk? site: wikipedia.org
> Evidence: [Nando's - Wikipedia] Nando's opened its first restaurants within the United Kingdom, in 1992, in the west London suburbs of Ealing and Earls Court, initially focusing on takeaway food.

The evidence suggests that Nando's first opened in the UK in 1992, which is consistent with the proposed answer. We can provide more detailed information in the answer."""


SELF_REFINE_CRITIQUE_INSTRUCTION_AMBIGNQ = """{examples}
(END OF EXAMPLES)

Question: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

"""


AMBIGNQ_REFINE_FEWSHOT_EXAMPLES = """Q: When did men's figure skating become a summer Olympic sport?
A: 1908

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

Q: When was the last time the USA men's national soccer team missed the World Cup?
A: 2018

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

Q: What does barium do in a ct scan?
A: to improve visualization of the gastrointestinal tract

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

Q: Where was the fort located at which the first shot of the civil war was fired?
A: Charleston Harbor, South Carolina

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

Q: When did nando's come to the uk?
A: 1992

What's the problem with the above answer?

1. Plausibility:

The question asks for a time, and the answer is "1992", which is a year. So it's plausible.

2. Truthfulness:

Let's search the question in google:

> Search Query: When did nando's come to the uk? site: wikipedia.org
> Evidence: [Nando's - Wikipedia] Nando's opened its first restaurants within the United Kingdom, in 1992, in the west London suburbs of Ealing and Earls Court, initially focusing on takeaway food.

The evidence suggests that Nando's first opened in the UK in 1992, which is consistent with the proposed answer. We can provide more detailed information in the answer.

Question: When did nando's come to the uk?
Here's the most possible answer: Nando's first opened its restaurants within the United Kingdom in 1992, specifically in the west London suburbs of Ealing and Earls Court. So the answer is: 1992"""


SELF_REFINE_REFINE_INSTRUCTION_AMBIGNQ = """{examples}
(END OF EXAMPLES)

Question: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

{critique}

Question: {question}
Here's the most possible answer: """


# ======================================================================== FEVER ======================================================================== #


SELF_REFINE_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. 

{examples}
(END OF EXAMPLES)

Claim: {question}
A: """


FEVER_CRITIQUE_FEWSHOT_EXAMPLES = """Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
A: SUPPORTS

What's the problem with the above answer?

1. Plausibility:

The answer "REFUTES" incorrectly negates the claim without supporting details.

2. Truthfulness:

> Search Query: Did Nikolaj Coster-Waldau work with Fox Broadcasting?
> Evidence: [Nikolaj Coster-Waldau - IMDb] Nikolaj Coster-Waldau appeared in the 2009 Fox television film Virtuality.

The evidence contradicts the proposed answer, confirming he did work with Fox in the television film Virtuality.

---

Claim: Stranger Things is set in Bloomington, Indiana.
A: REFUTES

What's the problem with the above answer?

1. Plausibility:

The answer is correct but lacks specific details that verify the claim.

2. Truthfulness:

> Search Query: Setting of Stranger Things
> Evidence: Stranger Things is set in the fictional town of Hawkins, Indiana, not Bloomington.

Although the proposed answer is correct, it could be more informative by mentioning the specific setting.

---

Claim: "Beautiful" by Christina Aguilera reached number two on the Billboard Hot 100 in 2003.
A: NOT ENOUGH INFO

What's the problem with the above answer?

1. Plausibility:

The answer "NOT ENOUGH INFO" is appropriate as it reflects the uncertainty due to incomplete information about the timing of the chart position.

2. Truthfulness:

> Search Query: Billboard Hot 100 position of "Beautiful" by Christina Aguilera in 2003
> Evidence: The song peaked at number two on the Billboard Hot 100, but the specific year it achieved this ranking was not directly specified in the sources found.

Given that the year 2003 is not verified in the available evidence, the proposed answer correctly reflects the uncertainty regarding the exact year of the chart position.

---

Claim: Tim Burton didn't direct the film "Edward Scissorhands."
A: REFUTES

What's the problem with the above answer?

1. Plausibility:
The answer "REFUTES" incorrectly negates the claim without supporting details.

2. Truthfulness:
> Search Query: Did Tim Burton direct the film "Edward Scissorhands"?
> Evidence: [Edward Scissorhands - IMDb] The film "Edward Scissorhands" was directed by Tim Burton and released in 1990.

The evidence supports the claim, confirming that Tim Burton directed "Edward Scissorhands."

---

Claim: "The Great Gatsby" was written by F. Scott Fitzgerald.
A: SUPPORTS

What's the problem with the above answer?

1. Plausibility:

The answer "SUPPORTS" is appropriate as it reflects the accuracy of the claim about the authorship of "The Great Gatsby".

2. Truthfulness:

> Search Query: Author of "The Great Gatsby"
> Evidence: [The Great Gatsby - Wikipedia] "The Great Gatsby is a 1925 novel written by American author F. Scott Fitzgerald that follows a cast of characters living in the fictional towns of West Egg and East Egg on prosperous Long Island in the summer of 1922."

The evidence confirms that F. Scott Fitzgerald is indeed the author of "The Great Gatsby"."""


SELF_REFINE_CRITIQUE_INSTRUCTION_FEVER = """{examples}
(END OF EXAMPLES)

Claim: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

"""


FEVER_REFINE_FEWSHOT_EXAMPLES = """Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
A: SUPPORTS

What's the problem with the above answer?

1. Plausibility:

The answer "REFUTES" incorrectly negates the claim without supporting details.

2. Truthfulness:

> Search Query: Did Nikolaj Coster-Waldau work with Fox Broadcasting?
> Evidence: [Nikolaj Coster-Waldau - IMDb] Nikolaj Coster-Waldau appeared in the 2009 Fox television film Virtuality.

The evidence contradicts the proposed answer, confirming he did work with Fox in the television film Virtuality.

Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Here's the most possible answer: Yes, Nikolaj Coster-Waldau worked with the Fox Broadcasting Company as he appeared in the 2009 Fox television film Virtuality. So the answer is: SUPPORTS.

---

Claim: Stranger Things is set in Bloomington, Indiana.
A: REFUTES

What's the problem with the above answer?

1. Plausibility:

The answer is correct but lacks specific details that verify the claim.

2. Truthfulness:

> Search Query: Setting of Stranger Things
> Evidence: Stranger Things is set in the fictional town of Hawkins, Indiana, not Bloomington.

Although the proposed answer is correct, it could be more informative by mentioning the specific setting.

Claim: Stranger Things is set in Bloomington, Indiana.
Here's the most possible answer: No, Stranger Things is set in the fictional town of Hawkins, Indiana. So the answer is: REFUTES.

---

Claim: "Beautiful" by Christina Aguilera reached number two on the Billboard Hot 100 in 2003.
A: NOT ENOUGH INFO

What's the problem with the above answer?

1. Plausibility:

The answer "NOT ENOUGH INFO" is appropriate as it reflects the uncertainty due to incomplete information about the timing of the chart position.

2. Truthfulness:

> Search Query: Billboard Hot 100 position of "Beautiful" by Christina Aguilera in 2003
> Evidence: The song peaked at number two on the Billboard Hot 100, but the specific year it achieved this ranking was not directly specified in the sources found.

Given that the year 2003 is not verified in the available evidence, the proposed answer correctly reflects the uncertainty regarding the exact year of the chart position.

Claim: "Beautiful" by Christina Aguilera reach number two on the Billboard Hot 100 in 2003.
Here's the most possible answer: The song "Beautiful" by Christina Aguilera peaked at number two on the Billboard Hot 100, but there is no specific evidence confirming this occurred in 2003. So the answer is: NOT ENOUGH INFO.

---

Claim: Tim Burton didn't direct the film "Edward Scissorhands."
A: REFUTES

What's the problem with the above answer?

1. Plausibility:
The answer "REFUTES" incorrectly negates the claim without supporting details.

2. Truthfulness:
> Search Query: Did Tim Burton direct the film "Edward Scissorhands"?
> Evidence: [Edward Scissorhands - IMDb] The film "Edward Scissorhands" was directed by Tim Burton and released in 1990.

The evidence supports the claim, confirming that Tim Burton directed "Edward Scissorhands."

Claim: Tim Burton didn't direct the film "Edward Scissorhands."
Here's the most possible answer: No. Tim Burton directed the film "Edward Scissorhands," which was released in 1990. So the answer is: REFUTES.

---

Claim: "The Great Gatsby" was written by F. Scott Fitzgerald.
A: SUPPORTS

What's the problem with the above answer?

1. Plausibility:

The answer "SUPPORTS" is appropriate as it reflects the accuracy of the claim about the authorship of "The Great Gatsby".

2. Truthfulness:

> Search Query: Author of "The Great Gatsby"
> Evidence: [The Great Gatsby - Wikipedia] "The Great Gatsby is a 1925 novel written by American author F. Scott Fitzgerald that follows a cast of characters living in the fictional towns of West Egg and East Egg on prosperous Long Island in the summer of 1922."

The evidence confirms that F. Scott Fitzgerald is indeed the author of "The Great Gatsby".

Claim: "The Great Gatsby" was written by F. Scott Fitzgerald.
Here's the most possible answer: The novel "The Great Gatsby" was written by F. Scott Fitzgerald, as confirmed by reliable sources. So the answer is: SUPPORTS."""


SELF_REFINE_REFINE_INSTRUCTION_FEVER = """{examples}
(END OF EXAMPLES)

Claim: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

{critique}

Claim: {question}
Here's the most possible answer: """


# ======================================================================== GSM8K ======================================================================== #


SELF_REFINE_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Question: {question}
# Python code, return answer"""


GSM8K_CRITIQUE_FEWSHOT_EXAMPLES = """Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
denny_lollipops = jason_lollipops_initial + jason_lollipops_after
answer = denny_lollipops
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
jason_lollipops_initial = 20
# looks good

# Let's check the other parts
jason_lollipops_after = 12
# looks good

# Let's check the other parts
denny_lollipops = jason_lollipops_initial + jason_lollipops_after
# wrong! Jason gave some lollipops to Denny, so we need to subtract the lollipops Jason has now from the lollipops he had initially. The correct calculation should be jason_lollipops_initial - jason_lollipops_after.

# Let's check the other parts
answer = denny_lollipops
# looks good

---

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
```python
trees_initial = 15
trees_after = 21
trees_added = trees_initial - trees_after
answer = trees_added
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
trees_initial = 15
# looks good

# Let's check the other parts
trees_after = 21
# looks good

# Let's check the other parts
trees_added = trees_initial - trees_after
# wrong! The number of trees added should be calculated by subtracting the initial number of trees from the number of trees after planting. The correct calculation should be trees_after - trees_initial.

# Let's check the other parts
answer = trees_added
# looks good

---

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
```python
toys_initial = 5
mom_toys = 2
dad_toys = 2
total_received = mom_toys + dad_toys
total_toys = toys_initial + total_received
answer = total_toys
```

# There is no error in the code above. The logic and calculations are correct. Let us verify step-by-step:

# Let us go through the calculation step-by-step
toys_initial = 5
# looks good

# Let's check the other parts
mom_toys = 2
# looks good

# Let's check the other parts
dad_toys = 2
# looks good

# Let's check the other parts
total_received = mom_toys + dad_toys
# looks good

# Let's check the other parts
total_toys = toys_initial + total_received
# looks good

# Let's check the other parts
answer = total_toys
# looks good

---

Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
```python
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between Monday and Thursday
computers_added = computers_per_day + num_days
computers_total = computers_initial + computers_added
answer = computers_total
```

There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
computers_initial = 9
#looks good

# Let's check the other parts
computers_per_day = 5
# looks good

# Let's check the other parts
num_days = 4 # 4 days between Monday and Thursday
# looks good

#Let's check the other parts
computers_added = computers_per_day + num_days

wrong! The error here is the incorrect addition. We should multiply the number of computers added per day by the number of days, not add them together. This causes the calculation of computers_added to be incorrect.

---

Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
```python
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
answer = golf_balls_left
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
golf_balls_initial = 58
# looks good

# Let's check the other parts
golf_balls_lost_tuesday = 23
# looks good

# Let's check the other parts
golf_balls_lost_wednesday = 2

# Let's check the other parts
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
# looks good

#Let's check the other parts
answer = golf_balls_left
# looks good

There is no error in the code! It is correct!"""


SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Question: {question}
```python
{answer}
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good."""


GSM8K_REFINE_FEWSHOT_EXAMPLES = """Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
denny_lollipops = jason_lollipops_initial + jason_lollipops_after
answer = denny_lollipops
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
jason_lollipops_initial = 20
# looks good

# Let's check the other parts
jason_lollipops_after = 12
# looks good

# Let's check the other parts
denny_lollipops = jason_lollipops_initial + jason_lollipops_after
# wrong! Jason gave some lollipops to Denny, so we need to subtract the lollipops Jason has now from the lollipops he had initially. The correct calculation should be jason_lollipops_initial - jason_lollipops_after.

# Let's check the other parts
answer = denny_lollipops
# looks good

Okay! Here is the rewrite:

```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
denny_lollipops = jason_lollipops_initial - jason_lollipops_after
answer = denny_lollipops
```

---

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
```python
trees_initial = 15
trees_after = 21
trees_added = trees_initial - trees_after
answer = trees_added
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
trees_initial = 15
# looks good

# Let's check the other parts
trees_after = 21
# looks good

# Let's check the other parts
trees_added = trees_initial - trees_after
# wrong! The number of trees added should be calculated by subtracting the initial number of trees from the number of trees after planting. The correct calculation should be trees_after - trees_initial.

# Let's check the other parts
answer = trees_added
# looks good

Okay! Here is the rewrite:

```python
trees_initial = 15
trees_after = 21
trees_added = trees_after - trees_initial
answer = trees_added
```

---

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
```python
toys_initial = 5
mom_toys = 2
dad_toys = 2
total_received = mom_toys + dad_toys
total_toys = toys_initial + total_received
answer = total_toys
```

# There is no error in the code above. The logic and calculations are correct. Let us verify step-by-step:

# Let us go through the calculation step-by-step
toys_initial = 5
# looks good

# Let's check the other parts
mom_toys = 2
# looks good

# Let's check the other parts
dad_toys = 2
# looks good

# Let's check the other parts
total_received = mom_toys + dad_toys
# looks good

# Let's check the other parts
total_toys = toys_initial + total_received
# looks good

# Let's check the other parts
answer = total_toys
# looks good

There is no error in the code! It is correct! 

Here is the rewrite (for the sake of completeness):

```python
toys_initial = 5
mom_toys = 2
dad_toys = 2
total_received = mom_toys + dad_toys
total_toys = toys_initial + total_received
answer = total_toys
```

---

Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
```python
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between Monday and Thursday
computers_added = computers_per_day + num_days
computers_total = computers_initial + computers_added
answer = computers_total
```

There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
computers_initial = 9
#looks good

# Let's check the other parts
computers_per_day = 5
# looks good

# Let's check the other parts
num_days = 4 # 4 days between Monday and Thursday
# looks good

#Let's check the other parts
computers_added = computers_per_day + num_days

wrong! The error here is the incorrect addition. We should multiply the number of computers added per day by the number of days, not add them together. This causes the calculation of computers_added to be incorrect.

Okay! Here is the rewrite:

```python
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between monday and thursday
computers_added = computers_per_day * num_days
computers_total = computers_initial + computers_added
answer = computers_total
```

---

Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
```python
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
answer = golf_balls_left
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
golf_balls_initial = 58
# looks good

# Let's check the other parts
golf_balls_lost_tuesday = 23
# looks good

# Let's check the other parts
golf_balls_lost_wednesday = 2

# Let's check the other parts
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
# looks good

#Let's check the other parts
answer = golf_balls_left
# looks good

There is no error in the code! It is correct!

Here is the rewrite (for the sake of completeness):

```python
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
answer = golf_balls_left
```"""


SELF_REFINE_REFINE_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Question: {question}
```python
{answer}
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good. Provide the improved solution. If there is no error, write out the entire solution again.

{critique}

Okay! Here is the rewrite:"""


# ======================================================================== SVAMP ======================================================================== #


SELF_REFINE_INSTRUCTION_SVAMP = """{examples}
(END OF EXAMPLES)

Question: {question}
# Python code, return answer"""


SVAMP_CRITIQUE_FEWSHOT_EXAMPLES = """Question: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers + used_red_stickers
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
original_red_stickers = 93
# looks good

# Let's check the other parts
used_red_stickers = 31
# looks good

# Let's check the other parts
answer = original_red_stickers + used_red_stickers
# wrong! The code is adding the used stickers instead of subtracting them from the original count.

# The corrected code should be:
# answer = original_red_stickers - used_red_stickers

---

Question: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy for each egg?
python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars + discount_dollars

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
original_egg_price_in_dollars = 80
# looks good

# Let's check the other parts
discount_dollars = 29
# looks good

# Let's check the other parts
answer = original_egg_price_in_dollars + discount_dollars
# wrong! The code is adding the discount to the original price instead of subtracting it.

# The corrected code should be:
# answer = original_egg_price_in_dollars - discount_dollars

---

Question: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books. How many books did danny have at first?
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now - num_books_bought_at_store
```

# There is no error in the code above. The logic and calculations are correct. Let us verify step-by-step:

# Let us go through the calculation step-by-step
num_books_bought_at_store = 5
# looks good

# Let's check the other parts
num_books_now = 25
# looks good

# Let's check the other parts
answer = num_books_now - num_books_bought_at_store
# looks good

---

Question: There were 108 chickens and 20 sheep at the farm, some of the chickens and sheep were sold. There are 87 chickens and 18 sheep left now. How many chickens were sold?
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before * num_chicken_now
```

# There is no error in the code above. The logic and calculations are correct. Let us verify step-by-step:

# Let us go through the calculation step-by-step
num_chicken_before = 108
# look good

# Let's check the other parts
num_chicken_now = 87
# look good

# Let's check the other parts
answer = num_chicken_before * num_chicken_now

The answer is incorrect because the code multiplies the initial number of chickens by the number of chickens left. This operation does not make sense in the context of finding out how many chickens were sold.

---

Question: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday. How many did Katty score on monday and wednesday?
```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday + num_goals_on_wednesday
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
num_goals_on_monday = 2
#look good

# Let's check the other parts
num_goals_on_wednesday = 9
#look good

# Let's check the other parts
answer = num_goals_on_monday + num_goals_on_wednesday
# look good

There is no error. The answer is correct!"""


SELF_REFINE_CRITIQUE_INSTRUCTION_SVAMP = """{examples}
(END OF EXAMPLES)

Question: {question}
```python
{answer}
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good."""


SVAMP_REFINE_FEWSHOT_EXAMPLES = """Question: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers + used_red_stickers
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
original_red_stickers = 93
# looks good

# Let's check the other parts
used_red_stickers = 31
# looks good

# Let's check the other parts
answer = original_red_stickers + used_red_stickers
# wrong! The code is adding the used stickers instead of subtracting them from the original count.

# The corrected code should be:
# answer = original_red_stickers - used_red_stickers

Okay! Here is the rewrite:

```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers - used_red_stickers
```

---

Question: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy for each egg?
python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars + discount_dollars

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
original_egg_price_in_dollars = 80
# looks good

# Let's check the other parts
discount_dollars = 29
# looks good

# Let's check the other parts
answer = original_egg_price_in_dollars + discount_dollars
# wrong! The code is adding the discount to the original price instead of subtracting it.

# The corrected code should be:
# answer = original_egg_price_in_dollars - discount_dollars

Okay! Here is the rewrite:

```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars - discount_dollars
```

---

Question: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books. How many books did danny have at first?
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now - num_books_bought_at_store
```

# There is no error in the code above. The logic and calculations are correct. Let us verify step-by-step:

# Let us go through the calculation step-by-step
num_books_bought_at_store = 5
# looks good

# Let's check the other parts
num_books_now = 25
# looks good

# Let's check the other parts
answer = num_books_now - num_books_bought_at_store
# looks good

Here is the rewrite (for the sake of completeness):

```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now - num_books_bought_at_store
```

---

Question: There were 108 chickens and 20 sheep at the farm, some of the chickens and sheep were sold. There are 87 chickens and 18 sheep left now. How many chickens were sold?
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before * num_chicken_now
```

# There is no error in the code above. The logic and calculations are correct. Let us verify step-by-step:

# Let us go through the calculation step-by-step
num_chicken_before = 108
# look good

# Let's check the other parts
num_chicken_now = 87
# look good

# Let's check the other parts
answer = num_chicken_before * num_chicken_now

The answer is incorrect because the code multiplies the initial number of chickens by the number of chickens left. This operation does not make sense in the context of finding out how many chickens were sold.

Okay! Here is the rewrite:

```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before - num_chicken_now
```

---

Question: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday. How many did Katty score on monday and wednesday?
```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday + num_goals_on_wednesday
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
num_goals_on_monday = 2
#look good

# Let's check the other parts
num_goals_on_wednesday = 9
#look good

# Let's check the other parts
answer = num_goals_on_monday + num_goals_on_wednesday
# look good

There is no error. The answer is correct!

Here is the rewrite (for the sake of completeness):

```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday + num_goals_on_wednesday
```"""


SELF_REFINE_REFINE_INSTRUCTION_SVAMP = """{examples}
(END OF EXAMPLES)

Question: {question}
```python
{answer}
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good. Provide the improved solution. If there is no error, write out the entire solution again.

{critique}

Okay! Here is the rewrite:"""


# ======================================================================== TABMWP ======================================================================== #


SELF_REFINE_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

{question}
# Python code, return answer"""


TABMWP_CRITIQUE_FEWSHOT_EXAMPLES = """Read the following table regarding "Coin collections" and then write Python code to answer a question:

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
```python
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
mean_number_of_coins = sum(number_of_coins_for_different_person) + len(number_of_coins_for_different_person)
answer = mean_number_of_coins
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
# looks good

# Let's check the other parts
mean_number_of_coins = sum(number_of_coins_for_different_person) + len(number_of_coins_for_different_person)
# wrong! To find the mean, we should divide the sum of the numbers by the count, not add them. The correct calculation should be sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person).

# Let's check the other parts
answer = mean_number_of_coins
# looks good

---

Read the following table regarding "Supply and Demand" and then write Python code to answer a question:

Price | Quantity demanded | Quantity supplied
$155 | 22,600 | 5,800
$275 | 20,500 | 9,400
$395 | 18,400 | 13,000
$515 | 16,300 | 16,600
$635 | 14,200 | 20,200

Question: Look at the table. Then answer the question. At a price of $155, is there a shortage or a surplus? Choose from the options: [shortage, surplus]
```python
quantity_demanded_at_price_155 = 22600
quantity_supplied_at_price_155 = 5800
if quantity_demanded_at_price_155 < quantity_supplied_at_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
quantity_demanded_at_price_155 = 22600
# looks good

# Let's check the other parts
quantity_supplied_at_price_155 = 5800
# looks good

# Let's check the other parts
if quantity_demanded_at_price_155 < quantity_supplied_at_price_155:
answer = 'shortage'
# wrong! The condition is incorrect. We should check if quantity_demanded_at_price_155 is greater than quantity_supplied_at_price_155 to determine a shortage. The correct condition should be if quantity_demanded_at_price_155 > quantity_supplied_at_price_155.

# Let's check the other parts
else:
answer = 'surplus'
# looks good

# Let's check the other parts
answer = answer
# looks good

---

Read the following table regarding "Cans of food collected" and then write Python code to answer a question:

Samir | 7
Kristen | 4
Dakota | 7
Jamie | 8
Maggie | 9

Question: Samir's class recorded how many cans of food each student collected for their canned food drive. What is the median of the numbers?
```python
cans = [7, 4, 5, 8, 9]
cans = sorted(cans)
middle1 = (len(cans) - 1) // 2
middle2 = len(cans) // 2
answer = (cans[middle1] + cans[middle2]) / 2
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
cans = [7, 4, 5, 8, 9]
# looks good

# Let's check the other parts
cans = sorted(cans)
# looks good

# Let's check the other parts
middle1 = (len(cans) - 1) // 2
# looks good

# Let's check the other parts
middle2 = len(cans) // 2
# looks good

# Let's check the other parts
answer = (cans[middle1] + cans[middle2]) / 2

The answer looks good. There is no error!

---

Read the following table regarding "" and then write Python code to answer a question:

toy boat | $5.54
toy guitar | $8.23
set of juggling balls | $5.01
trivia game | $8.18
jigsaw puzzle | $5.30
toy dinosaur | $3.00

Question: Lorenzo has $13.50. Does he have enough to buy a toy guitar and a set of juggling balls? Choose from the the options: ['yes', 'no']
```python
guitar_price = 8.23
juggling_balls = 5.01
total_money = 13.5
if total_money > juggling_balls - guitar_price:
    answer = "yes"
else:
    answer = "no\"
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
guitar_price = 8.23
# looks good

# Let's check the other parts
juggling_balls = 5.01
# looks good

# Let's check the other parts
total_money = 13.5
# looks good

# Let's check the other parts
if total_money > juggling_balls - guitar_price:
    answer = "yes"
else:
    answer = "no\"

The answer is incorrect. The logic here mistakenly subtracts the price of the guitar from the price of the juggling balls, which does not make sense in the context of determining if Lorenzo has enough money to buy both items.

---

Read the following table regarding "Apple harvest" and then write Python code to answer a question:

Number of apple trees | Number of apples harvested
1 | 50
2 | 100
3 | 150
4 | 200
5 | 250
6 | ?

Question: Each apple tree yields 50 apples. How many apples are harvested from 6 apple trees?
``` python
apples_per_tree = 50
number_of_trees = 6
answer = apples_per_tree / number_of_trees
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
apples_per_tree = 50
# looks good

# Let's check the other parts
number_of_trees = 6
# looks good

# Let's check the other parts
answer = apples_per_tree / number_of_trees

The answer looks good. There is no error!"""


SELF_REFINE_CRITIQUE_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

{question}
```python
{answer}
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good."""


TABMWP_REFINE_FEWSHOT_EXAMPLES = """Read the following table regarding "Coin collections" and then write Python code to answer a question:

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
```python
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
mean_number_of_coins = sum(number_of_coins_for_different_person) + len(number_of_coins_for_different_person)
answer = mean_number_of_coins
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
# looks good

# Let's check the other parts
mean_number_of_coins = sum(number_of_coins_for_different_person) + len(number_of_coins_for_different_person)
# wrong! To find the mean, we should divide the sum of the numbers by the count, not add them. The correct calculation should be sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person).

# Let's check the other parts
answer = mean_number_of_coins
# looks good

Okay! Here is the rewrite:

```python
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
answer = sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person)
```

---

Read the following table regarding "Supply and Demand" and then write Python code to answer a question:

Price | Quantity demanded | Quantity supplied
$155 | 22,600 | 5,800
$275 | 20,500 | 9,400
$395 | 18,400 | 13,000
$515 | 16,300 | 16,600
$635 | 14,200 | 20,200

Question: Look at the table. Then answer the question. At a price of $155, is there a shortage or a surplus? Choose from the options: [shortage, surplus]
```python
quantity_demanded_at_price_155 = 22600
quantity_supplied_at_price_155 = 5800
if quantity_demanded_at_price_155 < quantity_supplied_at_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
quantity_demanded_at_price_155 = 22600
# looks good

# Let's check the other parts
quantity_supplied_at_price_155 = 5800
# looks good

# Let's check the other parts
if quantity_demanded_at_price_155 < quantity_supplied_at_price_155:
answer = 'shortage'
# wrong! The condition is incorrect. We should check if quantity_demanded_at_price_155 is greater than quantity_supplied_at_price_155 to determine a shortage. The correct condition should be if quantity_demanded_at_price_155 > quantity_supplied_at_price_155.

# Let's check the other parts
else:
answer = 'surplus'
# looks good

# Let's check the other parts
answer = answer
# looks good

Okay! Here is the rewrite:

```python
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_at_price_155 > quantity_supplied_at_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'
```

---

Read the following table regarding "Cans of food collected" and then write Python code to answer a question:

Samir | 7
Kristen | 4
Dakota | 7
Jamie | 8
Maggie | 9

Question: Samir's class recorded how many cans of food each student collected for their canned food drive. What is the median of the numbers?
```python
cans = [7, 4, 5, 8, 9]
cans = sorted(cans)
middle1 = (len(cans) - 1) // 2
middle2 = len(cans) // 2
answer = (cans[middle1] + cans[middle2]) / 2
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
cans = [7, 4, 5, 8, 9]
# looks good

# Let's check the other parts
cans = sorted(cans)
# looks good

# Let's check the other parts
middle1 = (len(cans) - 1) // 2
# looks good

# Let's check the other parts
middle2 = len(cans) // 2
# looks good

# Let's check the other parts
answer = (cans[middle1] + cans[middle2]) / 2

The answer looks good. There is no error!

Here is the rewrite (for the sake of completeness):

```python
cans = [7, 4, 5, 8, 9]
cans = sorted(cans)
middle1 = (len(cans) - 1) // 2
middle2 = len(cans) // 2
answer = (cans[middle1] + cans[middle2]) / 2
```

---

Read the following table regarding "" and then write Python code to answer a question:

toy boat | $5.54
toy guitar | $8.23
set of juggling balls | $5.01
trivia game | $8.18
jigsaw puzzle | $5.30
toy dinosaur | $3.00

Question: Lorenzo has $13.50. Does he have enough to buy a toy guitar and a set of juggling balls? Choose from the the options: ['yes', 'no']
```python
guitar_price = 8.23
juggling_balls = 5.01
total_money = 13.5
if total_money > juggling_balls - guitar_price:
    answer = "yes"
else:
    answer = "no"
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
guitar_price = 8.23
# looks good

# Let's check the other parts
juggling_balls = 5.01
# looks good

# Let's check the other parts
total_money = 13.5
# looks good

# Let's check the other parts
if total_money > juggling_balls - guitar_price:
    answer = "yes"
else:
    answer = "no"

The answer is incorrect. The logic here mistakenly subtracts the price of the guitar from the price of the juggling balls, which does not make sense in the context of determining if Lorenzo has enough money to buy both items.

Okay! Here is the rewrite:

```python
guitar_price = 8.23
juggling_balls = 5.01
total_money = 13.5
if total_money >= guitar_price + juggling_balls:
    answer = "yes"
else:
    answer = "no"
```

---

Read the following table regarding "Apple harvest" and then write Python code to answer a question:

Number of apple trees | Number of apples harvested
1 | 50
2 | 100
3 | 150
4 | 200
5 | 250
6 | ?

Question: Each apple tree yields 50 apples. How many apples are harvested from 6 apple trees?
``` python
apples_per_tree = 50
number_of_trees = 6
answer = apples_per_tree / number_of_trees
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
apples_per_tree = 50
# looks good

# Let's check the other parts
number_of_trees = 6
# looks good

# Let's check the other parts
answer = apples_per_tree / number_of_trees

The answer looks good. There is no error!

Here is the rewrite (for the sake of completeness):

```python
apples_per_tree = 50
number_of_trees = 6
answer = apples_per_tree / number_of_trees
```"""


SELF_REFINE_REFINE_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

{question}
```python
{answer}
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good. Provide the improved solution. If there is no error, write out the entire solution again.

{critique}

Okay! Here is the rewrite:"""
