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

The question asks for the elevation range, and the answer provides "1,800 to 7,000 ft," which fits the expected format and seems plausible.

2. Truthfulness:

The eastern sector of the Colorado orogeny extends into regions with elevations ranging from approximately 1,800 ft to 7,000 ft. This matches the proposed answer, ensuring it is correct and truthful.
The proposed answer accurately identifies the elevation range for the area that the eastern sector of the Colorado orogeny extends into. However, the explanation can be improved for clarity.

---

Q: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?
A: Richard Nixon

What's the problem with the above answer?

1. Plausibility:

The question asks for the namesake of "The Simpsons" character Milhouse, and the answer provides "Richard Nixon," which fits the expected format and seems plausible given that Nixon's middle name was "Milhous."

2. Truthfulness:

The character Milhouse was indeed named after U.S. President Richard Nixon, whose middle name was Milhous. This confirms that the proposed answer, "Richard Nixon," is correct and truthful.
The proposed answer accurately identifies the namesake of "The Simpsons" character Milhouse. However, the explanation could be improved for clarity.

---

Q: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
A: The Saimaa Gesture

What's the problem with the above answer?

1. Plausibility:

The question asks which documentary is about Finnish rock groups. The Saimaa Gesture is a 1981 film documenting three Finnish rock groups touring around Lake Saimaa, making it a plausible answer. Conversely, Adam Clayton Powell focuses on an American pastor and politician, which is unrelated to Finnish rock music, confirming it as an implausible option.

2. Truthfulness:

The Saimaa Gesture is indeed a documentary about Finnish rock groups, while Adam Clayton Powell is about the life and career of the U.S. politician Adam Clayton Powell Jr. 
This confirms that the proposed answer, The Saimaa Gesture, is correct and truthful.

---

Q: What profession does Nicholas Ray and Elia Kazan have in common?
A: Director and screenwriter

What's the problem with the above answer?

1. Plausibility:

The question asks for shared professions between Nicholas Ray and Elia Kazan. Nicholas Ray was primarily a director and screenwriter, while Elia Kazan was a director, screenwriter, producer, and occasional actor. This makes "director and screenwriter" a plausible shared profession, while "actor" is less central to Ray's career.

2. Truthfulness:

Nicholas Ray is best known as a director and screenwriter, particularly for Rebel Without a Cause. Elia Kazan is similarly renowned for his work as a director and screenwriter in Hollywood and on Broadway. 
Both share these professions, confirming the answer "director and screenwriter" as correct and truthful.

---

Q: Which magazine was started first, Arthur's Magazine or First for Women?
A: Arthur's Magazine

What's the problem with the above answer?

1. Plausibility:

The question asks which magazine was started earlier. Arthur's Magazine, first published in 1844, predates First for Women, which began in 1989. This makes "Arthur's Magazine" a plausible answer.

2. Truthfulness:

Arthur's Magazine was indeed first published in 1844, significantly earlier than First for Women, which launched in 1989. 
This confirms the answer "Arthur's Magazine" as correct and truthful."""


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

The question asks for the elevation range, and the answer provides "1,800 to 7,000 ft," which fits the expected format and seems plausible.

2. Truthfulness:

The eastern sector of the Colorado orogeny extends into regions with elevations ranging from approximately 1,800 ft to 7,000 ft. This matches the proposed answer, ensuring it is correct and truthful.
The proposed answer accurately identifies the elevation range for the area that the eastern sector of the Colorado orogeny extends into. However, the explanation can be improved for clarity.

Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?

Provide a concise response to the question.
Answer: 1,800 to 7,000 ft

---

Q: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?
A: Richard Nixon

What's the problem with the above answer?

1. Plausibility:

The question asks for the namesake of "The Simpsons" character Milhouse, and the answer provides "Richard Nixon," which fits the expected format and seems plausible given that Nixon's middle name was "Milhous."

2. Truthfulness:

The character Milhouse was indeed named after U.S. President Richard Nixon, whose middle name was Milhous. This confirms that the proposed answer, "Richard Nixon," is correct and truthful.
The proposed answer accurately identifies the namesake of "The Simpsons" character Milhouse. However, the explanation could be improved for clarity.

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?

Provide a concise response to the question.
Answer: Richard Nixon

---

Q: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
A: The Saimaa Gesture

What's the problem with the above answer?

1. Plausibility:

The question asks which documentary is about Finnish rock groups. The Saimaa Gesture is a 1981 film documenting three Finnish rock groups touring around Lake Saimaa, making it a plausible answer. Conversely, Adam Clayton Powell focuses on an American pastor and politician, which is unrelated to Finnish rock music, confirming it as an implausible option.

2. Truthfulness:

The Saimaa Gesture is indeed a documentary about Finnish rock groups, while Adam Clayton Powell is about the life and career of the U.S. politician Adam Clayton Powell Jr. 
This confirms that the proposed answer, The Saimaa Gesture, is correct and truthful.

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?

Provide a concise response to the question.
Answer: The Saimaa Gesture

---

Q: What profession does Nicholas Ray and Elia Kazan have in common?
A: Director and screenwriter

What's the problem with the above answer?

1. Plausibility:

The question asks for shared professions between Nicholas Ray and Elia Kazan. Nicholas Ray was primarily a director and screenwriter, while Elia Kazan was a director, screenwriter, producer, and occasional actor. This makes "director and screenwriter" a plausible shared profession, while "actor" is less central to Ray's career.

2. Truthfulness:

Nicholas Ray is best known as a director and screenwriter, particularly for Rebel Without a Cause. Elia Kazan is similarly renowned for his work as a director and screenwriter in Hollywood and on Broadway. 
Both share these professions, confirming the answer "director and screenwriter" as correct and truthful.

Question: What profession does Nicholas Ray and Elia Kazan have in common?

Provide a concise response to the question.
Answer: Director and screenwriter

---

Q: Which magazine was started first, Arthur's Magazine or First for Women?
A: Arthur's Magazine

What's the problem with the above answer?

1. Plausibility:

The question asks which magazine was started earlier. Arthur's Magazine, first published in 1844, predates First for Women, which began in 1989. This makes "Arthur's Magazine" a plausible answer.

2. Truthfulness:

Arthur's Magazine was indeed first published in 1844, significantly earlier than First for Women, which launched in 1989. 
This confirms the answer "Arthur's Magazine" as correct and truthful.

Question: Which magazine was started first, Arthur's Magazine or First for Women?

Provide a concise response to the question.
Answer: Arthur's Magazine"""


SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA = """{examples}
(END OF EXAMPLES)

Question: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

{critique}

Question: {question}

Provide a concise response to the question.
Answer: """


# ======================================================================== TRIVIAQA ======================================================================== #


SELF_REFINE_INSTRUCTION_TRIVIAQA = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """


TRIVIAQA_CRITIQUE_FEWSHOT_EXAMPLES = """Q: Mendelssohn's 'Wedding March' was. originally written as incidental music for which Shakespeare play in 1842?
A: A Midsummer Night's Dream

What's the problem with the above answer?

1. Plausibility:

The question asks for the Shakespeare play for which Mendelssohn's Wedding March was composed. The answer "Hamlet" refers to a Shakespeare play, making it initially plausible. However, further investigation is necessary to verify its accuracy.

2. Truthfulness:

The Wedding March in C major, written in 1842, was composed as incidental music for Shakespeare's A Midsummer Night's Dream. While Hamlet is a tragedy written between 1599 and 1602. There is no record of Mendelssohn composing incidental music for Hamlet.
This evidence shows that the proposed answer "A Midsummer Night's Dream" is correct.

---

Q: \"\"\"Christ in the House of his Parents\"\" is one of the best known paintings of which artist?"
A: John Millais

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the artist, and the answer is "John Millais," which is a name of an artist. So it's plausible.

2. Truthfulness:

"Christ in the House of His Parents" is a painting by John Everett Millais, which depicts the Holy Family in Saint Joseph's carpentry workshop.
The evidence confirms that John Everett Millais is the artist of "Christ in the House of his Parents."

---

Q: Who designed the National Theatre building on the South Bank in London?
A: Sir Norman Foster

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the designer of the National Theatre building on the South Bank in London, and the answer "Sir Norman Foster" is a plausible name of an architect. However, it might not be accurate.

2. Truthfulness:

The National Theatre building on the South Bank in London was designed by Sir Denys Lasdun and opened in 1976.
The evidence shows that the National Theatre building was designed by Sir Denys Lasdun, not Sir Norman Foster.

---

Q: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?
A: Bodie Miller

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the skier, and the answer is "Bodie Miller," which is a name. So it's plausible.

2. Truthfulness:

At the 2010 Winter Olympics, Miller won the gold medal in the super combined.
The evidence confirms that Bode Miller won the gold medal in the super combined at the 2010 Winter Olympics.

---

Q: Which English author wrote the novel "1984"?
A: George Orwell wrote the novel "1984". So the answer is: George Orwell.

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the author, and the answer is "George Orwell," which is a name. So it's plausible.

2. Truthfulness:

"Nineteen Eighty-Four: A Novel, often published as '1984', is a dystopian social science fiction novel and cautionary tale, written by the English writer George Orwell.
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

The question asks for the Shakespeare play for which Mendelssohn's Wedding March was composed. The answer "Hamlet" refers to a Shakespeare play, making it initially plausible. However, further investigation is necessary to verify its accuracy.

2. Truthfulness:

The Wedding March in C major, written in 1842, was composed as incidental music for Shakespeare's A Midsummer Night's Dream. While Hamlet is a tragedy written between 1599 and 1602. There is no record of Mendelssohn composing incidental music for Hamlet.
This evidence shows that the proposed answer "A Midsummer Night's Dream" is correct.

Question: Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842?

Provide a concise response to the question.
Answer: A Midsummer Night's Dream

---

Q: \"\"\"Christ in the House of his Parents\"\" is one of the best known paintings of which artist?"
A: John Millais

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the artist, and the answer is "John Millais," which is a name of an artist. So it's plausible.

2. Truthfulness:

"Christ in the House of His Parents" is a painting by John Everett Millais, which depicts the Holy Family in Saint Joseph's carpentry workshop.
The evidence confirms that John Everett Millais is the artist of "Christ in the House of his Parents."

Question: "Christ in the House of his Parents" is one of the best known paintings of which artist?

Provide a concise response to the question.
Answer: John Millais

---

Q: Who designed the National Theatre building on the South Bank in London?
A: Sir Norman Foster

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the designer of the National Theatre building on the South Bank in London, and the answer "Sir Norman Foster" is a plausible name of an architect. However, it might not be accurate.

2. Truthfulness:

The National Theatre building on the South Bank in London was designed by Sir Denys Lasdun and opened in 1976.
The evidence shows that the National Theatre building was designed by Sir Denys Lasdun, not Sir Norman Foster.

Question: Who designed the National Theatre building on the South Bank in London?

Provide a concise response to the question.
Answer: Sir Denys Lasdun

---

Q: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?
A: Bodie Miller

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the skier, and the answer is "Bodie Miller," which is a name. So it's plausible.

2. Truthfulness:

At the 2010 Winter Olympics, Miller won the gold medal in the super combined.
The evidence confirms that Bode Miller won the gold medal in the super combined at the 2010 Winter Olympics.

Question: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?

Provide a concise response to the question.
Answer: Bodie Miller

---

Q: Which English author wrote the novel "1984"?
A: George Orwell

What's the problem with the above answer?

1. Plausibility:

The question asks for the name of the author, and the answer is "George Orwell," which is a name. So it's plausible.

2. Truthfulness:

"Nineteen Eighty-Four: A Novel, often published as '1984', is a dystopian social science fiction novel and cautionary tale, written by the English writer George Orwell.
The evidence confirms that George Orwell is indeed the author of the novel "1984".

Question: Which English author wrote the novel "1984"?

Provide a concise response to the question.
Answer: George Orwell"""


SELF_REFINE_REFINE_INSTRUCTION_TRIVIAQA = """{examples}
(END OF EXAMPLES)

Question: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

{critique}

Question: {question}

Provide a concise response to the question.
Answer: """


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

Figure skating was first contested in the Olympic Games at the 1908 Summer Olympics. Since 1924, the sport has been a part of the Winter Olympic Games.
The evidence suggests Figure skating became an Olympic sport at the 1908 Summer Olympics, and has been a part of the Winter Olympic Games in 1924.

---

Q: When was the last time the USA men's national soccer team missed the World Cup?
A: 2018

What's the problem with the above answer?

1. Plausibility:

The question asks for a year, and the answer is "1986", which is a year. So it's plausible.

2. Truthfulness:

The United States participated in every World Cup from 1990 through 2014, but did not qualify in 2018, marking first time the team had missed a World Cup since 1986. They returned to the World Cup by qualifying for the 2022 edition.
The evidence suggests that the USA men's national soccer team did not qualify for the World Cup in 2018, but qualify for 2022, 2018 > 1986, so the last time the USA men's national soccer team missed the World Cup was in 2018, not in 1986, which contradicts the proposed answer.

---

Q: What does barium do in a ct scan?
A: to improve visualization of the gastrointestinal tract

What's the problem with the above answer?

1. Plausibility:

The question asks for the function of barium in a CT scan, and the answer is "highlight the digestive system", which is a function. So it's plausible.

2. Truthfulness:

Barium sulfate works by coating the inside of your esophagus, stomach, or intestines which allows them to be seen more clearly on a CT scan or other radiologic (x-ray) examination.
The evidence suggests that barium is used to "improve visualization of the gastrointestinal tract", which includes the digestive system. Therefore, the proposed answer should be more specific by stating "improve visualization".

---

Q: Where was the fort located at which the first shot of the civil war was fired?
A: Charleston Harbor, South Carolina

What's the problem with the above answer?

1. Plausibility:

The question asks for the location of the fort, and the answer is "Charleston Harbor", which may be a location. So it's plausible.

2. Truthfulness:

April 12, 1861 At 4:30 a.m. on April 12, 1861, Confederate troops fired on Fort Sumter in South Carolina’s Charleston Harbor. Less than 34 hours later, Union forces surrendered.
The evidence suggests that the first shot of the Civil War was fired at Charleston Harbor, which is consistent with the proposed answer.

---

Q: When did nando's come to the uk?
A: 1992

What's the problem with the above answer?

1. Plausibility:

The question asks for a time, and the answer is "1992", which is a year. So it's plausible.

2. Truthfulness:

Nando's opened its first restaurants within the United Kingdom, in 1992, in the west London suburbs of Ealing and Earls Court, initially focusing on takeaway food.
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

Figure skating was first contested in the Olympic Games at the 1908 Summer Olympics. Since 1924, the sport has been a part of the Winter Olympic Games.
The evidence suggests Figure skating became an Olympic sport at the 1908 Summer Olympics, and has been a part of the Winter Olympic Games in 1924.

Question: When did men's figure skating become a summer Olympic sport?

Provide a concise response to the question.
Answer: 1908

---

Q: When was the last time the USA men's national soccer team missed the World Cup?
A: 2018

What's the problem with the above answer?

1. Plausibility:

The question asks for a year, and the answer is "1986", which is a year. So it's plausible.

2. Truthfulness:

The United States participated in every World Cup from 1990 through 2014, but did not qualify in 2018, marking first time the team had missed a World Cup since 1986. They returned to the World Cup by qualifying for the 2022 edition.
The evidence suggests that the USA men's national soccer team did not qualify for the World Cup in 2018, but qualify for 2022, 2018 > 1986, so the last time the USA men's national soccer team missed the World Cup was in 2018, not in 1986, which contradicts the proposed answer.

Question: When was the last time the USA men's national soccer team missed the World Cup?

Provide a concise response to the question.
Answer: 2018

---

Q: What does barium do in a ct scan?
A: to improve visualization of the gastrointestinal tract

What's the problem with the above answer?

1. Plausibility:

The question asks for the function of barium in a CT scan, and the answer is "highlight the digestive system", which is a function. So it's plausible.

2. Truthfulness:

Barium sulfate works by coating the inside of your esophagus, stomach, or intestines which allows them to be seen more clearly on a CT scan or other radiologic (x-ray) examination.
The evidence suggests that barium is used to "improve visualization of the gastrointestinal tract", which includes the digestive system. Therefore, the proposed answer should be more specific by stating "improve visualization".

Question: What does barium do in a ct scan?

Provide a concise response to the question.
Answer: to improve visualization of the gastrointestinal tract

---

Q: Where was the fort located at which the first shot of the civil war was fired?
A: Charleston Harbor, South Carolina

What's the problem with the above answer?

1. Plausibility:

The question asks for the location of the fort, and the answer is "Charleston Harbor", which may be a location. So it's plausible.

2. Truthfulness:

April 12, 1861 At 4:30 a.m. on April 12, 1861, Confederate troops fired on Fort Sumter in South Carolina’s Charleston Harbor. Less than 34 hours later, Union forces surrendered.
The evidence suggests that the first shot of the Civil War was fired at Charleston Harbor, which is consistent with the proposed answer.

Question: Where was the fort located at which the first shot of the civil war was fired?

Provide a concise response to the question.
Answer: Charleston Harbor, South Carolina

---

Q: When did nando's come to the uk?
A: 1992

What's the problem with the above answer?

1. Plausibility:

The question asks for a time, and the answer is "1992", which is a year. So it's plausible.

2. Truthfulness:

Nando's opened its first restaurants within the United Kingdom, in 1992, in the west London suburbs of Ealing and Earls Court, initially focusing on takeaway food.
The evidence suggests that Nando's first opened in the UK in 1992, which is consistent with the proposed answer. We can provide more detailed information in the answer.

Question: When did nando's come to the uk?

Provide a concise response to the question.
Answer: 1992"""


SELF_REFINE_REFINE_INSTRUCTION_AMBIGNQ = """{examples}
(END OF EXAMPLES)

Question: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

{critique}

Question: {question}

Provide a concise response to the question.
Answer: """


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

Nikolaj Coster-Waldau appeared in the 2009 Fox television film Virtuality.
The evidence contradicts the proposed answer, confirming he did work with Fox in the television film Virtuality.

---

Claim: Stranger Things is set in Bloomington, Indiana.
A: REFUTES

What's the problem with the above answer?

1. Plausibility:

The answer is correct but lacks specific details that verify the claim.

2. Truthfulness:

Stranger Things is set in the fictional town of Hawkins, Indiana, not Bloomington.
Although the proposed answer is correct, it could be more informative by mentioning the specific setting.

---

Claim: "Beautiful" by Christina Aguilera reached number two on the Billboard Hot 100 in 2003.
A: NOT ENOUGH INFO

What's the problem with the above answer?

1. Plausibility:

The answer "NOT ENOUGH INFO" is appropriate as it reflects the uncertainty due to incomplete information about the timing of the chart position.

2. Truthfulness:

The song peaked at number two on the Billboard Hot 100, but the specific year it achieved this ranking was not directly specified in the sources found.
Given that the year 2003 is not verified in the available evidence, the proposed answer correctly reflects the uncertainty regarding the exact year of the chart position.

---

Claim: Tim Burton didn't direct the film "Edward Scissorhands."
A: REFUTES

What's the problem with the above answer?

1. Plausibility:

The answer "REFUTES" incorrectly negates the claim without supporting details.

2. Truthfulness:

The film "Edward Scissorhands" was directed by Tim Burton and released in 1990.
The evidence supports the claim, confirming that Tim Burton directed "Edward Scissorhands."

---

Claim: "The Great Gatsby" was written by F. Scott Fitzgerald.
A: SUPPORTS

What's the problem with the above answer?

1. Plausibility:

The answer "SUPPORTS" is appropriate as it reflects the accuracy of the claim about the authorship of "The Great Gatsby".

2. Truthfulness:

"The Great Gatsby is a 1925 novel written by American author F. Scott Fitzgerald that follows a cast of characters living in the fictional towns of West Egg and East Egg on prosperous Long Island in the summer of 1922."
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

Nikolaj Coster-Waldau appeared in the 2009 Fox television film Virtuality.
The evidence contradicts the proposed answer, confirming he did work with Fox in the television film Virtuality.

Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.

Provide a concise response to the claim.
Answer: SUPPORTS

---

Claim: Stranger Things is set in Bloomington, Indiana.
A: REFUTES

What's the problem with the above answer?

1. Plausibility:

The answer is correct but lacks specific details that verify the claim.

2. Truthfulness:

Stranger Things is set in the fictional town of Hawkins, Indiana, not Bloomington.
Although the proposed answer is correct, it could be more informative by mentioning the specific setting.

Claim: Stranger Things is set in Bloomington, Indiana.

Provide a concise response to the claim.
Answer: REFUTES

---

Claim: "Beautiful" by Christina Aguilera reached number two on the Billboard Hot 100 in 2003.
A: NOT ENOUGH INFO

What's the problem with the above answer?

1. Plausibility:

The answer "NOT ENOUGH INFO" is appropriate as it reflects the uncertainty due to incomplete information about the timing of the chart position.

2. Truthfulness:

The song peaked at number two on the Billboard Hot 100, but the specific year it achieved this ranking was not directly specified in the sources found.
Given that the year 2003 is not verified in the available evidence, the proposed answer correctly reflects the uncertainty regarding the exact year of the chart position.

Claim: "Beautiful" by Christina Aguilera reach number two on the Billboard Hot 100 in 2003.

Provide a concise response to the claim.
Answer: NOT ENOUGH INFO

---

Claim: Tim Burton didn't direct the film "Edward Scissorhands."
A: REFUTES

What's the problem with the above answer?

1. Plausibility:
The answer "REFUTES" incorrectly negates the claim without supporting details.

2. Truthfulness:

The film "Edward Scissorhands" was directed by Tim Burton and released in 1990.
The evidence supports the claim, confirming that Tim Burton directed "Edward Scissorhands."

Claim: Tim Burton didn't direct the film "Edward Scissorhands."

Provide a concise response to the claim.
Answer: REFUTES

---

Claim: "The Great Gatsby" was written by F. Scott Fitzgerald.
A: SUPPORTS

What's the problem with the above answer?

1. Plausibility:

The answer "SUPPORTS" is appropriate as it reflects the accuracy of the claim about the authorship of "The Great Gatsby".

2. Truthfulness:

"The Great Gatsby is a 1925 novel written by American author F. Scott Fitzgerald that follows a cast of characters living in the fictional towns of West Egg and East Egg on prosperous Long Island in the summer of 1922."
The evidence confirms that F. Scott Fitzgerald is indeed the author of "The Great Gatsby".

Claim: "The Great Gatsby" was written by F. Scott Fitzgerald.

Provide a concise response to the claim.
Answer: SUPPORTS"""


SELF_REFINE_REFINE_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. 

{examples}
(END OF EXAMPLES)

Claim: {question}
Proposed Answer: {answer}

What's the problem with the above answer?

1. Plausibility:

{critique}

Claim: {question}

Provide a concise response to the claim.
Answer: """


# ======================================================================== GSM8K ======================================================================== #


SELF_REFINE_INSTRUCTION_GSM8K = """# Write Python Code to solve the following questions. Store your result as a variable named 'answer'.

{examples}
(END OF EXAMPLES)

Question: {question}
# Python code, return answer"""


GSM8K_CRITIQUE_FEWSHOT_EXAMPLES = """Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
lollipop_list = [jason_lollipops_initial]

for i in range(jason_lollipops_after):
lollipop_list.append(-1)

denny_lollipops = 0

for lollipop in lollipop_list:
denny_lollipops += lollipop

temp_result = jason_lollipops_initial - jason_lollipops_after
answer = temp_result
```

What's the problem with the above code?

1. The total lollipops Jason initially had and the lollipops left should be positive, 20 and 12 > 0, it's reasonable.

2. Let's check the code:

> jason_lollipops_initial = 20
> jason_lollipops_after = 12
> lollipop_list = [jason_lollipops_initial]

This defines the initial lollipops and creates a list starting with the initial count, which is correct.

> for i in range(jason_lollipops_after):
> lollipop_list.append(-1)

This loop adds negative values to the list to represent the lollipops given away, which is an inefficient approach.

> denny_lollipops = 0

It initializes the count for Denny's lollipops, which is correct.

> for lollipop in lollipop_list:
> denny_lollipops += lollipop

This loop adds up all the lollipops in the list, which is unnecessary for this task.

> temp_result = jason_lollipops_initial - jason_lollipops_after
> answer = temp_result

This assigns the correct total to the answer but the approach to calculate it is inefficient.

Overall, the code correctly understands the task but is inefficient and performs unnecessary operations. The correct and efficient approach is to directly subtract the remaining lollipops from the initial count.

---

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
```python
trees_initial = 15
trees_after = 21
trees_added = trees_after - initial_trees
answer = trees_added
```

What's the problem with the above code?

1. The above code causes the "NameError" because it use the variable `initial_trees` before it is defined.

2. The variable names in the code are a little bit confusing, becase both `trees_after` and "initial_trees" are used.

---

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
```python
toys_initial = 5
mom_toys = 2
dad_toys = 2
total_received = mom_toys + dad_toys
total_toys = toys_initial - total_received
answer = total_toys
```

What's the problem with the above code?

1. The total of toys should be greater than the initial number of toys, 1 < 5, so the answer is not reasonable. 

2. Let's check the code:

> # the total number of toys received from mom and dad
> mom_toys = 2
> dad_toys = 2
> total_received = mom_toys + dad_toys

It calculates the total number of received toys `total_received`, that's correct.

> toys_initial = 5
> total_toys = toys_initial - total_received

According to the question, Shawn receives the toys instead of giving , `toys_initial - total_received` means Shawns is giving away his toys, this is wrong.

---

Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
```python
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between Monday and Thursday
computers_added = computers_per_day * num_days
computers_total = computers_initial - computers_added
answer = computers_total
```

What's the problem with the above code?

1. The total number of computers should be a positive number, 9 > 0, it's reasonable.

2. Let's check the code:

> computers_initial = 9
> computers_per_day = 5
> num_days = 4

These define the initial number of computers, the number of computers added per day, and the number of days. These are correct.

> computers_added = computers_per_day * num_days

It calculates the total number of computers added over the 4 days, which is correct.

> computers_total = computers_initial - computers_added

This line incorrectly subtracts the added computers from the initial count, which is wrong.

> answer = computers_total

This assigns the incorrect total to the answer.

Overall, the code correctly calculates the number of added computers but incorrectly subtracts them from the initial count instead of adding them. The correct approach is to add the computers added to the initial count.

---

Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
```python
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
answer = golf_balls_left
```

What's the problem with the above code?

1. The number of golf ball should be a positive number, 33 > 0, it's reasonable.

2. Let's check the code:

> golf_balls_initial = 58
> golf_balls_lost_tuesday = 23
> golf_balls_lost_wednesday = 2

It defines the golf_balls_lost_wednesday as 2, but the number of golf balls lost on wednesday is 2 more than the number on Tuesday.

> golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
> answer = golf_balls_left

The initial number of golf balls subtracts the totla number of golf ball lost on tuesday and wednesday . That's correct.

Overall, the code correctly calculates the number of golf ball left but incorrectly defines the number of golf ball lost on wednesday. The correct approach is to add the lost on tuesday by 2."""


SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Question: {question}
```python
{answer}
```

What's the problem with the above code?"""


GSM8K_REFINE_FEWSHOT_EXAMPLES = """Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
lollipop_list = [jason_lollipops_initial]

for i in range(jason_lollipops_after):
lollipop_list.append(-1)

denny_lollipops = 0

for lollipop in lollipop_list:
denny_lollipops += lollipop

temp_result = jason_lollipops_initial - jason_lollipops_after
answer = temp_result
```

What's the problem with the above code?

1. The total lollipops Jason initially had and the lollipops left should be positive, 20 and 12 > 0, it's reasonable.

2. Let's check the code:

> jason_lollipops_initial = 20
> jason_lollipops_after = 12
> lollipop_list = [jason_lollipops_initial]

This defines the initial lollipops and creates a list starting with the initial count, which is correct.

> for i in range(jason_lollipops_after):
> lollipop_list.append(-1)

This loop adds negative values to the list to represent the lollipops given away, which is an inefficient approach.

> denny_lollipops = 0

It initializes the count for Denny's lollipops, which is correct.

> for lollipop in lollipop_list:
> denny_lollipops += lollipop

This loop adds up all the lollipops in the list, which is unnecessary for this task.

> temp_result = jason_lollipops_initial - jason_lollipops_after
> answer = temp_result

This assigns the correct total to the answer but the approach to calculate it is inefficient.

Overall, the code correctly understands the task but is inefficient and performs unnecessary operations. The correct and efficient approach is to directly subtract the remaining lollipops from the initial count.

If no changes are needed, return the same code. 
Here's a better solution:
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
trees_added = trees_after - initial_trees
answer = trees_added
```

What's the problem with the above code?

1. The above code causes the "NameError" because it use the variable `initial_trees` before it is defined.

2. The variable names in the code are a little bit confusing, becase both `trees_after` and "initial_trees" are used.

If no changes are needed, return the same code. 
Here's a better solution:
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
total_toys = toys_initial - total_received
answer = total_toys
```

What's the problem with the above code?

1. The total of toys should be greater than the initial number of toys, 1 < 5, so the answer is not reasonable. 

2. Let's check the code:

> # the total number of toys received from mom and dad
> mom_toys = 2
> dad_toys = 2
> total_received = mom_toys + dad_toys

It calculates the total number of received toys `total_received`, that's correct.

> toys_initial = 5
> total_toys = toys_initial - total_received

According to the question, Shawn receives the toys instead of giving , `toys_initial - total_received` means Shawns is giving away his toys, this is wrong.

If no changes are needed, return the same code. 
Here's a better solution:
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
computers_added = computers_per_day * num_days
computers_total = computers_initial - computers_added
answer = computers_total
```

What's the problem with the above code?

1. The total number of computers should be a positive number, 9 > 0, it's reasonable.

2. Let's check the code:

> computers_initial = 9
> computers_per_day = 5
> num_days = 4

These define the initial number of computers, the number of computers added per day, and the number of days. These are correct.

> computers_added = computers_per_day * num_days

It calculates the total number of computers added over the 4 days, which is correct.

> computers_total = computers_initial - computers_added

This line incorrectly subtracts the added computers from the initial count, which is wrong.

> answer = computers_total

This assigns the incorrect total to the answer.

Overall, the code correctly calculates the number of added computers but incorrectly subtracts them from the initial count instead of adding them. The correct approach is to add the computers added to the initial count.

If no changes are needed, return the same code. 
Here's a better solution:
```python
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between Monday and Thursday
computers_added = computers_per_day * num_days
computers_total = computers_initial + computers_added
answer = computers_total
```

---

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
```python
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
answer = golf_balls_left
```

What's the problem with the above code?

1. The number of golf ball should be a positive number, 33 > 0, it's reasonable.

2. Let's check the code:

> golf_balls_initial = 58
> golf_balls_lost_tuesday = 23
> golf_balls_lost_wednesday = 2

It defines the golf_balls_lost_wednesday as 2, but the number of golf balls lost on wednesday is 2 more than the number on Tuesday.

> golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
> answer = golf_balls_left

The initial number of golf balls subtracts the totla number of golf ball lost on tuesday and wednesday . That's correct.

Overall, the code correctly calculates the number of golf ball left but incorrectly defines the number of golf ball lost on wednesday. The correct approach is to add the lost on tuesday by 2.

If no changes are needed, return the same code. 
Here's a better solution:
```python
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
golf_balls_lost_total = golf_balls_lost_tuesday + golf_balls_lost_wednesday
golf_balls_left = golf_balls_initial + golf_balls_lost_total
answer = golf_balls_left
```"""


SELF_REFINE_REFINE_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Question: {question}
```python
{answer}
```

What's the problem with the above code?

{critique}

Return the better solution and if not, return the previous solution:"""


# ======================================================================== SVAMP ======================================================================== #


SELF_REFINE_INSTRUCTION_SVAMP = """Read the following passages to answer questions with Python code, store the result as a 'answer' variable:

{examples}
(END OF EXAMPLES)

Question: {question}
# Python code, return answer"""


SVAMP_CRITIQUE_FEWSHOT_EXAMPLES = """Question: James bought 93 red and 10 blue stickers, he used 31 red stickers on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
```python
original_red_stickers = 93
used_red_stickers = 31
original_blue_stickers = 10
used_blue_stickers = 7
answer = (original_red_stickers - used_red_stickers) + (original_blue_stickers - used_blue_stickers)
```

What's the problem with the above code?

1. The number of stickers should be positive, 65 > 0, it's reasonable.
2. Let's check the code:

> answer = (original_red_stickers - used_red_stickers) + (original_blue_stickers - used_blue_stickers)

The above code incorrectly combines the count of red and blue stickers. The question only asks for the number of red stickers James has left, so the calculation involving blue stickers is unnecessary and incorrect.

According to the question, James initially bought 93 red stickers and used 31 of them. To find the remaining number of red stickers, we only need to subtract the used red stickers from the original red stickers.

---

Question: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy for each egg?
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars - discount_dollars
```

What's the problem with the above code?

1. The answer, 51, is a reasonable result.
2. Let's check the code:

> answer = original_egg_price_in_dollars - discount_dollars

The code correctly calculates the final price of each egg after applying the discount. There is no problem with the above code.

The proposed answer is correct and accurately computes the discounted price of each egg.

---

Question: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books. How many books did danny have at first?
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now + num_books_bought_at_store
```

What's the problem with the above code?

1. The answer, 30, is not reasonable given the context.
2. Let's check the code:

> answer = num_books_now + num_books_bought_at_store

The code incorrectly calculates the number of books Danny had initially by adding the number of books bought at the store to the total number of books he has now. The correct approach is to subtract the number of books bought from the total number of books he has now.

---

Question: There were 108 chickens and 20 sheeps at the farm, some of chickens and sheeps were sold. There are 87 chickens and 18 sheeps left now. How many chickens were sold?
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before - num_chicken_now
```

What's the problem with the above code?

1. The answer, 21, is a reasonable result.
2. Let's check the code:

> answer = num_chicken_before - num_chicken_now

The code calculates the number of chicken sold by subtracting the original number of chicken from the number of chicken left. There is no problem with the above code. 

---

Question: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday. How many did Katty score on monday and wednesday?
```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday + num_goals_on_wednesday
```

What's the problem with the above code?

1. The answer, 11, is a reasonable result.
2. Let's check the code:

> answer = num_goals_on_monday + num_goals_on_wednesday
The code calculates the total number of goals , scored by Katty on monday and wednesday by adding the number of goals on monday and number of goals on wednesday. There is no problem with the above code."""


SELF_REFINE_CRITIQUE_INSTRUCTION_SVAMP = """{examples}
(END OF EXAMPLES)

Question: {question}
```python
{answer}
```

What's the problem with the above code?"""


SVAMP_REFINE_FEWSHOT_EXAMPLES = """Question: James bought 93 red and 10 blue stickers, he used 31 red stickers on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
```python
original_red_stickers = 93
used_red_stickers = 31
original_blue_stickers = 10
used_blue_stickers = 7
answer = (original_red_stickers - used_red_stickers) + (original_blue_stickers - used_blue_stickers)
```

What's the problem with the above code?

1. The number of stickers should be positive, 65 > 0, it's reasonable.
2. Let's check the code:

> answer = (original_red_stickers - used_red_stickers) + (original_blue_stickers - used_blue_stickers)

The above code incorrectly combines the count of red and blue stickers. The question only asks for the number of red stickers James has left, so the calculation involving blue stickers is unnecessary and incorrect.

According to the question, James initially bought 93 red stickers and used 31 of them. To find the remaining number of red stickers, we only need to subtract the used red stickers from the original red stickers.

Here's a better solution:
```python
original_red_stickers = 93
used_red_stickers = 31

# Calculate the number of red stickers James has left
remaining_red_stickers = original_red_stickers - used_red_stickers
answer = remaining_red_stickers
```

---

Question: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy for each egg?
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars - discount_dollars
```

What's the problem with the above code?

1. The answer, 51, is a reasonable result.
2. Let's check the code:

> answer = original_egg_price_in_dollars - discount_dollars

The code correctly calculates the final price of each egg after applying the discount. There is no problem with the above code.

The proposed answer is correct and accurately computes the discounted price of each egg.

Question: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy for each egg?
Answer:
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
final_price = original_egg_price_in_dollars - discount_dollars
answer = final_price
```

---

Question: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books. How many books did danny have at first?
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now + num_books_bought_at_store
```

What's the problem with the above code?

1. The answer, 30, is not reasonable given the context.
2. Let's check the code:

> answer = num_books_now + num_books_bought_at_store

The code incorrectly calculates the number of books Danny had initially by adding the number of books bought at the store to the total number of books he has now. The correct approach is to subtract the number of books bought from the total number of books he has now.

Here's a better solution:
```python
num_books_bought_at_store = 5
num_books_now = 25

# Calculate the number of books Dianna had at first
num_books_initially = num_books_now - num_books_bought_at_store
answer = num_books_initially
```

---

Question: There were 108 chickens and 20 sheeps at the farm, some of chickens and sheeps were sold. There are 87 chickens and 18 sheeps left now. How many chickens were sold?
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before - num_chicken_now
```

What's the problem with the above code?

1. The answer, 21, is a reasonable result.
2. Let's check the code:

> answer = num_chicken_before - num_chicken_now

The code calculates the number of chicken sold by subtracting the original number of chicken from the number of chicken left. There is no problem with the above code. 

Here's a better solution:
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

What's the problem with the above code?

1. The answer, 11, is a reasonable result.
2. Let's check the code:

> answer = num_goals_on_monday + num_goals_on_wednesday
The code calculates the total number of goals , scored by Katty on monday and wednesday by adding the number of goals on monday and number of goals on wednesday. There is no problem with the above code.

Here's a better solution:
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

What's the problem with the above code?

{critique}

Return the better solution and if not, return the previous solution:"""


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
mean = sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person)
```

What's the problem with the above code?

The problem with the above code is that it does not assign the result of the calculation to the variable answer. The code correctly calculates the mean of the number of coins, but it does not store the result in the answer variable.

---

Read the following table regarding "" and then write Python code to answer a question:

Price | Quantity demanded | Quantity supplied
$155 | 22,600 | 5,800
$275 | 20,500 | 9,400
$395 | 18,400 | 13,000
$515 | 16,300 | 16,600
$635 | 14,200 | 20,200

Question: Look at the table. Then answer the question. At a price of $155, is there a shortage or a surplus? Choose from the the options: [shortage, surplus]
```python
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_at_price_155 > quantity_supplied_at_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'
```

What's the problem with the above code?

1. The above code causes a "NameError" because it uses the variables `quantity_demanded_at_price_155` and `quantity_supplied_at_price_155` without defining them.
2. The variable names in the code are a little bit long 

To solve the problem, we should first properly the quantity_demanded_price_155 and quantity_supplied_price_155 and then compare them.

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

What's the problem with the above code?

There is no problem with the above code. It correctly answers the question and calculates the median of the numbers. However, the code can be simplified.

---

Read the following table regarding "Toy prices" and then write Python code to answer a question:

toy boat | $5.54
toy guitar | $8.23
set of juggling balls | $5.01
trivia game | $8.18
jigsaw puzzle | $5.30
toy dinosaur | $3.00

Question: Lorenzo has $13.50. Does he have enough to buy a toy guitar and a set of juggling balls? Choose from the following options: ['yes', 'no']
```python
guitar_price = 8.23
juggling_balls_price = 5.01
total_money = 13.5
if total_money >= juggling_balls_price + guitar_price:
    answer = True
else:
    answer = False
```

What's the problem with the above code?

The problem with the above code is that it returns the answer as a boolean value (True or False), while the question expects a string response ('yes' or 'no').

To fix this issue, we should return the answer as a string that matches one of the provided options.

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
```python
apples_from_1_tree = 50
apples_from_2_trees = 100
apples_from_3_trees = 150
apples_from_4_trees = 200
apples_from_5_trees = 250
apples_from_6_trees = apples_from_5_trees + apples_from_1_tree
answer = apples_from_6_trees
```

What's the problem with the above code?

There is no problem with the above code. It correctly answers the question and calculates the number of apples harvested from 6 apple trees. However, the code can be simplified, as there's a clear pattern in the relationship between the number of apple trees and the number of apples harvested."""


SELF_REFINE_CRITIQUE_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

{question}
```python
{answer}
```

What's the problem with the above code?"""


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
mean = sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person)
```

What's the problem with the above code?

The problem with the above code is that it does not assign the result of the calculation to the variable answer. The code correctly calculates the mean of the number of coins, but it does not store the result in the answer variable.

Here's a better solution:
```python
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
mean = sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person)
answer = mean  
```

---

Read the following table regarding "" and then write Python code to answer a question:

Price | Quantity demanded | Quantity supplied
$155 | 22,600 | 5,800
$275 | 20,500 | 9,400
$395 | 18,400 | 13,000
$515 | 16,300 | 16,600
$635 | 14,200 | 20,200

Question: Look at the table. Then answer the question. At a price of $155, is there a shortage or a surplus? Choose from the the options: [shortage, surplus]
```python
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_at_price_155 > quantity_supplied_at_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'
```

What's the problem with the above code?

1. The above code causes a "NameError" because it uses the variables `quantity_demanded_at_price_155` and `quantity_supplied_at_price_155` without defining them.
2. The variable names in the code are a little bit long 

To solve the problem, we should first properly the quantity_demanded_price_155 and quantity_supplied_price_155 and then compare them.

Here's a better solution:
```python
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_price_155 > quantity_supplied_price_155:
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

What's the problem with the above code?

There is no problem with the above code. It correctly answers the question and calculates the median of the numbers. However, the code can be simplified.

Here's a better solution:
```python
cans = sorted([7, 4, 5, 8, 9])
middle = len(cans) // 2
answer = (cans[middle] + cans[~middle]) / 2
```

---

Read the following table regarding "Toy prices" and then write Python code to answer a question:

toy boat | $5.54
toy guitar | $8.23
set of juggling balls | $5.01
trivia game | $8.18
jigsaw puzzle | $5.30
toy dinosaur | $3.00

Question: Lorenzo has $13.50. Does he have enough to buy a toy guitar and a set of juggling balls? Choose from the following options: ['yes', 'no']
```python
guitar_price = 8.23
juggling_balls_price = 5.01
total_money = 13.5
if total_money >= juggling_balls_price + guitar_price:
    answer = True
else:
    answer = False
```

What's the problem with the above code?

The problem with the above code is that it returns the answer as a boolean value (True or False), while the question expects a string response ('yes' or 'no').

To fix this issue, we should return the answer as a string that matches one of the provided options.

Here's a better solution:
```python
guitar_price = 8.23
juggling_balls_price = 5.01
total_money = 13.5

# Calculate if Lorenzo has enough money to buy both items
if total_money >= juggling_balls_price + guitar_price:
    answer = "yes"
else:
    answer = "no"
```

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
```python
apples_from_1_tree = 50
apples_from_2_trees = 100
apples_from_3_trees = 150
apples_from_4_trees = 200
apples_from_5_trees = 250
apples_from_6_trees = apples_from_5_trees + apples_from_1_tree
answer = apples_from_6_trees
```

What's the problem with the above code?

There is no problem with the above code. It correctly answers the question and calculates the number of apples harvested from 6 apple trees. However, the code can be simplified, as there's a clear pattern in the relationship between the number of apple trees and the number of apples harvested.

Here's a better solution:
```python
number_of_trees = 6
apples_per_tree = 50
total_apples = number_of_trees * apples_per_tree
answer = total_apples
```"""


SELF_REFINE_REFINE_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

{question}
```python
{answer}
```

What's the problem with the above code?

{critique}

Return the better solution and if not, return the previous solution as code examples:"""


# ======================================================================== HUMANEVAL ======================================================================== #


SELF_REFINE_INSTRUCTION_HUMANEVAL = """You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. 

```python
{question}
    pass
```"""


HUMANEVAL_CRITIQUE_FEWSHOT_EXAMPLES = """```python
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

Overall, the primary issue is that the function does not perform a case conversion before counting the characters, which is essential for a correct case-insensitive anagram comparison. This oversight leads to the function incorrectly determining that strings like 'Listen' and 'silent' are not anagrams due to case differences. The correct approach should involve converting both input strings to the same case (either all uppercase or all lowercase) before applying the `Counter`."""


SELF_REFINE_CRITIQUE_INSTRUCTION_HUMANEVAL = """{examples}
(END OF EXAMPLES)

{question}

```python
{answer}

{tests}
```

What's the problem with the above code?"""


HUMANEVAL_REFINE_FEWSHOT_EXAMPLES = """```python
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


SELF_REFINE_REFINE_INSTRUCTION_HUMANEVAL = """{examples}
(END OF EXAMPLES)

{question}

```python
{answer}

{tests}
```

What's the problem with the above code?

{critique}

If no changes are needed, return the same code. 
Here's a better solution (include only function implementation):"""


# ======================================================================== MBPP ======================================================================== #


SELF_REFINE_INSTRUCTION_MBPP = """{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

```python"""


MBPP_CRITIQUE_FEWSHOT_EXAMPLES = """You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists.
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

---

You are an expert Python programmer, and here is your task: Write a python function to check whether the two numbers differ at one bit position only or not.
Your code should pass these tests.

```python
def is_power_of_two(x):
    return x and (not(x & (x - 1)))

def differ_at_one_bit_pos(a, b):
    return is_power_of_two(a ^ b)

assert differ_at_one_bit_pos(13, 9) == True
assert differ_at_one_bit_pos(15, 8) == False
assert differ_at_one_bit_pos(2, 4) == False
assert differ_at_one_bit_pos(2, 3) == True
assert differ_at_one_bit_pos(5, 1) == True
assert differ_at_one_bit_pos(1, 5) == True
```

What's the problem with the above code?

1. There is no problem with the code above. It follows the statement which checks whether the two numbers differ at one bit positino only or not.

2. Let's check the code:

> a = 13
> b = 9

This should identify that 13 (1101 in binary) and 9 (1001 in binary) differ at more than one bit position and should return True.

> return a ^ b == 1

Overall, the function succeed to perform as expected in comparing the XOR result.

---

You are an expert Python programmer, and here is your task: Write a function to find all words which are at least 4 characters long in a string.
Your code should pass these tests.

```python
import re

def find_char_long(text):
    return re.findall(r"\b\w{4,}\b", text)

assert set(find_char_long('Please move back to stream')) == set(['Please', 'move', 'back', 'stream'])
assert set(find_char_long('Jing Eco and Tech')) == set(['Jing', 'Tech'])
assert set(find_char_long('Jhingai wulu road Zone 3')) == set(['Jhingai', 'wulu', 'road', 'Zone'])
```

What's the problem with the above code?

There is no problem with the above code. It correctly identifies and returns all words which are at least 4 characters long in the given string. The regex pattern `\b\w{4,}\b` correctly matches words that are 4 or more characters long."""


SELF_REFINE_CRITIQUE_INSTRUCTION_MBPP = """{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests.

```python
{answer}

{tests}
```

What's the problem with the above code?"""


MBPP_REFINE_FEWSHOT_EXAMPLES = """You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists.
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
def is_power_of_two(x):
    return x and (not(x & (x - 1)))

def differ_at_one_bit_pos(a, b):
    return is_power_of_two(a ^ b)

assert differ_at_one_bit_pos(13, 9) == True
assert differ_at_one_bit_pos(15, 8) == False
assert differ_at_one_bit_pos(2, 4) == False
assert differ_at_one_bit_pos(2, 3) == True
assert differ_at_one_bit_pos(5, 1) == True
assert differ_at_one_bit_pos(1, 5) == True
```

What's the problem with the above code?

1. There is no problem with the code above. It follows the statement which checks whether the two numbers differ at one bit positino only or not.

2. Let's check the code:

> a = 13
> b = 9

This should identify that 13 (1101 in binary) and 9 (1001 in binary) differ at more than one bit position and should return True.

> return a ^ b == 1

Overall, the function succeed to perform as expected in comparing the XOR result.

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
import re

def find_char_long(text):
    return re.findall(r"\b\w{4,}\b", text)

assert set(find_char_long('Please move back to stream')) == set(['Please', 'move', 'back', 'stream'])
assert set(find_char_long('Jing Eco and Tech')) == set(['Jing', 'Tech'])
assert set(find_char_long('Jhingai wulu road Zone 3')) == set(['Jhingai', 'wulu', 'road', 'Zone'])
```

What's the problem with the above code?

There is no problem with the above code. It correctly identifies and returns all words which are at least 4 characters long in the given string. The regex pattern `\b\w{4,}\b` correctly matches words that are 4 or more characters long.

Here's the solution again for clarity:
```python
import re

def find_char_long(text):
    return re.findall(r"\b\w{4,}\b", text)
```"""

SELF_REFINE_REFINE_INSTRUCTION_MBPP = """{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests.

```python
{answer}

{tests}
```

What's the problem with the above code?

{critique}

If no changes are needed, return the same code. 
Here's a better solution:"""
