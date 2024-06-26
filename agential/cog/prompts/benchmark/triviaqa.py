"""Few-shot examples for TriviaQA."""

NEW_TRIVIAQA_FEWSHOT_EXAMPLES_COT = """Q: Which innovation for the car was developed by Prince Henry of Prussia in 1911? 
A: Prince Henry of Prussia is associated with the development of long-distance automobile racing. In 1907, he established the Prince Henry Trials, which were long-distance automobile races aimed at promoting automotive innovation and reliability. These trials encouraged advancements such as better engines, tires, and overall vehicle durability, pushing the development of more reliable and powerful cars suitable for long-distance travel. So the answer is: Standardized car testing and reliability competitions.

---

Q: When did the founder of Jehovah's Witnesses say the world would end?
A: The founder of Jehovah's Witnesses, Charles Taze Russell, did not specify a specific date for the end of the world. However, the Jehovah's Witnesses organization has made predictions regarding the end times, notably in 1914, 1925, and more recently, the year 1975. These predictions were based on their interpretation of biblical prophecies and the belief that significant events marking the end times were imminent. So the answer is: 1914.

---

Q: What state had its bi-centenary of joining the Union a year after North Carolina?
A: The state that had its bicentenary of joining the Union a year after North Carolina is Rhode Island. North Carolina became the 12th state to join the Union on November 21, 1789, while Rhode Island became the 13th state on May 29, 1790. Therefore, Rhode Island celebrated its bicentenary (200th anniversary) of joining the Union a year after North Carolina did. So the answer is: Rhode Island.
"""

NEW_TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT = """Q: Which innovation for the car was developed by Prince Henry of Prussia in 1911? 
A: Standardized car testing and reliability competitions.

---

Q: When did the founder of Jehovah's Witnesses say the world would end?
A: 1914.

---

Q: What state had its bi-centenary of joining the Union a year after North Carolina?
A: Rhode Island.
"""

NEW_TRIVIAQA_FEWSHOT_EXAMPLES_REACT = """Question: Which innovation for the car was developed by Prince Henry of Prussia in 1911?
Thought 1: I need to determine the innovation for the car developed by Prince Henry of Prussia in 1911.
Action 1: Search[Prince Henry of Prussia car innovation 1911]
Observation 1: Prince Henry of Prussia established a prize in 1908 for a reliable, inexpensive automobile. This led to the development of standardized car testing and reliability competitions.
Thought 2: Prince Henry of Prussia's innovation was related to standardized car testing and reliability competitions.
Action 2: Finish[Standardized car testing and reliability competitions]

---

Question: When did the founder of Jehovah's Witnesses say the world would end?
Thought 1: I need to find out when the founder of Jehovah's Witnesses predicted the end of the world.
Action 1: Search[founder of Jehovah's Witnesses end of world prediction]
Observation 1: The founder of Jehovah's Witnesses, Charles Taze Russell, initially predicted that the end of the world and Christ's Second Coming would occur in 1914.
Thought 2: Charles Taze Russell, the founder of Jehovah's Witnesses, predicted the end of the world to occur in 1914.
Action 2: Finish[1914]

---

Question: What state had its bicentenary of joining the Union a year after North Carolina?
Thought 1: I need to find out which state celebrated its bicentenary of joining the Union one year after North Carolina.
Action 1: Search[state bicentenary joining Union after North Carolina]
Observation 1: North Carolina joined the Union on November 21, 1789. The state that celebrated its bicentenary of joining the Union one year after North Carolina would be Rhode Island.
Thought 2: Rhode Island celebrated its bicentenary of joining the Union in 1890, one year after North Carolina in 1889.
Action 2: Finish[Rhode Island]

"""

TRIVIAQA_FEWSHOT_EXAMPLES_COT = """Q: Mendelssohn's 'Wedding March' was. originally written as incidental music for which Shakespeare play in 1842? 
A: Mendelssohn's 'Wedding March' was originally written as incidental music for A Midsummer Night's Dream in 1842. So the answer is: A Midsummer Night's Dream.

---

Q: Christ in the House of his Parents"" is one of the best known paintings of which artist?" 
A: "Christ in the House of his Parents" is a painting by John Everett Millais. So the answer is: John Everett Millais.

---

Q: Who designed the National Theatre building on the South Bank in London ? 
A: The National Theatre building on the South Bank in London was designed by Denys Lasdun. So the answer is: Denys Lasdun.

---

Q: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics? 
A: The only American skier who won the gold medal in the Men's Combined at the 2010 Winter Olympics is Bode Miller. So the answer is: Bode Miller."""


TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT = """Q: Mendelssohn's 'Wedding March' was. originally written as incidental music for which Shakespeare play in 1842?
A: A Midsummer Night's Dream

---

Q: \"\"\"Christ in the House of his Parents\"\" is one of the best known paintings of which artist?"
A: John Millais

---

Q: Who designed the National Theatre building on the South Bank in London ?
A: Sir Denys Lasdun

---

Q: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?
A: Bodie Miller"""


TRIVIAQA_FEWSHOT_EXAMPLES_REACT = """Question: Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842?
Thought 1: Let's search the question in google
Action 1: Search[Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842? site:wikipedia.org]
Observation 1: [Wedding March (Mendelssohn) - Wikipedia] Mendelssohn's 'Wedding March' was originally written as incidental music for A Midsummer Night's Dream in 1842.
Thought 2: The evidence suggests that Mendelssohn's 'Wedding March' was originally written as incidental music for A Midsummer Night's Dream in 1842.
Action 2: Finish[A Midsummer Night's Dream]

---

Question: "Christ in the House of his Parents" is one of the best known paintings of which artist?
Thought 1: Let's search the question in google
Action 1: Search["Christ in the House of his Parents" is one of the best known paintings of which artist? site:wikipedia.org]
Observation 1: [Christ in the House of His Parents - Wikipedia] "Christ in the House of His Parents" is a painting by John Everett Millais.
Thought 2: The evidence suggests that "Christ in the House of His Parents" is a painting by John Everett Millais.
Action 2: Finish[John Everett Millais]

---

Question: Who designed the National Theatre building on the South Bank in London?
Thought 1: Let's search the question in google
Action 1: Search[Who designed the National Theatre building on the South Bank in London? site:wikipedia.org]
Observation 1: [National Theatre (London) - Wikipedia] The National Theatre building on the South Bank in London was designed by Denys Lasdun.
Thought 2: The evidence suggests that the National Theatre building on the South Bank in London was designed by Denys Lasdun.
Action 2: Finish[Denys Lasdun]

---

Question: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics?
Thought 1: Let's search the question in google
Action 1: Search[Which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics? site:wikipedia.org]
Observation 1: [Bode Miller - Wikipedia] The only American skier who won the gold medal in the Men's Combined at the 2010 Winter Olympics is Bode Miller.
Thought 2: The evidence suggests that Bode Miller is the only American skier who won the gold medal in the Men's Combined at the 2010 Winter Olympics.
Action 2: Finish[Bode Miller]"""
