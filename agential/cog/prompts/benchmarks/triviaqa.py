"""Few-shot examples for TriviaQA."""


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