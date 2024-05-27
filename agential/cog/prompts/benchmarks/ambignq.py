"""Few-shot examples for AmbigNQ"""


AMBIGNQ_FEWSHOT_EXAMPLES_COT = """Q: When did men's figure skating become a summer Olympic sport?
A: Figure skating was first contested in the Olympic Games at the 1908 Summer Olympics. So the answer is: 1908

---

Q: When was the last time the USA men's national soccer team missed the World Cup?
A: The USA men's national soccer team did not qualify for the World Cup in 2018. So the answer is: 2018

---

Q: What does barium do in a ct scan?
A: Barium is used to improve visualization of the gastrointestinal tract during a CT scan. So the answer is: to improve visualization of the gastrointestinal tract

---

Q: Where was the fort located at which the first shot of the civil war was fired?
A: The first shot of the Civil War was fired at Fort Sumter, which is located in Charleston Harbor, South Carolina. So the answer is: Charleston Harbor, South Carolina

---

Q: When did nando's come to the uk?
A: Nando's opened its first restaurants in the United Kingdom in 1992. So the answer is: 1992"""


AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT = """Q: When did men's figure skating become a summer Olympic sport?
A: 1908

---

Q: When was the last time the USA men's national soccer team missed the World Cup?
A: 2018

---

Q: What does barium do in a ct scan?
A: to improve visualization of the gastrointestinal tract

---

Q: Where was the fort located at which the first shot of the civil war was fired?
A: Charleston Harbor, South Carolina

---

Q: When did nando's come to the uk?
A: 1992"""


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
Action 2: Finish[1992]"""