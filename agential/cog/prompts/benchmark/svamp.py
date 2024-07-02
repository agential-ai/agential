"""Few-shot examples for SVAMP."""

SVAMP_FEWSHOT_EXAMPLES_POT = """Question: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
# Python code, return answer
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers - used_red_stickers

---

Question: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy for each egg?
# Python code, return answer
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars - discount_dollars

---

Question: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books. How many books did danny have at first?
# Python code, return answer
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now - num_books_bought_at_store

---

Question: There were 108 chickens and 20 sheeps at the farm, some of chickens and sheeps were sold. There are 87 chickens and 18 sheeps left now. How many chickens were sold?
# Python code, return answer
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before - num_chicken_now

---

Question: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday. How many did Katty score on monday and wednesday?
# Python code, return answer
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday + num_goals_on_wednesday

---

Question: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined.  How many more girls than boys are in the Masquerade?
# Python code, return answer
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
answer = total_girls - total_boys

---

Question: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now.  How much ice creasm did Getty purchase?
# Python code, return answer
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
answer = total_ice_creams - num_ice_creams_bought_by_joseph"""


SVAMP_FEWSHOT_EXAMPLES_COT = """Question: James bought 93 red and 10 blue stickers, he used 31 red stickers on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
Thought: Let's think step by step. James initially bought 93 red stickers. He used 31 red stickers on his fridge. So, the number of red stickers left is 93 - 31 = 62.
Action: Finish[
```python
original_red_stickers = 93
used_red_stickers = 31
red_stickers_left = original_red_stickers - used_red_stickers
answer = red_stickers_left
```
]

---

Question: Allen went to the supermarket to buy eggs. Each egg costs 80 dollars, and there is a discount of 29 dollars. How much does Allen have to pay for each egg?
Thought: Let's think step by step. The original cost of each egg is 80 dollars. There is a discount of 29 dollars. So, the amount Allen has to pay for each egg is 80 - 29 = 51 dollars.
Action: Finish[
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
price_after_discount = original_egg_price_in_dollars - discount_dollars
answer = price_after_discount
```
]

---

Question: Dianna collects both cases and books. She bought 22 cases and 5 books from the store. Now she has 57 cases and 25 books. How many books did Dianna have at first?
Thought: Let's think step by step. Dianna bought 5 books from the store, and now she has 25 books. So, the number of books she had at first is 25 - 5 = 20.
Action: Finish[
```python
num_books_bought_at_store = 5
num_books_now = 25
initial_books = num_books_now - num_books_bought_at_store
answer = initial_books
```
]

---

Question: There were 108 chickens and 20 sheep at the farm. Some of the chickens and sheep were sold. Now there are 87 chickens and 18 sheep left. How many chickens were sold?
Thought: Let's think step by step. Initially, there were 108 chickens. Now there are 87 chickens. So, the number of chickens sold is 108 - 87 = 21.
Action: Finish[
```python
num_chickens_before = 108
num_chickens_now = 87
chickens_sold = num_chickens_before - num_chickens_now
answer = chickens_sold
```
]

---

Question: Katty scored 2 goals on Monday, 8 goals on Tuesday, and 9 goals on Wednesday. How many goals did Katty score on Monday and Wednesday?
Thought: Let's think step by step. Katty scored 2 goals on Monday and 9 goals on Wednesday. So, the total number of goals scored on Monday and Wednesday is 2 + 9 = 11.
Action: Finish[
```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
total_goals = num_goals_on_monday + num_goals_on_wednesday
answer = total_goals
```
]

---

Question: There are 5 girls and 4 boys in the Masquerade. 12 more girls and 7 more boys joined. How many more girls than boys are in the Masquerade?
Thought: Let's think step by step. Initially, there were 5 girls and 4 boys. Then 12 more girls and 7 more boys joined. The total number of girls is 5 + 12 = 17, and the total number of boys is 4 + 7 = 11. So, the difference is 17 - 11 = 6.
Action: Finish[
```python
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
difference = total_girls - total_boys
answer = difference
```
]

---

Question: Joseph and Getty went to buy ice creams. Together, they bought 36 ice creams. On the way back, Joseph ate 12 of the ice creams, and he has 2 ice creams left now. How many ice creams did Getty purchase?
Thought: Let's think step by step. Joseph ate 12 ice creams and has 2 left, so he initially had 12 + 2 = 14 ice creams. Together, they bought 36 ice creams, so the number of ice creams Getty purchased is 36 - 14 = 22.
Action: Finish[
```python
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
ice_creams_bought_by_getty = total_ice_creams - num_ice_creams_bought_by_joseph
answer = ice_creams_bought_by_getty
```
]"""


SVAMP_FEWSHOT_EXAMPLES_REACT = """Question: James bought 93 red and 10 blue stickers, he used 31 red stickers on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
Thought 1: First, I need to find out how many red stickers James has left after using some on his fridge.
Action 1: Calculate[
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers + used_red_stickers
```
]
Observation 1: 
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers + used_red_stickers
```
Execution Status: Done
Output: answer = 124
Thought 2: The answer is incorrect. I need to subtract the used stickers from the original number of stickers to get the correct answer.
Action 2: Calculate[
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers - used_red_stickers
```
]
Observation 2: 
```python
original_red_stickers = 93
used_red_stickers = 31
answer = original_red_stickers - used_red_stickers
```
Execution Status: Done
Output: answer = 62
Thought 3: James has 62 red stickers left.
Action 3: Finish[
```python
answer = 62
```
]
Observation 3: 
```python
answer = 62
```

---

Question: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars. How much do you have to pay to buy each egg?
Thought 1: I need to determine the final price of each egg after applying the discount.
Action 1: Calculate[
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars + discount_dollars
```
]
Observation 1: 
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars + discount_dollars
```
Execution Status: Done
Output: answer = 109
Thought 2: The answer is incorrect. I need to subtract the discount from the original price to get the correct answer.
Action 2: Calculate[
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars - discount_dollars
```
]
Observation 2: 
```python
original_egg_price_in_dollars = 80
discount_dollars = 29
answer = original_egg_price_in_dollars - discount_dollars
```
Execution Status: Done
Output: answer = 51
Thought 3: Each egg costs 51 dollars after the discount.
Action 3: Finish[
```python
answer = 51
```
]
Observation 3: 
```python
answer = 51
```

---

Question: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books. How many books did Danny have at first?
Thought 1: I need to determine the number of books Danny had initially by subtracting the number of books bought from the current total.
Action 1: Calculate[
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now - num_books_bought_at_store
```
]
Observation 1: 
```python
num_books_bought_at_store = 5
num_books_now = 25
answer = num_books_now - num_books_bought_at_store
```
Execution Status: Done
Output: answer = 20
Thought 2: Danny had 20 books initially.
Action 2: Finish[
```python
answer = 20
```
]
Observation 2: 
```python
answer = 20
```

---

Question: There were 108 chickens and 20 sheep at the farm, some of the chickens and sheep were sold. There are 87 chickens and 18 sheep left now. How many chickens were sold?
Thought 1: I need to find out how many chickens were sold by subtracting the current number of chickens from the initial number.
Action 1: Calculate[
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before - num_chicken_now
```
]
Observation 1: 
```python
num_chicken_before = 108
num_chicken_now = 87
answer = num_chicken_before - num_chicken_now
```
Execution Status: Done
Output: answer = 21
Thought 2: 21 chickens were sold.
Action 2: Finish[
```python
answer = 21
```
]
Observation 2: 
```python
answer = 21
```

---

Question: Katty scored 2 goals on Monday, 8 goals on Tuesday and 9 goals on Wednesday. How many did Katty score on Monday and Wednesday?
Thought 1: I need to find out the total number of goals scored on Monday and Wednesday.
Action 1: Calculate[
```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday + num_goals_on_wednesday
```
]
Observation 1: 
```python
num_goals_on_monday = 2
num_goals_on_wednesday = 9
answer = num_goals_on_monday + num_goals_on_wednesday
```
Execution Status: Done
Output: answer = 11
Thought 2: Katty scored 11 goals on Monday and Wednesday combined.
Action 2: Finish[
```python
answer = 11
```
]
Observation 2: 
```python
answer = 11
```

---

Question: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined. How many more girls than boys are in the Masquerade?
Thought 1: I need to calculate the total number of girls and boys in the Masquerade and find the difference between them.
Action 1: Calculate[
```python
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
answer = total_girls - total_boys
```
]
Observation 1: 
```python
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
answer = total_girls - total_boys
```
Execution Status: Done
Output: answer = 6
Thought 2: There are 6 more girls than boys in the Masquerade.
Action 2: Finish[
```python
answer = 6
```
]
Observation 2: 
```python
answer = 6
```

---

Question: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creams, and he has 2 ice creams left now. How many ice creams did Getty purchase?
Thought 1: I need to calculate the number of ice creams Joseph had initially and subtract this from the total to find out how many Getty purchased.
Action 1: Calculate[
```python
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
answer = total_ice_creams - num_ice_creams_bought_by_joseph
```
]
Observation 1: 
```python
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
answer = total_ice_creams - num_ice_creams_bought_by_joseph
```
Execution Status: Done
Output: answer = 22
Thought 2: Getty purchased 22 ice creams.
Action 2: Finish[
```python
answer = 22
```
]
Observation 2: 
```python
answer = 22
```"""
