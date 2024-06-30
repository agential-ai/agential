"""Few-shot examples for SVAMP."""

NEW_SVAMP_FEWSHOT_EXAMPLES_POT = """Question: Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. How much did Marco's strawberries weigh?
# Python code, return answer
dad_strawberries_weight = 11
total_weight_together = 30
answer = total_weight_together - dad_strawberries_weight

---

Question: Zachary did 46 push-ups and 58 crunches in gym class today. David did 38 more push-ups but 62 less crunches than zachary. How many more crunches than push-ups did Zachary do?
# Python code, return answer
zachary_push_ups = 46
zachary_crunches = 58
david_push_ups = zachary_push_ups + 38
david_crunches = zachary_crunches - 62
# Calculate how many more crunches Zachary did than push-ups
answer = zachary_crunches - zachary_push_ups

---

Question: Faye had 35 packs of pencils each one having 4 pencils. She was placing her pencils into rows with 2 pencils in each row. How many rows could she make?
# Python code, return answer
num_packs_of_pencils = 35
pencils_per_pack = 4
pencils_total = num_packs_of_pencils * pencils_per_pack
pencils_per_row = 2
# Calculate how many rows Faye could make
answer = pencils_total // pencils_per_row

---

Question: Jack received 4 emails and sent 2 emails in the morning. He then received 6 emails and sent 8 emails in the afternoon. How many emails did Jack send in the day?
# Python code, return answer
morning_received = 4
morning_sent = 2
afternoon_received = 6
afternoon_sent = 8
# Calculate total emails sent by Jack in the day
answer = morning_sent + afternoon_sent



"""


NEW_SVAMP_FEWSHOT_EXAMPLES_REACT = """Question: Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. How much did Marco's strawberries weigh?
Thought 1: First, I need to find out how much did Marco's strawberries weigh.
Action 1: Calculate[
```python
dad_strawberries_weight = 11
total_weight_together = 30
answer = total_weight_together + dad_strawberries_weight
```
]
Observation 1: 
```python
dad_strawberries_weight = 11
total_weight_together = 30
answer = total_weight_together + dad_strawberries_weight
```
Execution Status: Done
Output: answer = 41
Thought 2: The answer is incorrect. I need to subtract the weight of Marco's dad's strawberries from the total weight of strawberries.
Action 2: Calculate[
```python
dad_strawberries_weight = 11
total_weight_together = 30
answer = total_weight_together - dad_strawberries_weight
```
]
Observation 2: 
```python
dad_strawberries_weight = 11
total_weight_together = 30
answer = total_weight_together - dad_strawberries_weight
```
Execution Status: Done
Output: answer = 19
Thought 3: Marco's strawberries weigh 19 pounds.
Action 3: Finish[
```python
answer = 19
```
]
Observation 3: 
```python
answer = 19
```

---

Question: Zachary did 46 push-ups and 58 crunches in gym class today. David did 38 more push-ups but 62 less crunches than zachary. How many more crunches than push-ups did Zachary do?
Thought 1: First, I need to find out the difference of crunches and push ups that Zachary did.
Action 1: Calculate[
```python
zachary_push_ups = 46
zachary_crunches = 58
david_push_ups = zachary_push_ups + 38
david_crunches = zachary_crunches - 62
answer = zachary_crunches - zachary_push_ups
```
]
Observation 1:
```python
zachary_push_ups = 46
zachary_crunches = 58
david_push_ups = zachary_push_ups + 38
david_crunches = zachary_crunches - 62
answer = zachary_crunches - zachary_push_ups
```
Execution Status: Done
Output: answer = -4
Thought 2: Zachary did 4 crunches less than push-ups.
Action 2: Finish[
```python
answer = -4
```
]
Observation 2:
```python
answer = -4
```

---

Question: Faye had 35 packs of pencils each one having 4 pencils. She was placing her pencils into rows with 2 pencils in each row. How many rows could she make?
Thought 1: First, I need to calculate the total number of pencils she has
Action 1: Calculate[
```python
num_packs_of_pencils = 35
pencils_per_pack = 4
pencils_total = num_packs_of_pencils * pencils_per_pack
pencils_per_row = 2
# Calculate how many rows Faye could make
answer = pencils_total // pencils_per_row
```
]
Observation 1:
```python
num_packs_of_pencils = 35
pencils_per_pack = 4
pencils_total = num_packs_of_pencils * pencils_per_pack
pencils_per_row = 2
# Calculate how many rows Faye could make
answer = pencils_total // pencils_per_row
```
Execution Status: Done
Output: answer = 70
Thought 2: Faye could make 70 rows.
Action 2: Finish[
```python
answer = 70
```
]
Observation 2:
```python
answer = 70
```

---

Question: Jack received 4 emails and sent 2 emails in the morning. He then received 6 emails and sent 8 emails in the afternoon. How many emails did Jack send in the day?
Thought 1: 
Action 1: Calculate[
```python
morning_received = 4
morning_sent = 2
afternoon_received = 6
afternoon_sent = 8
# Calculate total emails sent by Jack in the day
answer = morning_sent + afternoon_sent
```
]
Observation 1: 
```python
morning_received = 4
morning_sent = 2
afternoon_received = 6
afternoon_sent = 8
# Calculate total emails sent by Jack in the day
answer = morning_sent + afternoon_sent
```
Execution Status: Done
Output: answer = 10
Thought 2:  Jack sent in the day 10 emails.
Action 2: Finish[
```python
answer = 10
```
]
Observation 2:
```python
answer = 10
```

"""

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
