"""Few-shot examples for GSM8K."""

GSM8K_FEWSHOT_EXAMPLES_POT = """Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
# Python code, return answer
jason_lollipops_initial = 20
jason_lollipops_after = 12
denny_lollipops = jason_lollipops_initial - jason_lollipops_after
answer = denny_lollipops

---

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
# Python code, return answer
trees_initial = 15
trees_after = 21
trees_added = trees_after - trees_initial
answer = trees_added

---

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
# Python code, return answer
toys_initial = 5
mom_toys = 2
dad_toys = 2
total_received = mom_toys + dad_toys
total_toys = toys_initial + total_received
answer = total_toys

---

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
# Python code, return answer
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between monday and thursday
computers_added = computers_per_day * num_days
computers_total = computers_initial + computers_added
answer = computers_total

---

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
# Python code, return answer
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
answer = golf_balls_left

---

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
# Python code, return answer
cars_initial = 3
cars_arrived = 2
total_cars = cars_initial + cars_arrived
answer = total_cars

---

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
# Python code, return answer
money_initial = 23
bagels = 5
bagel_cost = 3
money_spent = bagels * bagel_cost
money_left = money_initial - money_spent
answer = money_left

---

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
# Python code, return answer
leah_chocolates = 32
sister_chocolates = 42
total_chocolates = leah_chocolates + sister_chocolates
chocolates_eaten = 35
chocolates_left = total_chocolates - chocolates_eaten
answer = chocolates_left"""


GSM8K_FEWSHOT_EXAMPLES_COT = """Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Thought: Let's think step by step. Jason had 20 lollipops initially and now he has 12 lollipops. So, he must have given 20 - 12 = 8 lollipops to Denny.
Action: Finish[
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
denny_lollipops = jason_lollipops_initial - jason_lollipops_after
answer = denny_lollipops
```
]

---

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Thought: Let's think step by step. There were initially 15 trees and after planting, there are 21 trees. So, the number of trees planted is 21 - 15 = 6.
Action: Finish[
```python
trees_initial = 15
trees_after = 21
trees_added = trees_after - trees_initial
answer = trees_added
```
]

---

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Thought: Let's think step by step. Shawn had 5 toys initially, and he got 2 toys from his mom and 2 toys from his dad. So, the total number of toys he has now is 5 + 2 + 2 = 9.
Action: Finish[
```python
toys_initial = 5
mom_toys = 2
dad_toys = 2
total_received = mom_toys + dad_toys
total_toys = toys_initial + total_received
answer = total_toys
```
]

---

Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
Thought: Let's think step by step. Initially, there were 9 computers. Five more computers were added each day from Monday to Thursday, which is 4 days. So, the total number of computers added is 5 * 4 = 20. Therefore, the total number of computers now is 9 + 20 = 29.
Action: Finish[
```python
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between Monday and Thursday
computers_added = computers_per_day * num_days
computers_total = computers_initial + computers_added
answer = computers_total
```
]

---

Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
Thought: Let's think step by step. Michael initially had 58 golf balls. He lost 23 golf balls on Tuesday and 2 more on Wednesday. So, the total number of golf balls lost is 23 + 2 = 25. Therefore, the number of golf balls he has left is 58 - 25 = 33.
Action: Finish[
```python
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
answer = golf_balls_left
```
]


---

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Thought: Let's think step by step. Initially, there are 3 cars in the parking lot. When 2 more cars arrive, the total number of cars is 3 + 2 = 5.
Action: Finish[
```python
cars_initial = 3
cars_arrived = 2
total_cars = cars_initial + cars_arrived
answer = total_cars
```
]

---

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Thought: Let's think step by step. Olivia had $23 initially. She bought 5 bagels for $3 each. The total cost of the bagels is 5 * 3 = $15. So, the money she has left is 23 - 15 = $8.
Action: Finish[
```python
money_initial = 23
bagels = 5
bagel_cost = 3
money_spent = bagels * bagel_cost
money_left = money_initial - money_spent
answer = money_left
```
]

---

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Thought: Let's think step by step. Leah had 32 chocolates and her sister had 42 chocolates. Together they had 32 + 42 = 74 chocolates. They ate 35 chocolates, so the number of chocolates left is 74 - 35 = 39.
Action: Finish[
```python
leah_chocolates = 32
sister_chocolates = 42
total_chocolates = leah_chocolates + sister_chocolates
chocolates_eaten = 35
chocolates_left = total_chocolates - chocolates_eaten
answer = chocolates_left
```
]"""


GSM8K_FEWSHOT_EXAMPLES_REACT = """Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Thought 1: First, I need to find out how many lollipops Jason gave to Denny.
Action 1: Calculate[
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
answer = jason_lollipops_after - jason_lollipops_initial
```
]
Observation 1: 
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
answer = jason_lollipops_after - jason_lollipops_initial
```
Execution Status: Done
Output: answer = -8
Thought 2: The answer should be positive because Jason gave lollipops to Denny. I need to reverse the subtraction.
Action 2: Calculate[
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
answer = jason_lollipops_initial - jason_lollipops_after
```
]
Observation 2: 
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
answer = jason_lollipops_initial - jason_lollipops_after
```
Execution Status: Done
Output: answer = 8
Thought 3: Jason gave 8 lollipops to Denny.
Action 3: Finish[
```python
answer = 8
```
]
Observation 3: 
```python
answer = 8
```

---

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Thought 1: I need to determine how many trees were planted.
Action 1: Calculate[
```python
trees_initial = 15
trees_after = 21
answer = trees_initial - trees_after
```
]
Observation 1: 
```python
trees_initial = 15
trees_after = 21
answer = trees_initial - trees_after
```
Execution Status: Done
Output: answer = -6
Thought 2: The answer should be positive because trees are being added, not removed. I need to reverse the subtraction.
Action 2: Calculate[
```python
trees_initial = 15
trees_after = 21
answer = trees_after - trees_initial
```
]
Observation 2: 
```python
trees_initial = 15
trees_after = 21
answer = trees_after - trees_initial
```
Execution Status: Done
Output: answer = 6
Thought 3: The workers planted 6 trees today.
Action 3: Finish[
```python
answer = 6
```
]
Observation 3: 
```python
answer = 6
```

---

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Thought 1: I need to calculate the total number of toys received.
Action 1: Calculate[
```pyhton 
toys_received_mom = 2
toys_received_dad = 2
toys_received = toys_received_mom + toys_received_dad
```
]
Observation 1:
```pyhton 
toys_received_mom = 2
toys_received_dad = 2
answer = toys_received_mom + toys_received_dad
```
Execution Status: Done
Output: answer = 4
Thought 2: I need to calculate the total number of toys Shawn has after receiving gifts from his parents.
Action 2: Calculate[
```python
toys_initial = 5
toys_received = 2 * 3
answer = toys_initial + toys_received
```
]
Observation 2: 
```python
toys_initial = 5
toys_received = 2 * 3
answer = toys_initial + toys_received
```
Execution Status: Done
Output: answer = 11
Thought 3: I incorrectly calculated the total toys received. Shawn got 2 toys each from his mom and dad, so the correct total received should be 2 + 2.
Action 3: Calculate[
```python
toys_initial = 5
toys_received = 2 + 2
answer = toys_initial + toys_received
```
]
Observation 3: 
```python
toys_initial = 5
toys_received = 2 + 2
answer = toys_initial + toys_received
```
Execution Status: Done
Output: answer = 9
Thought 4: Shawn now has 9 toys.
Action 4: Finish[
```python
answer = 9
```
]
Observation 4: 
```python
answer = 9
```

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

---

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
Thought 1: I need to calculate how much money Olivia has left after buying bagels.
Action 1: Calculate[
```python
money_initial = 23
bagels = 5
bagel_cost = 3
money_spent = bagels * bagel_cost
answer = money_initial - money_spent
```
]
Observation 1: 
```python
money_initial = 23
bagels = 5
bagel_cost = 3
money_spent = bagels * bagel_cost
answer = money_initial - money_spent
```
Execution Status: Done
Output: answer = 8
Thought 2: Olivia has $8 left.
Action 2: Finish[
```python
answer = 8
```
]
Observation 2: 
```python
answer = 8
```

---

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Thought 1: I need to find out the total number of chocolates left after Leah and her sister ate some.
Action 1: Calculate[
```python
leah_chocolates = 32
sister_chocolates = 42
chocolates_eaten = 35
answer = leah_chocolates - chocolates_eaten
```
]
Observation 1:
```python
leah_chocolates = 32
sister_chocolates = 42
chocolates_eaten = 35
answer = leah_chocolates - chocolates_eaten
```
Execution Status: Done
Output: answer = -3
Thought 2: The calculation is incorrect because it only considers Leah's chocolates. I need to include her sister's chocolates as well.
Action 2: Calculate[
```python
leah_chocolates = 32
sister_chocolates = 42
chocolates_eaten = 35
answer = sister_chocolates - chocolates_eaten
```
]
Observation 2:
```python
leah_chocolates = 32
sister_chocolates = 42
chocolates_eaten = 35
answer = sister_chocolates - chocolates_eaten
```
Execution Status: Done
Output: answer = 7
Thought 3: The calculation is still incorrect because it only considers the sister's chocolates. I need to consider the total number of chocolates initially.
Action 3: Calculate[
```python
leah_chocolates = 32
sister_chocolates = 42
chocolates_eaten = 35
answer = (leah_chocolates + sister_chocolates) - chocolates_eaten - sister_chocolates
```
]
Observation 3:
```python
leah_chocolates = 32
sister_chocolates = 42
chocolates_eaten = 35
answer = (leah_chocolates + sister_chocolates) - chocolates_eaten - sister_chocolates
```
Execution Status: Done
Output: answer = -3
Thought 4: The calculation is still incorrect because it incorrectly subtracts the sister's chocolates again. I need to correct this error.
Action 4: Calculate[
```python
leah_chocolates = 32
sister_chocolates = 42
total_chocolates = leah_chocolates + sister_chocolates
chocolates_eaten = 35
answer = total_chocolates - chocolates_eaten
```
]
Observation 4:
```python
leah_chocolates = 32
sister_chocolates = 42
total_chocolates = leah_chocolates + sister_chocolates
chocolates_eaten = 35
answer = total_chocolates - chocolates_eaten
```
Execution Status: Done
Output: answer = 39
Thought 5: They have 39 chocolates left.
Action 5: Finish[
```python
answer = 39
```
]
Observation 5:
```python
answer = 39
```"""
