"""Few-shot examples for GSM8K."""

NEW_GSM8K_FEWSHOT_EXAMPLES_POT = """Question: Janet hires six employees. Four of them are warehouse workers who make $15/hour, and the other two are managers who make $20/hour. Janet has to pay 10%% of her workers' salaries in FICA taxes. If everyone works 25 days a month and 8 hours a day, how much does Janet owe total for their wages and taxes for one month?
# Python code, return answer
# Define the variables
warehouse_workers = 4
manager_workers = 2
warehouse_hourly_rate = 15
manager_hourly_rate = 20
days_worked_per_month = 25
hours_per_day = 8
fica_tax_rate = 0.1
wages_warehouse = warehouse_workers * warehouse_hourly_rate * days_worked_per_month * hours_per_day
wages_managers = manager_workers * manager_hourly_rate * days_worked_per_month * hours_per_day
total_wages_before_tax = wages_warehouse + wages_managers
total_due = total_wages_before_tax * ( 1 + fica_tax_rate)
answer = total_due

---

Question: A farmer has twice as many pigs as cows, and 4 more cows than goats.  If the farmer has 56 animals total, how many pigs does he have?
# Python code, return answer
# let's define the number of goats is x
# then the number of cows is 4+x
# then the number of pigs is 2(4+x) 
# so, total animals is x + (4+x) + 2(4+x) = 56
# Now we can solve for "x":
# x + 4 + x + 2x + 8 = 56
# 4x + 12 = 56
# 4x = 44
# x = 11
num_of_goats = 11
num_of_cows = num_of_goats + 4
num_of_pigs = 2 * num_of_cows
answer = num_of_pigs

---

Question: Ann, Bill, Cate, and Dale each buy personal pan pizzas cut into 4 pieces. If Bill and Dale eat 50%% of their pizzas and Ann and Cate eat 75%% of the pizzas, how many pizza pieces are left uneaten?
# Python code, return answer
pizza_pieces = 4  # each person buys a personal pan pizza cut into 4 pieces
ann_pieces = 4 * 0.75  # Ann eats 75%% of her pizza
bill_pieces = 4 * 0.5  # Bill eats 50%% of his pizza
cate_pieces = 4 * 0.75  # Cate eats 75%% of her pizza
dale_pieces = 4 * 0.5  # Dale eats 50%% of his pizza
total_pieces_eaten = ann_pieces + bill_pieces + cate_pieces + dale_pieces
total_pieces = pizza_pieces * 4  # there are 4 people
pieces_left = total_pieces - total_pieces_eaten
answer = pieces_left

---

Question: Tommy is making 12 loaves of bread. He needs 4 pounds of flour per loaf. A 10-pound bag of flour costs $10 and a 12-pound bag costs $13. When he is done making his bread, he has no use for flour and so he will throw away whatever is left. How much does he spend on flour if he buys the cheapest flour to get enough?
# Python code, return answer
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

"""

NEW_GSM8K_FEWSHOT_EXAMPLES_REACT = """Question: Janet hires six employees. Four of them are warehouse workers who make $15/hour, and the other two are managers who make $20/hour. Janet has to pay 10%% of her workers' salaries in FICA taxes. If everyone works 25 days a month and 8 hours a day, how much does Janet owe total for their wages and taxes for one month?
Thought 1: First, I need to calculate the total wages for the warehouse workers.
Action 1: Calculate[
```pyhon
warehouse_workers = 4
warehouse_hourly_rate = 15
days_worked_per_month = 25
hours_per_day = 8
answer = warehouse_workers * warehouse_hourly_rate * days_worked_per_month * hours_per_day
```
]
Observation 1:
```python
warehouse_workers = 4
warehouse_hourly_rate = 15
days_worked_per_month = 25
hours_per_day = 8
answer = warehouse_workers * warehouse_hourly_rate * days_worked_per_month * hours_per_day
```
Execution Status: Done
Output: answer = 120000
Thought 2: Then , I need to calculate the total wages for the managers:
Action 2: Calculate[
```pyhon
manager_workers = 2
manager_hourly_rate = 20
days_worked_per_month = 25
hours_per_day = 8
answer = manager_workers * manager_hourly_rate * days_worked_per_month * hours_per_day
```
]
Observation 2:
```python
manager_workers = 2
warehouse_hourly_rate = 15
days_worked_per_month = 25
hours_per_day = 8
answer = manager_workers * manager_hourly_rate * days_worked_per_month * hours_per_day
```
Execution Status: Done
Output: answer = 8000
Thought 3: After that , I need to sum the total wages for all workers and calculate the FICA taxes:
Action 3: Calculate[
```python
warehouse_workers = 4
manager_workers = 2
warehouse_hourly_rate = 15
manager_hourly_rate = 20
days_worked_per_month = 25
hours_per_day = 8
fica_tax_rate = 0.1
wages_warehouse = warehouse_workers * warehouse_hourly_rate * days_worked_per_month * hours_per_day
wages_managers = manager_workers * manager_hourly_rate * days_worked_per_month * hours_per_day
total_wages_before_tax = wages_warehouse + wages_managers
total_wages_before_tax = wages_warehouse + wages_managers
answer = total_wages_before_tax * ( 1 + fica_tax_rate)
```
]
Observation 3:
```python
warehouse_workers = 4
manager_workers = 2
warehouse_hourly_rate = 15
manager_hourly_rate = 20
days_worked_per_month = 25
hours_per_day = 8
fica_tax_rate = 0.1
wages_warehouse = warehouse_workers * warehouse_hourly_rate * days_worked_per_month * hours_per_day
wages_managers = manager_workers * manager_hourly_rate * days_worked_per_month * hours_per_day
total_wages_before_tax = wages_warehouse + wages_managers
total_wages_before_tax = wages_warehouse + wages_managers
answer = total_wages_before_tax * ( 1 + fica_tax_rate)
```
Execution Status: Done
Output: answer = 22000
Thought 4: Janet owe total of 22000 for their wages and taxes for one month.
Action 4: Finish[
```python
answer = 22000
```
]
Observation 4: 
```python
answer = 22000
```

---

Question: A farmer has twice as many pigs as cows, and 4 more cows than goats.  If the farmer has 56 animals total, how many pigs does he have?
Thought 1: First, I need to set up equation based on the relationship.
Action 1: Calculate[
```python
# let's define the number of goats is x
# then the number of cows is 4+x
# then the number of pigs is 2(4+x) 
# so, total animals is x + (4+x) + 2(4+x) = 56
```
]
Observation 1:
```python
# let's define the number of goats is x
# then the number of cows is 4+x
# then the number of pigs is 2(4+x) 
# so, total animals is x + (4+x) + 2(4+x) = 56
```
Execution Status: Done
Output: x = 11
Thought 2: Then , I need to validate the answer.
Action 2: Calculate[
```python
answer = x + (4+x) + 2(4+x)
```
]
Observation 2:
```python
answer = x + (4+x) + 2(4+x)
```
Executaion Status: Done
Output: answer = 56
Thought 3: After , I need to find out the number of pigs.
Action 3: Calculate[
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
num_of_goats = 11
num_of_cows = num_of_goats + 4
num_of_pigs = 2 * num_of_cows
answer = num_of_pigs
```
]
Observation 3:
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
num_of_goats = 11
num_of_cows = num_of_goats + 4
num_of_pigs = 2 * num_of_cows
answer = num_of_pigs
```
Execution Status: Done
Output: answer = 30
Thought 4: The farmer has 30 pigs.
Action 4: Finish[
```python
answer = 30
```
]
Observation 4: 
```python
answer = 30
```

---
Question: Ann, Bill, Cate, and Dale each buy personal pan pizzas cut into 4 pieces. If Bill and Dale eat 50%% of their pizzas and Ann and Cate eat 75%% of the pizzas, how many pizza pieces are left uneaten?
Thought 1: First I need to calculate the total number of pizza pieces each person eats.
Action 1: Calculate[
```python
pizza_pieces = 4
ann_pieces = 4 * 0.75
bill_pieces = 4 * 0.5
cate_pieces = 4 * 0.75
dale_pieces = 4 * 0.5
answer = ann_pieces + bill_pieces + cate_pieces + dale_pieces
```
]
Observation 1:
```python
pizza_pieces = 4
ann_pieces = 4 * 0.75
bill_pieces = 4 * 0.5
cate_pieces = 4 * 0.75
dale_pieces = 4 * 0.5
answer = ann_pieces + bill_pieces + cate_pieces + dale_pieces
```
Execution Status Done
Output: answer = 10
Thought 2: Then , I need to subtract that from the total number of pieces.
Action 2: Calculate[
```python 
pizza_pieces = 4
ann_pieces = 4 * 0.75
bill_pieces = 4 * 0.5
cate_pieces = 4 * 0.75
dale_pieces = 4 * 0.5
total_pieces_eaten = ann_pieces + bill_pieces + cate_pieces + dale_pieces
total_pieces = pizza_pieces * 4  # there are 4 people
pieces_left = total_pieces - total_pieces_eaten
answer = pieces_left
```
]
Observation 2:
```python 
pizza_pieces = 4
ann_pieces = 4 * 0.75
bill_pieces = 4 * 0.5
cate_pieces = 4 * 0.75
dale_pieces = 4 * 0.5
total_pieces_eaten = ann_pieces + bill_pieces + cate_pieces + dale_pieces
total_pieces = pizza_pieces * 4  # there are 4 people
pieces_left = total_pieces - total_pieces_eaten
answer = pieces_left
```
Execution Statuts: Done
Output: answer = 6
Thought 3: There are 6 pizza pieces uneaten.
Action 4: Finish[
```python
answer = 6
```
]
Observation 4: 
```python
answer = 6
```

---

Question: Tommy is making 12 loaves of bread. He needs 4 pounds of flour per loaf. A 10-pound bag of flour costs $10 and a 12-pound bag costs $13. When he is done making his bread, he has no use for flour and so he will throw away whatever is left. How much does he spend on flour if he buys the cheapest flour to get enough?
Thought 1: First , I need to calculate the total amount of flour Tommy needs
Action 1: Calculate[
```python
num_of_loaves = 12
pounds_of_flour_per_loaf = 4
answer = num_of_loaves * pounds_of_flour_per_loaf
```
]
Observation 1:
```python
num_of_loaves = 12
pounds_of_flour_per_loaf = 4
answer = num_of_loaves * pounds_of_flour_per_loaf
```
Execution Status: Done
Output: answer = 48
Thought 2: Then , i need to find the cheapest flour and the total cost.
Action 2: Calculate[
```python
# Python code, return answer
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
```
]
Observation 2:
```python
# Python code, return answer
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
```
Execution Status: Done
Output: answer = 50
Thought 3: He spends $50 on flour.
Action 4: Finish[
```python
answer = 50
```
]
Observation 4: 
```python
answer = 50
```

"""

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
Thought 1: I need to calculate the total number of toys Shawn has after receiving gifts from his parents.
Action 1: Calculate[
```python
toys_initial = 5
toys_received = 2 * 3
answer = toys_initial + toys_received
```
]
Observation 1: 
```python
toys_initial = 5
toys_received = 2 * 3
answer = toys_initial + toys_received
```
Execution Status: Done
Output: answer = 11
Thought 2: I incorrectly calculated the total toys received. Shawn got 2 toys each from his mom and dad, so the correct total received should be 2 + 2.
Action 2: Calculate[
```python
toys_initial = 5
toys_received = 2 + 2
answer = toys_initial + toys_received
```
]
Observation 2: 
```python
toys_initial = 5
toys_received = 2 + 2
answer = toys_initial + toys_received
```
Execution Status: Done
Output: answer = 9
Thought 3: Shawn now has 9 toys.
Action 3: Finish[
```python
answer = 9
```
]
Observation 3: 
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
total_chocolates = leah_chocolates + sister_chocolates
chocolates_eaten = 35
answer = total_chocolates - chocolates_eaten
```
]
Observation 1: 
```python
leah_chocolates = 32
sister_chocolates = 42
total_chocolates = leah_chocolates + sister_chocolates
chocolates_eaten = 35
answer = total_chocolates - chocolates_eaten
```
Execution Status: Done
Output: answer = 39
Thought 2: They have 39 chocolates left.
Action 2: Finish[
```python
answer = 39
```
]
Observation 2: 
```python
answer = 39
```"""
