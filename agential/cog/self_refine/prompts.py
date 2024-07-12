"""Self-Refine prompts and fewshot examples."""

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


SELF_REFINE_REFINE_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

{question}
```python
{answer}
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good. Provide the improved solution. If there is no error, write out the entire solution again.

{critique}

Okay! Here is the rewrite:"""
