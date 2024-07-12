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
