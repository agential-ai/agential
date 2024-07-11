"""Self-Refine prompts and fewshot examples."""

# ======================================================================== GSM8K ======================================================================== #


SELF_REFINE_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Question: {question}
# Python code, return answer"""


GSM8K_CRITIQUE_FEWSHOT_EXAMPLES = """Question: Kelly is grocery shopping at a supermarket and is making sure she has enough in her budget for the items in her cart. Her 5 packs of bacon cost $10 in total and she has 6 packets of chicken which each cost twice as much as a pack of bacon. She also has 3 packs of strawberries, priced at $4 each, and 7 packs of apples, each priced at half the price of a pack of strawberries. If Kelly’s budget is $65 then how much money, in dollars, does she have left in her budget?
```python
budget = 65
bacon = 5
bacon_cost = 10
chicken = 6
chicken_cost = 2 * bacon_cost
strawberries = 3
strawberry_cost = 4
apples = 7
apple_cost = strawberry_cost / 2
total_cost = bacon_cost + chicken_cost + strawberry_cost + apple_cost
money_left = budget - total_cost
answer = money_left
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
budget = 65
# looks good

# Let's check the other parts
bacon = 5
bacon_cost = 10
# Warning: the variables should be named better as she got 5 packs of bacon, not 5 bacons.

# Let's check the other parts
chicken = 6
chicken_cost = 2 * bacon_cost
# wrong! 6 *packs* of chicken, and each pack costs twice as much as a pack of bacon. So we need to calculate the cost of one pack of bacon first (total bags of bacon / total cost) and use that.

# Let's check the other parts
strawberries = 3
strawberry_cost = 4
# wrong! There is a mistake here. She got 3 packs of strawberries, not 3 strawberries. And the cost is per pack! So total cost should be 3 * 4, not 4.

# Let's check the other parts
apples = 7
apple_cost = strawberry_cost / 2
# wrong! She got 7 packs of apples, not 7 apples. Also, the total apple cost should be (strawberry_cost / 2) * num_apple_packs

# Let's check the other parts
total_cost = bacon_cost + chicken_cost + strawberry_cost + apple_cost
# looks good

# Let's check the other parts
money_left = budget - total_cost
# looks good

---

Question: Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.
```python
plates = 6
plate_cost = 6000
cups = 12 * 20
cup_cost = plate_cost
answer = cup_cost
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
plates = 6
plate_cost = 6000
# looks good

# Let's check the other parts
cups = 12 * 20
cup_cost = plate_cost
# wrong! The cost of a cup is not the same as the cost of a plate. The cost of a cup is $1200 less than the total cost of half a dozen plates sold at $6000 each. So we need to calculate the cost of a cup first (total cost of half a dozen plates sold at $6000 each - $1200) and use that.

---

Question: Carly had 80 cards, 2/5 of the cards had the letter A on them, 1/2 of the remaining had the letter B, 5/8 of the rest had the letter C on them, and the others had the letter D. How many of the cards had the letter D on them?
```python
cards_initial = 80
cards_a = cards_initial * 2 / 5
cards_left = cards_initial - cards_a
cards_b = cards_left * 1 / 2
cards_left = cards_left - cards_b
cards_c = cards_left * 5 / 8
cards_d = cards_left * 3 / 8
answer = cards_d
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
cards_initial = 80
# looks good

# Let's check the other parts
cards_a = cards_initial * 2 / 5
cards_left = cards_initial - cards_a
# looks good

# Let's check the other parts
cards_b = cards_left * 1 / 2
cards_left = cards_left - cards_b
# looks good

# Let's check the other parts
cards_c = cards_left * 5 / 8
cards_d = cards_left * 3 / 8
# looks good

# Let's check the other parts
answer = cards_d
# looks good

There is no error in the code! It is correct!

---

Question: Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?
```python
chips_per_inch = 12
chips_per_bag = 72
bags = 2
height = 3
chips_needed = height * chips_per_inch
chips_available = bags * chips_per_bag
chips_left = chips_available - chips_needed
length = chips_left / chips_per_inch
answer = length
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
chips_per_inch = 12
# looks good

# Let's check the other parts
chips_per_bag = 72
bags = 2
# looks good

# Let's check the other parts
height = 3
# looks good

# Let's check the other parts
chips_needed = height * chips_per_inch
chips_available = bags * chips_per_bag
# wrong! 12 chips per *sqare inch*, not per inch. Need to calculate square inches he can build with chips_available. Then, divide that by the desired height to get the length. That will be chips_needed, not chips_available. The rest of the code is incorrect as a result."""


SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Question: {question}
```python
{answer}
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good."""


GSM8K_REFINE_FEWSHOT_EXAMPLES = """Question: Kelly is grocery shopping at a supermarket and is making sure she has enough in her budget for the items in her cart. Her 5 packs of bacon cost $10 in total and she has 6 packets of chicken which each cost twice as much as a pack of bacon. She also has 3 packs of strawberries, priced at $4 each, and 7 packs of apples, each priced at half the price of a pack of strawberries. If Kelly's budget is $65 then how much money, in dollars, does she have left in her budget?
```python
budget = 65
bacon = 5
bacon_cost = 10
chicken = 6
chicken_cost = 2 * bacon_cost
strawberries = 3
strawberry_cost = 4
apples = 7
apple_cost = strawberry_cost / 2
total_cost = bacon_cost + chicken_cost + strawberry_cost + apple_cost
money_left = budget - total_cost
answer = money_left
```
    
# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
budget = 65
# looks good

# Let's check the other parts
bacon = 5
bacon_cost = 10
# Warning: the variables should be named better as she got 5 packs of bacon, not 5 bacons.

# Let's check the other parts
chicken = 6
chicken_cost = 2 * bacon_cost
# wrong! 6 *packs* of chicken, and each pack costs twice as much as a pack of bacon. So we need to calculate the cost of one pack of bacon first (total bags of bacon / total cost) and use that.

# Let's check the other parts
strawberries = 3
strawberry_cost = 4
# wrong! There is a mistake here. She got 3 packs of strawberries, not 3 strawberries. And the cost is per pack! So total cost should be 3 * 4, not 4.

# Let's check the other parts
apples = 7
apple_cost = strawberry_cost / 2
# wrong! She got 7 packs of apples, not 7 apples. Also, the total apple cost should be (strawberry_cost / 2) * num_apple_packs

# Let's check the other parts
total_cost = bacon_cost + chicken_cost + strawberry_cost + apple_cost
# looks good

# Let's check the other parts
money_left = budget - total_cost
# looks good

Okay! Here is the rewrite:

```python
budget = 65
num_bacon_packs = 5
bacon_total_cost = 10
num_chicken_packs = 6
bacon_cost = bacon_total_cost / num_bacon_packs
chicken_cost = 2 * bacon_cost
num_strawberry_packs = 3
strawberry_cost = 4
num_apple_packs = 7
apple_cost = strawberry_cost / 2
total_cost = bacon_total_cost + (chicken_cost * num_chicken_packs) + (strawberry_cost * num_strawberry_packs) + (apple_cost * num_apple_packs)
money_left = budget - total_cost
answer = money_left
```

---

Question: Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.
```python
plates = 6
plate_cost = 6000
cups = 12 * 20
cup_cost = plate_cost
answer = cup_cost
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
plates = 6
plate_cost = 6000
# looks good

# Let's check the other parts
cups = 12 * 20
cup_cost = plate_cost
# wrong! The cost of a cup is not the same as the cost of a plate. The cost of a cup is $1200 less than the total cost of half a dozen plates sold at $6000 each. So we need to calculate the cost of a cup first (total cost of half a dozen plates sold at $6000 each - $1200) and use that.

Okay! Here is the rewrite:

```python
plates = 6
plate_cost = 6000
cups = 12 * 20
total_cup_cost = (plate_cost * plates) - 1200
cost_per_cup = total_cup_cost / cups
answer = cost_per_cup
```

---

Question: Carly had 80 cards, 2/5 of the cards had the letter A on them, 1/2 of the remaining had the letter B, 5/8 of the rest had the letter C on them, and the others had the letter D. How many of the cards had the letter D on them?
```python
cards_initial = 80
cards_a = cards_initial * 2 / 5
cards_left = cards_initial - cards_a
cards_b = cards_left * 1 / 2
cards_left = cards_left - cards_b
cards_c = cards_left * 5 / 8
cards_d = cards_left * 3 / 8
answer = cards_d
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
cards_initial = 80
# looks good

# Let's check the other parts
cards_a = cards_initial * 2 / 5
cards_left = cards_initial - cards_a
# looks good

# Let's check the other parts
cards_b = cards_left * 1 / 2
cards_left = cards_left - cards_b
# looks good

# Let's check the other parts
cards_c = cards_left * 5 / 8
cards_d = cards_left * 3 / 8
# looks good

# Let's check the other parts
answer = cards_d
# looks good

There is no error in the code! It is correct! 

Here is the rewrite (for the sake of completeness):

```python
cards_initial = 80
cards_a = cards_initial * 2 / 5
cards_left = cards_initial - cards_a
cards_b = cards_left * 1 / 2
cards_left = cards_left - cards_b
cards_c = cards_left * 5 / 8
cards_d = cards_left * 3 / 8
answer = cards_d
```

---

Question: Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?
```python
chips_per_inch = 12
chips_per_bag = 72
bags = 2
height = 3
chips_needed = height * chips_per_inch
chips_available = bags * chips_per_bag
chips_left = chips_available - chips_needed
length = chips_left / chips_per_inch
answer = length
```

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
chips_per_inch = 12
# looks good

# Let's check the other parts
chips_per_bag = 72
bags = 2
# looks good

# Let's check the other parts
height = 3
# looks good

# Let's check the other parts
chips_needed = height * chips_per_inch
chips_available = bags * chips_per_bag
# wrong! 12 chips per *sqare inch*, not per inch. Need to calculate square inches he can build with chips_available. Then, divide that by the desired height to get the length. That will be chips_needed, not chips_available. The rest of the code is incorrect as a result.

Okay! Here is the rewrite:

```python
chips_per_inch = 12
chips_per_bag = 72
bags = 2
height = 3
chips_available = bags * chips_per_bag
square_inches_available = chips_available / chips_per_inch
length = square_inches_available / height
answer = length
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