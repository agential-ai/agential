"""Self-Refine prompts and fewshot examples."""


# ======================================================================== GSM8K ======================================================================== #


GSM8K_FEWSHOT_EXAMPLES = """# Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
# solution using Python:

def solution():
    \"\"\"Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\"\"\"
    jason_lollipops_initial = 20
    jason_lollipops_after = 12
    denny_lollipops = jason_lollipops_initial - jason_lollipops_after
    result = denny_lollipops
    return result



# Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
# solution using Python:

def solution():
    \"\"\"There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\"\"\"
    trees_initial = 15
    trees_after = 21
    trees_added = trees_after - trees_initial
    result = trees_added
    return result



# Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
# solution using Python:

def solution():
    \"\"\"Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\"\"\"
    toys_initial = 5
    mom_toys = 2
    dad_toys = 2
    total_received = mom_toys + dad_toys
    total_toys = toys_initial + total_received
    result = total_toys
    return result



# Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
# solution using Python:

def solution():
    \"\"\"There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\"\"\"
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result



# Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
# solution using Python:

def solution():
    \"\"\"Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\"\"\"
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result



# Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
# solution using Python:

def solution():
    \"\"\"If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\"\"\"
    cars_initial = 3
    cars_arrived = 2
    total_cars = cars_initial + cars_arrived
    result = total_cars
    return result



# Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
# solution using Python:

def solution():
    \"\"\"Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\"\"\"
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result



# Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
# solution using Python:

def solution():
    \"\"\"Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\"\"\"
    leah_chocolates = 32
    sister_chocolates = 42
    total_chocolates = leah_chocolates + sister_chocolates
    chocolates_eaten = 35
    chocolates_left = total_chocolates - chocolates_eaten
    result = chocolates_left
    return result"""


SELF_REFINE_INSTRUCTION_GSM8K = """{examples}

{question_prefix}{question}{intra_example_sep}{answer_prefix}"""


GSM8K_FEEDBACK_FEWSHOT_EXAMPLES = """def solution():
    \"\"\"Kelly is grocery shopping at a supermarket and is making sure she has enough in her budget for the items in her cart. Her 5 packs of bacon cost $10 in total and she has 6 packets of chicken which each cost twice as much as a pack of bacon. She also has 3 packs of strawberries, priced at $4 each, and 7 packs of apples, each priced at half the price of a pack of strawberries. If Kelly’s budget is $65 then how much money, in dollars, does she have left in her budget?\"\"\"
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
    result = money_left
    return result

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

def solution():
    \"\"\"Kelly is grocery shopping at a supermarket and is making sure she has enough in her budget for the items in her cart. Her 5 packs of bacon cost $10 in total and she has 6 packets of chicken which each cost twice as much as a pack of bacon. She also has 3 packs of strawberries, priced at $4 each, and 7 packs of apples, each priced at half the price of a pack of strawberries. If Kelly’s budget is $65 then how much money, in dollars, does she have left in her budget?\"\"\"
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
    result = money_left
    return result

### END ###

def solution():
    \"\"\"Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.\"\"\"
    plates = 6
    plate_cost = 6000
    cups = 12 * 20
    cup_cost = plate_cost
    result = cup_cost
    return result

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

def solution():
    \"\"\"Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.\"\"\"
    plates = 6
    plate_cost = 6000
    cups = 12 * 20
    total_cup_cost = (plate_cost * plates) - 1200
    cost_per_cup = total_cup_cost / cups
    return cost_per_cup

### END ###

def solution():
    \"\"\"Carly had 80 cards, 2/5 of the cards had the letter A on them, 1/2 of the remaining had the letter B, 5/8 of the rest had the letter C on them, and the others had the letter D. How many of the cards had the letter D on them?\"\"\"
    cards_initial = 80
    cards_a = cards_initial * 2 / 5
    cards_left = cards_initial - cards_a
    cards_b = cards_left * 1 / 2
    cards_left = cards_left - cards_b
    cards_c = cards_left * 5 / 8
    cards_d = cards_left * 3 / 8
    result = cards_d
    return result

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
    result = cards_d
    return result
# looks good

There is no error in the code! It is correct! Here is the rewrite (for the sake of completeness):

def solution():
    \"\"\"Carly had 80 cards, 2/5 of the cards had the letter A on them, 1/2 of the remaining had the letter B, 5/8 of the rest had the letter C on them, and the others had the letter D. How many of the cards had the letter D on them?\"\"\"
    cards_initial = 80
    cards_a = cards_initial * 2 / 5
    cards_left = cards_initial - cards_a
    cards_b = cards_left * 1 / 2
    cards_left = cards_left - cards_b
    cards_c = cards_left * 5 / 8
    cards_d = cards_left * 3 / 8
    result = cards_d
    return result

### END ###

def solution():
    \"\"\"Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?\"\"\"
    chips_per_inch = 12
    chips_per_bag = 72
    bags = 2
    height = 3
    chips_needed = height * chips_per_inch
    chips_available = bags * chips_per_bag
    chips_left = chips_available - chips_needed
    length = chips_left / chips_per_inch
    result = length
    return result

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

def solution():
    \"\"\"Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?\"\"\"
    chips_per_inch = 12
    chips_per_bag = 72
    bags = 2
    height = 3
    chips_available = bags * chips_per_bag
    square_inches_available = chips_available / chips_per_inch
    length = square_inches_available / height
    result = length
    return result

### END ###"""


GSM8K_FEEDBACK_INSTRUCTION = "# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good."


SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K = """{examples}

{question_prefix}{solution}{intra_example_sep}{feedback_instruction}{answer_prefix}"""
