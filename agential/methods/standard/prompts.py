"""Prompts and few-shot examples for standard prompting."""

# ======================================================================== HOTPOTQA ======================================================================== #

STANDARD_INSTRUCTION_HOTPOTQA = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """

# ======================================================================== FEVER ======================================================================== #

STANDARD_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. 

{examples}
(END OF EXAMPLES)

Claim: {question}
A: """

# ======================================================================== TRIVIAQA ======================================================================== #

STANDARD_INSTRUCTION_TRIVIAQA = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """

# ======================================================================== AMBIGNQ ======================================================================== #

STANDARD_INSTRUCTION_AMBIGNQ = """{examples}
(END OF EXAMPLES)

Q: {question}
A: """

# ======================================================================== GSM8K ======================================================================== #

STANDARD_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Question: {question}
# Python code, return answer"""

# ======================================================================== SVAMP ======================================================================== #

STANDARD_INSTRUCTION_SVAMP = """{examples}
(END OF EXAMPLES)

Question: {question}
# Python code, return answer"""

# ======================================================================== TABMWP ======================================================================== #

STANDARD_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

Question: {question}
# Python code, return answer"""

# ======================================================================== HUMANEVAL ======================================================================== #

STANDARD_INSTRUCTION_HUMANEVAL = """Implement the function below. Provide the entire function implementation and all necessary imports.

{question}

```python"""

# ======================================================================== MBPP ======================================================================== #

STANDARD_INSTRUCTION_MBPP = """{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

```python"""


HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT = """Q: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
A: 1,800 to 7,000 ft

---

Q: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?
A: Richard Nixon

---

Q: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
A: The Saimaa Gesture

---

Q: What profession does Nicholas Ray and Elia Kazan have in common?
A: director, screenwriter, actor

---

Q: Which magazine was started first Arthur's Magazine or First for Women?
A: Arthur's Magazine

---

Q: Were Pavel Urysohn and Leonid Levin known for the same type of work?
A: Yes"""


FEVER_FEWSHOT_EXAMPLES_DIRECT = """Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
A: SUPPORTS

---

Claim: Stranger Things is set in Bloomington, Indiana.
A: REFUTES

---

Claim: "Beautiful" by Christina Aguilera reached number two on the Billboard Hot 100 in 2003.
A: NOT ENOUGH INFO"""


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

Question: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creams, and he has 2 ice creams left now.  How much ice creams did Getty purchase?
# Python code, return answer
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
answer = total_ice_creams - num_ice_creams_bought_by_joseph"""


TABMWP_FEWSHOT_EXAMPLES_POT = """Read the following table regarding "Coin collections" and then write Python code to answer a question:

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
# Python Code, return answer
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
answer = sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person)

---

Read the following table regarding "" and then write Python code to answer a question:

Price | Quantity demanded | Quantity supplied
$155 | 22,600 | 5,800
$275 | 20,500 | 9,400
$395 | 18,400 | 13,000
$515 | 16,300 | 16,600
$635 | 14,200 | 20,200

Question: Look at the table. Then answer the question. At a price of $155, is there a shortage or a surplus? Choose from the the options: [shortage, surplus]
# Python Code, return answer
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_price_155 > quantity_supplied_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'

---

Read the following table regarding "Cans of food collected" and then write Python code to answer a question:

Samir | 7
Kristen | 4
Dakota | 7
Jamie | 8
Maggie | 9

Question: Samir's class recorded how many cans of food each student collected for their canned food drive. What is the median of the numbers?
# Python Code, return answer
cans = [7, 4, 5, 8, 9]
cans = sorted(cans)
middle1 = (len(cans) - 1) // 2
middle2 = len(cans) // 2
answer = (cans[middle1] + cans[middle2]) / 2

---

Read the following table regarding "" and then write Python code to answer a question:

toy boat | $5.54
toy guitar | $8.23
set of juggling balls | $5.01
trivia game | $8.18
jigsaw puzzle | $5.30
toy dinosaur | $3.00

Question: Lorenzo has $13.50. Does he have enough to buy a toy guitar and a set of juggling balls? Choose from the the options: ['yes', 'no']
# Python Code, return answer
guitar_price = 8.23
juggling_balls = 5.01
total_money = 13.5
if total_money > juggling_balls + guitar_price:
    answer = "yes"
else:
    answer = "no\""""


HUMANEVAL_FEWSHOT_EXAMPLES_POT = """"""


MBPP_FEWSHOT_EXAMPLES_POT = """You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists.
Your code should pass these tests:

assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))

```python
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return res
```

---

You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers.
Your code should pass these tests:

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False

```python
import math

def is_not_prime(n):
    result = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
            break
    return result
```

---

You are an expert Python programmer, and here is your task: Write a function to find the n largest integers from a given list of numbers, returned in descending order.
Your code should pass these tests:

assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]

```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```"""