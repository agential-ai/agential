"""Prompts for CoT."""

# ======================================================================== HOTPOTQA ======================================================================== #

COT_INSTRUCTION_HOTPOTQA = """{examples}
(END OF EXAMPLES)

Question: {question}
Thought: """

# ======================================================================== FEVER ======================================================================== #

COT_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. 

{examples}
(END OF EXAMPLES)

Claim: {question}
Thought: """

# ======================================================================== TRIVIAQA ======================================================================== #

COT_INSTRUCTION_TRIVIAQA = """{examples}
(END OF EXAMPLES)

Question: {question}
Thought: """

# ======================================================================== AMBIGNQ ======================================================================== #

COT_INSTRUCTION_AMBIGNQ = """{examples}
(END OF EXAMPLES)

Question: {question}
Thought: """

# ======================================================================== GSM8K ======================================================================== #

COT_INSTRUCTION_GSM8K = """{examples}
(END OF EXAMPLES)

Question: {question}
Thought: """

# ======================================================================== SVAMP ======================================================================== #

COT_INSTRUCTION_SVAMP = """{examples}
(END OF EXAMPLES)

Question: {question}
Thought: """

# ======================================================================== TABMWP ======================================================================== #

COT_INSTRUCTION_TABMWP = """{examples}
(END OF EXAMPLES)

{question}
Thought: """

# ======================================================================== HUMANEVAL ======================================================================== #

COT_INSTRUCTION_HUMANEVAL = """Generate a thought statement for the following question. Then provide the code solution in the action step in this format (provide all necessary imports): Finish[\\n```python<your-code>\\n```\\n].
{question}

Thought: """

# ======================================================================== MBPP ======================================================================== #

COT_INSTRUCTION_MBPP = """{examples}
(END OF EXAMPLES)

You are an expert Python programmer, and here is your task: {question}.
Your code should pass these tests:

{tests}

Thought: """


AMBIGNQ_FEWSHOT_EXAMPLES_COT = """Question: When did men's figure skating become a summer Olympic sport?
Thought: Let's think step by step. Figure skating was first contested in the Olympic Games at the 1908 Summer Olympics. So the answer is: 1908.
Action: Finish[1908]

Question: When was the last time the USA men's national soccer team missed the World Cup?
Thought: Let's think step by step. The USA men's national soccer team did not qualify for the World Cup in 2018. So the answer is: 2018.
Action: Finish[2018]

Question: What does barium do in a ct scan?
Thought: Let's think step by step. Barium is used to improve visualization of the gastrointestinal tract during a CT scan. So the answer is: to improve visualization of the gastrointestinal tract.
Action: Finish[to improve visualization of the gastrointestinal tract]

Question: Where was the fort located at which the first shot of the civil war was fired?
Thought: Let's think step by step. The first shot of the Civil War was fired at Fort Sumter, which is located in Charleston Harbor, South Carolina. So the answer is: Charleston Harbor, South Carolina.
Action: Finish[Charleston Harbor, South Carolina]

Question: When did nando's come to the uk?
Thought: Let's think step by step. Nando's opened its first restaurants in the United Kingdom in 1992. So the answer is: 1992.
Action: Finish[1992]"""


FEVER_FEWSHOT_EXAMPLES_COT = """Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought: Let's think step by step. He appeared in the 2009 Fox television film Virtuality. So the answer is: SUPPORTS.
Action: Finish[SUPPORTS]

Claim: Stranger Things is set in Bloomington, Indiana.
Thought: Let's think step by step. It is set in the fictional town of Hawkins, Indiana. So the answer is: REFUTES.
Action: Finish[REFUTES]

Claim: "Beautiful" by Christina Aguilera reached number two on the Billboard Hot 100 in 2003.
Thought: Let's think step by step. The song peaked at number two on the Billboard Hot 100, but it does not specify that it was in 2003. So the answer is: NOT ENOUGH INFO.
Action: Finish[NOT ENOUGH INFO]"""


HOTPOTQA_FEWSHOT_EXAMPLES_COT = """Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: Let's think step by step. The eastern sector of Colorado orogeny extends into the High Plains. High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: Finish[1,800 to 7,000 ft]

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: Let's think step by step. Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action: Finish[Richard Nixon]

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought: Let's think step by step. Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action: Finish[The Saimaa Gesture]

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: Let's think step by step. Professions of Nicholas Ray are director, screenwriter, and actor. Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action: Finish[director, screenwriter, actor]

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought: Let's think step by step. Arthur's Magazine was started in 1844. First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action: Finish[Arthur's Magazine]

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: Let's think step by step. Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.
Action: Finish[Yes]"""


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

HUMANEVAL_FEWSHOT_EXAMPLES_COT = """"""


MBPP_FEWSHOT_EXAMPLES_COT = """You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists.
Your code should pass these tests:

assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))

Thought: Let's think step by step. We need to find the common elements between the two lists and return them as a tuple.
Action: Finish[
```python
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return res
```
]

---

You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers.
Your code should pass these tests:

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False

Thought: Let's think step by step. We need to check if a number is not a prime by testing if it is divisible by any number from 2 to the square root of the number.
Action: Finish[
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
]

---

You are an expert Python programmer, and here is your task: Write a function to find the n largest integers from a given list of numbers, returned in descending order.
Your code should pass these tests:

assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]

Thought: Let's think step by step. We need to find the n largest numbers in the list using the heapq.nlargest function.
Action: Finish[
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```
]"""


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


TABMWP_FEWSHOT_EXAMPLES_COT = """Read the following table regarding "Coin collections" and then write Python code to answer a question:

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
Thought: Let's think step by step. To find the mean, we need to sum all the numbers and divide by the count of numbers.
Action: Finish[
```python
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
answer = sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person)
```
]

---

Read the following table regarding "Price and Quantity" and then write Python code to answer a question:

Price | Quantity demanded | Quantity supplied
$155 | 22,600 | 5,800
$275 | 20,500 | 9,400
$395 | 18,400 | 13,000
$515 | 16,300 | 16,600
$635 | 14,200 | 20,200

Question: At a price of $155, is there a shortage or a surplus? Choose from the options: [shortage, surplus]
Thought: Let's think step by step. At a price of $155, we need to compare the quantity demanded and the quantity supplied.
Action: Finish[
```python
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_price_155 > quantity_supplied_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'
```
]

---

Read the following table regarding "Cans of food collected" and then write Python code to answer a question:

Name | Number of cans
Samir | 7
Kristen | 4
Dakota | 7
Jamie | 8
Maggie | 9

Question: Samir's class recorded how many cans of food each student collected for their canned food drive. What is the median of the numbers?
Thought: Let's think step by step. To find the median, we need to sort the numbers and find the middle value.
Action: Finish[
```python
cans = [7, 4, 7, 8, 9]
cans = sorted(cans)
middle1 = (len(cans) - 1) // 2
middle2 = len(cans) // 2
answer = (cans[middle1] + cans[middle2]) / 2
```
]

---

Read the following table regarding "Toy prices" and then write Python code to answer a question:

Item | Price
toy boat | $5.54
toy guitar | $8.23
set of juggling balls | $5.01
trivia game | $8.18
jigsaw puzzle | $5.30
toy dinosaur | $3.00

Question: Lorenzo has $13.50. Does he have enough to buy a toy guitar and a set of juggling balls? Choose from the options: ['yes', 'no']
Thought: Let's think step by step. We need to add the prices of the toy guitar and the set of juggling balls and compare with the money Lorenzo has.
Action: Finish[
```python
guitar_price = 8.23
juggling_balls_price = 5.01
total_money = 13.5
if total_money >= guitar_price + juggling_balls_price:
    answer = "yes"
else:
    answer = "no"
```
]"""


TRIVIAQA_FEWSHOT_EXAMPLES_COT = """Question: Mendelssohn's 'Wedding March' was originally written as incidental music for which Shakespeare play in 1842? 
Thought: Let's think step by step. Mendelssohn's 'Wedding March' was originally written as incidental music for A Midsummer Night's Dream in 1842. So the answer is: A Midsummer Night's Dream.
Action: Finish[A Midsummer Night's Dream]

Question: "Christ in the House of his Parents" is one of the best known paintings of which artist? 
Thought: Let's think step by step. "Christ in the House of his Parents" is a painting by John Everett Millais. So the answer is: John Everett Millais.
Action: Finish[John Everett Millais]

Question: Who designed the National Theatre building on the South Bank in London? 
Thought: Let's think step by step. The National Theatre building on the South Bank in London was designed by Denys Lasdun. So the answer is: Denys Lasdun.
Action: Finish[Denys Lasdun]

Question: Also a two-time World Champion, which American skier won the gold medal in the Men's Combined at the 2010 Winter Olympics? 
Thought: Let's think step by step. The only American skier who won the gold medal in the Men's Combined at the 2010 Winter Olympics is Bode Miller. So the answer is: Bode Miller.
Action: Finish[Bode Miller]"""