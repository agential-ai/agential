"""Few-shot examples for MBPP."""

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
```

---

You are an expert Python programmer, and here is your task: Write a python function to check whether the two numbers differ at one bit position only or not.
Your code should pass these tests.

assert differ_at_one_bit_pos(13, 9) == True
assert differ_at_one_bit_pos(15, 8) == False
assert differ_at_one_bit_pos(2, 4) == False
assert differ_at_one_bit_pos(2, 3) == True
assert differ_at_one_bit_pos(5, 1) == True
assert differ_at_one_bit_pos(1, 5) == True

```python
def differ_at_one_bit_pos(a, b):
    # XOR the two numbers; the result will have exactly one bit set if they differ by one bit position.
    xor_result = a ^ b
    # Check if xor_result has exactly one bit set by confirming it's a power of 2.
    return xor_result & (xor_result - 1) == 0 and xor_result != 0
```

---

You are an expert Python programmer, and here is your task: Write a function to find all words which are at least 4 characters long in a string.
Your code should pass these tests.

assert set(find_char_long('Please move back to stream')) == set(['Please', 'move', 'back', 'stream'])
assert set(find_char_long('Jing Eco and Tech')) == set(['Jing', 'Tech'])
assert set(find_char_long('Jhingai wulu road Zone 3')) == set(['Jhingai', 'wulu', 'road', 'Zone'])

```python
def find_char_long(s):
    # Split the string into words and filter words that are at least 4 characters long.
    words = s.split()
    long_words = [word for word in words if len(word) >= 4]
    return long_words

```"""


MBPP_FEWSHOT_EXAMPLES_REACT = """You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists.
Your code should pass these tests:

assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))

Thought 1: I need to write a function that accepts two tuples and finds the common elements between them. I will use a nested loop to check each element in the first tuple against each element in the second tuple.
Action 1: Implement[
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i, j in zip(test_tup1, test_tup2):
        if i == j:
            res.append(i)
    return tuple(res)
```
]
Observation 1: 
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i, j in zip(test_tup1, test_tup2):
        if i == j:
            res.append(i)
    return tuple(res)
```
Execution Status: Done
Thought 2: I need to test the function to ensure it works correctly with different test cases.
Action 2: Test[
```python
assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))
```
]
Observation 2: 
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i, j in zip(test_tup1, test_tup2):
        if i == j:
            res.append(i)
    return tuple(res)

assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))
```
Execution Status: AssertionError()
Thought 3: The function incorrectly uses `zip`, which only compares elements at the same positions in both tuples. I need to compare each element in the first tuple with all elements in the second tuple.
Action 3: Implement[
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i in test_tup1:
        if i in test_tup2:
            res.append(i)
    return tuple(res)
```
]
Observation 3: 
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i in test_tup1:
        if i in test_tup2:
            res.append(i)
    return tuple(res)
```
Thought 4: I need to test the updated function to ensure it works correctly with different test cases.
Action 4: Test[
```python
assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))
```
]
Observation 4: 
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i in test_tup1:
        if i in test_tup2:
            res.append(i)
    return tuple(res)

assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))
```
Execution Status: Done
Thought 5: The function now works correctly for the provided test cases.
Action 5: Finish[
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i in test_tup1:
        if i in test_tup2:
            res.append(i)
    return tuple(res)
```
]
Observation 5:
```python
def similar_elements(test_tup1, test_tup2):
    res = []
    for i in test_tup1:
        if i in test_tup2:
            res.append(i)
    return tuple(res)
```

---

You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers.
Your code should pass these tests:

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False

Thought 1: I need to write a function that can determine if a number is not a prime by checking its divisibility by any number up to its square root.
Action 1: Implement[
```python
import math

def is_not_prime(n):
    result = True
    for i in range(1, int(math.sqrt(n))):
        if n % i == 1:
            result = False
            break
    return result
```
]
Observation 1: 
```python
import math

def is_not_prime(n):
    result = True
    for i in range(1, int(math.sqrt(n))):
        if n % i == 1:
            result = False
            break
    return result
```
Execution Status: Done
Thought 2: I need to test the function to ensure it works correctly with different test cases.
Action 2: Test[
```python
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False
```
]
Observation 2: 
```python
import math

def is_not_prime(n):
    result = True
    for i in range(1, int(math.sqrt(n))):
        if n % i == 1:
            result = False
            break
    return result

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False
```
Execution Status: AssertionError()
Thought 3: The function is incorrect because it uses the wrong modulus condition (`n % i == 1` instead of `n % i == 0`) and starts the loop from 1 instead of 2. I need to correct these errors.
Action 3: Implement[
```python
import math

def is_not_prime(n):
    if n <= 1:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False
```
]
Observation 3: 
```python
import math

def is_not_prime(n):
    if n <= 1:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False
```
Thought 4: I need to test the updated function to ensure it works correctly with different test cases.
Action 4: Test[
```python
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False
```
]
Observation 4: 
```python
import math

def is_not_prime(n):
    if n <= 1:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False
```
Execution Status: Done
Thought 5: The function now works correctly for the provided test cases.
Action 5: Finish[
```python
import math

def is_not_prime(n):
    if n <= 1:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False
```
]
Observation 5:
```python
import math

def is_not_prime(n):
    if n <= 1:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False
```

---

You are an expert Python programmer, and here is your task: Write a function to find the n largest integers from a given list of numbers, returned in descending order.
Your code should pass these tests:

assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]

Thought 1: I need to write a function that finds the n largest integers in a list and returns them in descending order.
Action 1: Implement[
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```
]
Observation 1: 
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```
Execution Status: Done
Thought 2: I need to test the function to ensure it works correctly with different test cases.
Action 2: Test[
```python
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]
```
]
Observation 2: 
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums

assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]
```
Execution Status: Done
Thought 3: The function works correctly for the provided test cases.
Action 3: Finish[
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```
]
Observation 3:
```python
import heapq as hq

def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```"""
