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
```"""