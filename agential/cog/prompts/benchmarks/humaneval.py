"""Few-shot examples for HumanEval."""

HUMANEVAL_FEWSHOT_EXAMPLES_POT = """```python
def has_duplicate_names(names_list: List[str]) -> bool:
    \"\"\"Check if there is any name that appears more than once in the list.
    >>> has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Alice'])
    True
    >>> has_duplicate_names(['Alice', 'Bob', 'Charlie', 'Dave']) 
    False
    \"\"\"
    return len(names_list) != len(set(names_list))
```"""
