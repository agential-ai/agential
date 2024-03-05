"""General utility functions."""

from typing import List, Any
import math
import random

def shuffle_chunk_list(lst: List[Any], k: int) -> List[List[Any]]:
    """Shuffles and divides the list into chunks, each with maximum length k.

    Args:
        lst: The list to be divided.
        k: The maximum length of each chunk.

    Returns:
        A list of chunks.

    Ref: https://github.com/LeapLabTHU/ExpeL.
    """
    lst = random.sample(lst, len(lst))
    
    if len(lst) <= k:
        return [lst]
    
    num_chunks = math.ceil(len(lst) / k)
    chunk_size = math.ceil(len(lst) / num_chunks)
    return [lst[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]