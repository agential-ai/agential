"""General utility functions."""

import math
import random
import func_timeout
from typing import Any, List


def shuffle_chunk_list(lst: List[Any], k: int, seed: int = 42) -> List[List[Any]]:
    """Shuffles and divides the list into chunks, each with maximum length k.

    Args:
        lst (List[Any]): The list to be divided.
        k (int): The maximum length of each chunk.
        seed (int): The random seed. Defaults to 42.

    Returns:
        A list of chunks.

    Ref: https://github.com/LeapLabTHU/ExpeL.
    """
    random.seed(seed)

    lst = random.sample(lst, len(lst))

    if len(lst) <= k:
        return [lst]

    num_chunks = math.ceil(len(lst) / k)
    chunk_size = math.ceil(len(lst) / num_chunks)
    return [lst[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

# copy from https://github.com/microsoft/ProphetNet/blob/master/CRITIC/src/program/utils.py
def remove_comment(code):
    code = code.split("\n")
    code = [line for line in code if not line.startswith("#")]
    code = [line for line in code if line.strip() != ""]
    return "\n".join(code)




def safe_execute(code_string: str, keys=None):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            if keys is None:
                an = locals_.get('answer', None)
            else:
                an = [locals_.get(k, None) for k in keys]
            return an, "Done"
        except BaseException as e: # jump wrong case
            return None, repr(e)

    try:
        an, report = func_timeout.func_timeout(3, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        an = None
        report = "TimeoutError: execution timeout"

    return an, report