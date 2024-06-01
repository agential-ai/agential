"""General utility functions."""

import builtins
import math
import random
import sys

from typing import Any, Dict, List, Optional, Tuple

import func_timeout


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


# Ref: https://github.com/microsoft/ProphetNet/blob/master/CRITIC/src/tools/interpreter_api.py.
def safe_execute(
    code_string: str,
    keys: Optional[List[str]] = None,
    default_exec_status: str = "Done",
) -> Tuple[List[Any], str]:
    """Executes the provided Python code string in a safe manner with a timeout and returns specified variables from the execution.

    Args:
        code_string (str): Python code to execute.
        keys (Optional[List[str]]): A list of variable names whose values are to be returned after execution. If None, the function tries to return a variable named 'answer'.
        default_exec_status (str): Default execution status string to output. Defaults to "Done".

    Returns:
        tuple: A tuple containing the result(s) of the specified variable(s) and a status message. If an exception occurs or timeout happens, it returns None for the result.
    """
    safe_globals: Dict[str, Any] = {"__builtins__": builtins, "sys": sys}

    def execute(x: str) -> Tuple[Optional[Any], str]:
        """Executes the code string with python exec()."""
        try:
            exec(x, safe_globals)
            if keys is None:
                an = [safe_globals.get("answer", None)]
            else:
                an = [safe_globals.get(k, None) for k in keys]
            return an, "Done"
        except BaseException as e:
            return [None], repr(e)

    try:
        an, report = func_timeout.func_timeout(3, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        an = [None]
        report = "TimeoutError: execution timeout"

    return an, report
