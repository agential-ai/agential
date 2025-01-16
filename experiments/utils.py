"""Utility functions for experiments."""

import random
import numpy as np


def set_seed(seed: int) -> None:
    """Set seed for reproducibility across different libraries and frameworks.

    Args:
        seed (int): The seed value to use for reproducibility.
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    print(f"Seed set to: {seed}")
