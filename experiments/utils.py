"""Utility functions for experiments."""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seed for reproducibility across different libraries and frameworks.

    Args:
        seed (int): The seed value to use for reproducibility.
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility in PyTorch
    torch.backends.cudnn.benchmark = (
        False  # Disables auto-optimization for reproducibility
    )

    print(f"Seed set to: {seed}")
