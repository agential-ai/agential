from typing import Dict


def validate_overlapping_keys(dict_1: Dict[str, str], dict_2: Dict[str, str]) -> bool:
    overlapping_keys = dict_1.keys() & dict_2.keys()
    if overlapping_keys:
        raise ValueError(f"Overlapping keys detected: {overlapping_keys}")
