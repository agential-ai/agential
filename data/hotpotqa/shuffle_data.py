"""Shuffle and save HotpotQA dataset indices."""

import random
import json
import argparse
import pickle

parser = argparse.ArgumentParser(description="Shuffle HotpotQA dataset indices.")
parser.add_argument('--file_name', type=str, default='hotpot_dev_v1_simplified.json', help='HotpotQA dataset file json.')
parser.add_argument('--save_file_name', type=str, default='hotpot_dev_v1_simplified_s42_indices.pkl', help='Save file name.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling the dataset indices (default: 42).')
args = parser.parse_args()

random.seed(args.seed)

if __name__ == '__main__':
    with open(args.file_name, 'r') as file:
        data = json.load(file)

    numbers = list(range(len(data)))
    random.shuffle(numbers)

    with open(args.save_file_name, 'wb') as f:
        pickle.dump(numbers, f)