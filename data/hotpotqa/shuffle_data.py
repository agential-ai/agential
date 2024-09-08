"""Shuffle and save HotpotQA dataset indices."""

import random
import json
import argparse
import pickle

parser = argparse.ArgumentParser(description="Shuffle HotpotQA dataset indices.")
parser.add_argument('--file_name', type=str, default='hotpot_dev_v1_simplified.json', help='HotpotQA dataset file json.')
parser.add_argument('--save_file_name', type=str, default='hotpot_dev_v1_simplified', help='Save file name.')
parser.add_argument('--sample_size', type=int, default=500, help='Sample size (default: 500).')
parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling the dataset indices (default: 42).')
args = parser.parse_args()

random.seed(args.seed)

if __name__ == '__main__':
    with open(args.file_name, 'r') as file:
        data = json.load(file)

    indices = list(range(len(data)))
    random.shuffle(indices)

    with open(f"{args.save_file_name}_s{args.seed}_indices.pkl", 'wb') as f:
        pickle.dump(indices, f)

    data = [data[i] for i in indices[:args.sample_size]]

    with open(f"{args.save_file_name}_s{args.seed}_sample{args.sample_size}.json", 'w') as json_file:
        json.dump(data, json_file, indent=4)
