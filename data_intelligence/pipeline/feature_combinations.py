# pipeline/feature_combinations.py

from itertools import combinations
from config import MAX_COMB_SIZE

def generate_feature_combinations(features):
    all_combinations = []

    for r in range(2, MAX_COMB_SIZE + 1):
        all_combinations.extend(list(combinations(features, r)))

    print(f"[INFO] Feature combinations: {len(all_combinations)}")

    return all_combinations