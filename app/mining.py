"""Exact Apriori frequent-itemset mining over possession baskets."""
from __future__ import annotations
from itertools import combinations

def mine_frequent_itemsets(baskets: list[list[str]], min_support: float = 0.08,
                           max_size: int = 4) -> list[dict]:
    if not baskets:
        raise ValueError("At least one basket is required")
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1]")
    transactions = [set(b) for b in baskets]
    threshold = max(1, int(min_support * len(transactions) + 0.999999))
    counts = {}
    items = sorted(set().union(*transactions))
    current = [(item,) for item in items]
    results = []
    size = 1
    while current and size <= max_size:
        frequent = []
        for candidate in current:
            count = sum(set(candidate).issubset(basket) for basket in transactions)
            if count >= threshold:
                frequent.append(candidate)
                counts[candidate] = count
                results.append({
                    "itemset": list(candidate),
                    "support_count": count,
                    "support": count / len(transactions)
                })
        size += 1
        candidates = set()
        for left, right in combinations(frequent, 2):
            merged = tuple(sorted(set(left) | set(right)))
            if len(merged) == size and all(tuple(c) in counts for c in combinations(merged, size - 1)):
                candidates.add(merged)
        current = sorted(candidates)
    return sorted(results, key=lambda x: (-x["support"], len(x["itemset"]), x["itemset"]))

def save_itemsets(itemsets: list[dict], path):
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(itemsets, indent=2), encoding="utf-8")
