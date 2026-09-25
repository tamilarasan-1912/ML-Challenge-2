import re
import numpy as np
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio, WRatio

FEATURE_NAMES = [
    "name_ratio", "name_token_set", "name_token_sort", "name_wratio",
    "address_ratio", "address_token_set", "address_token_sort", "address_wratio",
    "name_exact", "address_exact", "country_match",
    "name_length_ratio", "address_length_ratio",
    "name_shared_tokens", "address_shared_tokens",
    "name_jaccard", "address_jaccard",
    "name_digit_overlap", "address_digit_overlap"
]

def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _digit_set(s):
    return set(re.findall(r"\d+", s or ""))

def _length_ratio(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))

def pair_features(a, b):
    n1, n2 = a["name_norm"], b["name_norm"]
    a1, a2 = a["address_norm"], b["address_norm"]
    nt1, nt2 = a["name_tokens"], b["name_tokens"]
    at1, at2 = a["address_tokens"], b["address_tokens"]

    return [
        ratio(n1, n2) / 100,
        token_set_ratio(n1, n2) / 100,
        token_sort_ratio(n1, n2) / 100,
        WRatio(n1, n2) / 100,
        ratio(a1, a2) / 100,
        token_set_ratio(a1, a2) / 100,
        token_sort_ratio(a1, a2) / 100,
        WRatio(a1, a2) / 100,
        float(bool(n1) and n1 == n2),
        float(bool(a1) and a1 == a2),
        float(a["country"] == b["country"]),
        _length_ratio(n1, n2),
        _length_ratio(a1, a2),
        float(len(nt1 & nt2)),
        float(len(at1 & at2)),
        _jaccard(nt1, nt2),
        _jaccard(at1, at2),
        float(len(_digit_set(n1) & _digit_set(n2))),
        float(len(_digit_set(a1) & _digit_set(a2))),
    ]

def build_matrix(pairs, left, right):
    X = np.empty((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)
    for k, (li, ri) in enumerate(pairs):
        X[k] = pair_features(left.iloc[li], right.iloc[ri])
    return X
