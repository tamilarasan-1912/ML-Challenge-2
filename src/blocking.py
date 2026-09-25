from collections import defaultdict
from rapidfuzz.fuzz import ratio, token_set_ratio

def _index(df, key):
    out = defaultdict(list)
    for i, value in enumerate(df[key].tolist()):
        if value:
            out[(df["country"].iat[i], value)].append(i)
    return out

def _token_index(df, column, min_len=4):
    out = defaultdict(list)
    for i, values in enumerate(df[column].tolist()):
        for token in values:
            if len(token) >= min_len:
                out[(df["country"].iat[i], token)].append(i)
    return out

def build_indexes(target):
    return {
        "name": _index(target, "name_norm"),
        "name_prefix": _index(target, "name_prefix"),
        "address_prefix": _index(target, "address_prefix"),
        "name_token": _token_index(target, "name_tokens"),
        "address_token": _token_index(target, "address_tokens"),
    }

def candidates_for_row(row, target, indexes, max_candidates=50):
    country = row["country"]
    ids = set()

    def add(index, key):
        if key:
            ids.update(index.get((country, key), []))

    add(indexes["name"], row["name_norm"])
    add(indexes["name_prefix"], row["name_prefix"])
    add(indexes["address_prefix"], row["address_prefix"])

    for token in sorted(row["name_tokens"], key=len, reverse=True)[:3]:
        ids.update(indexes["name_token"].get((country, token), []))
    for token in sorted(row["address_tokens"], key=len, reverse=True)[:3]:
        ids.update(indexes["address_token"].get((country, token), []))

    if len(ids) <= max_candidates:
        return sorted(ids)

    scored = []
    for j in ids:
        t = target.iloc[j]
        ns = max(ratio(row["name_norm"], t["name_norm"]),
                 token_set_ratio(row["name_norm"], t["name_norm"]))
        ads = max(ratio(row["address_norm"], t["address_norm"]),
                  token_set_ratio(row["address_norm"], t["address_norm"]))
        exact_bonus = 1000 if row["name_norm"] and row["name_norm"] == t["name_norm"] else 0
        scored.append((exact_bonus + 0.7 * ns + 0.3 * ads, j))
    scored.sort(reverse=True)
    return sorted(j for _, j in scored[:max_candidates])

def generate_candidate_pairs(source1, target, max_candidates=50, indexes=None):
    if indexes is None:
        indexes = build_indexes(target)
    for i in range(len(source1)):
        yield i, candidates_for_row(source1.iloc[i], target, indexes, max_candidates)
