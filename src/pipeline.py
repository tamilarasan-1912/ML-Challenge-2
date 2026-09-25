from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

from .data import load_split
from .normalize import prepare_frame
from .blocking import build_indexes, generate_candidate_pairs
from .features import pair_features, FEATURE_NAMES
from .metrics import macro_f05

def parse_truth(gt):
    truth = {}
    for _, row in gt.iterrows():
        truth[row["source1_entity_id"]] = {x for x in str(row["matched_entity_ids"]).split(",") if x}
    return truth

def collect_pairs(roots, targets, truth=None, max_candidates=50, indexes=None):
    pair_meta = []
    labels = []
    for source_idx, target in enumerate(targets):
        idx = indexes[source_idx] if indexes else build_indexes(target)
        for li, candidates in generate_candidate_pairs(roots, target, max_candidates, idx):
            sid = roots.iloc[li]["entity_id"]
            for ci in candidates:
                pair_meta.append((li, source_idx, ci))
                if truth is not None:
                    labels.append(int(target.iloc[ci]["entity_id"] in truth.get(sid, set())))
    return pair_meta, np.asarray(labels, dtype=np.int8) if truth is not None else None

def build_features(pair_meta, roots, targets):
    import numpy as np
    X = np.empty((len(pair_meta), len(FEATURE_NAMES)), dtype=np.float32)
    for k, (li, source_idx, ci) in enumerate(pair_meta):
        X[k] = pair_features(roots.iloc[li], targets[source_idx].iloc[ci])
    return X

def sample_training(pair_meta, y, max_pos, max_neg, seed):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) > max_pos:
        pos = rng.choice(pos, max_pos, replace=False)
    if len(neg) > max_neg:
        neg = rng.choice(neg, max_neg, replace=False)
    keep = np.concatenate([pos, neg])
    rng.shuffle(keep)
    return keep

def train(args):
    s1, s2, s3, gt = load_split(args.data_dir, "train")
    s1, s2, s3 = prepare_frame(s1), prepare_frame(s2), prepare_frame(s3)
    truth = parse_truth(gt)

    rng = np.random.default_rng(args.seed)
    ids = s1["entity_id"].to_numpy()
    n_val = max(1, int(len(ids) * args.val_fraction))
    val_ids = set(rng.choice(ids, n_val, replace=False))

    train_roots = s1[~s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    val_roots = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    train_roots = train_roots.iloc[:args.train_roots].reset_index(drop=True)
    val_roots = val_roots.iloc[:args.val_roots].reset_index(drop=True)

    train_truth = {k: v for k, v in truth.items() if k in set(train_roots["entity_id"])}
    target_tables = [s2, s3]
    target_indexes = [build_indexes(s2), build_indexes(s3)]

    pairs, y = collect_pairs(
        train_roots, target_tables, train_truth, args.max_candidates, target_indexes
    )
    keep = sample_training(pairs, y, args.max_pos, args.max_neg, args.seed)
    sampled_pairs = [pairs[i] for i in keep]
    X = build_features(sampled_pairs, train_roots, target_tables)
    ys = y[keep]

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=max(1, (os.cpu_count() or 4) - 1),
    )
    model.fit(X, ys)

    val_truth = {k: v for k, v in truth.items() if k in set(val_roots["entity_id"])}
    vpairs, _ = collect_pairs(
        val_roots, target_tables, val_truth, args.max_candidates, target_indexes
    )
    VX = build_features(vpairs, val_roots, target_tables)
    probabilities = model.predict_proba(VX)[:, 1]

    best_threshold = 0.90
    best_score = -1.0
    for threshold in np.arange(0.50, 0.991, 0.01):
        pred = {}
        for (li, source_idx, ci), p in zip(vpairs, probabilities):
            if p >= threshold:
                pred.setdefault(val_roots.iloc[li]["entity_id"], set()).add(
                    target_tables[source_idx].iloc[ci]["entity_id"]
                )
        score = macro_f05(pred, val_truth)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(args.model_dir) / "matcher.joblib")
    with open(Path(args.model_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "threshold": best_threshold,
            "validation_macro_f05": best_score,
            "features": FEATURE_NAMES,
        }, f, indent=2)

    print(json.dumps({
        "validation_macro_f05": best_score,
        "threshold": best_threshold,
        "training_roots": len(train_roots),
        "validation_roots": len(val_roots),
        "training_pairs_sampled": len(ys),
        "positive_pairs_sampled": int(ys.sum()),
    }, indent=2))

def predict(args):
    s1, s2, s3, _ = load_split(args.data_dir, "test")
    s1, s2, s3 = prepare_frame(s1), prepare_frame(s2), prepare_frame(s3)
    targets = [s2, s3]
    indexes = [build_indexes(s2), build_indexes(s3)]
    model = joblib.load(Path(args.model_dir) / "matcher.joblib")
    with open(Path(args.model_dir) / "config.json", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    candidate_path = Path(args.output_dir) / "candidate_pairs.tsv"
    match_path = Path(args.output_dir) / "matching_results.tsv"

    with open(candidate_path, "w", encoding="utf-8") as cf, open(match_path, "w", encoding="utf-8") as mf:
        cf.write("source1_entity_id\tcandidate_entity_ids\n")
        mf.write("source1_entity_id\tmatched_entity_ids\n")

        for start in range(0, len(s1), args.batch_roots):
            roots = s1.iloc[start:start + args.batch_roots].reset_index(drop=True)
            candidate_rows = {sid: [] for sid in roots["entity_id"]}

            scored_matches = {sid: [] for sid in roots["entity_id"]}

            for source_idx, target in enumerate(targets):
                local = list(generate_candidate_pairs(
                    roots, target, args.max_candidates, indexes[source_idx]
                ))
                pairs = [(li, source_idx, ci) for li, cs in local for ci in cs]
                if not pairs:
                    continue

                X = build_features(pairs, roots, targets)
                probs = model.predict_proba(X)[:, 1]

                for (li, _, ci), p in zip(pairs, probs):
                    sid = roots.iloc[li]["entity_id"]
                    tid = target.iloc[ci]["entity_id"]
                    candidate_rows[sid].append(tid)
                    if p >= threshold:
                        scored_matches[sid].append(tid)

            for sid in roots["entity_id"]:
                cands = sorted(set(candidate_rows[sid]))
                matches = sorted(set(scored_matches[sid]))
                cf.write(sid + "\t" + ",".join(cands) + "\n")
                mf.write(sid + "\t" + ",".join(matches) + "\n")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data-dir", default="dataset")
    t.add_argument("--model-dir", default="models")
    t.add_argument("--max-candidates", type=int, default=50)
    t.add_argument("--train-roots", type=int, default=100000)
    t.add_argument("--val-roots", type=int, default=25000)
    t.add_argument("--max-pos", type=int, default=250000)
    t.add_argument("--max-neg", type=int, default=500000)
    t.add_argument("--val-fraction", type=float, default=0.10)
    t.add_argument("--n-estimators", type=int, default=700)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=train)

    q = sub.add_parser("predict")
    q.add_argument("--data-dir", default="dataset")
    q.add_argument("--model-dir", default="models")
    q.add_argument("--output-dir", default="output")
    q.add_argument("--max-candidates", type=int, default=50)
    q.add_argument("--batch-roots", type=int, default=5000)
    q.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
