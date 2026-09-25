from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .data import load_split
from .normalize import prepare_frame
from .blocking import generate_candidate_pairs
from .features import build_matrix, FEATURE_NAMES
from .metrics import macro_f05

def parse_truth(gt):
    truth = {}
    for _, row in gt.iterrows():
        raw = row["matched_entity_ids"]
        truth[row["source1_entity_id"]] = {x for x in str(raw).split(",") if x}
    return truth

def collect_pairs(roots, targets, truth=None, max_candidates=50, root_limit=None):
    pair_meta = []
    y = []
    target_parts = []
    target_offset = 0
    roots = roots.iloc[:root_limit].reset_index(drop=True) if root_limit else roots.reset_index(drop=True)

    for target in targets:
        local_rows = []
        for li, candidates in generate_candidate_pairs(roots, target, max_candidates):
            for ci in candidates:
                local_rows.append((li, target_offset + ci))
                if truth is not None:
                    sid = roots.iloc[li]["entity_id"]
                    y.append(int(target.iloc[ci]["entity_id"] in truth.get(sid, set())))
        pair_meta.extend(local_rows)
        target_parts.append(target)
        target_offset += len(target)

    target_all = pd.concat(target_parts, ignore_index=True)
    return pair_meta, target_all, np.asarray(y, dtype=np.int8) if truth is not None else None

def build_features(pair_meta, roots, target_all):
    return build_matrix(pair_meta, roots, target_all)

def sample_training(pair_meta, target_all, y, max_pos, max_neg, seed):
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

    # Entity-level split prevents candidate leakage across train/validation.
    rng = np.random.default_rng(args.seed)
    all_ids = s1["entity_id"].to_numpy()
    n_val = max(1, int(len(all_ids) * args.val_fraction))
    val_ids = set(rng.choice(all_ids, n_val, replace=False))
    train_roots = s1[~s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    val_roots = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    # Large source data should not force all candidates into RAM.
    # Train on a deterministic root sample; increase this on high-RAM machines.
    train_roots = train_roots.iloc[:args.train_roots].reset_index(drop=True)
    val_roots = val_roots.iloc[:args.val_roots].reset_index(drop=True)

    train_truth = {k: v for k, v in truth.items() if k in set(train_roots["entity_id"])}
    pairs, target_all, y = collect_pairs(
        train_roots, [s2, s3], train_truth, args.max_candidates
    )
    keep = sample_training(pairs, target_all, y, args.max_pos, args.max_neg, args.seed)
    sampled_pairs = [pairs[i] for i in keep]
    X = build_features(sampled_pairs, train_roots, target_all)
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
    vpairs, vtargets, _ = collect_pairs(
        val_roots, [s2, s3], val_truth, args.max_candidates
    )
    VX = build_features(vpairs, val_roots, vtargets)
    probabilities = model.predict_proba(VX)[:, 1]

    best_threshold = 0.90
    best_score = -1.0
    for threshold in np.arange(0.50, 0.991, 0.01):
        pred = {}
        for (li, ri), p in zip(vpairs, probabilities):
            if p >= threshold:
                pred.setdefault(val_roots.iloc[li]["entity_id"], set()).add(
                    vtargets.iloc[ri]["entity_id"]
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
    model = joblib.load(Path(args.model_dir) / "matcher.joblib")
    with open(Path(args.model_dir) / "config.json", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])

    matches = {sid: [] for sid in s1["entity_id"]}
    candidates = {sid: [] for sid in s1["entity_id"]}

    for target in (s2, s3):
        for start in range(0, len(s1), args.batch_roots):
            roots = s1.iloc[start:start + args.batch_roots].reset_index(drop=True)
            local = list(generate_candidate_pairs(roots, target, args.max_candidates))
            pairs = [(li, ci) for li, cs in local for ci in cs]
            if not pairs:
                continue
            X = build_features(pairs, roots, target)
            probs = model.predict_proba(X)[:, 1]
            for (li, ci), p in zip(pairs, probs):
                sid = roots.iloc[li]["entity_id"]
                tid = target.iloc[ci]["entity_id"]
                candidates[sid].append(tid)
                if p >= threshold:
                    matches[sid].append(tid)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(args.output_dir) / "candidate_pairs.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1["entity_id"]:
            f.write(sid + "\t" + ",".join(sorted(set(candidates[sid]))) + "\n")

    with open(Path(args.output_dir) / "matching_results.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1["entity_id"]:
            f.write(sid + "\t" + ",".join(sorted(set(matches[sid]))) + "\n")

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
