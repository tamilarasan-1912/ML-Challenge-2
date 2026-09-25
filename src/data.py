from pathlib import Path
import pandas as pd

REQUIRED = ["entity_id", "business_name", "business_address", "country"]

def read_source(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[REQUIRED].copy()

def read_ground_truth(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

def load_split(data_dir, split):
    base = Path(data_dir) / split
    prefix = "train" if split == "train" else "test"
    s1 = read_source(base / f"{prefix}_source1.tsv")
    s2 = read_source(base / f"{prefix}_source2.tsv")
    s3 = read_source(base / f"{prefix}_source3.tsv")
    gt = read_ground_truth(base / "train_ground_truth.tsv") if split == "train" else None
    return s1, s2, s3, gt
