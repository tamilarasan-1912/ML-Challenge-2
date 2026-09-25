# Reproducible final package

Copy the repository src/ implementation into this folder for the final submission ZIP.

Required runtime:
- dataset/train and dataset/test from the official challenge resource
- Python dependencies listed in requirements.txt

Run:
python -m src.pipeline train --data-dir dataset --model-dir models
python -m src.pipeline predict --data-dir dataset --model-dir models --output-dir output
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

The matcher is XGBoost, which is Apache-2.0 licensed and far below the 8B parameter limit. No external entity data is used.
