# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains a data-only entity-resolution pipeline for the Amazon ML Challenge 2026.

Pipeline:
1. Normalize names and addresses.
2. Generate country-aware candidates with multiple blocking keys.
3. Build string/token similarity features.
4. Train an XGBoost binary matcher.
5. Tune the threshold for macro F0.5 on an entity-level validation split.
6. Generate matching_results.tsv and candidate_pairs.tsv.
7. Run the organizer validator before submission.

Data must be extracted outside Git:
dataset/train/{train_source1.tsv,train_source2.tsv,train_source3.tsv,train_ground_truth.tsv}
dataset/test/{test_source1.tsv,test_source2.tsv,test_source3.tsv}

Do not commit the 1+ GB challenge dataset.

Install:
pip install -r requirements.txt

Train:
python -m src.pipeline train --data-dir dataset --model-dir models

Predict:
python -m src.pipeline predict --data-dir dataset --model-dir models --output-dir output

The pipeline uses only supplied challenge data and no external entity-resolution service, registry, geocoder, or web lookup.
