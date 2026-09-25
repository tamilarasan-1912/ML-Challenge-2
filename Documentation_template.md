# ML Challenge 2026: Business Entity Resolution Solution

Team Name: [Your Team Name]
Team Members: [List team members]
Submission Date: [Date]

## 1. Executive Summary
Two-stage entity resolution using multi-key blocking and an XGBoost pair classifier.

## 2. Methodology
### 2.1 Problem Analysis
Names and addresses contain punctuation changes, legal-suffix variants, abbreviations, typos, token reordering and missing components.

### 2.2 Solution Strategy
Approach Type: Blocking + Classifier.

## 3. Candidate Generation (Blocking)
Country-aware blocks:
- normalized exact name
- normalized name prefix
- normalized address prefix
- informative name tokens
- informative address tokens

A deterministic similarity score caps each root's final candidate set before ML scoring.

## 4. Matching Model
Name features: RapidFuzz ratio, token-set ratio, token-sort ratio, WRatio, exact equality, token Jaccard.
Address features: the same similarity families plus digit overlap.
Other features: country equality and normalized length ratios.

Model: XGBoost binary classifier (Apache-2.0).
Threshold: selected on an entity-level validation split by macro F0.5.

## 5. Results & Error Analysis
Validation macro F0.5: [fill after training]
Best threshold: [fill after training]
Candidate recall: [measure after training]
False positives: [fill after error analysis]
False negatives: [fill after error analysis]

## 6. Conclusion
The pipeline is reproducible from the supplied challenge data and produces the two required TSV files.

## Appendix
All runnable source code is included under code/business_entity_resolution/.
