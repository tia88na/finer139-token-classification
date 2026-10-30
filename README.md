# FiNER-139: Financial Numeric Entity Recognition (XBRL Tagging)

Token classification on SEC financial filings. ~95% of labeled tokens are numbers, so the correct XBRL tag depends on context, not the number itself.

## Dataset
- Source: nlpaueb/finer-139 (ACL 2022) — https://huggingface.co/datasets/nlpaueb/finer-139
- Original: 1.1M sentences (900,384 / 112,494 / 108,378)
- 139 entity types -> 279 BIO tags; 167 active in train
- data/raw/      = original, untouched
- data/cleaned/  = after cleaning (dedup + leakage removal + O-only reduction)
- data/processed/= cleaned data tokenized (bert-base-cased) + label-aligned, model-ready

## Cleaning (train / val / test)
Aggressive cleaning only on TRAIN; val/test kept close to original for honest evaluation.

| Split | Before | After | % kept |
|-------|--------|-------|--------|
| train | 900,384 | 261,610 | 29.1% |
| validation | 112,494 | 101,125 | 89.9% |
| test | 108,378 | 95,030 | 87.7% |

Train removal breakdown: 296,717 duplicates + 41,099 train/val-test leakage + 300,958 O-only sentences.
O-only (no-entity) train sentences: 721,189 -> 128,887 (80% -> 49%).

## Key facts
- Severe imbalance: 99.22% of tokens are O
- 94.7% of labeled tokens numeric
- 10.6% of words split into sub-pieces (number fragmentation)
- max_length = 128 (covers 97.4% of sentences)

## Evaluation
Entity-level F1 (seqeval). Accuracy is misleading due to the O majority.

## Team
See CONTRIBUTORS.md.
