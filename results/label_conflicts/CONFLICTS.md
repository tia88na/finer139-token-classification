# Conflicting labels

Identical sentences (same tokens) that FiNER-139 labels in more than one way. Deduplication in `01_data_preprocessing.ipynb` keeps the first copy, so only one version survives. Produced by `python src/label_conflicts.py`; nothing is retrained and `data/cleaned/` is unchanged.

| Split | sentences in data/cleaned | conflicting sentences kept | share |
|---|---:|---:|---:|
| train | 261,610 | 725 | 0.28% |
| validation | 101,125 | 8 | 0.01% |
| test | 95,030 | 29 | 0.03% |

## What kind of disagreement

| Kind | sentences |
|---|---:|
| entity vs O | 624 |
| entity type | 127 |
| entity vs O and type | 11 |

- Words that differ per conflicting sentence: median 1, max 9.
- The kept (first) copy is the most frequent version, or tied for most frequent, in 654 of 762 sentences, and a less frequent version in 108. In 477 sentences the top versions are tied, so no majority vote could decide the label.
- `entity vs O`: one annotation tags a number, another leaves it untagged. `entity type`: both tag it, with different XBRL types.

Entity types most often involved:

- DebtInstrumentInterestRateStatedPercentage (76)
- LineOfCreditFacilityMaximumBorrowingCapacity (67)
- DerivativeNotionalAmount (65)
- DebtInstrumentFaceAmount (50)
- DebtInstrumentBasisSpreadOnVariableRate1 (41)
- NumberOfReportableSegments (33)
- OperatingLeaseRightOfUseAsset (33)
- DerivativeFixedInterestRate (32)

## Effect on the test results

Main view (92,608 sentences) vs main view without the 29 conflicting test sentences (92,579).

| Model | micro-F1 | without conflicts | macro-F1 | without conflicts |
|---|---:|---:|---:|---:|
| Most-frequent tag | 3.07 | 3.08 | 0.56 | 0.56 |
| BERT-base-cased | 70.44 | 70.48 | 66.01 | 66.04 |
| BiLSTM-CRF | 71.45 | 71.48 | 67.68 | 67.70 |
| CRF | 59.03 | 59.04 | 52.86 | 52.86 |
| HMM + Viterbi | 3.51 | 3.52 | 0.98 | 0.98 |
| SEC-BERT-SHAPE + CRF | 73.52 | 73.55 | 70.03 | 70.05 |
| SEC-BERT-SHAPE | 72.95 | 72.98 | 69.15 | 69.17 |

## Interpretation and limits

- These sentences are label noise in the source data: the same text received different annotations, so at least one version is wrong or the sentence is genuinely ambiguous without the surrounding filing.
- They are a very small share of each split, so retraining without them was not done; the table above shows whether they move the test scores.
- Only exact duplicates are detected. Near-duplicates (one word different) with conflicting labels are not counted.
