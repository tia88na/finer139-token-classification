# Ece Mina Örenler

- Student ID: 220901539
- GitHub: [@eceorenler](https://github.com/eceorenler)
- Repository: [tia88na/finer139-token-classification](https://github.com/tia88na/finer139-token-classification)

## Summary

My contribution is the **comparative error analysis of the saved model predictions** and the documentation of the findings. I added a reproducible analysis pipeline for BiLSTM-CRF, BERT-base-cased and SEC-BERT-SHAPE, separating missed entities, spurious predictions, wrong entity types and boundary overlaps. I compared results by training entity frequency and sentence length, extracted real examples and label confusions, and checked that the analysis reproduces the team's shared evaluation results. No new model training was needed.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Reproducible analysis of three models' saved word-level predictions | `src/analyze_errors.py` |
| 2 | Separate false-negative and false-positive error partitions | `results/analysis/summary.csv`, `results/analysis/ANALYSIS.md` |
| 3 | Per-class, training-frequency and sentence-length comparisons | `results/analysis/per_class.csv`, `by_train_frequency.csv`, `by_length.csv` |
| 4 | Word-level type confusions and illustrative real examples | `results/analysis/token_confusions.csv`, `examples.json` |
| 5 | Exact gold-span recovery comparison between BERT and SEC-BERT-SHAPE | `results/analysis/bert_shape_disagreements.json` |
| 6 | Tests of strict IOB2 span extraction and error accounting | `tests/test_error_analysis.py` |
| 7 | Evaluation-protocol clarification, contributor index and notebook documentation fix | `README.md`, `CONTRIBUTORS.md`, `01_data_preprocessing.ipynb` |

## 1. Analysis protocol

The analysis uses the same **main test view** as the team's final table: 92,608 sentences, excluding exact token duplicates of validation. Gold and predicted labels are compared on original words with strict IOB2 entity spans. Prediction row counts, per-sentence word counts and tag IDs are validated before analysis. Aggregate precision, recall and micro-F1 must match the existing saved shared-evaluator results.

The pipeline reads the existing cleaned data and prediction files. It writes new outputs under `results/analysis/`, without changing the saved model predictions or original benchmark scores.

### Error categories

- **Missed entity:** a gold entity has no overlapping predicted entity.
- **Spurious prediction:** a predicted entity has no overlapping gold entity.
- **Wrong type:** the boundaries match exactly, but the entity type differs.
- **Boundary overlap:** the spans overlap but their boundaries differ; the type can also differ.

Exact-boundary wrong-type matches take precedence over boundary overlaps. Gold-side categories partition false negatives; prediction-side categories partition false positives. A wrong type can count on both sides, so these counts must not be added as unique error events. Token-level confusion counts remove BIO prefixes and are reported separately from entity-level errors.

## 2. Results and findings

Main test view, entity-level strict IOB2; scores below are percentages.

| Model | Precision | Recall | Micro-F1 | Macro-F1 |
|---|---:|---:|---:|---:|
| BiLSTM-CRF | 66.80 | 76.79 | 71.45 | 67.68 |
| BERT-base-cased | 63.05 | 79.80 | 70.44 | 66.01 |
| SEC-BERT-SHAPE | 66.23 | 81.19 | 72.95 | 69.15 |

### Precision and recall

Spurious predictions and wrong entity types are much more common than boundary overlaps. SEC-BERT-SHAPE has 1,857 completely missed gold entities, 8,222 spurious predictions, 3,364 gold-side wrong-type errors and 81 gold-side boundary-overlap errors. This describes the observed precision/recall gap; it does not establish its cause.

Compared with BERT, SEC-BERT-SHAPE produces **1,195 fewer spurious predictions** and **319 fewer exact-boundary wrong-type errors**. It correctly recovers 1,412 gold spans that BERT misses, while BERT correctly recovers 1,019 that SEC-BERT-SHAPE misses. These recovery counts are not net F1 changes because false positives also affect F1.

### Training frequency and sentence length

- SEC-BERT-SHAPE reaches **66.51 F1** for types occurring 100–999 times in cleaned training data, versus **74.59 F1** for types occurring at least 1,000 times. Counts refer to complete entity occurrences after merging B-/I- labels, not individual BIO-tag frequencies.
- Performance does not decrease uniformly as sentence length increases. Length is measured in original words; these groups do not directly identify tokenizer truncation.
- Common confusions include `ShareBasedCompensation` versus `AllocatedShareBasedCompensationExpense`, and current versus maximum borrowing capacity.

The report includes the first two main-test examples per model/error category. These deterministic examples illustrate behavior and are not a representative random sample.

## 3. Validation

I added three tests in `tests/test_error_analysis.py`:

- strict IOB2 behavior for orphan continuation tags and changes of entity type;
- false-negative and false-positive error partition accounting;
- span-based precision/recall/F1 agreement with strict seqeval across all 125 three-token predictions from a five-tag vocabulary.

The three new tests and eight existing evaluator/HMM tests passed: **11 tests in total**. The full GPU/model test suite was not run in this environment. The generated per-class, length-group and frequency-group entity totals were also checked against the overall totals, and each error partition was checked against its false-negative or false-positive total.

## 4. Documentation and scope

- Added README instructions for reproducing the error analysis and locating its outputs.
- Clarified why original transformer scores and the final strict-IOB2 main-view scores differ.
- Documented that SEC-BERT-SHAPE is an existing pretrained model, and that the BERT comparison changes both domain pretraining and number representation.
- Distinguished duplicate/leakage removal from O-only downsampling, whose effect on precision/recall has not been tested with a controlled experiment.
- Added `CONTRIBUTORS.md` linking the existing member contribution records.
- Removed the notebook statement that overwrote the maintained README during preprocessing.

The shared evaluator, trained models, original LaTeX results table and preprocessing pipeline were existing team contributions. My work builds on them. No new training, latency measurement, slides or speech notes are claimed here. Bootstrap intervals and sample assessment were subsequently added as described below. Presentation materials are maintained outside the repository. Trained model weights were not available for latency measurement.

## 5. How to reproduce

From the repository root, with the project dependencies installed:

```bash
python src/analyze_errors.py
python -m pytest tests/test_error_analysis.py tests/test_finer.py -q
```

## Files I added or changed

- `src/analyze_errors.py`
- `tests/test_error_analysis.py`
- `results/analysis/ANALYSIS.md`
- `results/analysis/summary.csv`, `per_class.csv`, `by_length.csv`, `by_train_frequency.csv`, `token_confusions.csv`
- `results/analysis/examples.json`, `bert_shape_disagreements.json`
- `README.md`, `CONTRIBUTORS.md`, `01_data_preprocessing.ipynb`
- this contribution file

## Additional analysis: bootstrap and sample assessment

- Added `src/assess_results.py` and `tests/test_assess_results.py`: 10,000 paired
  sentence-bootstrap replicates on the same main test view, with 95% percentile
  intervals and Bonferroni-adjusted intervals for three model comparisons.
- SEC-BERT-SHAPE minus BERT: +2.51 F1 points, 95% interval [2.16, 2.85].
  SEC-BERT-SHAPE minus BiLSTM-CRF: +1.51 [1.07, 1.94].
  BiLSTM-CRF minus BERT: +1.00 [0.57, 1.44]. These intervals are conditional on
  the saved predictions; they do not cover training-seed variability or annotation uncertainty.
- Prepared and assessed 50 randomly sampled SEC-BERT-SHAPE predictions with no
  overlap with a gold entity (seed 42, population 8,222 spans). Each example has a
  rationale and a follow-up verification step in `results/assessment/DEGERLENDIRME.md`.
- Checked exact sentence copies in raw train/validation/test for target-label
  variation. The single contextual assessment is provisional; no independent
  second review, expert re-annotation or corrected test score is claimed.
- Added `src/render_assessment.py` to regenerate the report from saved sample-specific
  notes. Method metadata, source hashes, per-sentence counts and bootstrap replicates
  are retained under `results/assessment/`.

Reproduce the numerical results and render the saved assessments:

```bash
python src/assess_results.py --replicates 10000 --sample-size 50 --seed 42
python src/render_assessment.py
python -m pytest tests/test_assess_results.py tests/test_error_analysis.py tests/test_finer.py -q
```

## Commit evidence

The implementation and results are recorded in the following local commits on `main` (not yet pushed):

- `380590c` — Add comparative error analysis for saved predictions.
- `295718b` — Add paired bootstrap confidence intervals and audit sampling.
- `cecfaa4` — Document assessment of 50 sampled entity predictions.
- `f0f292f` — Preserve maintained README when preprocessing data.

This contribution record and the README instructions are included in a subsequent documentation commit.
