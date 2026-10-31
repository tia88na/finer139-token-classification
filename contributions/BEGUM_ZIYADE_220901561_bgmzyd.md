# Begüm Ziyade

- Student ID: 220901561
- GitHub: [@bgmzyd](https://github.com/bgmzyd)
- Repository: [tia88na/finer139-token-classification](https://github.com/tia88na/finer139-token-classification)

## Summary

My contribution is the **HMM baseline and the shared evaluation** of the project. I wrote one evaluation module that every model in the team is scored with (word-level, full-length sentences, entity-level micro and macro F1 with seqeval), removed the validation–test leakage from the test view, added a most-frequent-tag baseline and a first-order **HMM with Viterbi decoding**, and analysed why a token-only model fails on FiNER-139. I re-scored the team's transformer predictions with the same code and built the final results table (Markdown and LaTeX). The HMM reaches only **3.5 micro-F1**, against **73.0** for SEC-BERT-SHAPE: the tag of a number is decided by its context, not by the number.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Shared evaluation module used by all models | `src/finer.py` |
| 2 | Validation–test leakage removed from the main test view (2,422 sentences) | `src/finer.py` (`load_splits`, `evaluate`) |
| 3 | Most-frequent-tag baseline and HMM + Viterbi | `src/baseline_hmm.py`, `results/baseline_mft_*`, `results/hmm_*` |
| 4 | Analysis: why the HMM fails | `results/hmm_analysis.txt` |
| 5 | Re-scoring of all saved predictions, final results table (Markdown + LaTeX) | `src/evaluate_predictions.py`, `results/RESULTS.md`, `results/results_table.tex` |
| 6 | Tests of the evaluator and the HMM against the lecture examples | `tests/test_finer.py` |
| 7 | README sections: Evaluation, Models, Results | `README.md` |

## 1. Shared evaluation (`src/finer.py`)

All models must be compared on exactly the same gold labels, so I put loading and scoring in one module:

- **Label names** (279 BIO tags) are read from the parquet metadata, the same source as `train_transformer22.py`.
- **Word level, full-length sentences.** Scores are computed on the original words. If a model could not label a word (for example, the sentence was cut at 128 or 512 sub-word pieces), the word counts as `O`: a missed entity stays a false negative and is never dropped from the gold set. With the 128-piece cut of `data/processed/`, **3.8% of the labelled test words** (1,108 of 29,568) would otherwise disappear from the evaluation.
- **Metric.** Entity-level precision, recall, micro-F1 and macro-F1 with seqeval in **strict IOB2** mode (an entity counts only if both boundaries and the type match).
- **Two test views.** I found that **2,422 cleaned test sentences also occur word for word in validation**, which is used to pick the epoch. Duplicates between selection data and test inflate scores (lecture W6, slide 19). The **main view** leaves them out (**92,608** sentences); the **full cleaned test** (95,030) is reported next to it for reference.

## 2. Models

Both models see the same normalised tokens: numbers -> digit shape (`645.1` -> `XXX.X`, `1,250` -> `X,XXX`), words -> lowercase.

- **Most-frequent tag.** Each token gets the tag it had most often in train; unseen tokens get `O`.
- **HMM + Viterbi.** First-order HMM, 168 states (the tags seen in train), vocabulary of 26,729 normalised tokens seen at least twice, add-k smoothing (k = 0.01). Rare words are mapped to three unknown classes (`<NUM>`, `<CAP>`, `<UNK>`), so the model learns how likely each tag is to emit an unseen number. Decoding uses Viterbi in log space. Trained on the full cleaned train set (261,610 sentences); training and decoding the whole test set take about 6 minutes on a CPU.

## 3. Results

Main view: 92,608 test sentences; full test: 95,030.

| Model | P | R | micro-F1 | macro-F1 | micro-F1 (full test) |
|---|---:|---:|---:|---:|---:|
| Most-frequent tag | 37.1 | 1.6 | 3.1 | 0.6 | 3.1 |
| **HMM + Viterbi** | 5.8 | 2.5 | **3.5** | 1.0 | 3.5 |
| BERT-base-cased (Zehra) | 63.0 | 79.8 | 70.4 | 66.0 | 70.4 |
| SEC-BERT-SHAPE (Zehra) | 66.2 | 81.2 | 73.0 | 69.2 | 72.9 |

The transformer rows are Zehra's saved test predictions, re-scored with the shared code. Her own script reports 70.25 / 72.80 on the full test; the small difference (0.1–0.2) comes from seqeval's default mode versus the strict IOB2 mode used here. Removing the validation duplicates changes the scores by at most 0.1 point, so the leakage is real but not what drives the results.

## 4. Analysis: why the HMM fails (`results/hmm_analysis.txt`)

- **A number's shape does not decide its tag.** In train, the shape `XX` occurs 306,226 times, is `O` 91.8% of the time, and belongs to **134 of the 139 entity types**. `X.X` is an entity 70% of the time but spread over 127 types.
- **Viterbi stays in `O`.** `P(O -> O) = 0.981`, and the only evidence for an entity is an ambiguous shape, so the best path almost never leaves `O`: the HMM predicts `O` for **91.8% of the gold entity words**. All ten most frequent errors are "entity -> O" (for example LineOfCreditFacilityMaximumBorrowingCapacity, 1,229 words).
- **What is missing is context.** The words that decide the tag ("interest rate of", "matures in", "notional amount of") are not visible to an HMM emission `b(tag, word)`. Models that read the whole sentence (the transformers) reach 70–73 micro-F1. This motivates the context features of the CRF and the BiLSTM-CRF.
- **Baseline precision is misleadingly high (37.1)** because it only tags the few tokens that are almost always the same entity; its recall is 1.6.

## 5. Tests (`tests/test_finer.py`)

`python -m pytest tests -q` (8 tests) checks:
- the label set (279 tags, 139 types with both `B-` and `I-`),
- the evaluator against the lecture examples: Ada / New York gives P = R = F1 = 0.50 (W6 slides 20–21), all-`O` prediction gives zero entity recall (slide 17),
- that a truncated prediction counts as a missed entity,
- the HMM's Viterbi against the "can fish" example: best path V N (W6 slides 26–29).

## 6. How to reproduce

    pip install -r requirements.txt
    python src/baseline_hmm.py            # baseline + HMM, ~7 min on CPU
    python src/evaluate_predictions.py    # scores all *_test_predictions.parquet, builds the tables
    python -m pytest tests -q

To add a model to the table, save its word-level predicted tag ids as `results/<model>_test_predictions.parquet` (one row per row of `data/cleaned/test.parquet`, column `predicted_tags`) and run `src/evaluate_predictions.py` again.

## Files I added or changed

- `src/finer.py`, `src/baseline_hmm.py`, `src/evaluate_predictions.py`
- `tests/test_finer.py`
- `results/baseline_mft_*`, `results/hmm_*`, `results/*_eval.json`, `results/RESULTS.md`, `results/results_table.tex`
- `README.md` (Evaluation, Models, Results, repository structure), `requirements.txt` (pyarrow, pytest), `.gitignore`

## Commit evidence

Commits are in the repository ([tia88na/finer139-token-classification](https://github.com/tia88na/finer139-token-classification), branch `main`):

- *(add the commit link after pushing)*
