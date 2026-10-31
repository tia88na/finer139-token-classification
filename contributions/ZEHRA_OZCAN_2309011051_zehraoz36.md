# Zehra Özcan

- Student ID: 2309011051
- GitHub: [@zehraoz36](https://github.com/zehraoz36)
- Repository: [tia88na/finer139-token-classification](https://github.com/tia88na/finer139-token-classification)

## Summary

I built and ran the transformer part of the benchmark: a single training script for token classification on the cleaned FiNER-139 data, the **BERT-base (cased)** baseline, and the **new model, SEC-BERT-SHAPE**, a BERT model pre-trained on SEC filings that replaces every number with its shape (e.g. `645.1` -> `[XXX.X]`). SEC-BERT-SHAPE beats the BERT baseline by **+2.5 micro-F1** and **+3.1 macro-F1** on the test set and is better on 122 of the 139 entity types.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Training and evaluation script for transformer token classification | `train_transformer22.py` |
| 2 | BERT-base-cased baseline run (Colab, Tesla T4) | `results/bert-base-cased_*` |
| 3 | SEC-BERT-SHAPE new model run, including number-shape preprocessing | `results/sec-bert-shape_*` |
| 4 | Benchmark table, per-entity reports, saved test predictions for error analysis | `results/transformer_benchmark.csv`, `*_report.txt`, `*_test_predictions.parquet` |
| 5 | Comparison and analysis of the two models | this file |

## 1. Training script (`train_transformer22.py`)

- **Input:** `data/cleaned/` (word-level `tokens` and `ner_tags`). I read the cleaned data instead of `data/processed/`, because the processed files are tokenized for `bert-base-cased` only; SEC-BERT has a different vocabulary. Label names (279 BIO tags) are read from the parquet metadata.
- **Label alignment:** each word's tag goes on its first sub-word piece; the other pieces and special tokens get `-100` and are ignored by the loss.
- **SEC-BERT-SHAPE preprocessing:** every number token (regex `(\d+[\d,.]*)|([,.]\d+)`, as in the model card) is replaced by its shape token (`[XX.X]`, `[XXXX]`, ...) if it is in the model vocabulary, otherwise by `[NUM]`. This removes the number-fragmentation problem found in the EDA (`645.1` -> 4 pieces in BERT, 1 token in SEC-BERT-SHAPE). It is switched on automatically when the model name contains `shape`.
- **Truncation handled at evaluation:** training uses up to 128 pieces; validation and test use up to 512, so long sentences are not cut. Words that are still beyond 512 pieces are predicted as `O` (1,181 test sentences for BERT, 1,142 for SEC-BERT-SHAPE, about 1.2% of the test set).
- **Predictions mapped back to words:** scores are computed on the original word-level tags, so every model in the team is evaluated on exactly the same words.
- **Metric:** entity-level micro and macro F1, precision and recall with `seqeval`. Token accuracy is not used, because about 99% of tokens are `O`.
- **Model selection:** best epoch chosen on validation micro-F1 (20,000 validation sentences, fixed seed); the test set (95,030 sentences) is evaluated once with the selected epoch.
- **Settings (same for both models):** AdamW, learning rate 3e-5, weight decay 0.01, linear schedule with 6% warm-up, batch size 32, 2 epochs, gradient clipping 1.0, mixed precision (fp16), seed 42.

Before the full runs I tested the script with `--limit 2000` (quick check that loading, training, evaluation and saving work end to end).

## 2. Results

Test set: 95,030 sentences, 28,500 gold entities, 139 entity types.

| Model | Role | Val micro-F1 | Test micro-F1 | Test macro-F1 | Precision | Recall | Time (T4) |
|---|---|---:|---:|---:|---:|---:|---:|
| bert-base-cased | baseline | 72.95 | 70.25 | 66.07 | 62.73 | 79.82 | 60.7 min |
| **sec-bert-shape** | **new model** | **75.07** | **72.80** | **69.18** | **65.96** | **81.22** | 60.1 min |

Validation scores per epoch:

| Model | Epoch 1 micro / macro | Epoch 2 micro / macro |
|---|---|---|
| bert-base-cased | 70.41 / 61.88 | 72.95 / 67.17 |
| sec-bert-shape | 72.49 / 65.27 | 75.07 / 70.26 |

Both models picked epoch 2. Scores were still rising, so more epochs may help.

## 3. Analysis

- **SEC-BERT-SHAPE is better overall:** +2.5 micro-F1, +3.1 macro-F1. The larger macro gain means the improvement is not only on frequent labels.
- **Per entity type:** SEC-BERT-SHAPE is better on **122 of 139** types, worse on 15, equal on 2.
  - Largest gains: StockRepurchasedDuringPeriodShares (0.00 -> 0.28), RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty (0.47 -> 0.63), LineOfCreditFacilityCurrentBorrowingCapacity (0.26 -> 0.39).
  - Largest drop: DebtInstrumentMaturityDate (0.18 -> 0.11). A likely reason is that shape replacement turns the year into `[XXXX]`, so the model loses the actual value.
- **Recall is much higher than precision for both models** (about 80 vs 63-66). The models find most tagged numbers but often choose a wrong, similar tag (for example StockRepurchasedDuringPeriodShares vs StockRepurchasedAndRetiredDuringPeriodShares). One possible cause is the cleaning step: no-entity sentences are 49% of the cleaned train set but about 80% of the test set, so the models may predict entities too often. This is a hypothesis, not tested.
- **Hardest types for both models:** ShareBasedCompensation (F1 < 0.08), DebtInstrumentMaturityDate, OperatingLeasesRentExpenseNet, StockRepurchasedDuringPeriodShares.
- **The benchmark is not saturated:** the best model reaches 72.8 micro-F1 and 69.2 macro-F1.

## 4. Limitations

- Only 2 epochs and one learning rate (3e-5) for both models, because each run takes about one hour on a free Colab T4.
- One seed per model; the difference between the models was not tested for significance.
- About 1.2% of test sentences are longer than 512 pieces; their tail words are counted as `O`.
- Epoch selection uses 20,000 of the 101,125 validation sentences to save time.

## Files I added

- `train_transformer22.py` - training and evaluation script for both models
- `results/transformer_benchmark.csv` - test scores of both models
- `results/bert-base-cased_report.txt`, `results/sec-bert-shape_report.txt` - per-entity precision, recall, F1
- `results/bert-base-cased_metrics.json`, `results/sec-bert-shape_metrics.json` - settings, versions, per-epoch scores
- `results/bert-base-cased_test_predictions.parquet`, `results/sec-bert-shape_test_predictions.parquet` - word-level test predictions (row i = row i of `data/cleaned/test.parquet`), for error analysis

## How to reproduce (Colab, T4 GPU)

    !git clone https://github.com/tia88na/finer139-token-classification
    %cd finer139-token-classification
    !pip install -q seqeval
    !python train_transformer22.py --model bert-base-cased --limit 2000   # quick check
    !python train_transformer22.py --model bert-base-cased
    !python train_transformer22.py --model nlpaueb/sec-bert-shape
