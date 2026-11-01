# Zehra Özcan

- Student ID: 2309011051
- GitHub: [@zehraoz36](https://github.com/zehraoz36)
- Repository: [tia88na/finer139-token-classification](https://github.com/tia88na/finer139-token-classification)

## Summary

I built and ran the transformer part of the benchmark: a single training script for token classification on the cleaned FiNER-139 data, the **BERT-base (cased)** baseline, and the **new model, SEC-BERT-SHAPE**, a BERT model pre-trained on SEC filings that replaces every number with its shape (e.g. `645.1` -> `[XXX.X]`). SEC-BERT-SHAPE beats the BERT baseline by **+2.5 micro-F1** and **+3.1 macro-F1** on the test set and is better on 122 of the 139 entity types.

I then added the team's linear-chain CRF layer (`src/crf_layer.py`) on top of SEC-BERT-SHAPE. **SEC-BERT-SHAPE + CRF is the best single model of the project: 73.5 micro-F1 and 70.0 macro-F1** (shared evaluator, main test view), +0.6 / +0.9 over SEC-BERT-SHAPE. It matches the three-model ensemble on micro-F1 (73.52 vs 73.51) and beats it on macro-F1 (70.03 vs 69.77) while running one model instead of three.

Finally I trained SEC-BERT-SHAPE for 4 epochs instead of 2 (the BiLSTM-CRF was trained for up to 12). **SEC-BERT-SHAPE (4 epochs) is the best model of the project: 75.9 micro-F1 and 73.2 macro-F1** (shared evaluator, main test view), +3.0 / +4.0 over the 2-epoch run and +4.5 / +5.5 over the BiLSTM-CRF. I also trained the BERT baseline for 4 epochs (73.3 / 70.4), so the new model and the baseline are compared at equal training: SEC-BERT-SHAPE is +2.5 micro-F1 better at 2 epochs and +2.6 at 4 epochs.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Training and evaluation script for transformer token classification | `train_transformer22.py` |
| 2 | BERT-base-cased baseline run (Colab, Tesla T4) | `results/bert-base-cased_*` |
| 3 | SEC-BERT-SHAPE new model run, including number-shape preprocessing | `results/sec-bert-shape_*` |
| 4 | Benchmark table, per-entity reports, saved test predictions for error analysis | `results/transformer_benchmark.csv`, `*_report.txt`, `*_test_predictions.parquet` |
| 5 | Comparison and analysis of the two models | this file |
| 6 | CRF option in the training script: transformer emissions + the team's CRF layer, Viterbi decoding with IOB2 constraints | `train_transformer22.py --crf` |
| 7 | SEC-BERT-SHAPE + CRF run (best single model) | `results/sec-bert-shape-crf_*` |
| 8 | Added the new models to the shared results table | `src/evaluate_predictions.py`, `results/RESULTS.md`, `results/results_table.tex` |
| 9 | SEC-BERT-SHAPE trained for 4 epochs (best model) | `results/sec-bert-shape-4ep_*` |
| 10 | BERT-base-cased baseline trained for 4 epochs (equal-training comparison) | `results/bert-base-cased-4ep_*` |

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

## 2. SEC-BERT-SHAPE + CRF (`--crf`)

- **Emissions:** the transformer's 279 scores on each word's first sub-word piece are gathered into a word-level sequence (one score vector per word).
- **CRF layer:** the same `CRF` class as the feature CRF and the BiLSTM-CRF (`src/crf_layer.py`): start, end and tag-transition scores, IOB2 constraints (an `I-X` may only follow `B-X` / `I-X`).
- **Training:** loss = mean negative log-likelihood of the gold tag sequence (forward algorithm), computed in float32/64 outside mixed precision. The CRF scores get their own learning rate (5e-3, no weight decay) because they start at zero; the encoder keeps 3e-5.
- **Prediction:** Viterbi decoding of the best tag sequence per sentence.
- **Everything else unchanged:** same data, number shapes, truncation handling, 2 epochs, seed, epoch selection on validation.
- **Checks:** with gold-label emissions ("oracle"), the word gathering and Viterbi decoding reproduce the gold tags of all 3,000 sampled test sentences exactly; a `--limit 2000` run on Colab trained, decoded and saved end to end.
- **Run names:** every run is saved under its own name (`sec-bert-shape-crf`, `...-4ep`), so earlier results are never overwritten, and is also scored with the team's shared evaluator (`results/<run>_eval.json`, `<run>_per_class.csv`).

## 3. Results

Test set: 95,030 sentences, 28,500 gold entities, 139 entity types.

| Model | Role | Val micro-F1 | Test micro-F1 | Test macro-F1 | Precision | Recall | Time (T4) |
|---|---|---:|---:|---:|---:|---:|---:|
| bert-base-cased | baseline | 72.95 | 70.25 | 66.07 | 62.73 | 79.82 | 60.7 min |
| sec-bert-shape | new model | 75.07 | 72.80 | 69.18 | 65.96 | 81.22 | 60.1 min |
| sec-bert-shape + CRF | new model + CRF | 76.04 | 73.51 | 70.13 | 66.39 | 82.35 | 72.5 min |
| bert-base-cased, 4 epochs | baseline, longer training | 76.48 | 73.09 | 70.32 | 66.25 | 81.50 | 116.0 min |
| **sec-bert-shape, 4 epochs** | **new model, longer training** | **78.48** | **75.77** | **73.13** | **69.84** | **82.81** | 113.6 min |

The table above uses my script's own scoring (default seqeval, all 95,030 test sentences). The team's final table (`results/RESULTS.md`) uses the shared evaluator: strict IOB2 and the main view without the 2,422 test sentences that also occur in validation. On that protocol:

| Model | P | R | micro-F1 | macro-F1 |
|---|---:|---:|---:|---:|
| BERT-base-cased | 63.05 | 79.80 | 70.44 | 66.01 |
| SEC-BERT-SHAPE | 66.23 | 81.19 | 72.95 | 69.15 |
| Majority-vote ensemble (BiLSTM-CRF + BERT + SEC-BERT-SHAPE, by Yunus) | 67.10 | 81.29 | 73.51 | 69.77 |
| SEC-BERT-SHAPE + CRF | 66.43 | 82.32 | 73.52 | 70.03 |
| BERT-base-cased, 4 epochs | 66.69 | 81.47 | 73.34 | 70.35 |
| **SEC-BERT-SHAPE, 4 epochs** | **70.12** | **82.78** | **75.93** | **73.19** |

Where the gains come from (main-view micro-F1 / macro-F1):

| Effect | Comparison | Gain |
|---|---|---|
| Domain model (SEC-BERT-SHAPE vs BERT), 2 epochs | 72.95 vs 70.44 | +2.5 / +3.1 |
| Domain model (SEC-BERT-SHAPE vs BERT), 4 epochs | 75.93 vs 73.34 | +2.6 / +2.8 |
| Longer training (4 vs 2 epochs), BERT | 73.34 vs 70.44 | +2.9 / +4.3 |
| Longer training (4 vs 2 epochs), SEC-BERT-SHAPE | 75.93 vs 72.95 | +3.0 / +4.0 |
| CRF layer (SEC-BERT-SHAPE, 2 epochs) | 73.52 vs 72.95 | +0.6 / +0.9 |

Validation scores per epoch:

| Model | Epoch 1 micro / macro | Epoch 2 micro / macro |
|---|---|---|
| bert-base-cased | 70.41 / 61.88 | 72.95 / 67.17 |
| sec-bert-shape | 72.49 / 65.27 | 75.07 / 70.26 |
| sec-bert-shape + CRF | 73.00 / 66.21 | 76.04 / 70.97 |

SEC-BERT-SHAPE, 4 epochs (validation micro / macro): epoch 1 72.25 / 63.97, epoch 2 75.36 / 69.49, epoch 3 77.44 / 73.18, epoch 4 78.48 / 74.50.

BERT-base-cased, 4 epochs (validation micro / macro): epoch 1 69.66 / 60.33, epoch 2 71.66 / 65.91, epoch 3 75.04 / 69.98, epoch 4 76.48 / 71.81.

The 2-epoch models picked epoch 2 and the 4-epoch model picked epoch 4: validation scores were still rising at the last epoch in every run.

## 4. Analysis

- **SEC-BERT-SHAPE is better overall:** +2.5 micro-F1, +3.1 macro-F1. The larger macro gain means the improvement is not only on frequent labels.
- **Per entity type:** SEC-BERT-SHAPE is better on **122 of 139** types, worse on 15, equal on 2.
  - Largest gains: StockRepurchasedDuringPeriodShares (0.00 -> 0.28), RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty (0.47 -> 0.63), LineOfCreditFacilityCurrentBorrowingCapacity (0.26 -> 0.39).
  - Largest drop: DebtInstrumentMaturityDate (0.18 -> 0.11). A likely reason is that shape replacement turns the year into `[XXXX]`, so the model loses the actual value.
- **Recall is much higher than precision for both models** (about 80 vs 63-66). The models find most tagged numbers but often choose a wrong, similar tag (for example StockRepurchasedDuringPeriodShares vs StockRepurchasedAndRetiredDuringPeriodShares). One possible cause is the cleaning step: no-entity sentences are 49% of the cleaned train set but about 80% of the test set, so the models may predict entities too often. This is a hypothesis, not tested.
- **Hardest types for both models:** ShareBasedCompensation (F1 < 0.08), DebtInstrumentMaturityDate, OperatingLeasesRentExpenseNet, StockRepurchasedDuringPeriodShares.
- **What the CRF adds:** SEC-BERT-SHAPE + CRF is better than SEC-BERT-SHAPE on 87 of 139 types, worse on 49, equal on 3. The largest gain is DebtInstrumentMaturityDate (0.11 -> 0.26): maturity dates span several words (`December 31 , 2025`), and the CRF's tag transitions keep such multi-word entities together. Other large gains: BusinessCombination...AssumedIntangibles (0.46 -> 0.60), SaleOfStockNumberOfSharesIssuedInTransaction (0.45 -> 0.53).
- **Precision vs recall, tested:** my hypothesis that the O-only reduction makes the models over-predict entities was tested by Begüm's cleaning ablation with the feature CRF (`results/ablation/ABLATION.md`). Adding the dropped no-entity sentences back raises precision (57.5 -> 63.9) and lowers recall (60.6 -> 54.9), but micro-F1 does not change (59.0 vs 59.0). So the reduction shifts the precision/recall balance as expected, without hurting F1.
- **Significance:** Ece's paired bootstrap (`results/assessment/BOOTSTRAP.md`) shows SEC-BERT-SHAPE minus BERT = +2.51 (95% interval 2.16-2.85) and SEC-BERT-SHAPE minus BiLSTM-CRF = +1.51 (1.07-1.94), so both differences are not due to test-sentence sampling.
- **Longer training matters most:** 4 epochs instead of 2 gives +3.0 micro-F1 and +4.0 macro-F1, more than the CRF layer (+0.6 / +0.9). Precision rises the most (66.2 -> 70.1): the longer-trained model separates similar types better. It is better on 118 of 139 types, worse on 19; the largest gains are on types that were hardest before: ShareBasedCompensation (0.07 -> 0.32), StockRepurchasedDuringPeriodShares (0.28 -> 0.49), LineOfCreditFacilityCurrentBorrowingCapacity (0.39 -> 0.60), DebtInstrumentFairValue (0.49 -> 0.65).
- **Why BERT was below the BiLSTM-CRF:** with 2 epochs the transformers were under-trained compared with the BiLSTM-CRF (up to 12 epochs). With 4 epochs even the BERT baseline is above the BiLSTM-CRF (73.3 vs 71.5), and SEC-BERT-SHAPE is 4.5 points above it.
- **The two effects add up:** the domain-model gain is the same at 2 and 4 epochs (+2.5 / +2.6), and the longer-training gain is the same for both models (+2.9 / +3.0). So SEC-BERT-SHAPE's advantage does not come from training time.
- **The benchmark is not saturated:** the best model reaches 75.9 micro-F1 and 73.2 macro-F1.

## 5. Limitations

- One learning rate (3e-5) and at most 4 epochs, because each epoch takes about 27 minutes on a free Colab T4. Scores were still rising at epoch 4, so more epochs may help. SEC-BERT-SHAPE + CRF was not re-run with 4 epochs.
- One seed per model; the bootstrap intervals cover test-sentence sampling, not training randomness. The CRF gain over SEC-BERT-SHAPE (+0.6 micro-F1) has no interval yet.
- SEC-BERT-SHAPE is an existing pre-trained model (Loukas et al., ACL 2022); its gain over BERT mixes domain pre-training and number shapes.
- About 1.2% of test sentences are longer than 512 pieces; their tail words are counted as `O`.
- Epoch selection uses 20,000 of the 101,125 validation sentences to save time.

## Files I added

- `train_transformer22.py` - training and evaluation script for all three models (`--crf` added later)
- `results/transformer_benchmark.csv` - test scores of the three models (script scoring)
- `results/sec-bert-shape-crf_{report.txt, metrics.json, eval.json, per_class.csv, test_predictions.parquet}` - SEC-BERT-SHAPE + CRF results
- `results/sec-bert-shape-4ep_{report.txt, metrics.json, eval.json, per_class.csv, test_predictions.parquet}` - SEC-BERT-SHAPE, 4 epochs
- `results/bert-base-cased-4ep_{report.txt, metrics.json, eval.json, per_class.csv, test_predictions.parquet}` - BERT-base-cased, 4 epochs
- `src/evaluate_predictions.py` - added the display name of the new model; `results/RESULTS.md` and `results/results_table.tex` regenerated with it
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
    !python train_transformer22.py --model nlpaueb/sec-bert-shape --crf
    !python train_transformer22.py --model nlpaueb/sec-bert-shape --epochs 4
    !python train_transformer22.py --model bert-base-cased --epochs 4
    !python src/evaluate_predictions.py      # rebuilds the shared results table
   
 ## Commit evidence

Commits are in the repository ([tia88na/finer139-token-classification](https://github.com/tia88na/finer139-token-classification), branch `main`):

- [`4a2e833`](https://github.com/tia88na/finer139-token-classification/commit/4a2e833) (2026-10-31) - added `train_transformer22.py`, all files in `results/` and this contribution file (9 files)
- [`07b6b40`](https://github.com/tia88na/finer139-token-classification/commit/07b6b40) (2026-11-01) - `--crf` option, SEC-BERT-SHAPE + CRF results, results table row, updated contribution file (11 files)
- [`a24e5b4`](https://github.com/tia88na/finer139-token-classification/commit/a24e5b4) (2026-11-01) - SEC-BERT-SHAPE 4-epoch results, results table row, updated contribution file (10 files)
