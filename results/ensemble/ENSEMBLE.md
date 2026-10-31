# Majority-vote ensemble (exploratory)

**Exploratory analysis.** Only saved test predictions exist, so the voting rule could not be chosen or checked on validation data. It was fixed in advance and no alternative rule was tried. The rule and its tests were committed (`7d8dc33`) before this run, and the run used that committed version unchanged. The result is one descriptive comparison on a test set whose results had already been inspected; it is not unbiased confirmation that ensembling helps.

## Method (fixed in advance)

1. **Voters**: the saved word-level test predictions of BiLSTM-CRF, BERT-base-cased and SEC-BERT-SHAPE. No model was retrained or re-run.
2. **Orphan tags**: each voter's tags are rebuilt from its strict IOB2 spans, so an I- tag that strict seqeval ignores (one that does not follow B-X or I-X of the same type) becomes O. Every voter then votes with exactly what it is scored on; its strict scores do not change (checked below).
3. **Vote**: each word gets the full BIO tag, O included, that at least two of the three voters chose. If all three differ, SEC-BERT-SHAPE's tag is used.
4. **Repair**: an I-X that does not follow B-X or I-X becomes B-X, keeping the type the vote chose.
5. **Validity**: every output sentence is checked to be valid IOB2 (95,030 of 95,030 sentences pass).
6. **Scoring**: the shared evaluator `finer.evaluate` (strict IOB2, word level, full-length sentences; main view and full test).
7. **Primary comparison**: main-view micro-F1, ensemble minus SEC-BERT-SHAPE, with a 95% percentile interval from 10,000 paired sentence-bootstrap replicates (seed 42; `assess_results.paired_bootstrap`).

Because SEC-BERT-SHAPE also breaks ties, the ensemble can differ from it only where BiLSTM-CRF and BERT-base-cased agree with each other against it, or where such a change makes a following I- tag need repair.

## Result

| Model | P | R | micro-F1 | macro-F1 | micro-F1 (full test) |
|---|---:|---:|---:|---:|---:|
| BiLSTM-CRF | 66.80 | 76.79 | 71.45 | 67.68 | 71.42 |
| BERT-base-cased | 63.05 | 79.80 | 70.44 | 66.01 | 70.40 |
| SEC-BERT-SHAPE | 66.23 | 81.19 | 72.95 | 69.15 | 72.93 |
| Majority-vote ensemble | 67.10 | 81.29 | 73.51 | 69.77 | 73.49 |

| Main-view micro-F1 | Value | 95% interval |
|---|---:|---|
| Majority-vote ensemble | 73.51 | [72.97, 74.06] |
| SEC-BERT-SHAPE | 72.95 | [72.40, 73.50] |
| **Ensemble minus SEC-BERT-SHAPE** | **+0.56** | [0.34, 0.79] |

The ensemble scores 0.56 micro-F1 points above SEC-BERT-SHAPE and the 95% interval excludes zero. On this test set it improves the score; because the test set had already been inspected and no validation check was possible, this is exploratory evidence, not confirmation. Relative to SEC-BERT-SHAPE, precision changes by +0.87 points, recall by +0.10 and macro-F1 by +0.62. Only the micro-F1 difference was specified in advance and has an interval.

## Orphan-tag cleaning leaves the strict scores unchanged

Every model was scored with the shared evaluator before and after its orphan I- tags were set to O. All scores (main and full test: P, R, micro-F1, macro-F1 and every per-class value) are identical, and they reproduce each model's saved `results/<model>_eval.json`.

| Model | Orphan I- tags set to O | Main micro-F1 before / after | Full-test micro-F1 before / after |
|---|---:|---|---|
| BiLSTM-CRF | 0 | 71.45 / 71.45 | 71.42 / 71.42 |
| BERT-base-cased | 155 | 70.44 / 70.44 | 70.40 / 70.40 |
| SEC-BERT-SHAPE | 147 | 72.95 / 72.95 | 72.93 / 72.93 |

## How the votes went (main view)

- 92,608 sentences, 4,493,708 words; on 42,024 words at least one model predicts an entity tag. On all other words the three models agree on O.
- Of those 42,024 words: all three agree on 26,408 (62.8%), two agree on 14,617 (34.8%), and all three differ (SEC-BERT-SHAPE decides) on 999 (2.4%).
- BIO repairs: 2 words.
- The ensemble differs from SEC-BERT-SHAPE on 3,259 words in 2,558 sentences.

## Entity changes relative to SEC-BERT-SHAPE (main view)

| | SEC-BERT-SHAPE | Ensemble |
|---|---:|---:|
| True positives | 22,887 | 22,915 |
| False positives | 11,669 | 11,238 |
| False negatives | 5,302 | 5,274 |
| Predicted entities | 34,556 | 34,153 |

The ensemble finds 559 gold entities that SEC-BERT-SHAPE misses and misses 531 that SEC-BERT-SHAPE finds. It removes 1,605 of SEC-BERT-SHAPE's false positives and adds 1,174 new ones.

## Limitations

- Exploratory and test-only: no validation predictions exist, so neither the rule nor the choice of tie-breaker could be checked on held-out data. One rule was fixed in advance; none was tuned.
- Two of the three voters are BERT-base-size transformers fine-tuned with the same script (`train_transformer22.py`), so their errors are likely correlated; the vote is not three independent opinions.
- The interval reflects sentence sampling conditional on these saved predictions. It does not cover training-seed variability (one seed per model) or annotation uncertainty, and it treats sentences as independent, as in `results/assessment/BOOTSTRAP.md`.
- The ensemble needs all three models at prediction time; its cost was not measured.
- Ensemble files are in `results/ensemble/`, not `results/`, so `src/evaluate_predictions.py` and the main table in `results/RESULTS.md` are unaffected.

Reproduce: `python src/ensemble.py --replicates 10000 --seed 42`
