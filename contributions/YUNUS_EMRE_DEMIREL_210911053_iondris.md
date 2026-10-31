# Yunus Emre Demirel

- Student ID: 210911053
- GitHub: [@iondris](https://github.com/iondris)
- Repository: [tia88na/finer139-token-classification](https://github.com/tia88na/finer139-token-classification)

## Summary

My contribution is the two **context models** between the HMM and the transformers:

- a linear-chain **CRF with hand-made features** (word shape, number detection, surrounding words);
- a **BiLSTM-CRF** (BiLSTM over word, character and casing embeddings, with a CRF output layer).

Both share one CRF layer, which I wrote and tested against brute-force enumeration. Both were trained on the cleaned train set, with settings and epoch chosen on validation only, and scored with the team's shared evaluator.

On the main test view, the CRF reaches **59.0 micro-F1** and the BiLSTM-CRF **71.5**, against **3.5** for the HMM and 70.4 / 73.0 for BERT-base / SEC-BERT-SHAPE. Reading the surrounding words is what turns a number into the right XBRL tag.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Linear-chain CRF layer: forward algorithm with exact gradient, Viterbi, IOB2 constraints | `src/crf_layer.py` |
| 2 | Hand-made CRF features: word shape, number detection, surrounding words, word pairs | `src/crf_features.py` |
| 3 | Feature CRF: training, selection on validation, test predictions | `src/baseline_crf.py`, `results/crf_*` |
| 4 | BiLSTM-CRF: training, selection on validation, test predictions | `src/bilstm_crf.py`, `results/bilstm_crf_*` |
| 5 | Shared training code for both models: batching, early stopping, saving in the team format | `src/seq_utils.py` |
| 6 | Tests: CRF layer vs brute force, features, both models end to end (18 new tests) | `tests/test_crf_layer.py`, `tests/test_crf_features.py`, `tests/test_seq_models.py` |
| 7 | Results table rebuilt with both models (existing scores unchanged) | `results/RESULTS.md`, `results/results_table.tex` |
| 8 | README sections for the two models, setup for the GPU | `README.md` |

## 1. Shared CRF layer (`src/crf_layer.py`)

For a sentence and a tag sequence y, the CRF score is

    s(y) = start[y_1] + sum_i e_i[y_i] + sum_i trans[y_{i-1}, y_i] + end[y_n],   P(y | x) = exp s(y) / Z.

The two models differ only in the emission scores `e_i`.

- **Training** minimises -log P(gold) = log Z - s(gold).
- **log Z** comes from the forward algorithm. Its gradient is the expected feature count: d log Z / d e_t[j] = P(y_t = j), and d log Z / d trans[i, j] = sum over t of P(y_{t-1} = i, y_t = j). Both are computed with the backward algorithm.
- **Why the gradient is written out:**
  - My first version let autograd unroll the loop over words. It spent about 90% of each training step in that loop: around 45 small GPU operations per word, at about 16 µs launch cost each on Windows.
  - The written-out version needs about 7 operations per word (one matrix product per word in exp space, float64 so nothing underflows), which made training about 2x faster.
  - The tests check log Z and every gradient against brute-force enumeration over all tag sequences.
- **Decoding** is Viterbi. IOB2 constraints forbid `I-X` at the start of a sentence or after anything other than `B-X` / `I-X`, so no ill-formed entity is produced (strict seqeval would not count one). The gold train tags have 0 IOB2 violations, so the constraints are also used in training.
- **Tag set:** the 168 tags seen in train, as for the HMM. Only 3 test entity words have a tag never seen in train.

## 2. CRF with hand-made features (`src/crf_features.py`, `src/baseline_crf.py`)

Each word gets seven attributes, computed once per distinct word:
- normalised form (`finer.norm_token`, the same as the HMM: `645.1` -> `XXX.X`, words lowercased)
- short shape (`645.1` -> `d.d`, `Revenue` -> `Xx`)
- casing
- number type (integer / thousands / decimal / percent / year / not a number; based on `finer.is_number`)
- digit count
- 3-letter prefix and suffix

Feature templates read these attributes at positions relative to the word. Each template has its own weight table (values x tags):

| Group | Templates |
|---|---|
| current word | all seven attributes + bias |
| surrounding words | normalised word at -3..+3, shape at -2..+2 (`<BOS>` / `<EOS>` outside the sentence) |
| word pairs | `w-2 \| w-1`, `w+1 \| w+2`, `w-1 \| shape` |
| left context | bag of the 8 previous words, e.g. "interest rate of approximately 5.0%" |

- **Rare values:** values seen fewer than 2 times in train become unknown, as `<NUM>` / `<CAP>` / `<UNK>` for words, like the HMM. At most 200,000 word pairs are kept per pair template.
- **Size:** 691,716 feature values x 168 tags (443 MB of weights). The emission score is a sum of feature weights (an `EmbeddingBag`), so this is exactly a log-linear CRF.
- **Training:** AdamW, lr 0.01, 8,192 words per batch, all-zero start, i.e. the L2-regularised conditional log-likelihood with mini-batch gradients (CRFsuite's `l2sgd` objective). I did not use CPU CRFsuite: with 168 tags and 13.1M training words, one pass costs about 168² operations per word, single-threaded, which means hours per L-BFGS iteration.

## 3. BiLSTM-CRF (`src/bilstm_crf.py`)

Based on Lample et al. (2016) and Ma & Hovy (2016).

- **Input per word:**
  - a 100-d word embedding of the normalised form (26,733 words seen at least twice in train; the rest become `<NUM>` / `<CAP>` / `<UNK>`)
  - a character CNN over the raw word (30-d characters, 50 filters of width 3, max-pooling), which keeps capitals, digits, `%` and `$`
  - an 8-d casing embedding
- **Network:** one BiLSTM layer (256 units per direction), dropout, then a linear layer gives the emission scores, followed by the shared CRF layer. There is no per-word softmax: the loss is the CRF negative log-likelihood and prediction is Viterbi.
- **Training:** Adam, lr 2e-3, 8,192 words per batch, gradient clipping 5.0, word dropout 0.05 (so the unknown-word vectors are learned). All embeddings are trained from scratch; nothing is downloaded.

## 4. Training and model selection

- **Data:** cleaned train (261,610 sentences, 13.1M words), no truncation (the longest sentence has 1,165 words).
- **Selection:** every setting and every epoch is scored on validation micro-F1 (strict IOB2, `finer.scores`), on the same 20,000 validation sentences the transformers used (`sample(n=20000, random_state=42)`). The best weights are kept.
- **Test set:** decoded **once**, with the selected setting and epoch.
- **Seed:** 42 everywhere.
- **Smoke checks first:**
  - `--limit 2000` and `--limit 20000` runs checked the pipeline end to end; in this mode nothing is saved and test is only checked for format.
  - A dry run of the save path wrote to a scratch folder, with the validation sample standing in for test.
  - lr 0.01 for the CRF was chosen over 0.003 on the 20k check: validation 53.5 vs 51.1.
  - For the BiLSTM-CRF, 8,192 words per batch with lr 2e-3 matched 4,096 words with lr 1e-3 on validation (36.5 vs 36.7) at half the steps.
- **Hardware:** one RTX 4070 (12 GB), shared by the two searches. The CRF search took 75 min and the BiLSTM-CRF search 125 min. Peak GPU memory was 2.2 GB (CRF) and 0.7 GB (BiLSTM-CRF).

CRF search (best validation micro-F1 per setting; each peaked at epoch 2 and stopped at epoch 4, patience 2):

| Window | Weight decay 0 | 0.01 | 0.1 |
|---|---:|---:|---:|
| 2 | 60.7 | 60.5 | 58.2 |
| 3 | **63.2** | 63.1 | 60.1 |

BiLSTM-CRF search (12 epochs each, patience 3):

| | dropout 0.33 | dropout 0.5 |
|---|---:|---:|
| 1 BiLSTM layer | **74.9** (epoch 12) | 73.0 (epoch 12) |
| 2 BiLSTM layers | 73.8 (epoch 12) | 71.6 (epoch 12) |

## 5. Results

Shared evaluator (`src/evaluate_predictions.py`), entity level, strict IOB2. Main view: 92,608 test sentences; full test: 95,030.

| Model | P | R | micro-F1 | macro-F1 | micro-F1 (full test) |
|---|---:|---:|---:|---:|---:|
| HMM + Viterbi (Begüm) | 5.8 | 2.5 | 3.5 | 1.0 | 3.5 |
| **CRF** | 57.5 | 60.6 | **59.0** | 52.9 | 59.0 |
| **BiLSTM-CRF** | 66.8 | 76.8 | **71.5** | 67.7 | 71.4 |
| BERT-base-cased (Zehra) | 63.0 | 79.8 | 70.4 | 66.0 | 70.4 |
| SEC-BERT-SHAPE (Zehra) | 66.2 | 81.2 | 73.0 | 69.2 | 72.9 |

Rebuilding the table with `src/evaluate_predictions.py` left every existing model's `*_eval.json` byte-identical; only the two new rows were added.

## 6. Analysis

**Context is what the HMM was missing.** Gold entity words on the main view (29,251), in %:

| Model | predicted `O` | exact tag right |
|---|---:|---:|
| HMM | 91.8 | 5.0 |
| CRF | 23.7 | 59.9 |
| BiLSTM-CRF | 12.5 | 76.5 |
| BERT-base-cased | 7.9 | 79.3 |
| SEC-BERT-SHAPE | 7.8 | 80.5 |

- **Learned context beats hand-made context.** The BiLSTM-CRF beats the CRF on **130 of the 139** entity types. The largest gains are on long share-based-compensation types and AmortizationOfIntangibleAssets (0.47 -> 0.93), where the deciding words are far from the number or vary in wording. The CRF only sees 3 words on each side, word pairs and an unordered bag of 8.
- **The CRF overfits fast.** All six settings peaked at epoch 2 while the training loss kept falling. Weight decay did not help, and 0.1 hurt. The extra surrounding word (window 3 vs 2) added +2.4 validation points.
- **Most remaining errors are missed entities.** The top confusion for both models is DebtInstrumentMaturityDate -> `O` (456 words for the CRF, 256 for the BiLSTM-CRF).
- **Hardest types** are the same as for the transformers: ShareBasedCompensation (F1 0.14), OperatingLeasesRentExpenseNet (0.26), DebtInstrumentMaturityDate (0.27).
- **Recall is higher than precision** for both models, as for the transformers.
- **BiLSTM-CRF vs BERT-base:** the BiLSTM-CRF is +1.1 micro-F1 and +1.7 macro-F1 ahead, with no pre-training. This is not an equal-compute comparison: BERT was trained for 2 epochs, the BiLSTM-CRF for up to 12, and each has one seed.

## 7. Limitations and known issues

- **Epoch cap:** every BiLSTM-CRF setting reached its best score at epoch 12, the maximum I allowed, so more epochs would probably help. I did not re-run, because the settings were fixed before the search.
- **Seeds and determinism:** one seed per model, and no significance test. CPU runs repeat exactly (tested). On the GPU, some CUDA kernels may differ in the last floating-point bits.
- **Tag coverage:** like the HMM, both models cannot predict the tags never seen in train (3 test entity words).
- **Validation sample:** selection uses 20,000 of the 101,125 validation sentences, the same sample as the transformers.
- **Exit code 127 (BiLSTM-CRF run):**
  - After printing its last line (`Saved results/bilstm_crf_test_predictions.parquet, ...`), the training process exited with code 127 and wrote no traceback. The CRF run, which ends with the same code, exited with 0. I did not find the cause.
  - Everything is written before that last line, and all saved files were checked afterwards: the prediction file has 95,030 rows in test order, every row as long as its sentence, valid tag ids; `src/evaluate_predictions.py` reproduces the saved scores exactly; `results/bilstm_crf_metrics.json` parses; the checkpoint `models/bilstm_crf.pt` loads.

## 8. Tests

`python -m pytest tests -q` runs 26 tests: Begüm's 8 plus my 18.

- **CRF layer:**
  - forward-algorithm log Z, the negative log-likelihood and **every gradient** (emissions, transitions, start, end) equal brute-force enumeration, with and without IOB2 constraints, on mixed-length padded batches
  - no underflow with scores in the hundreds
  - Viterbi equals the brute-force best path
  - decoding never breaks IOB2
  - batched and single decoding agree
- **Features:**
  - shapes and number types of `645.1`, `12%`, `1,250`, `2019`, `Revenue`, `(`
  - context at sentence edges (`<BOS>` / `<EOS>`)
  - word pairs and the left bag
  - unknown words
  - deterministic vocabularies
- **Models:**
  - both models learn that the same number (`5.0`) is an entity after "rate of" and `O` after "the year", which a token-only model cannot do
  - same seed, same predictions
  - output format (one row per sentence, input order, full length, global tag ids)
  - batching covers every sentence once

## 9. How to reproduce

    pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build first (GPU)
    pip install -r requirements.txt                                         # no extra packages needed
    python -m pytest tests -q                                               # 26 tests
    python src/baseline_crf.py --limit 2000   # quick check: nothing saved, test not scored
    python src/baseline_crf.py --grid         # 6 settings on validation, then test (~75 min, RTX 4070)
    python src/bilstm_crf.py --limit 2000
    python src/bilstm_crf.py --grid           # 4 settings on validation, then test (~2 h)
    python src/evaluate_predictions.py        # rebuild results/RESULTS.md and the LaTeX table

Tested with Python 3.12.10, PyTorch 2.11.0 (CUDA 12.8), numpy 2.5, pandas 3.1, seqeval 1.2.2. The exact versions and settings of each run are stored in `results/{crf,bilstm_crf}_metrics.json`.

Checkpoints are not in git: `models/bilstm_crf.pt` (15 MB, `models/` is gitignored), and the CRF's 470 MB checkpoint was written outside the repository with `--model-dir`.

## Files I added or changed

- `src/crf_layer.py`, `src/crf_features.py`, `src/seq_utils.py`, `src/baseline_crf.py`, `src/bilstm_crf.py` (new)
- `tests/test_crf_layer.py`, `tests/test_crf_features.py`, `tests/test_seq_models.py` (new)
- `results/crf_*`, `results/bilstm_crf_*`: test predictions, `_eval.json`, `_per_class.csv`, `_metrics.json`, `_search.csv` (new)
- `results/RESULTS.md`, `results/results_table.tex`: two rows added by `src/evaluate_predictions.py`
- `README.md`: Tests bullet, Models table, "CRF and BiLSTM-CRF" section, Results table, repository structure, GPU setup
- this file

I did not change other members' code (`src/finer.py`, `src/baseline_hmm.py`, `src/evaluate_predictions.py`, `train_transformer22.py`, `tests/test_finer.py`), the data, or their results. My code imports their shared functions (`load_splits`, `evaluate`, `save_results`, `norm_token`, `is_number`, `save_predictions`, the HMM's unknown-word classes).

## Commit evidence

Commits are in the repository ([tia88na/finer139-token-classification](https://github.com/tia88na/finer139-token-classification), branch `feature/crf-bilstm-crf`):

- [`a4d9e04`](https://github.com/tia88na/finer139-token-classification/commit/a4d9e04) (2026-10-31) - added the CRF layer, the feature CRF and the BiLSTM-CRF (`src/`), their tests, results (`results/crf_*`, `results/bilstm_crf_*`), the rebuilt results tables, README sections and this contribution file (22 files)
