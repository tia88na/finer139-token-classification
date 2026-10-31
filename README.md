# FiNER-139: Financial Numeric Entity Recognition (XBRL Tagging)

Token classification on US SEC financial filings. Each numeric token in a filing
is tagged with its XBRL entity type. The challenge: ~95% of labeled tokens are
numbers, so the correct tag depends on the **context** around the number, not the
number itself.

## Task

Given a sentence from a financial filing, assign each token a label: either `O`
(no entity) or one of the XBRL entity types in BIO format. This is a sequence
labeling / token classification problem with 139 entity types.

## Dataset

- Source: FiNER-139 (Loukas et al., ACL 2022) — https://huggingface.co/datasets/nlpaueb/finer-139
- Original size: 1.1M sentences (900,384 train / 112,494 validation / 108,378 test)
- 139 entity types -> 279 BIO tags; 167 appear in train
- `data/raw/`       — original dataset, untouched
- `data/cleaned/`   — after cleaning (dedup + leakage removal + O-only reduction)
- `data/processed/` — cleaned data tokenized (bert-base-cased) + label-aligned, model-ready

## Data cleaning

Aggressive cleaning was applied only to TRAIN; validation and test were
deduplicated only, so evaluation stays honest and reflects the real distribution.

| Split | Before | After | % kept |
|-------|-------:|------:|-------:|
| train | 900,384 | 261,610 | 29.1% |
| validation | 112,494 | 101,125 | 89.9% |
| test | 108,378 | 95,030 | 87.7% |

Train removal breakdown: 296,717 exact duplicates + 41,099 train/val-test leakage
sentences + 300,958 O-only sentences. No-entity sentences in train fell from
721,189 to 128,887 (80% -> 49%).

## Key facts (from EDA)

- Severe imbalance: 99.22% of tokens are `O`, only 0.78% carry a real entity label
- 94.7% of labeled tokens are numeric
- Long-tailed labels: most frequent appears 15,208 times, rarest 1-2 times
- The tokenizer splits 10.6% of words into sub-pieces (number fragmentation)
- `max_length = 128` covers 97.4% of sentences without truncation

## Evaluation

Metric: entity-level precision, recall and F1 (seqeval, strict IOB2), micro and
macro. Token accuracy is misleading here because of the 99% `O` majority.

Every model is scored with the same code, `src/finer.py` (`evaluate()`), so the
numbers in the results table are comparable:

- **Word level, full-length sentences.** Scores are computed on the original words,
  never on sub-word pieces. If a model could not label a word (for example because
  the sentence was cut at 128 or 512 pieces), that word counts as `O`, so a missed
  entity is a false negative and is never removed from the gold set. Cutting at 128
  pieces would otherwise drop 3.8% of the labelled test words.
- **Main test view without validation duplicates.** 2,422 cleaned test sentences
  also occur, word for word, in validation, which is used to pick the epoch. Such
  duplicates inflate test scores (lecture W6, slide 19), so the main view leaves
  them out (**92,608** sentences). The full cleaned test (95,030) is reported next
  to it for reference.
- **Saved predictions.** Each model saves word-level predicted tag ids as
  `results/<model>_test_predictions.parquet` (row i = row i of
  `data/cleaned/test.parquet`). `python src/evaluate_predictions.py` scores all of
  them and writes `results/RESULTS.md` and the LaTeX table
  `results/results_table.tex`. No model has to be re-trained to be re-scored.
- **Tests.** `python -m pytest tests -q` checks the evaluator against the lecture
  examples (W6 slides 17, 20-21: F1 = 0.50 for the Ada / New York example) and the
  HMM's Viterbi against the "can fish" example (slides 26-29: best path V N). It also
  checks the CRF layer (log Z, every gradient and Viterbi against brute-force
  enumeration), the CRF features, and that the CRF and the BiLSTM-CRF learn from
  context and save predictions in the shared format. Further tests cover the error
  analysis, the bootstrap, the label families and the ensemble (54 tests in total).

## Models

| Model | What it sees | Code |
|---|---|---|
| Most-frequent tag | the token only | `src/baseline_hmm.py` (CPU) |
| HMM + Viterbi | the token + the previous tag | `src/baseline_hmm.py` (CPU, about 7 min) |
| CRF | hand-made features of the word and of the 3 words on each side (shape, number type, word pairs, 8 previous words) + tag transitions | `src/baseline_crf.py` (GPU, about 75 min with the search) |
| BiLSTM-CRF | whole sentence (BiLSTM over word, character and casing embeddings) + tag transitions | `src/bilstm_crf.py` (GPU, about 2 h with the search) |
| BERT-base-cased | whole sentence, numbers split into sub-word pieces | `train_transformer22.py` (GPU) |
| SEC-BERT-SHAPE | whole sentence, numbers replaced by their shape (`645.1` -> `[XXX.X]`) | `train_transformer22.py` (GPU) |

The baseline and the HMM normalise tokens the same way: numbers -> digit shape
(`645.1` -> `XXX.X`), words -> lowercase. The CRF and the BiLSTM-CRF use the same
normalised form for their word features / word embeddings.

### CRF and BiLSTM-CRF

Both models share one linear-chain CRF layer (`src/crf_layer.py`): tag-transition,
start and end scores on top of per-word emission scores. Training minimises the
negative log-likelihood of the gold tag sequence (forward algorithm); prediction is
Viterbi. IOB2 constraints forbid `I-X` after anything other than `B-X` / `I-X`
(strict seqeval would not count such an entity). The two models differ only in the
emission scores. Like the HMM, they can predict the 168 tags seen in train.

- **CRF** (`src/crf_features.py`, `src/baseline_crf.py`): one weight per (feature value, tag)
  for binary features.
  - Current word: normalised form, short word shape (`645.1` -> `d.d`), casing,
    number type (integer / thousands / decimal / percent / year), digit count,
    3-letter prefix and suffix.
  - Surrounding words: normalised words at -3..+3 and shapes at -2..+2.
  - Word pairs: `w-2|w-1`, `w+1|w+2`, `w-1|shape`.
  - Left context: a bag of the 8 previous words.
  - Size: 691,716 feature values x 168 tags.
  - Training: on the GPU with AdamW (lr 0.01, 8,192 words per batch). This is the
    regularised conditional log-likelihood of CRFsuite's `l2sgd` trainer. A CPU
    CRFsuite would need hours per pass over 13.1M training words with 168 tags.
- **BiLSTM-CRF** (`src/bilstm_crf.py`; Lample et al. 2016, Ma & Hovy 2016):
  - Each word: a 100-d word embedding of the normalised form (trained from scratch,
    no pre-trained vectors), a character CNN (30-d characters, 50 filters of
    width 3) and an 8-d casing embedding.
  - Network: one BiLSTM layer (256 units per direction), dropout, then a linear
    layer gives the emission scores.
  - Training: Adam (lr 2e-3, 8,192 words per batch), gradient clipping 5.0,
    word dropout 0.05.

Settings and the epoch are chosen on validation micro-F1 (strict IOB2, the same 20,000
validation sentences as the transformers); the test set is decoded once, with the
selected model. Seed 42, one RTX 4070 (the two searches shared it).

| Model | Settings tried on validation | Selected | Val micro-F1 | Time |
|---|---|---|---:|---:|
| CRF | window {2, 3} x weight decay {0, 0.01, 0.1}; max 8 epochs, patience 2 | window 3, weight decay 0, epoch 2 | 63.2 | 75 min |
| BiLSTM-CRF | dropout {0.33, 0.5} x BiLSTM layers {1, 2}; max 12 epochs, patience 3 | dropout 0.33, 1 layer, epoch 12 | 74.9 | 125 min |

Every setting and epoch is listed in `results/{crf,bilstm_crf}_search.csv` and
`results/{crf,bilstm_crf}_metrics.json`. Every BiLSTM-CRF setting was still improving
at epoch 12, so more epochs may help. Reproduce (GPU):

    python src/baseline_crf.py --limit 2000   # quick check: nothing saved, test not scored
    python src/baseline_crf.py --grid         # search on validation, then test
    python src/bilstm_crf.py --grid           # search on validation, then test

Note on the BiLSTM-CRF run: after printing its last line (`Saved ...`) the process
exited with code 127 and wrote no traceback; the CRF run, which ends with the same
code, exited with 0. All saved BiLSTM-CRF files were checked afterwards:
- the prediction file has 95,030 rows in test order, one valid tag id per word;
- `src/evaluate_predictions.py` reproduces the saved scores exactly;
- the checkpoint loads.

## Results

Entity-level, strict IOB2 (seqeval), word level, full-length sentences.
**Main** = cleaned test without the sentences that also occur in validation (92,608 sentences). **Full test** = all 95,030 cleaned test sentences.

| Model | P | R | micro-F1 | macro-F1 | micro-F1 (full test) |
|---|---:|---:|---:|---:|---:|
| Most-frequent tag | 37.1 | 1.6 | **3.1** | 0.6 | 3.1 |
| HMM + Viterbi | 5.8 | 2.5 | **3.5** | 1.0 | 3.5 |
| CRF | 57.5 | 60.6 | **59.0** | 52.9 | 59.0 |
| BiLSTM-CRF | 66.8 | 76.8 | **71.5** | 67.7 | 71.4 |
| BERT-base-cased | 63.0 | 79.8 | **70.4** | 66.0 | 70.4 |
| SEC-BERT-SHAPE | 66.2 | 81.2 | **73.0** | 69.2 | 72.9 |

Why the token-only models fail (`results/hmm_analysis.txt`): a number's shape does
not decide its tag. In train, the shape `XX` occurs 306,226 times, is `O` 91.8% of
the time and belongs to 134 different entity types. `P(O -> O) = 0.981`, so the
Viterbi path almost never leaves `O`: the HMM predicts `O` for 91.8% of the gold
entity words. The words that decide the tag ("interest rate of", "matures in") are
not visible to an HMM emission. Models that read the context reach 59.0 (CRF with
hand-made context features), 71.5 (BiLSTM-CRF) and 70-73 micro-F1 (the transformers).
On the main view the HMM predicts `O` for 91.8% of the gold entity words, the CRF
for 23.7%, and the BiLSTM-CRF for 12.5%.

## Error analysis

Run `python src/analyze_errors.py` to compare the saved BiLSTM-CRF, BERT-base-cased
and SEC-BERT-SHAPE predictions without training or model weights. Outputs are in
`results/analysis/`: `ANALYSIS.md`, overall and per-class scores, sentence-length
and training-frequency breakdowns, word-level type confusions, and illustrative
examples with original test row IDs. Analysis uses only the main test view and
checks its precision/recall/F1 against the saved shared-evaluator results.

False negatives and false positives have separate error partitions. Word-level
confusions are diagnostic counts, not entity-level errors. Length is measured in
original words, so length buckets do not establish tokenizer truncation effects.

The final comparison table is `results/RESULTS.md` (strict IOB2, main test view).
`transformer_benchmark.csv` and transformer `*_metrics.json` preserve the original
full-test, default-seqeval scores; those numbers use a different protocol and must
not be mixed with the final table.

SEC-BERT-SHAPE is an existing pretrained model from Loukas et al. (ACL 2022),
not a new architecture introduced by this project. Comparing it to BERT-base
changes both domain pretraining and number representation; the improvement
cannot be attributed to shape replacement alone.

Removing duplicates/leakage and downsampling O-only training sentences are
different operations. Downsampling changes the training distribution; its
effect on precision/recall has not been established by a controlled experiment.
All training runs use one seed. Latency is not measured because trained model
weights are not included. Slides and speech notes are maintained outside the repo.

## Bootstrap and sample assessment

`python src/assess_results.py --replicates 10000 --sample-size 50 --seed 42`
computes paired sentence-bootstrap confidence intervals from existing predictions
and samples 50 of SEC-BERT-SHAPE's non-overlapping false-positive spans. Results,
per-sentence counts, replicates and source-file hashes are in `results/assessment/`.
`BOOTSTRAP.md` explains marginal and family-wise intervals and their limits:
sentence sampling conditional on these models, not training-seed or annotation uncertainty.

`DEGERLENDIRME.md` documents the contextual assessment of all 50 sampled spans.
It distinguishes possible annotation gaps/scope differences, apparent model errors,
and insufficient context. These are provisional interpretations, not verified gold
corrections. Exact raw sentence duplicates are checked separately for conflicting
target tags. A single assessment was performed; no independent second review is claimed.
`python src/render_assessment.py` regenerates the report from the saved sample-specific
`review_notes.json`. Changing the sample requires new assessments.

## Label families

`python src/label_families.py` groups the 139 entity types into 11 financial
families (debt & interest, credit facilities, equity & share capital, share-based
compensation, revenue & customer contracts, income tax, leases, acquisitions &
investments, long-lived assets, contingencies & guarantees, other) and compares
BiLSTM-CRF, BERT-base-cased and SEC-BERT-SHAPE per family. It reuses
`results/analysis/per_class.csv`; no model is re-run. Every type is in exactly one
family, assigned from its XBRL name before any family score was computed, and the
close calls are documented. Pooled family F1 (from summed TP, gold and predicted
counts) is reported separately from the unweighted mean of per-label F1. Summed
over all families, the counts reproduce each model's saved micro- and macro-F1.
Mapping, tables and observations are in `results/label_families/LABEL_FAMILIES.md`.

## Majority-vote ensemble (exploratory)

`python src/ensemble.py --replicates 10000 --seed 42` combines the saved BiLSTM-CRF,
BERT-base-cased and SEC-BERT-SHAPE test predictions without retraining. Each word
gets the BIO tag chosen by at least two models, and SEC-BERT-SHAPE breaks three-way
ties. Orphan `I-` tags are set to `O` before the vote (this leaves every strict score
unchanged, which the script checks) and repaired after it, so every output sentence
is valid IOB2. The ensemble is scored with the shared evaluator and compared with
SEC-BERT-SHAPE using the paired bootstrap above.

On the main view the ensemble reaches 73.5 micro-F1 against 73.0 for SEC-BERT-SHAPE
(+0.56 points, 95% interval [0.34, 0.79]), mainly through fewer false positives.
Only test predictions exist, so this is exploratory: the rule was fixed and committed
before the test run and was not tuned, but the test set had already been inspected,
so the gain is not unbiased confirmation. Method, checks and vote statistics are in
`results/ensemble/ENSEMBLE.md`. The files are kept in `results/ensemble/`, not
`results/`, so `src/evaluate_predictions.py` and `results/RESULTS.md` are unchanged.

## Repository structure

    finer139-token-classification/
    |- README.md
    |- LICENSE                      # MIT (code)
    |- CONTRIBUTORS.md
    |- requirements.txt
    |- .gitignore
    |- 01_data_preprocessing.ipynb  # EDA + cleaning + tokenization
    |- train_transformer22.py       # BERT / SEC-BERT-SHAPE training + evaluation
    |- src/
    |  |- finer.py                  # shared data loading + evaluation (all models)
    |  |- baseline_hmm.py           # most-frequent-tag baseline + HMM/Viterbi
    |  |- crf_layer.py              # linear-chain CRF layer (forward algorithm, Viterbi, IOB2)
    |  |- crf_features.py           # hand-made features for the CRF
    |  |- baseline_crf.py           # feature CRF: training, selection, test predictions
    |  |- bilstm_crf.py             # BiLSTM-CRF: training, selection, test predictions
    |  |- seq_utils.py              # batching, early stopping, saving (CRF + BiLSTM-CRF)
    |  |- evaluate_predictions.py   # scores every saved prediction file, builds tables
    |  |- label_families.py         # 139 types -> 11 families, family-level scores
    |  |- ensemble.py               # exploratory majority-vote ensemble of saved predictions
    |- tests/                       # checks for the evaluator, HMM, CRF layer, CRF, BiLSTM-CRF
    |- results/                     # metrics, per-class scores, test predictions
    |- data/
    |  |- LICENSE                   # CC BY-SA 4.0 (dataset) + attribution
    |  |- raw/                      # original parquet files
    |  |- cleaned/                  # cleaned parquet files
    |  |- processed/                # tokenized + aligned parquet files
    |- contributions/               # per-member contribution notes

## Setup

    pip install -r requirements.txt

The CRF and the BiLSTM-CRF need no packages beyond `requirements.txt`. For the GPU,
install a CUDA build of PyTorch before the requirements, for example
`pip install torch --index-url https://download.pytorch.org/whl/cu128` (on Windows,
plain `pip install torch` gives a CPU-only build). Tested with Python 3.12, PyTorch 2.11
(CUDA 12.8), seqeval 1.2.2.

## Reproduce preprocessing

Run `01_data_preprocessing.ipynb`: it loads the raw data, runs the EDA, cleans the
dataset, tokenizes with label alignment, and writes `data/cleaned/` and
`data/processed/`.

## License

- Code (notebook, scripts): MIT — see `LICENSE`.
- Data (`data/`): derived from FiNER-139, licensed CC BY-SA 4.0 — see `data/LICENSE`.

## Team

See `CONTRIBUTORS.md`.
