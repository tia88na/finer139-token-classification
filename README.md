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
  HMM's Viterbi against the "can fish" example (slides 26-29: best path V N).

## Models

| Model | What it sees | Code |
|---|---|---|
| Most-frequent tag | the token only | `src/baseline_hmm.py` (CPU) |
| HMM + Viterbi | the token + the previous tag | `src/baseline_hmm.py` (CPU, about 7 min) |
| BERT-base-cased | whole sentence, numbers split into sub-word pieces | `train_transformer22.py` (GPU) |
| SEC-BERT-SHAPE | whole sentence, numbers replaced by their shape (`645.1` -> `[XXX.X]`) | `train_transformer22.py` (GPU) |

The baseline and the HMM normalise tokens the same way: numbers -> digit shape
(`645.1` -> `XXX.X`), words -> lowercase.

## Results

Entity-level, strict IOB2 (seqeval), word level, full-length sentences.
**Main** = cleaned test without the sentences that also occur in validation (92,608 sentences). **Full test** = all 95,030 cleaned test sentences.

| Model | P | R | micro-F1 | macro-F1 | micro-F1 (full test) |
|---|---:|---:|---:|---:|---:|
| Most-frequent tag | 37.1 | 1.6 | **3.1** | 0.6 | 3.1 |
| HMM + Viterbi | 5.8 | 2.5 | **3.5** | 1.0 | 3.5 |
| BERT-base-cased | 63.0 | 79.8 | **70.4** | 66.0 | 70.4 |
| SEC-BERT-SHAPE | 66.2 | 81.2 | **73.0** | 69.2 | 72.9 |

Why the token-only models fail (`results/hmm_analysis.txt`): a number's shape does
not decide its tag. In train, the shape `XX` occurs 306,226 times, is `O` 91.8% of
the time and belongs to 134 different entity types. `P(O -> O) = 0.981`, so the
Viterbi path almost never leaves `O`: the HMM predicts `O` for 91.8% of the gold
entity words. The words that decide the tag ("interest rate of", "matures in") are
not visible to an HMM emission. Models that read the context (the transformers)
reach 70-73 micro-F1.

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
    |  |- evaluate_predictions.py   # scores every saved prediction file, builds tables
    |- tests/                       # checks for the evaluator and the HMM
    |- results/                     # metrics, per-class scores, test predictions
    |- data/
    |  |- LICENSE                   # CC BY-SA 4.0 (dataset) + attribution
    |  |- raw/                      # original parquet files
    |  |- cleaned/                  # cleaned parquet files
    |  |- processed/                # tokenized + aligned parquet files
    |- contributions/               # per-member contribution notes

## Setup

    pip install -r requirements.txt

## Reproduce preprocessing

Run `01_data_preprocessing.ipynb`: it loads the raw data, runs the EDA, cleans the
dataset, tokenizes with label alignment, and writes `data/cleaned/` and
`data/processed/`.

## License

- Code (notebook, scripts): MIT — see `LICENSE`.
- Data (`data/`): derived from FiNER-139, licensed CC BY-SA 4.0 — see `data/LICENSE`.

## Team

See `CONTRIBUTORS.md`.
