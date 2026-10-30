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

Metric: entity-level F1 (seqeval). Token accuracy is misleading here because of
the 99% `O` majority.

## Repository structure

    finer139-token-classification/
    |- README.md
    |- LICENSE                      # MIT (code)
    |- CONTRIBUTORS.md
    |- requirements.txt
    |- .gitignore
    |- 01_data_preprocessing.ipynb  # EDA + cleaning + tokenization
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
