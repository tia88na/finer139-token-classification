# Tuana Harmankaya

- Student ID: 2309111029
- GitHub: [@tia88na](https://github.com/tia88na)
- Repository: [tia88na/finer139-token-classification](https://github.com/tia88na/finer139-token-classification)

## Summary

My contribution is the data and preprocessing foundation of the project. I selected and integrated the FiNER-139 dataset into the shared repository, ran the exploratory data analysis that motivates the task, built the tokenization and label-alignment pipeline that every model in the team uses, cleaned the dataset (deduplication, train/val–test leakage removal, and no-entity sentence reduction), and published the data in three reproducible stages (raw, cleaned, processed) together with the repository structure.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Selected and integrated the FiNER-139 dataset and set up the repository structure | Repo root: `README.md`, `CONTRIBUTORS.md`, `requirements.txt`, `.gitignore`, `data/`, notebook |
| 2 | Exploratory data analysis: label distribution, imbalance, sentence lengths, numeric ratio, number fragmentation | `01_data_preprocessing.ipynb`, README "Key facts" |
| 3 | Tokenization and label alignment (bert-base-cased, word-to-subword, max_length 128) | `01_data_preprocessing.ipynb`, `data/processed/` |
| 4 | Data cleaning: deduplication, train/val–test leakage removal, no-entity sentence reduction | `data/cleaned/`, README before/after table |
| 5 | Published data in three stages (raw, cleaned, processed) as Parquet and pushed to GitHub | `data/raw/`, `data/cleaned/`, `data/processed/` |

## 1. Dataset integration and repository setup

I selected FiNER-139 (Loukas et al., ACL 2022), an XBRL financial numeric entity recognition dataset, as a non-trivial token-classification task: 139 entity types (279 BIO tags) over 1.1M sentences from ~10k US SEC filings, where ~95% of labeled tokens are numeric and the correct tag depends on context rather than the token itself.

I loaded it with the `datasets` library (`nlpaueb/finer-139`) and set up the repository: `README.md` (task, dataset, EDA facts, evaluation), `CONTRIBUTORS.md`, `requirements.txt`, `.gitignore`, a `data/` tree, and the preprocessing notebook.

| Split | Sentences |
|---|---:|
| train | 900,384 |
| validation | 112,494 |
| test | 108,378 |

## 2. Exploratory data analysis

I measured the properties that justify the task and the evaluation choices:

| Finding | Value |
|---|---|
| Total tokens (train) | 40,827,499 |
| `O` (no-entity) token ratio | 99.22% |
| Labeled token ratio | 0.78% |
| Active labels in train (of 279 BIO) | 167 |
| Most frequent label | B-DebtInstrumentInterestRateStatedPercentage (15,208) |
| Rarest labels | 1–2 occurrences (long-tailed distribution) |
| Labeled tokens that are numeric | 94.7% |
| Words split into sub-pieces by the tokenizer | 10.6% |
| Sentence length (mean / median / 95th / 99th) | 45.3 / 37 / 104 / 171 |
| Sentences longer than 128 tokens | 2.6% |

Two findings drive later decisions: the 99.22% `O` majority means token accuracy is misleading, so evaluation must use entity-level F1 (seqeval); and because 94.7% of labeled tokens are numbers, the model must rely on context, not the token string.

## 3. Tokenization and label alignment

I built the preprocessing function used by all models. Each sentence is tokenized with `bert-base-cased` (`is_split_into_words=True`, truncation at `max_length=128`, which covers 97.4% of sentences). Because the tokenizer fragments numbers (e.g. "645.1" splits into 4 sub-pieces; 10.6% of words split overall), I align labels with `word_ids()`: the label goes on the first sub-piece of each word, and the remaining sub-pieces receive `-100` so they are ignored in the loss. This keeps the one-word-one-label correspondence intact despite fragmentation.

## 4. Data cleaning

I cleaned the dataset in three steps. Aggressive cleaning was applied **only to train**; validation and test were deduplicated only, so evaluation stays honest and reflects the real distribution.

| Split | Before | After | % kept |
|---|---:|---:|---:|
| train | 900,384 | 261,610 | 29.1% |
| validation | 112,494 | 101,125 | 89.9% |
| test | 108,378 | 95,030 | 87.7% |

Train removal breakdown:

| Step | Removed |
|---|---:|
| Exact-duplicate sentences | 296,717 |
| Train/val–test leakage (train sentences also in val or test) | 41,099 |
| No-entity (`O`-only) sentences dropped | 300,958 |

No-entity sentences in train fell from 721,189 to 128,887 (80% to 49% of train), reducing the sentence-level imbalance so the model sees entity-bearing sentences far more often. Leakage removal prevents artificially inflated test scores.

## 5. Published data stages

I saved all three stages as Parquet and pushed them to GitHub, so the full provenance (raw to model-ready) is reproducible:

| Folder | Contents |
|---|---|
| `data/raw/` | Original FiNER-139, untouched (`tokens`, `ner_tags`) |
| `data/cleaned/` | After deduplication, leakage removal, and `O`-only reduction |
| `data/processed/` | Cleaned data tokenized and label-aligned (`input_ids`, `attention_mask`, `labels`), model-ready |

## Files I created

- `01_data_preprocessing.ipynb` — EDA, tokenization, label alignment, cleaning
- `README.md` — task, dataset, EDA facts, before/after cleaning table, evaluation note
- `requirements.txt`, `.gitignore`, repository structure
- `data/raw/`, `data/cleaned/`, `data/processed/` — three-stage Parquet data

## Commit evidence

Commits are in the repository ([tia88na/finer139-token-classification](https://github.com/tia88na/finer139-token-classification), branch `main`):

- Initial commit — repository and README
- Add preprocessing notebook, project files, and dataset
- Add data cleaning step (dedup, leakage removal, O-only reduction)
