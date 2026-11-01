# Tuana Harmankaya

- Student ID: 2309111029
- GitHub: [@tia88na](https://github.com/tia88na)
- Repository: [tia88na/finer139-token-classification](https://github.com/tia88na/finer139-token-classification)

## Summary

My contribution is the data foundation of the project plus two analyses of the
dataset's limits. I selected and integrated the FiNER-139 dataset into the shared
repository, ran the exploratory data analysis that motivates the task, cleaned the
dataset (deduplication, train/val–test leakage removal, and no-entity sentence
reduction) into the word-level `data/cleaned/` that every model is trained on, and
set up the repository structure. I also wrote the tokenization and label-alignment
step and produced a `data/processed/` version, measured how well a non-finetuned
LLM does on the task (Qwen2.5-7B-Instruct, few-shot), and analysed the conflicting
gold labels in the dataset.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Selected and integrated the FiNER-139 dataset and set up the repository structure | Repo root: `README.md`, `CONTRIBUTORS.md`, `requirements.txt`, `.gitignore`, `data/`, notebook |
| 2 | Exploratory data analysis: label distribution, imbalance, sentence lengths, numeric ratio, number fragmentation | `01_data_preprocessing.ipynb`, README "Key facts" |
| 3 | Data cleaning (dedup, leakage removal, no-entity reduction) producing the shared `data/cleaned/` the models train on | `data/cleaned/`, README before/after table |
| 4 | Tokenization and label-alignment step + a tokenized `data/processed/` version | `01_data_preprocessing.ipynb`, `data/processed/` |
| 5 | Published the data in three reproducible stages (raw, cleaned, processed) as Parquet | `data/raw/`, `data/cleaned/`, `data/processed/` |
| 6 | LLM few-shot comparison: how a non-finetuned Qwen2.5-7B does vs the trained models | `src/llm_fewshot.py`, `results/llm/` |
| 7 | Label-conflict analysis: identical sentences with different gold labels, and their effect on the test scores | `src/label_conflicts.py`, `results/label_conflicts/` |

## 1. Dataset integration and repository setup

I selected FiNER-139 (Loukas et al., ACL 2022), an XBRL financial numeric entity
recognition dataset, as a non-trivial token-classification task: 139 entity types
(279 BIO tags) over 1.1M sentences from ~10k US SEC filings, where ~95% of labeled
tokens are numeric and the correct tag depends on context rather than the token
itself.

I loaded it with the `datasets` library (`nlpaueb/finer-139`) and set up the
repository: `README.md`, `CONTRIBUTORS.md`, `requirements.txt`, `.gitignore`, a
`data/` tree, and the preprocessing notebook.

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
| Words split into sub-pieces by a WordPiece tokenizer | 10.6% |
| Sentence length (mean / median / 95th / 99th) | 45.3 / 37 / 104 / 171 |
| Sentences longer than 128 tokens | 2.6% |

Two findings drive later decisions: the 99.22% `O` majority means token accuracy is
misleading, so evaluation uses entity-level F1 (seqeval); and because 94.7% of
labeled tokens are numbers, the model must rely on context, not the token string.

## 3. Data cleaning

I cleaned the dataset in three steps. Aggressive cleaning was applied **only to
train**; validation and test were deduplicated only, so evaluation stays honest and
reflects the real distribution. The result is `data/cleaned/` (word-level: `tokens`,
`ner_tags`), which is the data every model in the project is trained and evaluated
on.

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

No-entity sentences in train fell from 721,189 to 128,887 (80% to 49% of train),
reducing the sentence-level imbalance so the model sees entity-bearing sentences far
more often. Leakage removal prevents artificially inflated test scores.

## 4. Tokenization and label alignment

I wrote the tokenization and label-alignment step in the notebook. Each sentence is
tokenized with `bert-base-cased` (`is_split_into_words=True`, truncation at
`max_length=128`, which covers 97.4% of sentences). Because a WordPiece tokenizer
fragments numbers (e.g. "645.1" splits into 4 sub-pieces; 10.6% of words split
overall), labels are aligned with `word_ids()`: the label goes on the first
sub-piece of each word, and the remaining sub-pieces receive `-100` so they are
ignored in the loss. This keeps the one-word-one-label correspondence intact despite
fragmentation. I saved this as `data/processed/`.

Note on how the team actually used the data: the training scripts standardised on
the word-level `data/cleaned/` and let each model handle its own tokenization — the
transformer trainer (`train_transformer22.py`) defaults to `--data-dir data/cleaned`
and loads its own tokenizer with `AutoTokenizer.from_pretrained(args.model)` (so
SEC-BERT-SHAPE uses its own `[NUM]`/shape tokenizer, not `bert-base-cased`), and the
classical models read `data/cleaned/` through `src/finer.py`. So `data/processed/`
is a reference, model-ready artifact and a record of the fragmentation analysis; it
is not the path the final models consume. The alignment logic (first-sub-piece
label, `-100` for the rest) is the same technique each transformer applies at train
time.

## 5. Published data stages

I saved all three stages as Parquet and pushed them to GitHub, so the full
provenance (raw to model-ready) is reproducible:

| Folder | Contents |
|---|---|
| `data/raw/` | Original FiNER-139, untouched (`tokens`, `ner_tags`) |
| `data/cleaned/` | After deduplication, leakage removal, and `O`-only reduction — the data the models train on |
| `data/processed/` | The cleaned data tokenized (bert-base-cased) and label-aligned; a model-ready reference version |

## 6. LLM few-shot comparison

To see whether a general, non-finetuned LLM can do this task, I compared
`Qwen/Qwen2.5-7B-Instruct` (4-bit, greedy, 3 labelled train examples in the prompt,
no training) against the trained models (`src/llm_fewshot.py`). All models are scored
on the same 300 main-view test sentences (seed 42: 200 with at least one entity, 100
without), with the shared strict-IOB2 evaluator, so these numbers are sample numbers,
not the full-test numbers in `results/RESULTS.md`.

| Model | micro-F1 | macro-F1 |
|---|---:|---:|
| Qwen2.5-7B-Instruct (3-shot) | 39.4 | 27.9 |
| Most-frequent tag | 1.9 | 0.3 |
| HMM + Viterbi | 3.2 | 1.3 |
| CRF | 63.9 | 44.5 |
| BiLSTM-CRF | 79.4 | 67.7 |
| BERT-base-cased | 80.0 | 67.7 |
| SEC-BERT-SHAPE + CRF | 80.4 | 66.6 |
| SEC-BERT-SHAPE | 81.1 | 68.9 |
| Ceiling of this LLM setup (gold answer for every asked number) | 95.7 | 90.7 |

Findings: the few-shot LLM (39.4 micro-F1) beats only the trivial baselines (most-
frequent, HMM) and is far below every trained model; the best trained model
(SEC-BERT-SHAPE, 81.1) roughly doubles it. The LLM was asked about 1,168 numbers and
gave 33 invalid types; since 89.8% of gold entity words are numbers, the ceiling row
shows that even a perfect answer to every asked number tops out at 95.7, i.e. the
format itself caps the setup. Limitations: one prompt, one model, one sample;
macro-F1 on a few hundred sentences is noisy, and the LLM is not finetuned while every
other model was trained on 261,610 sentences.

## 7. Label-conflict analysis

I checked how often FiNER-139 labels the *same* sentence (identical tokens) in more
than one way, which is label noise in the source data (`src/label_conflicts.py`;
nothing is retrained and `data/cleaned/` is unchanged). Because deduplication keeps
the first copy, only one version of each survives.

| Split | sentences in data/cleaned | conflicting sentences kept | share |
|---|---:|---:|---:|
| train | 261,610 | 725 | 0.28% |
| validation | 101,125 | 8 | 0.01% |
| test | 95,030 | 29 | 0.03% |

Kind of disagreement: `entity vs O` (one copy tags a number, another leaves it
untagged) 624; `entity type` (both tag it, different XBRL types) 127; both 11. The
kept (first) copy is the most frequent version, or tied, in 654 of 762 sentences; in
477 the top versions are tied, so a majority vote could not decide the label. The
types most often involved are the debt- and derivative-related ones
(DebtInstrumentInterestRateStatedPercentage 76, LineOfCreditFacilityMaximumBorrowing-
Capacity 67, DerivativeNotionalAmount 65).

Effect on the test scores: removing the 29 conflicting test sentences moves every
model's micro- and macro-F1 by at most ~0.03 points (e.g. SEC-BERT-SHAPE 72.95 ->
72.98 micro). So this is genuine label noise but too small a share to distort the
evaluation; that is why no retraining without them was done. Only exact duplicates
are detected — near-duplicates with conflicting labels are not counted.

## Files I created

- `01_data_preprocessing.ipynb` — EDA, cleaning, tokenization + label alignment
- `README.md` — task, dataset, EDA facts, before/after cleaning table, evaluation note
- `requirements.txt`, `.gitignore`, repository structure
- `data/raw/`, `data/cleaned/`, `data/processed/` — three-stage Parquet data
- `LICENSE` (MIT, code) and `data/LICENSE` (CC BY-SA 4.0, dataset attribution)
- `src/llm_fewshot.py` and `results/llm/` — LLM few-shot comparison
- `src/label_conflicts.py` and `results/label_conflicts/` — label-conflict analysis

## Commit evidence

Commits are on `main` of
[tia88na/finer139-token-classification](https://github.com/tia88na/finer139-token-classification):

- [Initial commit (repository and README)](https://github.com/tia88na/finer139-token-classification/commit/33d33bb9e6008e3603ee8f44116f1280389ab151)
- [Add preprocessing notebook, project files, and dataset](https://github.com/tia88na/finer139-token-classification/commit/3c351a2d370396f854b35dc194b0ee211cc33478)
- [Add data cleaning step (dedup, leakage removal, O-only reduction)](https://github.com/tia88na/finer139-token-classification/commit/288d1eed560a65ee50cdba48f6c6243651cad3a1)
- [Add MIT code license and CC BY-SA 4.0 data license](https://github.com/tia88na/finer139-token-classification/commit/899c562608683ff179389df70d0c235bd1e3e338)
- [Revise README for clarity and additional details](https://github.com/tia88na/finer139-token-classification/commit/57c37e9620b19ebfc4ecb885199294c549b2b491)
- [Add numeric-ratio and number-fragmentation EDA cells](https://github.com/tia88na/finer139-token-classification/commit/59c503f8fe2b74f8c249d35d36462979f2d01bb2)
- [Add LLM few-shot comparison (Qwen2.5-7B-Instruct, 300 test sentences)](https://github.com/tia88na/finer139-token-classification/commit/dbd86e2563a060a8242cfb2075a39ef7b07485ac)
- [Add label conflict analysis (identical sentences with different labels)](https://github.com/tia88na/finer139-token-classification/commit/273568bd963307ac53518dbc65ec735f7ecefdfa)
