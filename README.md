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
  analysis, the original and extended bootstrap, the label families, the ensemble,
  and the latency/ablation helpers. The latest bootstrap update passed 18 targeted
  evaluator/HMM and analysis tests; this is not a claim that the full GPU suite was rerun.

## Models

| Model | What it sees | Code |
|---|---|---|
| Most-frequent tag | the token only | `src/baseline_hmm.py` (CPU) |
| HMM + Viterbi | the token + the previous tag | `src/baseline_hmm.py` (CPU, about 7 min) |
| CRF | hand-made features of the word and of the 3 words on each side (shape, number type, word pairs, 8 previous words) + tag transitions | `src/baseline_crf.py` (GPU, about 75 min with the search) |
| BiLSTM-CRF | whole sentence (BiLSTM over word, character and casing embeddings) + tag transitions | `src/bilstm_crf.py` (GPU, about 2 h with the search) |
| BERT-base-cased | whole sentence, numbers split into sub-word pieces; 2 epochs | `train_transformer22.py` (GPU) |
| SEC-BERT-SHAPE | whole sentence, numbers replaced by their shape (`645.1` -> `[XXX.X]`); 2 epochs | `train_transformer22.py --model nlpaueb/sec-bert-shape` (GPU) |
| SEC-BERT-SHAPE + CRF | shape-normalised transformer emissions plus CRF tag transitions; 2 epochs | `train_transformer22.py --model nlpaueb/sec-bert-shape --crf` (GPU) |
| BERT-base-cased (4 epochs) | the BERT baseline with a larger training budget | `train_transformer22.py --model bert-base-cased --epochs 4` (GPU) |
| SEC-BERT-SHAPE (4 epochs) | the shape-normalised model with a larger training budget | `train_transformer22.py --model nlpaueb/sec-bert-shape --epochs 4` (GPU) |

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
| SEC-BERT-SHAPE + CRF | 66.4 | 82.3 | **73.5** | 70.0 | 73.5 |
| BERT-base-cased (4 epochs) | 66.7 | 81.5 | **73.3** | 70.4 | 73.3 |
| SEC-BERT-SHAPE (4 epochs) | 70.1 | 82.8 | **75.9** | 73.2 | 75.9 |

Why the token-only models fail (`results/hmm_analysis.txt`): a number's shape does
not decide its tag. In train, the shape `XX` occurs 306,226 times, is `O` 91.8% of
the time and belongs to 134 different entity types. `P(O -> O) = 0.981`, so the
Viterbi path almost never leaves `O`: the HMM predicts `O` for 91.8% of the gold
entity words. The words that decide the tag ("interest rate of", "matures in") are
not visible to an HMM emission. Models that read the context reach 59.0 (CRF with
hand-made context features), 71.5 (BiLSTM-CRF) and 70.4-75.9 micro-F1 (the transformer runs).
On the main view the HMM predicts `O` for 91.8% of the gold entity words, the CRF
for 23.7%, and the BiLSTM-CRF for 12.5%.

The best saved run is **SEC-BERT-SHAPE (4 epochs): 75.9 micro-F1 and
73.2 macro-F1**. At four epochs, BERT scores 73.3 / 70.4. From the unrounded
predictions, SHAPE leads BERT by 2.58 micro-F1 points at this epoch budget;
four versus two epochs adds 2.90 points for BERT and 2.97 for SHAPE.
Adding CRF to the two-epoch SHAPE model adds 0.57 points. Confidence intervals
are reported below. Equal epoch counts do not imply equal wall-clock time,
and these single-seed results do not isolate domain pretraining from number shapes.

## Error analysis

Run `python src/analyze_errors.py` to compare saved BiLSTM-CRF predictions
and the original two-epoch BERT/SEC-BERT-SHAPE runs without training or model weights.
This error analysis does not include the CRF transformer or four-epoch runs. Outputs are in
`results/analysis/`: `ANALYSIS.md`, overall and per-class scores, sentence-length
and training-frequency breakdowns, word-level type confusions, and illustrative
examples with original test row IDs. Analysis uses only the main test view and
checks its precision/recall/F1 against the saved shared-evaluator results.

For **SEC-BERT-SHAPE (2 epochs)**, precision is 66.23% and recall is 81.19%.
The analysis counts 8,222 spurious predictions and 1,857 completely missed
gold entities: about 4.4 times as many spurious as completely missed entities.
This compares two error categories, not total false positives versus total
false negatives; wrong-type and boundary errors are counted separately.
These figures do not describe the best four-epoch SHAPE run.

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
different operations; their effects are measured separately in the cleaning
ablation below. All training runs use one seed. Slides and speech notes are
maintained outside the repo.

## Bootstrap and sample assessment

### Current six-model comparison

Run `python src/compare_models_bootstrap.py --replicates 10000 --seed 42`.
The [extended bootstrap report](results/assessment/extended/BOOTSTRAP.md) compares
BiLSTM-CRF, the original two-epoch BERT and SEC-BERT-SHAPE, SHAPE+CRF,
and both four-epoch transformers on 92,608 main-test sentences.
It reproduces each model's saved precision, recall and micro-F1 before computing
10,000 paired sentence-bootstrap replicates. Source hashes, sentence counts and
replicate scores are retained in `results/assessment/extended/`.

The saved bootstrap reports do not provide model-level confidence intervals
for MFT, HMM or the feature CRF. Report their benchmark point estimates
without an interval unless a separate, reproducible bootstrap result is
available. The feature-CRF cleaning-ablation intervals describe differences
between cleaning conditions, not confidence intervals for these baseline scores.

| Comparison | Micro-F1 gain (points) | 95% interval | Family-wise 95% interval |
|---|---:|---|---|
| SHAPE+CRF minus SHAPE, both 2 epochs | +0.57 | [0.34, 0.80] | [0.25, 0.89] |
| BERT: 4 minus 2 epochs | +2.90 | [2.55, 3.25] | [2.42, 3.37] |
| SHAPE: 4 minus 2 epochs | +2.97 | [2.63, 3.32] | [2.50, 3.45] |
| SHAPE minus BERT, both 4 epochs | +2.58 | [2.22, 2.95] | [2.07, 3.08] |

The report contains eight contrasts in total; the family-wise intervals adjust
for all eight with Bonferroni correction. All eight adjusted intervals are above
zero on these saved predictions. They describe test-sentence sampling uncertainty,
not training-seed variability or annotation uncertainty. Sentences are treated as
independent, which can underestimate uncertainty within filings. The test results
had already been inspected, so these are descriptive comparisons.

### Original three-model comparison and 50-example assessment

The original report and sample remain unchanged and concern the two-epoch
transformers and BiLSTM-CRF; they are separate from the extended comparison.

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

## Inference speed

`python src/latency.py --device cuda` and `--device cpu` time the six original
model architectures on the same machine (Colab: Tesla T4 GPU and its CPU with
1 thread), using 2,000 main-view test sentences (96,002 words). Input preparation,
model execution and decoding are included; the reported throughput is based on
the median of three runs after a warm-up. The `ms / sentence` column in the full
report is amortised batch time, not a batch-size-one request-latency percentile.

MFT and HMM use fitted weights. CRF and BiLSTM-CRF use untrained models built with
the selected settings and training vocabulary; the transformers use pretrained
encoders with new 279-tag heads. These are architecture-level speed measurements,
not measurements of the saved fine-tuned checkpoints. See
[the full speed report](results/LATENCY.md) for weight provenance and
`05_latency_and_ablation_colab.ipynb` for the workflow.

| Model | micro-F1 | sentences / s (T4 setup) | ms / sentence (T4 setup) | sentences / s (CPU, 1 thread) | parameters |
|---|---:|---:|---:|---:|---:|
| Most-frequent tag | 3.1 | 33,046 | 0.030 | 35,751 | 0 |
| HMM + Viterbi | 3.5 | 223 | 4.480 | 214 | 4.5 M |
| CRF | 59.0 | 5,344 | 0.187 | 203 | 116.2 M |
| BiLSTM-CRF | 71.5 | 4,704 | 0.213 | 142 | 3.6 M |
| BERT-base-cased | 70.4 | 679 | 1.472 | 5.7 | 107.9 M |
| SEC-BERT-SHAPE | 73.0 | 819 | 1.221 | 9.2 | 108.7 M |
| SEC-BERT-SHAPE + CRF | 73.5 | Not measured | Not measured | Not measured | — |
| BERT-base-cased (4 epochs) | 73.3 | 679† | 1.472† | 5.7† | 107.9 M |
| SEC-BERT-SHAPE (4 epochs) | 75.9 | 819† | 1.221† | 9.2† | 108.7 M |

`sentences / s` is throughput; `ms / sentence` is amortised batch processing
time, including input preparation and decoding, copied from
`results/LATENCY.md`. It is not single-request latency. MFT and HMM run on
the CPU even in the T4 setup.

† Reused architecture-level measurements from the corresponding two-epoch
rows, not separate measurements of the four-epoch trained checkpoints.
Matching architecture and parameter count make these useful speed proxies;
they do not establish the exact runtime of each final checkpoint.
SHAPE+CRF adds Viterbi decoding and cannot inherit the SHAPE speed.
Its measurement code and `06_latency_shape_crf_colab.ipynb` are available,
but no measured result is included in the current speed report.

- **Speed versus F1.** The measured BiLSTM-CRF architecture is about 5.7 times
  faster than the SHAPE architecture on the T4 (15 times on this CPU), the same
  architecture as the best four-epoch SHAPE run. Its benchmark F1 is 4.5 points
  below that run. Model selection and training budgets differ;
  this is a practical trade-off, not an equal-compute comparison.
- In this setup SHAPE is faster than BERT (1.2 times on GPU, 1.6 on CPU).
  Shorter number representations are a possible explanation; their contribution
  to runtime was not isolated in an experiment.
- HMM decoding runs on the CPU even in the table labelled T4; it does not use the GPU.

## Cleaning ablation

**Scope: all results in this section are for the feature CRF only.** The other
models were not retrained on the alternative training sets, so these effects are
not claimed for them.

`python src/cleaning_ablation.py` (Colab T4, ~70 min) trains the team's feature CRF
(selected settings: window 3, weight decay 0) three times, changing only the training
data. It first rebuilds the notebook's cleaning steps from `data/raw/` and checks that
the result is identical to `data/cleaned/train.parquet`. Main view, strict IOB2;
differences with 10,000 paired sentence-bootstrap replicates. Details:
`results/ablation/ABLATION.md`.

| Training data | sentences | no-entity share | P | R | micro-F1 | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| A cleaned (used by all models) | 261,610 | 49% | 57.5 | 60.6 | **59.0** | 52.9 |
| B = A + the dropped no-entity sentences | 562,568 | 76% | 63.9 | 54.9 | **59.0** | 52.5 |
| C = A + the leaked sentences (also in val/test) | 302,709 | 54% | 60.9 | 61.6 | **61.3** | 56.3 |

- **No-entity downsampling (B vs A), feature CRF: precision for recall, same F1.** Keeping all
  no-entity sentences raises precision by 6.3 points and lowers recall by 5.7;
  micro-F1 changes by -0.02 (95% interval [-0.47, +0.42]), providing no clear
  evidence of an F1 difference in this feature-CRF experiment. The precision/recall
  shift supports the proposed direction for this model; it was not tested for the
  transformers. The recorded training time falls from 30 to 11 minutes.
  A reproduces the CRF's published score exactly (59.0, validation 0.6317).
- **Leakage removal (C vs A), feature CRF: without it the CRF's score would be inflated.** 17,992 of the 92,608
  main-view test sentences (19%) occur word for word among the train sentences that
  the leakage step removed. Training on them raises micro-F1 by 2.2 points
  (95% interval [+1.76, +2.68]); on those 17,992 sentences F1 jumps from 62.0 to 76.5,
  consistent with an advantage from seeing evaluation sentences during training.
  C's validation sample is contaminated too (4,516 of its 20,000 sentences), so
  both training exposure and epoch selection are affected. C is a leakage
  demonstration, not a valid improved benchmark result. The size of the inflation
  for the BiLSTM-CRF and the transformers was not measured.

## Additional saved analyses

- [Conflicting labels](results/label_conflicts/CONFLICTS.md): 762 retained sentences
  have exact raw copies with different annotations (725 train, 8 validation, 29 test).
  Removing the 29 conflicting main-test sentences changes the reported models'
  scores only slightly. This analysis predates the four-epoch runs.
- [LLM comparison](results/llm/LLM.md): Qwen2.5-7B-Instruct with three training
  examples in the prompt scores 39.4 micro-F1 on a 300-sentence test subset;
  the two-epoch SHAPE model scores 81.1 on that same subset. These are not full-test
  scores. The LLM is not fine-tuned, is asked only about detected numbers, and
  this one-model, one-prompt experiment does not establish a general LLM ranking.

## Repository structure

    finer139-token-classification/
    |- README.md
    |- LICENSE                      # MIT (code)
    |- CONTRIBUTORS.md
    |- requirements.txt
    |- .gitignore
    |- 01_data_preprocessing.ipynb  # EDA + cleaning + tokenization
    |- train_transformer22.py       # BERT / SEC-BERT-SHAPE training + evaluation
    |- 05_latency_and_ablation_colab.ipynb  # runs latency.py + cleaning_ablation.py on Colab
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
    |  |- analyze_errors.py         # original three-model error analysis
    |  |- assess_results.py         # original bootstrap and 50-example sample
    |  |- compare_models_bootstrap.py # six-model bootstrap and eight comparisons
    |  |- render_assessment.py      # renders the saved sample assessment
    |  |- label_conflicts.py        # exact-sentence annotation inconsistencies
    |  |- llm_fewshot.py            # small Qwen few-shot comparison
    |  |- latency.py                # speed of the six original architectures
    |  |- cleaning_ablation.py      # CRF trained on cleaned / + no-entity / + leaked train data
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
