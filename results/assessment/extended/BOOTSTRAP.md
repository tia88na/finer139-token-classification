# Extended bootstrap comparison

10,000 paired sentence-bootstrap replicates; seed 42; 92,608 main-test sentences. Strict IOB2 entity micro-F1, reported in percentage points.

| Model | F1 | 95% percentile interval |
|---|---:|---|
| bilstm_crf | 71.45 | [70.88, 72.02] |
| bert-base-cased | 70.44 | [69.87, 71.02] |
| sec-bert-shape | 72.95 | [72.39, 73.50] |
| sec-bert-shape-crf | 73.52 | [72.97, 74.08] |
| bert-base-cased-4ep | 73.34 | [72.79, 73.88] |
| sec-bert-shape-4ep | 75.93 | [75.40, 76.45] |

| A minus B | Difference | 95% interval | Family-wise 95% interval (8 comparisons) |
|---|---:|---|---|
| sec-bert-shape minus bert-base-cased | 2.51 | [2.16, 2.85] | [2.01, 2.99] |
| sec-bert-shape minus bilstm_crf | 1.51 | [1.08, 1.94] | [0.90, 2.12] |
| bilstm_crf minus bert-base-cased | 1.00 | [0.57, 1.43] | [0.41, 1.60] |
| sec-bert-shape-crf minus sec-bert-shape | 0.57 | [0.34, 0.80] | [0.25, 0.89] |
| bert-base-cased-4ep minus bert-base-cased | 2.90 | [2.55, 3.25] | [2.42, 3.37] |
| sec-bert-shape-4ep minus sec-bert-shape | 2.97 | [2.63, 3.32] | [2.50, 3.45] |
| sec-bert-shape-4ep minus bert-base-cased-4ep | 2.58 | [2.22, 2.95] | [2.07, 3.08] |
| sec-bert-shape-4ep minus sec-bert-shape-crf | 2.40 | [2.06, 2.73] | [1.93, 2.87] |

## Method and interpretation

- The same N sentences are drawn with replacement for every model in each replicate. Micro-F1 is recomputed from pooled TP, gold and predicted counts, not averaged sentence F1.
- Reuses assess_results.paired_bootstrap: identical sufficient-statistic rows are grouped and drawn with multinomial multiplicities. This is distributionally equivalent to ordinary sentence resampling. All-O sentences remain in the population.
- Every model reproduces the saved main-view precision, recall and micro-F1 before resampling. Rows are aligned by the repository prediction-file convention (test row order); lengths and tag IDs are checked. Equal-length row permutations cannot be ruled out from these files alone.
- Eight named contrasts form one comparison family. Marginal intervals use 2.5/97.5 percentiles; Bonferroni-adjusted intervals use 0.3125/99.6875 percentiles for nominal family-wise 95% coverage. Finite bootstrap draws approximate these tail quantiles.
- Difference intervals come directly from paired replicate differences. Individual-model interval overlap is not a significance test.
- Intervals are conditional on the saved predictions and gold labels. They do not cover training-seed variability or annotation uncertainty. All training runs used one seed.
- Sentences are treated as independent; dependence within a financial filing may make intervals optimistic. Document-level grouping was not available in these analysis inputs.
- The test point estimates were already inspected. These are descriptive comparisons, not a preregistered confirmatory test or an unbiased estimate after model selection.
- Four versus two epochs compares the saved training-budget configurations, not equal wall-clock time. BERT versus SEC-BERT-SHAPE changes both domain pretraining and number representation. The 4-epoch SHAPE versus 2-epoch SHAPE+CRF contrast changes both budget and decoder; it does not isolate the CRF effect.
- The original three-model bootstrap and 50-example assessment remain unchanged in the parent directory. New joint count patterns change the seeded draws, so rerun marginal intervals can differ slightly; family-wise intervals now adjust for eight rather than three contrasts.
- No model was trained, gold label edited, or original prediction overwritten.

Reproduce: `python src/compare_models_bootstrap.py --replicates 10000 --seed 42`
