# Bootstrap comparison

10,000 paired bootstrap replicates; seed 42; 92,608 main-test sentences. Strict IOB2 entity micro-F1, in percentage points.

| Model | F1 | 95% percentile interval |
|---|---:|---|
| bilstm_crf | 71.45 | [70.87, 72.01] |
| bert-base-cased | 70.44 | [69.88, 71.01] |
| sec-bert-shape | 72.95 | [72.41, 73.50] |

| A minus B | Difference | 95% interval | Family-wise 95% interval (3 comparisons) |
|---|---:|---|---|
| sec-bert-shape minus bert-base-cased | 2.51 | [2.16, 2.85] | [2.09, 2.94] |
| sec-bert-shape minus bilstm_crf | 1.51 | [1.07, 1.94] | [0.98, 2.04] |
| bilstm_crf minus bert-base-cased | 1.00 | [0.57, 1.44] | [0.47, 1.52] |

## Method and interpretation

- Each replicate draws N sentences with replacement and uses the same draw for every model. Micro-F1 is recomputed from pooled TP, gold and predicted entity counts; sentence F1 values are not averaged.
- Identical count patterns are grouped. Multinomial sampling of their multiplicities is distributionally equivalent to ordinary sentence resampling. All-O sentences remain in the population.
- The difference interval is computed directly from paired replicate differences. Overlapping individual model intervals are not used as a significance test.
- Marginal 95% intervals use 2.5/97.5 percentiles. The three comparison intervals also include Bonferroni-adjusted 0.8333/99.1667 percentiles for family-wise 95% coverage.
- Intervals describe sentence-sampling uncertainty conditional on these saved predictions and labels. They do not measure training-seed variability or annotation uncertainty.
- Sentences are treated as independent. Correlation within financial filings may make intervals optimistic; no filing-level grouping was available in the analysis inputs. IDs were not assumed to be document IDs.
- Observed test results have already been inspected. These are descriptive comparisons, not evidence of unbiased performance for a new method selected against this test set.
- Model scores reproduce the shared saved main-view results. Original predictions and gold labels are unchanged.

Reproduce: `python src/assess_results.py --replicates 10000 --sample-size 50 --seed 42`
