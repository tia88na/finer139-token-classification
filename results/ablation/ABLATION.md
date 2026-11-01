# Cleaning ablation (feature CRF)

Same model, settings, seed and model selection; only the training data changes. Scores: shared evaluator, strict IOB2, main view (92,608 test sentences). Bootstrap: 10,000 paired sentence replicates, seed 42. Produced by `python src/cleaning_ablation.py`.

| Variant | train sentences | no-entity share | P | R | micro-F1 | macro-F1 | epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| A  cleaned train (as used by all models) | 261,610 | 49% | 57.5 | 60.6 | **59.0** | 52.9 | 2 |
| B  A + dropped no-entity sentences | 562,568 | 76% | 63.9 | 54.9 | **59.0** | 52.5 | 3 |
| C  A + leaked sentences (also in val/test) | 302,709 | 54% | 60.9 | 61.6 | **61.3** | 56.3 | 6 |

| Comparison (main-view micro-F1) | difference | 95% interval |
|---|---:|---|
| B_plus_o_only minus A_cleaned | -0.02 | [-0.47, +0.42] |
| C_plus_leaked minus A_cleaned | +2.22 | [+1.76, +2.68] |

Main-view test sentences that are among the leaked train sentences: 17,992. Micro-F1 on this subset:

| Variant | micro-F1 on leaked subset |
|---|---:|
| A  cleaned train (as used by all models) | 62.0 |
| B  A + dropped no-entity sentences | 58.8 |
| C  A + leaked sentences (also in val/test) | 76.5 |

Limitations: one seed per variant; feature vocabularies are fitted on each variant's own train set, as they would be in practice; the bootstrap covers test-sentence sampling only, not training variability.
