"""Compare six saved models without changing the original 50-example assessment.

Run: python src/compare_models_bootstrap.py --replicates 10000 --seed 42
Outputs: results/assessment/extended/. No training or model weights required.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_errors import entities
from assess_results import digest, paired_bootstrap, pooled_f1
from finer import ID2LABEL

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    'bilstm_crf', 'bert-base-cased', 'sec-bert-shape',
    'sec-bert-shape-crf', 'bert-base-cased-4ep', 'sec-bert-shape-4ep',
)
# Fixed before this bootstrap run, after the test point estimates were inspected.
# One family of eight comparisons, including the original three.
PAIRS = (
    ('sec-bert-shape', 'bert-base-cased'),
    ('sec-bert-shape', 'bilstm_crf'),
    ('bilstm_crf', 'bert-base-cased'),
    ('sec-bert-shape-crf', 'sec-bert-shape'),
    ('bert-base-cased-4ep', 'bert-base-cased'),
    ('sec-bert-shape-4ep', 'sec-bert-shape'),
    ('sec-bert-shape-4ep', 'bert-base-cased-4ep'),
    ('sec-bert-shape-4ep', 'sec-bert-shape-crf'),
)


def comparison_rows(samples, point, models=MODELS, pairs=PAIRS):
    """Named contrasts prevent column-order mistakes; all share one family size."""
    rows = []
    tail = .05 / (2 * len(pairs))
    for a, b in pairs:
        ai, bi = models.index(a), models.index(b)
        delta = 100 * (samples[:, ai] - samples[:, bi])
        lo, hi = np.quantile(delta, [.025, .975])
        adjlo, adjhi = np.quantile(delta, [tail, 1-tail])
        rows.append(dict(model_a=a, model_b=b,
                         delta_f1_points=100*(point[ai]-point[bi]),
                         ci95_low=lo, ci95_high=hi,
                         family95_low=adjlo, family95_high=adjhi))
    return rows


def write_report(out, meta, model_rows, differences):
    lines = ['# Extended bootstrap comparison', '',
             f'{meta["replicates"]:,} paired sentence-bootstrap replicates; seed {meta["seed"]}; '
             f'{meta["main_sentences"]:,} main-test sentences. Strict IOB2 entity micro-F1, '
             'reported in percentage points.', '',
             '| Model | F1 | 95% percentile interval |', '|---|---:|---|']
    for r in model_rows:
        lines.append(f'| {r["model"]} | {r["micro_f1"]:.2f} | '
                     f'[{r["ci95_low"]:.2f}, {r["ci95_high"]:.2f}] |')
    lines += ['', f'| A minus B | Difference | 95% interval | Family-wise 95% interval ({len(PAIRS)} comparisons) |',
              '|---|---:|---|---|']
    for r in differences:
        lines.append(f'| {r["model_a"]} minus {r["model_b"]} | {r["delta_f1_points"]:.2f} | '
                     f'[{r["ci95_low"]:.2f}, {r["ci95_high"]:.2f}] | '
                     f'[{r["family95_low"]:.2f}, {r["family95_high"]:.2f}] |')
    lines += ['', '## Method and interpretation', '',
              '- The same N sentences are drawn with replacement for every model in each replicate. '
              'Micro-F1 is recomputed from pooled TP, gold and predicted counts, not averaged sentence F1.',
              '- Reuses assess_results.paired_bootstrap: identical sufficient-statistic rows are grouped '
              'and drawn with multinomial multiplicities. This is distributionally equivalent to ordinary '
              'sentence resampling. All-O sentences remain in the population.',
              '- Every model reproduces the saved main-view precision, recall and micro-F1 before resampling. '
              'Rows are aligned by the repository prediction-file convention (test row order); lengths and '
              'tag IDs are checked. Equal-length row permutations cannot be ruled out from these files alone.',
              '- Eight named contrasts form one comparison family. Marginal intervals use 2.5/97.5 '
              'percentiles; Bonferroni-adjusted intervals use 0.3125/99.6875 percentiles for nominal '
              'family-wise 95% coverage. Finite bootstrap draws approximate these tail quantiles.',
              '- Difference intervals come directly from paired replicate differences. Individual-model '
              'interval overlap is not a significance test.',
              '- Intervals are conditional on the saved predictions and gold labels. They do not cover '
              'training-seed variability or annotation uncertainty. All training runs used one seed.',
              '- Sentences are treated as independent; dependence within a financial filing may make '
              'intervals optimistic. Document-level grouping was not available in these analysis inputs.',
              '- The test point estimates were already inspected. These are descriptive comparisons, '
              'not a preregistered confirmatory test or an unbiased estimate after model selection.',
              '- Four versus two epochs compares the saved training-budget configurations, not equal '
              'wall-clock time. BERT versus SEC-BERT-SHAPE changes both domain pretraining and number '
              'representation. The 4-epoch SHAPE versus 2-epoch SHAPE+CRF contrast changes both budget '
              'and decoder; it does not isolate the CRF effect.',
              '- The original three-model bootstrap and 50-example assessment remain unchanged in the '
              'parent directory. New joint count patterns change the seeded draws, so rerun marginal '
              'intervals can differ slightly; family-wise intervals now adjust for eight rather than three contrasts.',
              '- No model was trained, gold label edited, or original prediction overwritten.', '',
              'Reproduce: `python src/compare_models_bootstrap.py --replicates 10000 --seed 42`', '']
    (out/'BOOTSTRAP.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replicates', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.replicates < 1:
        ap.error('--replicates must be positive')
    test_path = ROOT/'data/cleaned/test.parquet'
    val_path = ROOT/'data/cleaned/validation.parquet'
    test = pd.read_parquet(test_path, columns=['tokens', 'ner_tags'])
    validation = pd.read_parquet(val_path, columns=['tokens'])
    signatures = {tuple(t) for t in validation.tokens}
    keep = np.array([i for i, t in enumerate(test.tokens) if tuple(t) not in signatures])
    del validation, signatures
    gold = [entities([ID2LABEL[int(t)] for t in test.ner_tags.iloc[i]]) for i in keep]
    counts = np.zeros((len(keep), len(MODELS), 3), dtype=np.int64)
    paths = [test_path, val_path, Path(__file__), ROOT/'src/assess_results.py',
             ROOT/'src/analyze_errors.py', ROOT/'src/finer.py']
    for m, model in enumerate(MODELS):
        path = ROOT/'results'/f'{model}_test_predictions.parquet'
        eval_path = ROOT/'results'/f'{model}_eval.json'
        paths.extend([path, eval_path])
        pred = pd.read_parquet(path).predicted_tags
        if len(pred) != len(test):
            raise ValueError(f'{model}: row count mismatch')
        for i, raw in enumerate(pred):
            if len(raw) != len(test.tokens.iloc[i]) or any(int(t) < 0 or int(t) >= len(ID2LABEL) for t in raw):
                raise ValueError(f'{model}: invalid prediction row {i}')
        for j, i in enumerate(keep):
            ps = entities([ID2LABEL[int(t)] for t in pred.iloc[i]])
            gs = gold[j]
            counts[j, m] = len(gs & ps), len(gs), len(ps)
        tp, ng, npred = counts[:, m].sum(axis=0)
        observed = {'micro_P': tp/npred if npred else 0.,
                    'micro_R': tp/ng if ng else 0.,
                    'micro_F1': float(pooled_f1([tp, ng, npred]))}
        saved = json.loads(eval_path.read_text())['main']
        if len(keep) != saved['sentences'] or any(round(v, 4) != saved[k] for k, v in observed.items()):
            raise AssertionError(f'{model}: does not reproduce saved main results')
        print(f'{model}: checked {len(keep)} sentences, F1={observed["micro_F1"]:.6f}', flush=True)

    samples, patterns = paired_bootstrap(counts, args.replicates, args.seed)
    point = pooled_f1(counts.sum(axis=0))
    lows, highs = np.quantile(samples, [.025, .975], axis=0)
    model_rows = [dict(model=m, micro_f1=100*point[j], ci95_low=100*lows[j], ci95_high=100*highs[j])
                  for j, m in enumerate(MODELS)]
    differences = comparison_rows(samples, point)
    meta = dict(seed=args.seed, replicates=args.replicates, main_sentences=len(keep),
                models=list(MODELS), comparisons=[list(p) for p in PAIRS],
                family_size=len(PAIRS), grouped_patterns=patterns, numpy_version=np.__version__,
                bootstrap_unit='sentence; paired across six models; multinomial grouped sufficient statistics',
                file_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in paths})
    out = ROOT/'results/assessment/extended'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(model_rows).to_csv(out/'bootstrap_models.csv', index=False)
    pd.DataFrame(differences).to_csv(out/'bootstrap_differences.csv', index=False)
    sentence_rows = {'test_row': keep}
    for m, model in enumerate(MODELS):
        for j, key in enumerate(('tp', 'gold', 'predicted')):
            sentence_rows[f'{model}_{key}'] = counts[:, m, j]
    pd.DataFrame(sentence_rows).to_csv(out/'sentence_counts.csv', index=False)
    pd.DataFrame(100*samples, columns=MODELS).to_csv(out/'bootstrap_replicates.csv', index=False)
    (out/'method.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    write_report(out, meta, model_rows, differences)
    print(f'Finished: {args.replicates} replicates, {patterns} grouped patterns; {out}', flush=True)


if __name__ == '__main__':
    main()
