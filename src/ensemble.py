"""Exploratory majority-vote ensemble of saved test predictions; no model is retrained or re-run.

Run from the repository root: python src/ensemble.py --replicates 10000 --seed 42
Voters are the saved word-level predictions of BiLSTM-CRF, BERT-base-cased and SEC-BERT-SHAPE.
The rule was fixed before it was applied to the test predictions and is not tuned:

1. Each voter's tags are rebuilt from its strict IOB2 spans, so an orphan I- tag (one that
   strict seqeval ignores) becomes O. Every voter then votes with exactly what it is scored on.
2. Every word gets the full BIO tag (O included) that at least two of the three voters chose;
   if all three differ, SEC-BERT-SHAPE's tag.
3. An I-X that does not follow B-X or I-X becomes B-X, so the output is valid IOB2.

Only test predictions exist, so the comparison with SEC-BERT-SHAPE is exploratory, not unbiased
confirmation. Outputs go to results/ensemble/, not results/, so src/evaluate_predictions.py and
results/RESULTS.md are unaffected. Original predictions and scores are only read.
"""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from analyze_errors import MODELS, entities
from assess_results import digest, paired_bootstrap, pooled_f1
from evaluate_predictions import NAMES
from finer import ID2LABEL, LABEL2ID, evaluate, load_splits

ROOT = Path(__file__).resolve().parents[1]
VOTERS = MODELS
TIE_BREAKER = 'sec-bert-shape'
KEY, NAME = 'majority_vote', 'Majority-vote ensemble'


def encode(spans, length):
    """IOB2 tags of `length` words for non-overlapping (start, exclusive end, type) spans."""
    tags = ['O'] * length
    for start, end, kind in spans:
        tags[start:end] = [f'B-{kind}'] + [f'I-{kind}'] * (end - start - 1)
    return tags


def canonical(tags):
    """The tags as strict IOB2 scoring reads them: orphan I- tags become O."""
    return encode(entities(tags), len(tags))


def is_valid(tags):
    """Valid IOB2: every I-X follows B-X or I-X, so rebuilding from the spans changes nothing."""
    return canonical(tags) == list(tags)


def vote(sequences, tie_breaker):
    """Per word, the tag of a strict majority of the sequences, otherwise sequences[tie_breaker]'s."""
    if len({len(s) for s in sequences}) != 1:
        raise ValueError('voters disagree on the sentence length')
    out = []
    for tags in zip(*sequences):
        tag, count = Counter(tags).most_common(1)[0]
        out.append(tag if 2 * count > len(tags) else tags[tie_breaker])
    return out


def repair(tags):
    """Valid IOB2: an I-X that does not follow B-X or I-X becomes B-X."""
    out, prev = [], 'O'
    for tag in tags:
        if tag.startswith('I-') and prev[2:] != tag[2:]:
            tag = 'B-' + tag[2:]
        out.append(tag)
        prev = tag
    return out


def combine(sequences, tie_breaker):
    """Ensemble tags of one sentence: canonical votes, majority, IOB2 repair."""
    return repair(vote([canonical(s) for s in sequences], tie_breaker))


def run(test, predictions, replicates=10000, seed=42, verbose=True):
    """Clean, vote, check and score.

    test is the test DataFrame of finer.load_splits() (tags, in_validation); predictions maps
    every voter to its tag lists, row i = test row i. Returns everything the outputs need.
    """
    tie = VOTERS.index(TIE_BREAKER)
    gold = list(test.tags)
    keep = np.flatnonzero(~test.in_validation.to_numpy())
    clean, base, cleaning = {}, {}, []
    for model in VOTERS:
        raw = predictions[model]
        clean[model] = [canonical(p) for p in raw]
        before = evaluate(test, raw, NAMES[model], verbose=False)
        after = evaluate(test, clean[model], NAMES[model], verbose=False)
        if before != after:
            raise AssertionError(f'{model}: cleaning orphan I- tags changed the strict scores')
        base[model] = after
        cleaning.append(dict(model=model, orphan_tags_to_O=sum(a != b for p, c in zip(raw, clean[model])
                                                               for a, b in zip(p, c)),
                             main_micro_F1_before=before['main']['micro_F1'], main_micro_F1_after=after['main']['micro_F1'],
                             full_micro_F1_before=before['reference']['micro_F1'],
                             full_micro_F1_after=after['reference']['micro_F1'], all_scores_identical=True))

    voted = [vote([clean[m][i] for m in VOTERS], tie) for i in range(len(gold))]
    ensemble = [repair(v) for v in voted]
    if not all(len(p) == len(g) and is_valid(p) for p, g in zip(ensemble, gold)):
        raise AssertionError('ensemble output is not valid IOB2')
    shape = clean[TIE_BREAKER]
    stats = Counter(sentences=len(keep))
    for i in keep:
        for tags in zip(*(clean[m][i] for m in VOTERS)):
            if any(t != 'O' for t in tags):
                stats['entity_words'] += 1
                stats[f'agree_{max(Counter(tags).values())}'] += 1
        differ = sum(a != b for a, b in zip(ensemble[i], shape[i]))
        stats['words'] += len(gold[i])
        stats['repairs'] += sum(a != b for a, b in zip(voted[i], ensemble[i]))
        stats['words_differ_from_tie_breaker'] += differ
        stats['sentences_differ_from_tie_breaker'] += differ > 0
    result = evaluate(test, ensemble, NAME, verbose=verbose)

    # Main-view entity counts per sentence: [ensemble, tie-breaker] x (TP, gold, predicted).
    counts = np.zeros((len(keep), 2, 3), dtype=np.int64)
    change = Counter()
    for j, i in enumerate(keep):
        g, e, s = entities(gold[i]), entities(ensemble[i]), entities(shape[i])
        counts[j] = [(len(g & e), len(g), len(e)), (len(g & s), len(g), len(s))]
        change['tp_gained'] += len((g & e) - s)
        change['tp_lost'] += len((g & s) - e)
        change['fp_added'] += len((e - g) - s)
        change['fp_removed'] += len((s - g) - e)
    totals = counts.sum(axis=0)
    point = pooled_f1(totals)
    for k, res in ((0, result), (1, base[TIE_BREAKER])):
        if round(float(point[k]), 4) != res['main']['micro_F1']:
            raise AssertionError('span counts disagree with the shared evaluator')
    samples, patterns = paired_bootstrap(counts, replicates, seed)
    lows, highs = np.quantile(samples, [.025, .975], axis=0)
    delta = 100 * (samples[:, 0] - samples[:, 1])
    lo, hi = np.quantile(delta, [.025, .975])
    bootstrap = dict(model_a=KEY, model_b=TIE_BREAKER, replicates=replicates, seed=seed, grouped_patterns=patterns,
                     f1_a=100 * float(point[0]), ci95_low_a=100 * float(lows[0]), ci95_high_a=100 * float(highs[0]),
                     f1_b=100 * float(point[1]), ci95_low_b=100 * float(lows[1]), ci95_high_b=100 * float(highs[1]),
                     delta_f1_points=100 * float(point[0] - point[1]), ci95_low=float(lo), ci95_high=float(hi))
    spans = {name: dict(tp=int(t[0]), gold=int(t[1]), predicted=int(t[2])) for name, t in zip((KEY, TIE_BREAKER), totals)}
    return dict(base=base, ensemble=result, predictions=ensemble, cleaning=cleaning, stats=dict(stats),
                change=dict(change), spans=spans, bootstrap=bootstrap)


def git_state():
    """Checked-out commit and whether src/ensemble.py differs from it; None without git."""
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        changed = subprocess.run(['git', 'status', '--porcelain', '--', 'src/ensemble.py'], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return dict(commit=commit, ensemble_script_modified=bool(changed))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replicates', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    output = ROOT / 'results' / 'ensemble'
    output.mkdir(exist_ok=True)
    inputs = [ROOT / 'data' / 'cleaned' / f'{split}.parquet' for split in ('test', 'validation')]
    inputs += [ROOT / 'results' / f'{model}_test_predictions.parquet' for model in VOTERS]
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in inputs}
    test = load_splits()['test']
    predictions = {}
    for model in VOTERS:
        frame = pd.read_parquet(ROOT / 'results' / f'{model}_test_predictions.parquet')
        if len(frame) != len(test):
            raise ValueError(f'{model}: prediction row count does not match test')
        rows = []
        for i, (raw, tokens) in enumerate(zip(frame.predicted_tags, test.tokens)):
            if len(raw) != len(tokens) or any(int(t) < 0 or int(t) >= len(ID2LABEL) for t in raw):
                raise ValueError(f'{model}: invalid prediction row {i}')
            rows.append([ID2LABEL[int(t)] for t in raw])
        predictions[model] = rows
    result = run(test, predictions, args.replicates, args.seed)
    for model in VOTERS:
        saved = json.loads((ROOT / 'results' / f'{model}_eval.json').read_text())
        for view in ('main', 'reference'):
            if any(result['base'][model][view][k] != v for k, v in saved[view].items()):
                raise AssertionError(f'{model}: does not reproduce the saved {view} scores')
    if {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in inputs} != hashes:
        raise AssertionError('an input file changed during the run')
    meta = dict(git=git_state(), file_sha256=hashes, python=platform.python_version(),
                numpy=np.__version__, pandas=pd.__version__, seqeval=version('seqeval'))
    write_outputs(output, result, meta)
    b = result['bootstrap']
    print(f'Ensemble minus SEC-BERT-SHAPE: {b["delta_f1_points"]:+.2f} micro-F1 points, '
          f'95% interval [{b["ci95_low"]:.2f}, {b["ci95_high"]:.2f}]')


def write_outputs(output, r, meta):
    """Predictions in the shared format plus scores, checks, vote counts, bootstrap and the report."""
    pd.DataFrame({'predicted_tags': [[LABEL2ID[t] for t in p] for p in r['predictions']]}).to_parquet(
        output / f'{KEY}_test_predictions.parquet')
    res = r['ensemble']
    (output / f'{KEY}_eval.json').write_text(json.dumps({k: v for k, v in res.items() if k != 'report'}, indent=2))
    pd.DataFrame([{'label': k, **v} for k, v in res['report'].items() if isinstance(v, dict)]).round(4).to_csv(
        output / f'{KEY}_per_class.csv', index=False)
    rows = []
    for key, res in [*r['base'].items(), (KEY, r['ensemble'])]:
        for view in ('main', 'reference'):
            rows.append(dict(model=key, view=view, **res[view]))
    pd.DataFrame(rows).to_csv(output / 'scores.csv', index=False)
    pd.DataFrame(r['cleaning']).to_csv(output / 'cleaning_check.csv', index=False)
    pd.DataFrame([dict(statistic=k, value=v) for k, v in {**r['stats'], **r['change']}.items()]).to_csv(
        output / 'vote_stats.csv', index=False)
    pd.DataFrame([r['bootstrap']]).to_csv(output / 'bootstrap.csv', index=False)
    method = dict(voters=list(VOTERS), tie_breaker=TIE_BREAKER,
                  rule=['rebuild each voter from its strict IOB2 spans (orphan I- tags -> O)',
                        'per word, the full BIO tag chosen by at least two voters; if all differ, the tie-breaker',
                        'I-X not after B-X or I-X -> B-X'],
                  primary_comparison='main-view micro-F1, ensemble minus tie-breaker, paired sentence bootstrap 95% '
                                     'percentile interval',
                  exploratory='only test predictions exist; rule fixed in advance, not tuned', **meta,
                  bootstrap=r['bootstrap'], vote_stats=r['stats'], span_changes=r['change'])
    (output / 'method.json').write_text(json.dumps(method, indent=2), encoding='utf-8')
    write_report(output, r, meta)


def write_report(output, r, meta):
    b, st, ch, spans = r['bootstrap'], r['stats'], r['change'], r['spans']
    git = meta.get('git') or {}
    shape, ens = r['base'][TIE_BREAKER]['main'], r['ensemble']['main']
    if git.get('commit') and not git.get('ensemble_script_modified'):
        fixed = (f'The rule and its tests were committed (`{git["commit"][:7]}`) before this run, '
                 'and the run used that committed version unchanged.')
    else:
        fixed = 'The run did not use a committed version of `src/ensemble.py`.'
    lines = ['# Majority-vote ensemble (exploratory)', '',
             '**Exploratory analysis.** Only saved test predictions exist, so the voting rule could not be chosen '
             f'or checked on validation data. It was fixed in advance and no alternative rule was tried. {fixed} '
             'The result is one descriptive comparison on a test set whose results had already been inspected; '
             'it is not unbiased confirmation that ensembling helps.', '',
             '## Method (fixed in advance)', '',
             '1. **Voters**: the saved word-level test predictions of BiLSTM-CRF, BERT-base-cased and SEC-BERT-SHAPE. '
             'No model was retrained or re-run.',
             '2. **Orphan tags**: each voter\'s tags are rebuilt from its strict IOB2 spans, so an I- tag that strict '
             'seqeval ignores (one that does not follow B-X or I-X of the same type) becomes O. Every voter then '
             'votes with exactly what it is scored on; its strict scores do not change (checked below).',
             '3. **Vote**: each word gets the full BIO tag, O included, that at least two of the three voters chose. '
             'If all three differ, SEC-BERT-SHAPE\'s tag is used.',
             '4. **Repair**: an I-X that does not follow B-X or I-X becomes B-X, keeping the type the vote chose.',
             f'5. **Validity**: every output sentence is checked to be valid IOB2 ({len(r["predictions"]):,} of '
             f'{len(r["predictions"]):,} sentences pass).',
             '6. **Scoring**: the shared evaluator `finer.evaluate` (strict IOB2, word level, full-length sentences; '
             'main view and full test).',
             '7. **Primary comparison**: main-view micro-F1, ensemble minus SEC-BERT-SHAPE, with a 95% percentile '
             f'interval from {b["replicates"]:,} paired sentence-bootstrap replicates (seed {b["seed"]}; '
             '`assess_results.paired_bootstrap`).', '',
             'Because SEC-BERT-SHAPE also breaks ties, the ensemble can differ from it only where BiLSTM-CRF and '
             'BERT-base-cased agree with each other against it, or where such a change makes a following I- tag '
             'need repair.', '',
             '## Result', '',
             '| Model | P | R | micro-F1 | macro-F1 | micro-F1 (full test) |', '|---|---:|---:|---:|---:|---:|']
    for key, res in [*r['base'].items(), (KEY, r['ensemble'])]:
        m, f = res['main'], res['reference']
        lines.append(f'| {NAMES.get(key, NAME)} | {100*m["micro_P"]:.2f} | {100*m["micro_R"]:.2f} | '
                     f'{100*m["micro_F1"]:.2f} | {100*m["macro_F1"]:.2f} | {100*f["micro_F1"]:.2f} |')
    d, lo, hi = b['delta_f1_points'], b['ci95_low'], b['ci95_high']
    if lo > 0:
        verdict = (f'The ensemble scores {d:.2f} micro-F1 points above SEC-BERT-SHAPE and the 95% interval excludes '
                   'zero. On this test set it improves the score; because the test set had already been inspected and '
                   'no validation check was possible, this is exploratory evidence, not confirmation.')
    elif hi < 0:
        verdict = (f'The ensemble scores {-d:.2f} micro-F1 points below SEC-BERT-SHAPE and the 95% interval excludes '
                   'zero: it does not improve the score.')
    else:
        verdict = (f'The difference is {d:+.2f} micro-F1 points and the 95% interval includes zero: '
                   'no improvement over SEC-BERT-SHAPE can be claimed.')
    lines += ['', '| Main-view micro-F1 | Value | 95% interval |', '|---|---:|---|',
              f'| {NAME} | {b["f1_a"]:.2f} | [{b["ci95_low_a"]:.2f}, {b["ci95_high_a"]:.2f}] |',
              f'| SEC-BERT-SHAPE | {b["f1_b"]:.2f} | [{b["ci95_low_b"]:.2f}, {b["ci95_high_b"]:.2f}] |',
              f'| **Ensemble minus SEC-BERT-SHAPE** | **{d:+.2f}** | [{lo:.2f}, {hi:.2f}] |', '',
              verdict + f' Relative to SEC-BERT-SHAPE, precision changes by '
              f'{100*(ens["micro_P"]-shape["micro_P"]):+.2f} points, recall by '
              f'{100*(ens["micro_R"]-shape["micro_R"]):+.2f} and macro-F1 by '
              f'{100*(ens["macro_F1"]-shape["macro_F1"]):+.2f}. Only the micro-F1 difference was specified in '
              'advance and has an interval.', '',
              '## Orphan-tag cleaning leaves the strict scores unchanged', '',
              'Every model was scored with the shared evaluator before and after its orphan I- tags were set to O. '
              'All scores (main and full test: P, R, micro-F1, macro-F1 and every per-class value) are identical, '
              "and they reproduce each model's saved `results/<model>_eval.json`.", '',
              '| Model | Orphan I- tags set to O | Main micro-F1 before / after | Full-test micro-F1 before / after |',
              '|---|---:|---|---|']
    for c in r['cleaning']:
        lines.append(f'| {NAMES[c["model"]]} | {c["orphan_tags_to_O"]:,} | {100*c["main_micro_F1_before"]:.2f} / '
                     f'{100*c["main_micro_F1_after"]:.2f} | {100*c["full_micro_F1_before"]:.2f} / '
                     f'{100*c["full_micro_F1_after"]:.2f} |')
    n = st.get('entity_words', 0)
    pct = lambda k: f'{st.get(k, 0):,} ({100*st.get(k, 0)/n:.1f}%)' if n else '0'
    lines += ['', '## How the votes went (main view)', '',
              f'- {st["sentences"]:,} sentences, {st["words"]:,} words; on {n:,} words at least one model '
              'predicts an entity tag. On all other words the three models agree on O.',
              f'- Of those {n:,} words: all three agree on {pct("agree_3")}, two agree on {pct("agree_2")}, '
              f'and all three differ (SEC-BERT-SHAPE decides) on {pct("agree_1")}.',
              f'- BIO repairs: {st.get("repairs", 0):,} words.',
              f'- The ensemble differs from SEC-BERT-SHAPE on {st.get("words_differ_from_tie_breaker", 0):,} words '
              f'in {st.get("sentences_differ_from_tie_breaker", 0):,} sentences.', '',
              '## Entity changes relative to SEC-BERT-SHAPE (main view)', '',
              '| | SEC-BERT-SHAPE | Ensemble |', '|---|---:|---:|']
    for label, f in (('True positives', lambda s: s['tp']), ('False positives', lambda s: s['predicted'] - s['tp']),
                     ('False negatives', lambda s: s['gold'] - s['tp']), ('Predicted entities', lambda s: s['predicted'])):
        lines.append(f'| {label} | {f(spans[TIE_BREAKER]):,} | {f(spans[KEY]):,} |')
    lines += ['', f'The ensemble finds {ch.get("tp_gained", 0):,} gold entities that SEC-BERT-SHAPE misses and misses '
              f'{ch.get("tp_lost", 0):,} that SEC-BERT-SHAPE finds. It removes {ch.get("fp_removed", 0):,} of '
              f'SEC-BERT-SHAPE\'s false positives and adds {ch.get("fp_added", 0):,} new ones.', '',
              '## Limitations', '',
              '- Exploratory and test-only: no validation predictions exist, so neither the rule nor the choice of '
              'tie-breaker could be checked on held-out data. One rule was fixed in advance; none was tuned.',
              '- Two of the three voters are BERT-base-size transformers fine-tuned with the same script '
              '(`train_transformer22.py`), so their errors are likely correlated; the vote is not three '
              'independent opinions.',
              '- The interval reflects sentence sampling conditional on these saved predictions. It does not cover '
              'training-seed variability (one seed per model) or annotation uncertainty, and it treats sentences '
              'as independent, as in `results/assessment/BOOTSTRAP.md`.',
              '- The ensemble needs all three models at prediction time; its cost was not measured.',
              '- Ensemble files are in `results/ensemble/`, not `results/`, so `src/evaluate_predictions.py` and '
              'the main table in `results/RESULTS.md` are unaffected.', '',
              f'Reproduce: `python src/ensemble.py --replicates {b["replicates"]} --seed {b["seed"]}`', '']
    (output / 'ENSEMBLE.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
