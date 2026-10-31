"""Analyse saved word-level predictions; no training or model weights required.

Run from the repository root: python src/analyze_errors.py
All outputs go to results/analysis/; original predictions and scores are untouched.
"""
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from finer import ID2LABEL, scores

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('bilstm_crf', 'bert-base-cased', 'sec-bert-shape')


def entities(tags):
    """Strict IOB2 spans as (start, exclusive end, type); ignore orphan I tags."""
    spans, start, kind = set(), None, None
    for i, tag in enumerate(list(tags) + ['O']):
        prefix, _, label = tag.partition('-')
        continuation = prefix == 'I' and start is not None and label == kind
        if start is not None and not continuation:
            spans.add((start, i, kind))
            start = kind = None
        if prefix == 'B':
            start, kind = i, label
    return spans


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def error_counts(gold, pred):
    """Separate gold-side FN and prediction-side FP partitions.

    Exact-boundary wrong types take precedence over overlaps. Overlap errors may
    include both a boundary and type error. Counts on the two sides must not be
    added as if they were unique error events (one wrong type causes FN and FP).
    """
    out = Counter(tp=len(gold & pred), gold=len(gold), predicted=len(pred))
    for source, target, side in ((gold, pred, 'gold'), (pred, gold, 'pred')):
        for a in source - target:
            if any(a[:2] == b[:2] for b in target):
                error = 'wrong_type'
            elif any(overlap(a, b) for b in target):
                error = 'boundary_overlap'
            else:
                error = 'missed' if side == 'gold' else 'spurious'
            out[f'{side}_{error}'] += 1
    return out


def metrics(tp, gold, predicted):
    return dict(precision=tp / predicted if predicted else 0.,
                recall=tp / gold if gold else 0.,
                f1=2 * tp / (gold + predicted) if gold + predicted else 0.)


def frequency_bucket(n):
    if n == 0:
        return '0_unseen'
    if n < 10:
        return '1_1-9'
    if n < 100:
        return '2_10-99'
    if n < 1000:
        return '3_100-999'
    return '4_1000+'


def length_bucket(n):
    return '1_1-32' if n <= 32 else '2_33-64' if n <= 64 else '3_65-128' if n <= 128 else '4_129+'


def main():
    output = ROOT / 'results' / 'analysis'
    output.mkdir(exist_ok=True)
    data = ROOT / 'data' / 'cleaned'
    train_counts = Counter()
    train = pd.read_parquet(data / 'train.parquet', columns=['ner_tags'])
    for tags in train.ner_tags:
        train_counts.update(e[2] for e in entities([ID2LABEL[int(t)] for t in tags]))
    del train
    validation = pd.read_parquet(data / 'validation.parquet', columns=['tokens'])
    signatures = {tuple(t) for t in validation.tokens}
    del validation
    test = pd.read_parquet(data / 'test.parquet', columns=['tokens', 'ner_tags'])
    keep = [i for i, t in enumerate(test.tokens) if tuple(t) not in signatures]
    del signatures
    gold_tags = [[ID2LABEL[int(t)] for t in tags] for tags in test.ner_tags]
    gold_spans = [entities(tags) for tags in gold_tags]
    summaries, class_rows, length_rows, freq_rows, confusion_rows, examples = [], [], [], [], [], []
    exact_sets = {}
    for model in MODELS:
        frame = pd.read_parquet(ROOT / 'results' / f'{model}_test_predictions.parquet')
        if len(frame) != len(test):
            raise ValueError(f'{model}: prediction row count does not match test')
        totals, per_class, lengths, frequencies, confusions = Counter(), {}, {}, {}, Counter()
        selected_gold, selected_pred = [], []
        model_exact = []
        example_counts = Counter()
        for i, raw in enumerate(frame.predicted_tags):
            if len(raw) != len(gold_tags[i]):
                raise ValueError(f'{model}: row {i} is not word aligned')
            if any(int(t) < 0 or int(t) >= len(ID2LABEL) for t in raw):
                raise ValueError(f'{model}: invalid tag id at row {i}')
        for i in keep:
            pred = [ID2LABEL[int(t)] for t in frame.predicted_tags.iloc[i]]
            gold, gs, ps = gold_tags[i], gold_spans[i], entities(pred)
            selected_gold.append(gold)
            selected_pred.append(pred)
            counts = error_counts(gs, ps)
            totals.update(counts)
            model_exact.append(gs & ps)
            lb = length_bucket(len(gold))
            lengths.setdefault(lb, Counter()).update(counts)
            lengths[lb]['sentences'] += 1
            for group, spans in (('gold', gs), ('predicted', ps), ('tp', gs & ps)):
                for span in spans:
                    label = span[2]
                    per_class.setdefault(label, Counter())[group] += 1
                    fb = frequency_bucket(train_counts[label])
                    frequencies.setdefault(fb, Counter())[group] += 1
            for a, b in zip(gold, pred):
                ga, pb = a[2:] if a != 'O' else 'O', b[2:] if b != 'O' else 'O'
                if ga != pb:
                    confusions[(ga, pb)] += 1
            for category in ('gold_missed', 'gold_wrong_type', 'gold_boundary_overlap', 'pred_spurious'):
                if counts[category] and example_counts[category] < 2:
                    examples.append(dict(model=model, category=category, test_row=i,
                                         tokens=[str(w) for w in test.tokens.iloc[i]],
                                         gold=[list(s) for s in sorted(gs)],
                                         predicted=[list(s) for s in sorted(ps)]))
                    example_counts[category] += 1
        overall = metrics(totals['tp'], totals['gold'], totals['predicted'])
        reference = scores(selected_gold, selected_pred)
        if round(overall['f1'], 4) != reference['micro_F1']:
            raise AssertionError(f'{model}: span extraction disagrees with shared evaluator')
        saved = json.loads((ROOT / 'results' / f'{model}_eval.json').read_text())['main']
        for k, ref in (('precision', 'micro_P'), ('recall', 'micro_R'), ('f1', 'micro_F1')):
            if round(overall[k], 4) != saved[ref]:
                raise AssertionError(f'{model}: {k} disagrees with saved main result')
        summaries.append(dict(model=model, sentences=len(keep), **totals, **overall,
                              macro_f1=reference['macro_F1']))
        for label, counts in sorted(per_class.items()):
            class_rows.append(dict(model=model, label=label, train_entities=train_counts[label],
                                   **counts, **metrics(counts['tp'], counts['gold'], counts['predicted'])))
        for groups, rows in ((lengths, length_rows), (frequencies, freq_rows)):
            for bucket, counts in sorted(groups.items()):
                rows.append(dict(model=model, bucket=bucket, **counts,
                                 **metrics(counts['tp'], counts['gold'], counts['predicted'])))
        for (g, p), count in confusions.most_common():
            confusion_rows.append(dict(model=model, gold_type=g, predicted_type=p, words=count))
        exact_sets[model] = model_exact
        print(f'{model}: verified main F1={overall["f1"]:.4f}', flush=True)
        del frame, selected_gold, selected_pred

    for name, rows in (('summary', summaries), ('per_class', class_rows), ('by_length', length_rows),
                       ('by_train_frequency', freq_rows), ('token_confusions', confusion_rows)):
        pd.DataFrame(rows).fillna(0).to_csv(output / f'{name}.csv', index=False)
    (output / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
    comparison = []
    for position, i in enumerate(keep):
        recovered = exact_sets['sec-bert-shape'][position] - exact_sets['bert-base-cased'][position]
        lost = exact_sets['bert-base-cased'][position] - exact_sets['sec-bert-shape'][position]
        if recovered or lost:
            comparison.append(dict(test_row=i, shape_only_correct=[list(s) for s in sorted(recovered)],
                                   bert_only_correct=[list(s) for s in sorted(lost)]))
    (output / 'bert_shape_disagreements.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    write_report(output, summaries, class_rows, length_rows, freq_rows, confusion_rows, examples, comparison)


def write_report(output, summaries, classes, lengths, frequencies, confusions, examples, comparison):
    lines = ['# Error analysis', '',
             'Main test view only: exact token duplicates of validation are excluded. '
             'Strict IOB2 entity spans, original words, full sentences. '
             'Prediction row counts, word counts and tag IDs are checked; aggregate P/R/F1 '
             'must reproduce the saved common evaluator results.', '',
             '## Overall results', '',
             '| Model | P | R | F1 | Macro-F1 | TP | Gold entities | Predicted entities |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f'| {r["model"]} | {100*r["precision"]:.2f} | {100*r["recall"]:.2f} | '
                     f'{100*r["f1"]:.2f} | {100*r["macro_f1"]:.2f} | {r["tp"]} | {r["gold"]} | {r["predicted"]} |')
    baseline = next(r for r in summaries if r['model'] == 'bert-base-cased')
    shape = next(r for r in summaries if r['model'] == 'sec-bert-shape')
    lines += ['', '## Findings', '',
              f'- SEC-BERT-SHAPE improves on BERT by {100*(shape["f1"]-baseline["f1"]):.2f} '
              f'F1 points. It has {baseline["pred_spurious"]-shape["pred_spurious"]:,} fewer '
              f'spurious predictions and {baseline["gold_wrong_type"]-shape["gold_wrong_type"]:,} '
              'fewer exact-boundary wrong-type errors.',
              '- For every model, spurious entity predictions outnumber completely missed gold entities. '
              'This supports the observed precision/recall gap; it does not establish why overprediction occurs.',
              '- Boundary-overlap errors are much less frequent than wrong-type errors and spurious predictions.',
              '- Training frequency here counts complete entity types after merging B-/I- tags. '
              'Rare BIO continuation tags in the EDA must not be confused with rare complete entity types. '
              'Absent frequency buckets have no evaluated gold entities or predictions.',
              '- The length breakdown does not show a uniform decline as sentences grow longer. '
              'Do not claim that long sentences are the principal failure mode from these results alone.']
    lines += ['', '## Error partitions', '',
              'Gold-side categories partition false negatives; prediction-side categories partition false positives. '
              'A wrong type at the correct boundaries causes both a false negative and a false positive. '
              'Do not sum the two sides as unique errors. Boundary overlap may also involve a wrong type. '
              'Orphan I tags do not form entities under strict IOB2.', '',
              '| Model | Missed (gold) | Wrong type (gold) | Boundary overlap (gold) | Spurious (prediction) | Wrong type (prediction) | Boundary overlap (prediction) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        fields = ('gold_missed', 'gold_wrong_type', 'gold_boundary_overlap', 'pred_spurious', 'pred_wrong_type', 'pred_boundary_overlap')
        lines.append('| ' + r['model'] + ' | ' + ' | '.join(str(r.get(k, 0)) for k in fields) + ' |')
    for title, rows in [('Sentence length (original words)', lengths), ('Training entity frequency', frequencies)]:
        lines += ['', f'## {title}', '', '| Model | Bucket | Gold entities | Predicted entities | F1 |', '|---|---|---:|---:|---:|']
        for r in rows:
            lines.append(f'| {r["model"]} | {r["bucket"]} | {r.get("gold",0)} | {r.get("predicted",0)} | {100*r["f1"]:.2f} |')
    lines += ['', 'Frequency buckets use strict entity occurrences in cleaned training data, not test frequency. '
              'Predictions are assigned to the bucket of their predicted type, preserving false positives. '
              'These are pooled micro-F1 scores. Word-length buckets do not identify tokenizer truncation; '
              'a long-sentence association alone does not establish its cause.', '', '## Most common type confusions', '',
              'Word-level diagnostics, with BIO prefixes removed; these counts are not entity-level errors. '
              'O-related errors are in token_confusions.csv but excluded from this table.', '',
              '| Model | Gold type | Predicted type | Words |', '|---|---|---|---:|']
    for model in MODELS:
        rows = [r for r in confusions if r['model'] == model and r['gold_type'] != 'O' and r['predicted_type'] != 'O'][:5]
        for r in rows:
            lines.append(f'| {model} | {r["gold_type"]} | {r["predicted_type"]} | {r["words"]} |')
    gained = sum(len(r['shape_only_correct']) for r in comparison)
    lost = sum(len(r['bert_only_correct']) for r in comparison)
    lines += ['', '## BERT versus SEC-BERT-SHAPE', '',
              f'SEC-BERT-SHAPE gets {gained:,} gold spans correct that BERT misses; '
              f'BERT gets {lost:,} correct that SEC-BERT-SHAPE misses. '
              'These are exact gold-span recoveries, not net F1 changes; false positives also matter.', '',
              'Both domain pretraining and number representation differ between these models, '
              'so the difference cannot be attributed to shape replacement alone.', '',
              '## Examples', '',
              'Deterministic illustrative cases: first two main-test sentences per model/category. '
              'Examples are not a random sample and do not establish prevalence. '
              'Spans below use zero-based word indices with an exclusive end. Full tokens and all spans are in examples.json.', '']
    for r in examples:
        lines += [f'### {r["model"]}: {r["category"]}, test row {r["test_row"]}', '',
                  'Sentence: ' + ' '.join(r['tokens']).replace('|', '\\|'), '',
                  'Gold: ' + json.dumps(r['gold']), '', 'Prediction: ' + json.dumps(r['predicted']), '']
    lines += ['## Limitations and scope', '',
              '- Observational analysis of saved predictions from one training seed; no new training or latency claims.',
              '- O-only downsampling changes the training distribution. Its effect on precision/recall is a hypothesis, not a demonstrated cause.',
              '- No tokenizer-level truncation attribution is possible from saved labels alone.',
              '- Test-set analysis is descriptive; selecting changes against these errors would require fresh validation and an untouched final evaluation.',
              '- No presentation slides or speech notes are stored here.', '']
    (output / 'ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
