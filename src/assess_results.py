"""Paired sentence bootstrap and a reproducible random sample for data assessment.

python src/assess_results.py --replicates 10000 --sample-size 50 --seed 42
Reads existing data/predictions; never changes gold labels or model results.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analyze_errors import MODELS, entities, overlap
from finer import ID2LABEL

ROOT = Path(__file__).resolve().parents[1]


def pooled_f1(counts):
    """Last axis: TP, gold, predicted. Zero denominators have F1=0."""
    c = np.asarray(counts)
    denom = c[..., 1] + c[..., 2]
    return np.divide(2.0*c[..., 0], denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)


def paired_bootstrap(counts, replicates=10000, seed=42):
    """Exact multinomial implementation of ordinary paired sentence bootstrap.

    counts has shape (sentences, models, 3). Identical sufficient-statistic rows
    are grouped for speed; sampling category multiplicities is distributionally
    identical to sampling N sentence indices with replacement. All-O sentences
    remain in the population. Every model receives the same resampled sentences.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 3 or counts.shape[-1] != 3 or not len(counts):
        raise ValueError('Expected nonempty (sentences, models, 3) counts')
    if replicates <= 0 or np.any(counts < 0):
        raise ValueError('Positive replicates and nonnegative counts required')
    if np.any(counts[..., 0] > counts[..., 1]) or np.any(counts[..., 0] > counts[..., 2]):
        raise ValueError('TP cannot exceed gold or predicted counts')
    n, models, _ = counts.shape
    patterns, frequency = np.unique(counts.reshape(n, -1), axis=0, return_counts=True)
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, models))
    for start in range(0, replicates, 128):
        end = min(start+128, replicates)
        weights = rng.multinomial(n, frequency/n, size=end-start)
        totals = (weights @ patterns).reshape(end-start, models, 3)
        samples[start:end] = pooled_f1(totals)
    return samples, len(patterns)


def sample_candidates(candidates, size, seed):
    if size < 1 or size > len(candidates):
        raise ValueError('Sample size must be within the candidate population')
    indices = np.random.default_rng(seed).choice(len(candidates), size=size, replace=False)
    return [candidates[int(i)] for i in indices]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replicates', type=int, default=10000)
    ap.add_argument('--sample-size', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    out = ROOT/'results'/'assessment'
    out.mkdir(exist_ok=True)
    test_path = ROOT/'data'/'cleaned'/'test.parquet'
    val_path = ROOT/'data'/'cleaned'/'validation.parquet'
    test = pd.read_parquet(test_path, columns=['tokens', 'ner_tags'])
    validation = pd.read_parquet(val_path, columns=['tokens'])
    val_signatures = {tuple(t) for t in validation.tokens}
    keep = np.array([i for i,t in enumerate(test.tokens) if tuple(t) not in val_signatures])
    del validation, val_signatures
    gold = [entities([ID2LABEL[int(t)] for t in tags]) for tags in test.ner_tags]
    counts = np.zeros((len(keep), len(MODELS), 3), dtype=np.int64)
    all_predictions = {}
    candidates = []
    paths = [test_path, val_path]
    for m, model in enumerate(MODELS):
        path = ROOT/'results'/f'{model}_test_predictions.parquet'
        paths.append(path)
        pred = pd.read_parquet(path).predicted_tags
        if len(pred) != len(test):
            raise ValueError(f'{model}: row count mismatch')
        spans = {}
        for j,i in enumerate(keep):
            raw = pred.iloc[i]
            if len(raw) != len(test.tokens.iloc[i]) or any(int(t)<0 or int(t)>=len(ID2LABEL) for t in raw):
                raise ValueError(f'{model}: invalid prediction row {i}')
            ps, gs = entities([ID2LABEL[int(t)] for t in raw]), gold[i]
            spans[int(i)] = ps
            counts[j,m] = (len(gs & ps), len(gs), len(ps))
            if model == 'sec-bert-shape':
                candidates.extend((int(i), s) for s in sorted(ps) if not any(overlap(s,g) for g in gs))
        all_predictions[model] = spans
        saved = json.loads((ROOT/'results'/f'{model}_eval.json').read_text())['main']
        observed = float(pooled_f1(counts[:,m].sum(axis=0)))
        if len(keep) != saved['sentences'] or round(observed,4) != saved['micro_F1']:
            raise AssertionError(f'{model}: does not reproduce saved results')
        print(f'{model}: checked {len(keep)} sentences, F1={observed:.6f}', flush=True)

    samples, patterns = paired_bootstrap(counts, args.replicates, args.seed)
    point = pooled_f1(counts.sum(axis=0))
    lows, highs = np.quantile(samples, [.025, .975], axis=0)
    pd.DataFrame([dict(model=m, micro_f1=100*point[j], ci95_low=100*lows[j], ci95_high=100*highs[j])
                  for j,m in enumerate(MODELS)]).to_csv(out/'bootstrap_models.csv', index=False)
    # Positive difference means A is better. Three comparisons are specified in advance.
    pairs = [(2,1), (2,0), (0,1)]
    differences = []
    for a,b in pairs:
        delta = 100*(samples[:,a]-samples[:,b])
        lo,hi = np.quantile(delta,[.025,.975])
        # Family-wise 95% coverage via Bonferroni, in addition to marginal 95% CIs.
        adjlo,adjhi = np.quantile(delta,[.05/(2*len(pairs)),1-.05/(2*len(pairs))])
        differences.append(dict(model_a=MODELS[a],model_b=MODELS[b],delta_f1_points=100*(point[a]-point[b]),
                                ci95_low=lo,ci95_high=hi,family95_low=adjlo,family95_high=adjhi))
    pd.DataFrame(differences).to_csv(out/'bootstrap_differences.csv',index=False)
    sentence_rows = {'test_row':keep}
    for m,model in enumerate(MODELS):
        for j,key in enumerate(('tp','gold','predicted')):
            sentence_rows[f'{model}_{key}'] = counts[:,m,j]
    pd.DataFrame(sentence_rows).to_csv(out/'sentence_counts.csv',index=False)
    pd.DataFrame(100*samples,columns=MODELS).to_csv(out/'bootstrap_replicates.csv',index=False)
    print(f'Bootstrap finished: {args.replicates} replicates, {patterns} grouped count patterns.',flush=True)

    selected = sample_candidates(candidates,args.sample_size,args.seed)
    records = []
    for j,(row,span) in enumerate(selected,1):
        tokens = [str(t) for t in test.tokens.iloc[row]]
        start,end,label = span
        marked = tokens[:start]+['[[']+tokens[start:end]+[']]']+tokens[end:]
        records.append(dict(sample_id=f'Q{j:03d}',test_row=row,start=start,end=end,
                            predicted_type=label,target_text=' '.join(tokens[start:end]),
                            sentence=' '.join(tokens),marked_sentence=' '.join(marked),tokens=tokens,
                            gold_spans=[list(s) for s in sorted(gold[row])],
                            model_spans={m:[list(s) for s in sorted(all_predictions[m][row])] for m in MODELS},
                            raw_matches=[]))
    # Check exact duplicate sentences in the raw releases for annotation variants.
    # Conflicting annotations are evidence of inconsistency, not proof of which is right.
    signatures = {}
    for r in records:
        signatures.setdefault(tuple(r['tokens']),[]).append(r)
    for split in ('train','validation','test'):
        path = ROOT/'data'/'raw'/f'{split}.parquet'
        paths.append(path)
        offset=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=['id','tokens','ner_tags']):
            for j,item in enumerate(batch.to_pylist()):
                matches=signatures.get(tuple(item['tokens']))
                if matches:
                    raw_spans=entities([ID2LABEL[int(t)] for t in item['ner_tags']])
                    for record in matches:
                        start,end=record['start'],record['end']
                        record['raw_matches'].append(dict(split=split,raw_row=offset+j,source_id=str(item['id']),
                            target_tags=[ID2LABEL[int(t)] for t in item['ner_tags'][start:end]],
                            overlapping_spans=[list(s) for s in sorted(raw_spans) if overlap(s,(start,end,''))]))
            offset+=batch.num_rows
        print(f'Checked exact raw annotation matches in {split}.',flush=True)
    (out/'sample.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([dict(sample_id=r['sample_id'],test_row=r['test_row'],target_text=r['target_text'],
                       predicted_type=r['predicted_type'],marked_sentence=r['marked_sentence'],
                       raw_target_variants=len({tuple(x['target_tags']) for x in r['raw_matches']}))
                  for r in records]).to_csv(out/'sample.csv',index=False,encoding='utf-8-sig')
    meta=dict(seed=args.seed,replicates=args.replicates,sample_size=args.sample_size,
              main_sentences=len(keep),models=list(MODELS),sample_population=len(candidates),
              sample_unit='SEC-BERT-SHAPE predicted entity span with no overlap with any gold entity',
              distinct_sample_sentences=len({r['test_row'] for r in records}),
              bootstrap_unit='sentence; paired across models; multinomial grouped sufficient statistics',
              grouped_patterns=patterns,numpy_version=np.__version__,
              file_sha256={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in paths})
    (out/'method.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    write_bootstrap_report(out,meta,point,lows,highs,differences)


def write_bootstrap_report(out,meta,point,lows,highs,differences):
    lines=['# Bootstrap comparison','',
           f'{meta["replicates"]:,} paired bootstrap replicates; seed {meta["seed"]}; '
           f'{meta["main_sentences"]:,} main-test sentences. Strict IOB2 entity micro-F1, in percentage points.', '',
           '| Model | F1 | 95% percentile interval |','|---|---:|---|']
    for j,m in enumerate(MODELS):
        lines.append(f'| {m} | {100*point[j]:.2f} | [{100*lows[j]:.2f}, {100*highs[j]:.2f}] |')
    lines += ['', '| A minus B | Difference | 95% interval | Family-wise 95% interval (3 comparisons) |',
              '|---|---:|---|---|']
    for r in differences:
        lines.append(f'| {r["model_a"]} minus {r["model_b"]} | {r["delta_f1_points"]:.2f} | '
                     f'[{r["ci95_low"]:.2f}, {r["ci95_high"]:.2f}] | [{r["family95_low"]:.2f}, {r["family95_high"]:.2f}] |')
    lines += ['', '## Method and interpretation', '',
              '- Each replicate draws N sentences with replacement and uses the same draw for every model. '
              'Micro-F1 is recomputed from pooled TP, gold and predicted entity counts; sentence F1 values are not averaged.',
              '- Identical count patterns are grouped. Multinomial sampling of their multiplicities is distributionally '
              'equivalent to ordinary sentence resampling. All-O sentences remain in the population.',
              '- The difference interval is computed directly from paired replicate differences. '
              'Overlapping individual model intervals are not used as a significance test.',
              '- Marginal 95% intervals use 2.5/97.5 percentiles. The three comparison intervals also include '
              'Bonferroni-adjusted 0.8333/99.1667 percentiles for family-wise 95% coverage.',
              '- Intervals describe sentence-sampling uncertainty conditional on these saved predictions and labels. '
              'They do not measure training-seed variability or annotation uncertainty.',
              '- Sentences are treated as independent. Correlation within financial filings may make intervals optimistic; '
              'no filing-level grouping was available in the analysis inputs. IDs were not assumed to be document IDs.',
              '- Observed test results have already been inspected. These are descriptive comparisons, not evidence of '
              'unbiased performance for a new method selected against this test set.',
              '- Model scores reproduce the shared saved main-view results. Original predictions and gold labels are unchanged.', '',
              'Reproduce: `python src/assess_results.py --replicates 10000 --sample-size 50 --seed 42`', '']
    (out/'BOOTSTRAP.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
