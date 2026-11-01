"""Cleaning ablation: what did each cleaning step of 01_data_preprocessing.ipynb change?

The same model (the team's feature CRF, src/baseline_crf.py, with its selected settings:
window 3, weight decay 0) is trained three times, changing ONLY the training data:

  A  cleaned     data/cleaned/train.parquet (dedup + leakage removal + O-only reduction)
  B  +O-only     A + the 300,958 no-entity train sentences the O-only reduction dropped
                 (= raw train after dedup and leakage removal, no O-only reduction)
  C  +leaked     A + the 41,099 train sentences removed because they also occur in
                 validation or test (leakage step undone)

Everything else is identical: features fitted on the variant's own train set, same seed,
same optimiser, epoch chosen on the same 20,000-sentence validation sample, test decoded
once, shared evaluator (main view = 92,608 test sentences not in validation).

Questions answered
  B vs A: does dropping no-entity sentences make the model over-predict entities
          (lower precision, higher recall)?  (Zehra's hypothesis in her contribution file)
  C vs A: how much does training on sentences that also occur in test inflate the score?
          Reported on the whole main view and on the main-view test sentences that are
          among the leaked ones.

Run (GPU, ~1 hour on a T4):   python src/cleaning_ablation.py
Quick check (CPU):            python src/cleaning_ablation.py --limit 3000 --epochs 2 --out /tmp/ablation_check
Writes results/ablation/{ABLATION.md, scores.csv, bootstrap.csv, variants.csv,
<variant>_test_predictions.parquet, <variant>_eval.json, <variant>_history.json}.
Note: with --limit the test set is cut to its first N sentences, so the scores are only a format check.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assess_results import paired_bootstrap  # noqa: E402
from baseline_crf import Batches, FeatureCRF  # noqa: E402
from crf_features import CRFFeatures  # noqa: E402
from finer import DATA, ID2LABEL, LABEL2ID, RESULTS, ROOT, evaluate, load_splits, scores  # noqa: E402
from seq_utils import (TagSet, add_common_args, iob2_violations, predict, search,  # noqa: E402
                       set_seed, val_sample)

OUT = os.path.join(RESULTS, "ablation")
SEP = "␟"
VARIANTS = {"A_cleaned": "A  cleaned train (as used by all models)",
            "B_plus_o_only": "B  A + dropped no-entity sentences",
            "C_plus_leaked": "C  A + leaked sentences (also in val/test)"}


def sig(tokens):
    return SEP.join(tokens)


def frame(df):
    return pd.DataFrame({"tokens": [[str(w) for w in t] for t in df["tokens"]],
                         "tags": [[ID2LABEL[int(i)] for i in t] for t in df["ner_tags"]]})


def build_variants():
    """Re-run the notebook's cleaning steps on the raw parquet files and return the three
    training sets. Checks that A rebuilt here is exactly data/cleaned/train.parquet."""
    raw = {s: pd.read_parquet(os.path.join(ROOT, "data", "raw", f"{s}.parquet"), columns=["tokens", "ner_tags"])
           for s in ["train", "validation", "test"]}

    def dedup(df):  # notebook step 1: keep the first occurrence of each exact token sequence
        seen, keep = set(), []
        for i, t in enumerate(df["tokens"]):
            s = sig(t)
            if s not in seen:
                seen.add(s)
                keep.append(i)
        return df.iloc[keep].reset_index(drop=True)

    tr, va, te = dedup(raw["train"]), dedup(raw["validation"]), dedup(raw["test"])
    vt = {sig(t) for t in va["tokens"]} | {sig(t) for t in te["tokens"]}
    leaked = np.array([sig(t) in vt for t in tr["tokens"]])
    no_leak = tr[~leaked].reset_index(drop=True)                       # notebook step 2
    random.seed(42)                                                    # notebook step 3 (same draws)
    keep = [i for i, tags in enumerate(no_leak["ner_tags"]) if any(t != 0 for t in tags) or random.random() < 0.30]
    a = no_leak.iloc[keep].reset_index(drop=True)

    cleaned = pd.read_parquet(os.path.join(DATA, "train.parquet"), columns=["tokens", "ner_tags"])
    same = len(a) == len(cleaned) and all(sig(x) == sig(y) for x, y in zip(a["tokens"], cleaned["tokens"]))
    if not same:
        raise SystemExit("Rebuilt cleaned train differs from data/cleaned/train.parquet; stop.")
    variants = {"A_cleaned": a,
                "B_plus_o_only": no_leak,
                "C_plus_leaked": pd.concat([a, tr[leaked]], ignore_index=True)}
    leaked_sigs = {sig(t) for t in tr[leaked]["tokens"]}
    return {k: frame(v) for k, v in variants.items()}, leaked_sigs


def spans(seq):
    """Strict IOB2 entity spans (start, end, type); an I- tag without a matching B-/I- starts nothing."""
    out, i = set(), 0
    while i < len(seq):
        if seq[i].startswith("B-"):
            typ, j = seq[i][2:], i + 1
            while j < len(seq) and seq[j] == "I-" + typ:
                j += 1
            out.add((i, j, typ))
            i = j
        else:
            i += 1
    return out


def counts(gold, pred):
    """Per sentence: (true positives, gold entities, predicted entities)."""
    out = np.zeros((len(gold), 3), dtype=np.int64)
    for k, (g, p) in enumerate(zip(gold, pred)):
        G, P = spans(list(g)), spans(list(p))
        out[k] = (len(G & P), len(G), len(P))
    return out


def train_variant(name, train, val, test, args, device):
    set_seed(args.seed)
    tagset = TagSet(train.tags)
    feats = CRFFeatures(window=args.window, min_count=2, max_pairs=200_000).fit(train.tokens)
    f_train, s_train = feats.transform(train.tokens)
    f_val, s_val = feats.transform(val.tokens)
    train_tags = [np.array(tagset.encode(t)) for t in train.tags]
    keep = feats.keep_columns(args.window)
    constrain = iob2_violations(train.tags) == 0
    print(f"[{name}] {len(train):,} train sentences, {int(s_train[-1]):,} words, {len(tagset)} tags, "
          f"{feats.n_features:,} features", flush=True)

    def build(cfg):
        model = FeatureCRF(feats.n_features, tagset.names, constrain).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg["weight_decay"])
        return model, opt, Batches(f_train, s_train, keep, device, train_tags), Batches(f_val, s_val, keep, device)

    t0 = time.perf_counter()
    (cfg, out, state), trials = search([dict(window=args.window, weight_decay=args.weight_decay)], build,
                                       np.diff(s_train), list(val.tags), tagset, args, device)
    f_train = train_tags = None  # free memory before the test features are built
    model = FeatureCRF(feats.n_features, tagset.names, constrain).to(device)
    model.load_state_dict(state)
    f_test, s_test = feats.transform(test.tokens)
    pred = predict(model, Batches(f_test, s_test, keep, device), [len(t) for t in test.tokens], 4 * args.batch_tokens)
    pred = [tagset.decode(p) for p in pred]
    info = dict(variant=name, train_sentences=len(train), train_words=int(s_train[-1]),
                o_only_share=round(float(np.mean([all(t == "O" for t in s) for s in train.tags])), 4),
                tags=len(tagset), features=int(feats.n_features), selected_epoch=out["selected_epoch"],
                val_micro_F1=round(out["val_micro_F1"], 4), minutes=round((time.perf_counter() - t0) / 60, 1))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred, info, out["history"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, epochs=8, patience=2, batch_tokens=8192, lr=0.01)   # baseline_crf.py defaults
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--out", default=OUT, help="output folder (results/ablation)")
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.limit and args.out == OUT:
        raise SystemExit("With --limit pass --out <scratch folder> so the real results are not overwritten.")
    os.makedirs(args.out, exist_ok=True)

    d = load_splits()
    test = d["test"]
    val = val_sample(d["validation"], args.val_limit, args.seed)
    variants, leaked_sigs = build_variants()
    if args.limit:
        variants = {k: v.sample(n=min(args.limit, len(v)), random_state=args.seed).reset_index(drop=True)
                    for k, v in variants.items()}
        val = val.head(min(args.limit, len(val)))
        test = test.head(min(args.limit, len(test))).reset_index(drop=True)  # quick check only
    main_mask = ~test["in_validation"].to_numpy()
    leaked_mask = main_mask & np.array([sig(t) in leaked_sigs for t in test["tokens"]])
    print(f"variants: { {k: len(v) for k, v in variants.items()} } | main-view test sentences that are among "
          f"the leaked train sentences: {int(leaked_mask.sum()):,}", flush=True)

    preds, infos = {}, []
    for name in args.variants:
        pred, info, history = train_variant(name, variants[name], val, test, args, device)
        preds[name] = pred
        res = evaluate(test, pred, VARIANTS[name])
        g_leak = [g for g, m in zip(test.tags, leaked_mask) if m]
        p_leak = [p for p, m in zip(pred, leaked_mask) if m]
        leak = scores(g_leak, p_leak) if g_leak else {}
        info.update({f"main_{k}": v for k, v in res["main"].items() if k != "sentences"})
        info.update({f"leaked_subset_{k}": v for k, v in leak.items()})
        infos.append(info)
        pd.DataFrame({"predicted_tags": [[LABEL2ID[t] for t in p] for p in pred]}).to_parquet(
            os.path.join(args.out, f"{name}_test_predictions.parquet"))
        with open(os.path.join(args.out, f"{name}_eval.json"), "w") as f:
            json.dump({k: v for k, v in res.items() if k != "report"}, f, indent=2)
        with open(os.path.join(args.out, f"{name}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    table = pd.DataFrame(infos)
    print(table.to_string())
    table.to_csv(os.path.join(args.out, "scores.csv"), index=False)

    # paired sentence bootstrap of micro-F1 differences vs A, main view (same draws for every model)
    names = [n for n in args.variants]
    gold_main = [g for g, m in zip(test.tags, main_mask) if m]
    c = np.stack([counts(gold_main, [p for p, m in zip(preds[n], main_mask) if m]) for n in names], axis=1)
    samples, _ = paired_bootstrap(c, replicates=args.replicates, seed=args.seed)
    tot = c.sum(0)
    point = 2 * tot[:, 0] / (tot[:, 1] + tot[:, 2])
    boot = []
    if "A_cleaned" in names:
        a = names.index("A_cleaned")
        for i, n in enumerate(names):
            if i == a:
                continue
            diff = samples[:, i] - samples[:, a]
            boot.append(dict(comparison=f"{n} minus A_cleaned", micro_F1_difference=round(100 * (point[i] - point[a]), 2),
                             ci95_low=round(100 * np.percentile(diff, 2.5), 2),
                             ci95_high=round(100 * np.percentile(diff, 97.5), 2)))
    pd.DataFrame(boot).to_csv(os.path.join(args.out, "bootstrap.csv"), index=False)
    write_report(table, boot, int(leaked_mask.sum()), args)


def write_report(table, boot, n_leaked, args):
    L = ["# Cleaning ablation (feature CRF)\n",
         "Same model, settings, seed and model selection; only the training data changes. "
         "Scores: shared evaluator, strict IOB2, main view (92,608 test sentences). "
         f"Bootstrap: {args.replicates:,} paired sentence replicates, seed {args.seed}. "
         "Produced by `python src/cleaning_ablation.py`.\n",
         "| Variant | train sentences | no-entity share | P | R | micro-F1 | macro-F1 | epoch |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in table.iterrows():
        L.append(f"| {VARIANTS[r['variant']]} | {r['train_sentences']:,} | {100 * r['o_only_share']:.0f}% | "
                 f"{100 * r['main_micro_P']:.1f} | {100 * r['main_micro_R']:.1f} | **{100 * r['main_micro_F1']:.1f}** | "
                 f"{100 * r['main_macro_F1']:.1f} | {r['selected_epoch']} |")
    L += ["", "| Comparison (main-view micro-F1) | difference | 95% interval |", "|---|---:|---|"]
    for b in boot:
        L.append(f"| {b['comparison']} | {b['micro_F1_difference']:+.2f} | [{b['ci95_low']:+.2f}, {b['ci95_high']:+.2f}] |")
    if "leaked_subset_micro_F1" in table:
        L += ["", f"Main-view test sentences that are among the leaked train sentences: {n_leaked:,}. "
              "Micro-F1 on this subset:", "", "| Variant | micro-F1 on leaked subset |", "|---|---:|"]
        for _, r in table.iterrows():
            L.append(f"| {VARIANTS[r['variant']]} | {100 * r['leaked_subset_micro_F1']:.1f} |")
    L += ["", "Limitations: one seed per variant; feature vocabularies are fitted on each variant's own train "
          "set, as they would be in practice; the bootstrap covers test-sentence sampling only, not training "
          "variability."]
    with open(os.path.join(args.out, "ABLATION.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
