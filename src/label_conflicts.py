"""Conflicting labels: identical sentences that FiNER-139 labels in more than one way.

The cleaning notebook removes exact duplicate sentences and keeps the FIRST copy. When the
copies of a sentence carry different tag sequences, the first copy is not necessarily the
right one, and the other versions are silently dropped. This script measures that:

1. In each raw split, groups identical token sequences and finds the ones with more than
   one tag sequence ("conflicting sentences").
2. Finds those sentences in data/cleaned/ and describes each conflict: how many label
   versions, how many words differ, whether the difference is entity-vs-O or one entity
   type vs another, and whether the kept version is the most frequent one.
3. Re-scores every saved test prediction file with the shared evaluator (strict IOB2,
   main view) after leaving out the conflicting test sentences, to show whether they
   change any result.

Nothing is retrained and data/cleaned/ is not modified.

Run (CPU, about 2 minutes):   python src/label_conflicts.py
Writes results/label_conflicts/{CONFLICTS.md, conflicts.csv, scores.csv}.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finer import ID2LABEL, RESULTS, ROOT, load_splits, scores  # noqa: E402

SEP = "␟"  # same separator as finer.py and the cleaning notebook
OUT = os.path.join(RESULTS, "label_conflicts")
SPLITS = ["train", "validation", "test"]
NAMES = {"baseline_mft": "Most-frequent tag", "hmm": "HMM + Viterbi", "crf": "CRF",
         "bilstm_crf": "BiLSTM-CRF", "bert-base-cased": "BERT-base-cased",
         "sec-bert-shape": "SEC-BERT-SHAPE", "sec-bert-shape-crf": "SEC-BERT-SHAPE + CRF"}


def entity_types(tags):
    return {t[2:] for t in tags if t != "O"}


def describe(versions):
    """Kind of disagreement between the label versions of one sentence, and the words that differ."""
    names = [[ID2LABEL[i] for i in v] for v in versions]
    differ = [j for j in range(len(names[0])) if len({v[j] for v in names}) > 1]
    entity_vs_o = any("O" in {v[j] for v in names} for j in differ)
    type_change = any(len({v[j][2:] for v in names if v[j] != "O"}) > 1 for j in differ)
    kind = ("entity vs O and type" if entity_vs_o and type_change else
            "entity vs O" if entity_vs_o else
            "entity type" if type_change else "B/I boundary")
    types = sorted(set().union(*(entity_types(v) for v in names)))
    return kind, len(differ), types


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for split in SPLITS:
        raw = pd.read_parquet(os.path.join(ROOT, "data", "raw", f"{split}.parquet"))
        clean = pd.read_parquet(os.path.join(ROOT, "data", "cleaned", f"{split}.parquet"))
        groups = defaultdict(Counter)
        for tokens, tags in zip(raw["tokens"], raw["ner_tags"]):
            groups[SEP.join(tokens)][tuple(int(t) for t in tags)] += 1
        conflicting = {k: v for k, v in groups.items() if len(v) > 1}
        for row, (tokens, tags) in enumerate(zip(clean["tokens"], clean["ner_tags"])):
            key = SEP.join(tokens)
            if key not in conflicting:
                continue
            counts = conflicting[key]
            kept = tuple(int(t) for t in tags)
            kind, n_diff, types = describe(list(counts))
            top = max(counts.values())
            rows.append(dict(split=split, cleaned_row=row, words=len(tokens),
                             label_versions=len(counts), copies_in_raw=sum(counts.values()),
                             words_that_differ=n_diff, kind=kind,
                             kept_is_most_frequent=counts[kept] == top,
                             majority_is_tied=sum(c == top for c in counts.values()) > 1,
                             entity_types=";".join(types),
                             sentence=" ".join(tokens)[:300]))
        print(f"{split}: {len(conflicting):,} raw sentences with conflicting labels, "
              f"{sum(r['split'] == split for r in rows):,} of them still in data/cleaned/ "
              f"({len(clean):,} sentences)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "conflicts.csv"), index=False)

    # Re-score every saved prediction file without the conflicting test sentences
    test = load_splits()["test"]
    bad_test = set(table.loc[table.split == "test", "cleaned_row"])
    main = ~test["in_validation"].to_numpy()
    keep_all = [i for i in range(len(test)) if main[i]]
    keep_ok = [i for i in keep_all if i not in bad_test]
    gold = list(test["tags"])
    score_rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*_test_predictions.parquet"))):
        key = os.path.basename(path)[: -len("_test_predictions.parquet")]
        pred = [[ID2LABEL[int(i)] for i in r] for r in pd.read_parquet(path)["predicted_tags"]]
        if len(pred) != len(test):
            continue
        saved = os.path.join(RESULTS, f"{key}_eval.json")
        if os.path.exists(saved):  # main-view scores already computed by the shared evaluator
            a = json.load(open(saved))["main"]
        else:
            a = scores([gold[i] for i in keep_all], [pred[i] for i in keep_all])
        b = scores([gold[i] for i in keep_ok], [pred[i] for i in keep_ok])
        score_rows.append(dict(model=NAMES.get(key, key),
                               main_micro_f1=round(100 * a["micro_F1"], 2),
                               without_conflicts_micro_f1=round(100 * b["micro_F1"], 2),
                               main_macro_f1=round(100 * a["macro_F1"], 2),
                               without_conflicts_macro_f1=round(100 * b["macro_F1"], 2)))
        print(f"  {key:24s} main {100 * a['micro_F1']:.2f} -> without conflicts {100 * b['micro_F1']:.2f}",
              flush=True)
    scored = pd.DataFrame(score_rows)
    scored.to_csv(os.path.join(OUT, "scores.csv"), index=False)

    # Report
    md = ["# Conflicting labels\n",
          "Identical sentences (same tokens) that FiNER-139 labels in more than one way. "
          "Deduplication in `01_data_preprocessing.ipynb` keeps the first copy, so only one "
          "version survives. Produced by `python src/label_conflicts.py`; nothing is retrained "
          "and `data/cleaned/` is unchanged.\n",
          "| Split | sentences in data/cleaned | conflicting sentences kept | share |",
          "|---|---:|---:|---:|"]
    sizes = {s: len(pd.read_parquet(os.path.join(ROOT, "data", "cleaned", f"{s}.parquet"),
                                    columns=["tokens"])) for s in SPLITS}
    for s in SPLITS:
        n = int((table.split == s).sum()) if len(table) else 0
        md.append(f"| {s} | {sizes[s]:,} | {n:,} | {100 * n / sizes[s]:.2f}% |")
    if len(table):
        md += ["", "## What kind of disagreement", "",
               "| Kind | sentences |", "|---|---:|"]
        for kind, n in table["kind"].value_counts().items():
            md.append(f"| {kind} | {n:,} |")
        kept_top = int(table["kept_is_most_frequent"].sum())
        tied = int(table["majority_is_tied"].sum())
        md += ["",
               f"- Words that differ per conflicting sentence: median "
               f"{int(table['words_that_differ'].median())}, max {int(table['words_that_differ'].max())}.",
               f"- The kept (first) copy is the most frequent version, or tied for most frequent, in "
               f"{kept_top:,} of {len(table):,} sentences, and a less frequent version in "
               f"{len(table) - kept_top:,}. In {tied:,} sentences the top versions are tied, so no "
               f"majority vote could decide the label.",
               "- `entity vs O`: one annotation tags a number, another leaves it untagged. "
               "`entity type`: both tag it, with different XBRL types."]
        common = Counter(t for ts in table["entity_types"] for t in ts.split(";") if t)
        md += ["", "Entity types most often involved:", ""]
        md += [f"- {t} ({n})" for t, n in common.most_common(8)]
    if len(scored):
        md += ["", "## Effect on the test results", "",
               f"Main view ({len(keep_all):,} sentences) vs main view without the "
               f"{len(keep_all) - len(keep_ok)} conflicting test sentences ({len(keep_ok):,}).", "",
               "| Model | micro-F1 | without conflicts | macro-F1 | without conflicts |",
               "|---|---:|---:|---:|---:|"]
        for r in score_rows:
            md.append(f"| {r['model']} | {r['main_micro_f1']:.2f} | {r['without_conflicts_micro_f1']:.2f} | "
                      f"{r['main_macro_f1']:.2f} | {r['without_conflicts_macro_f1']:.2f} |")
    md += ["", "## Interpretation and limits", "",
           "- These sentences are label noise in the source data: the same text received "
           "different annotations, so at least one version is wrong or the sentence is "
           "genuinely ambiguous without the surrounding filing.",
           "- They are a very small share of each split, so retraining without them was "
           "not done; the table above shows whether they move the test scores.",
           "- Only exact duplicates are detected. Near-duplicates (one word different) with "
           "conflicting labels are not counted."]
    with open(os.path.join(OUT, "CONFLICTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {OUT}/CONFLICTS.md, conflicts.csv, scores.csv")


if __name__ == "__main__":
    main()
