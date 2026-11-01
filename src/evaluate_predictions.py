"""Score every saved prediction file with the shared evaluator and build the results table.

Any model can be added without re-running it: save word-level predicted tag ids as
results/<key>_test_predictions.parquet with one column `predicted_tags`, where row i
is row i of data/cleaned/test.parquet (the format used by train_transformer22.py and
baseline_hmm.py). Then run:

    python src/evaluate_predictions.py

Writes results/<key>_eval.json for every prediction file, and results/RESULTS.md
(markdown) + results/results_table.tex (LaTeX) with all models in one table.
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finer import ID2LABEL, RESULTS, evaluate, load_splits  # noqa: E402

# Display names and table order; unknown files are appended at the end.
NAMES = {
    "baseline_mft": "Most-frequent tag",
    "hmm": "HMM + Viterbi",
    "crf": "CRF",
    "bilstm_crf": "BiLSTM-CRF",
    "bert-base-cased": "BERT-base-cased",
    "sec-bert-shape": "SEC-BERT-SHAPE",
    "sec-bert-shape-crf": "SEC-BERT-SHAPE + CRF",
    "bert-base-cased-4ep": "BERT-base-cased (4 epochs)",
    "sec-bert-shape-4ep": "SEC-BERT-SHAPE (4 epochs)",
}


def main():
    test = load_splits()["test"]
    files = sorted(glob.glob(os.path.join(RESULTS, "*_test_predictions.parquet")))
    keys = [os.path.basename(f)[: -len("_test_predictions.parquet")] for f in files]
    keys = [k for k in NAMES if k in keys] + [k for k in keys if k not in NAMES]
    rows = []
    for key in keys:
        p = pd.read_parquet(os.path.join(RESULTS, f"{key}_test_predictions.parquet"))
        if len(p) != len(test):
            print(f"skip {key}: {len(p)} rows, test has {len(test)}")
            continue
        pred = [[ID2LABEL[int(i)] for i in r] for r in p["predicted_tags"]]
        res = evaluate(test, pred, NAMES.get(key, key))
        with open(os.path.join(RESULTS, f"{key}_eval.json"), "w") as f:
            json.dump({k: v for k, v in res.items() if k != "report"}, f, indent=2)
        rows.append(res)

    n_main, n_all = rows[0]["main"]["sentences"], rows[0]["reference"]["sentences"]
    md = [f"# Test results\n",
          f"Entity-level, strict IOB2 (seqeval), word level, full-length sentences.",
          f"**Main** = cleaned test without the sentences that also occur in validation "
          f"({n_main:,} sentences). **Full test** = all {n_all:,} cleaned test sentences.\n",
          "| Model | P | R | micro-F1 | macro-F1 | micro-F1 (full test) |",
          "|---|---:|---:|---:|---:|---:|"]
    tex = [r"\begin{table}[t]", r"\centering\small",
           r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Model & P & R & micro-F1 & macro-F1 & micro-F1 (full) \\", r"\midrule"]
    for r in rows:
        m, f = r["main"], r["reference"]
        md.append(f"| {r['model']} | {100*m['micro_P']:.1f} | {100*m['micro_R']:.1f} | "
                  f"**{100*m['micro_F1']:.1f}** | {100*m['macro_F1']:.1f} | {100*f['micro_F1']:.1f} |")
        tex.append(f"{r['model']} & {100*m['micro_P']:.1f} & {100*m['micro_R']:.1f} & "
                   f"{100*m['micro_F1']:.1f} & {100*m['macro_F1']:.1f} & {100*f['micro_F1']:.1f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            rf"\caption{{FiNER-139 test results (entity-level, strict IOB2). Main view: {n_main:,} test "
            rf"sentences not in validation; full: all {n_all:,}.}}",
            r"\label{tab:results}", r"\end{table}"]
    with open(os.path.join(RESULTS, "RESULTS.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(RESULTS, "results_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
