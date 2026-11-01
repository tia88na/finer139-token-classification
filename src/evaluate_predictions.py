"""Score every saved prediction file with the shared evaluator and build the results table.

Any model can be added without re-running it: save word-level predicted tag ids as
results/<key>_test_predictions.parquet with one column `predicted_tags`, where row i
is row i of data/cleaned/test.parquet (the format used by train_transformer22.py and
baseline_hmm.py). Then run:

    python src/evaluate_predictions.py

Writes results/<key>_eval.json for every prediction file, and results/RESULTS.md
(markdown) + results/results_table.tex (LaTeX, booktabs) with all models in one table:
precision, recall, micro-F1 with its 95% confidence interval, macro-F1, full-test micro-F1
and inference speed.

95% CI: 10,000 paired sentence-bootstrap replicates of the main view (seed 42), all models
on the same resampled sentences (`assess_results.paired_bootstrap`, percentile interval);
also saved to results/results_ci.csv.
Speed: sentences per second on the T4 GPU from results/latency.csv (`src/latency.py`).
A 4-epoch run has the same architecture as its 2-epoch run, so it shows that run's speed
(marked with a dagger); the HMM runs on the CPU (double dagger); "-" = not measured.
"""
import glob
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_errors import entities  # noqa: E402
from assess_results import paired_bootstrap  # noqa: E402
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
SAME_ARCHITECTURE = {"bert-base-cased-4ep": "bert-base-cased", "sec-bert-shape-4ep": "sec-bert-shape"}
CPU_ONLY = {"hmm"}


def one(value) -> str:
    """A proportion as a percentage with one decimal: rounded half up to two decimals first
    (the precision of the team's reports, e.g. BiLSTM-CRF 71.45), then to one (71.5)."""
    two = (Decimal(value) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return str(two.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def sentence_counts(gold, pred):
    """Per sentence: (true positives, gold entities, predicted entities), strict IOB2 spans."""
    out = np.zeros((len(gold), 3), dtype=np.int64)
    for i, (g, p) in enumerate(zip(gold, pred)):
        G, P = entities(g), entities(p)
        out[i] = (len(G & P), len(G), len(P))
    return out


def speeds():
    """key -> speed label (sentences/s on the T4) from results/latency.csv."""
    path = os.path.join(RESULTS, "latency.csv")
    if not os.path.exists(path):
        return {}
    lat = pd.read_csv(path)
    gpu = dict(zip(lat[lat.device == "cuda"].key, lat[lat.device == "cuda"].sentences_per_s))
    out = {}
    for key in NAMES:
        src = SAME_ARCHITECTURE.get(key, key)
        if src in gpu:
            mark = "†" if key in SAME_ARCHITECTURE else ("‡" if key in CPU_ONLY else "")
            out[key] = (f"{gpu[src]:,.0f}", mark)
    return out


def main():
    test = load_splits()["test"]
    files = sorted(glob.glob(os.path.join(RESULTS, "*_test_predictions.parquet")))
    keys = [os.path.basename(f)[: -len("_test_predictions.parquet")] for f in files]
    keys = [k for k in NAMES if k in keys] + [k for k in keys if k not in NAMES]
    rows, used, counts = [], [], []
    keep = ~test["in_validation"].to_numpy()
    gold_main = [g for g, k in zip(test["tags"], keep) if k]
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
        used.append(key)
        counts.append(sentence_counts(gold_main, [p for p, k in zip(pred, keep) if k]))

    samples, _ = paired_bootstrap(np.stack(counts, axis=1), replicates=10000, seed=42)
    ci = {k: (100 * np.percentile(samples[:, i], 2.5), 100 * np.percentile(samples[:, i], 97.5))
          for i, k in enumerate(used)}
    pd.DataFrame([dict(key=k, model=NAMES.get(k, k), micro_F1=round(100 * r["main"]["micro_F1"], 2),
                       ci95_low=round(ci[k][0], 2), ci95_high=round(ci[k][1], 2)) for k, r in zip(used, rows)]
                 ).to_csv(os.path.join(RESULTS, "results_ci.csv"), index=False)
    speed = speeds()

    n_main, n_all = rows[0]["main"]["sentences"], rows[0]["reference"]["sentences"]
    md = ["# Test results\n",
          "Entity-level, strict IOB2 (seqeval), word level, full-length sentences.",
          f"**Main** = cleaned test without the sentences that also occur in validation "
          f"({n_main:,} sentences). **Full test** = all {n_all:,} cleaned test sentences.",
          "95% CI: 10,000 paired sentence-bootstrap replicates of the main view. "
          "Speed: sentences/s on a Tesla T4 (`results/LATENCY.md`); † same architecture as the "
          "2-epoch run (not timed separately); ‡ runs on the CPU; - not measured.\n",
          "| Model | P | R | micro-F1 | 95% CI | macro-F1 | micro-F1 (full test) | sent./s (T4) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    tex = [r"\begin{table}[t]", r"\centering\small",
           r"\begin{tabular}{lrrrcrrr}", r"\toprule",
           r"Model & P & R & micro-F1 & 95\% CI & macro-F1 & micro-F1 (full) & sent./s \\", r"\midrule"]
    tex_mark = {"†": r"$^\dagger$", "‡": r"$^\ddagger$", "": ""}
    for i, (key, r) in enumerate(zip(used, rows)):
        # display values from UNROUNDED numbers (pooled counts, seqeval's macro average);
        # re-rounding the 4-decimal JSON values with float formatting turned 70.35 into 70.3
        tp, n_gold, n_pred = counts[i].sum(0)
        P, R, F1 = one(tp / n_pred), one(tp / n_gold), one(2 * tp / (n_gold + n_pred))
        macro = one(r["report"]["macro avg"]["f1-score"])
        full = one(Decimal(str(r["reference"]["micro_F1"])))
        lo, hi = one(ci[key][0] / 100), one(ci[key][1] / 100)
        sp, mark = speed.get(key, ("-", ""))
        md.append(f"| {r['model']} | {P} | {R} | **{F1}** | [{lo}, {hi}] | {macro} | {full} | {sp}{mark} |")
        tex.append(f"{r['model']} & {P} & {R} & {F1} & [{lo}, {hi}] & {macro} & {full} & "
                   f"{sp.replace(',', '{,}')}{tex_mark[mark]} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            rf"\caption{{FiNER-139 test results (entity-level, strict IOB2). Main view: {n_main:,} test "
            rf"sentences not in validation; full: all {n_all:,}. 95\% CI: 10{{,}}000 paired bootstrap "
            r"replicates. Speed: sentences per second on a Tesla T4; $^\dagger$ same architecture as the "
            r"2-epoch run; $^\ddagger$ CPU; - not measured.}",
            r"\label{tab:results}", r"\end{table}"]
    with open(os.path.join(RESULTS, "RESULTS.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(RESULTS, "results_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
