"""Shared data loading and evaluation for every model in the project.

Every model is scored with `evaluate()` so the numbers in the final table are comparable:

* word-level labels (never sub-word pieces),
* full-length sentences: if a model could not label a word (e.g. it was cut by
  truncation), that word counts as `O`, so a missed entity is a false negative and
  is never silently dropped from the gold set,
* entity-level precision / recall / F1 with seqeval, micro and macro,
* two test views:
    - MAIN      = cleaned test minus the 2,422 sentences that also occur in
                  validation (92,608 sentences). Duplicates between the data used for
                  model selection and the test set inflate scores (lecture W6, slide 19).
    - REFERENCE = the whole cleaned test set (95,030 sentences), for comparison
                  with numbers reported before this fix.

Usage
-----
    from finer import load_splits, evaluate
    d = load_splits()                    # train / validation / test DataFrames
    res = evaluate(d["test"], predictions, "My model")
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "cleaned")
RESULTS = os.path.join(ROOT, "results")
_SEP = "␟"  # joins tokens into a sentence signature for exact-duplicate checks


def label_names(path: str | None = None) -> list[str]:
    """The 279 BIO tag names, read from the parquet metadata written by `datasets`."""
    meta = pq.read_schema(path or os.path.join(DATA, "test.parquet")).metadata
    feats = json.loads(meta[b"huggingface"])["info"]["features"]
    return feats["ner_tags"]["feature"]["names"]


ID2LABEL = label_names()
LABEL2ID = {name: i for i, name in enumerate(ID2LABEL)}


def load_splits() -> dict[str, pd.DataFrame]:
    """Cleaned train / validation / test as DataFrames with columns
    `tokens` (list[str]) and `tags` (list[str]).
    The test DataFrame has an extra boolean column `in_validation` that marks the
    sentences that also occur (identical tokens) in the validation split."""
    out = {}
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(os.path.join(DATA, f"{split}.parquet"), columns=["tokens", "ner_tags"])
        out[split] = pd.DataFrame({
            "tokens": [[str(w) for w in t] for t in df["tokens"]],
            "tags": [[ID2LABEL[int(i)] for i in t] for t in df["ner_tags"]],
        })
    val = {_SEP.join(t) for t in out["validation"]["tokens"]}
    out["test"]["in_validation"] = [_SEP.join(t) in val for t in out["test"]["tokens"]]
    return out


# ---------------------------------------------------------------- number handling
_NUM = re.compile(r"^[+-]?(\d+[.,]?)+\d*%?$|^[+-]?\.\d+$")


def is_number(word: str) -> bool:
    return bool(_NUM.match(word))


def shape(word: str) -> str:
    """Digit shape that keeps the number of digits: 645.1 -> XXX.X, 12% -> XX%."""
    return re.sub(r"\d", "X", word)


def norm_token(word: str) -> str:
    """Normalisation used by the baseline and the HMM: numbers -> shape, words -> lowercase."""
    return shape(word) if is_number(word) else word.lower()


# ---------------------------------------------------------------- evaluation
def _pad(gold: list[list[str]], pred: list[list[str]]) -> list[list[str]]:
    """Make every predicted sentence as long as its gold sentence (missing words -> O)."""
    if len(gold) != len(pred):
        raise ValueError(f"{len(pred)} predicted sentences for {len(gold)} gold sentences")
    fixed = []
    for g, p in zip(gold, pred):
        p = list(p)[: len(g)]
        fixed.append(p + ["O"] * (len(g) - len(p)))
    return fixed


def scores(gold: list[list[str]], pred: list[list[str]], report: bool = False) -> dict:
    """Entity-level scores. Strict IOB2 mode: an entity counts only if both
    boundaries and the type match exactly."""
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2

    pred = _pad(gold, pred)
    kw = dict(mode="strict", scheme=IOB2, zero_division=0)
    res = {
        "micro_P": precision_score(gold, pred, average="micro", **kw),
        "micro_R": recall_score(gold, pred, average="micro", **kw),
        "micro_F1": f1_score(gold, pred, average="micro", **kw),
        "macro_F1": f1_score(gold, pred, average="macro", **kw),
    }
    res = {k: round(float(v), 4) for k, v in res.items()}
    if report:
        res["report"] = classification_report(gold, pred, output_dict=True, **kw)
    return res


def evaluate(test: pd.DataFrame, pred: list[list[str]], name: str, verbose: bool = True) -> dict:
    """Score predictions for the WHOLE cleaned test DataFrame from `load_splits()`.
    Returns {'model', 'main': {...}, 'reference': {...}, 'report': per-class (main)}."""
    gold = list(test["tags"])
    keep = ~test["in_validation"].to_numpy()
    main = scores([g for g, k in zip(gold, keep) if k], [p for p, k in zip(pred, keep) if k], report=True)
    ref = scores(gold, pred)
    res = {"model": name,
           "main": {"sentences": int(keep.sum()), **{k: v for k, v in main.items() if k != "report"}},
           "reference": {"sentences": len(gold), **ref},
           "report": main["report"]}
    if verbose:
        m, r = res["main"], res["reference"]
        print(f"{name:32s} MAIN micro-F1 {m['micro_F1']:.4f}  macro-F1 {m['macro_F1']:.4f}  "
              f"(P {m['micro_P']:.4f} R {m['micro_R']:.4f})   | full test micro-F1 {r['micro_F1']:.4f}")
    return res


def save_results(res: dict, key: str) -> None:
    """Write results/<key>_eval.json (summary) and results/<key>_per_class.csv (main view)."""
    os.makedirs(RESULTS, exist_ok=True)
    rows = [{"label": k, **v} for k, v in res.get("report", {}).items() if isinstance(v, dict)]
    pd.DataFrame(rows).round(4).to_csv(os.path.join(RESULTS, f"{key}_per_class.csv"), index=False)
    with open(os.path.join(RESULTS, f"{key}_eval.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "report"}, f, indent=2)


def confusions(gold: list[list[str]], pred: list[list[str]], top: int = 15):
    """Most frequent (gold entity type -> predicted type or O) errors on gold entity words."""
    c = Counter()
    for g, p in zip(gold, _pad(gold, pred)):
        for a, b in zip(g, p):
            if a != "O" and a[2:] != (b[2:] if b != "O" else "O"):
                c[(a[2:], b[2:] if b != "O" else "O")] += 1
    return c.most_common(top)
