"""Checks for src/cleaning_ablation.py and src/latency.py.   Run:  python -m pytest tests -q"""
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cleaning_ablation import counts, spans  # noqa: E402
from finer import scores  # noqa: E402
import latency  # noqa: E402


def test_spans_strict():
    assert spans(["B-A", "I-A", "O", "B-B"]) == {(0, 2, "A"), (3, 4, "B")}
    assert spans(["O", "I-A", "I-A"]) == set()            # orphan I- starts nothing (strict IOB2)
    assert spans(["B-A", "I-B"]) == {(0, 1, "A")}          # type change ends the entity


def test_counts_match_shared_evaluator():
    # pooled TP / gold / predicted counts must give the same micro P, R, F1 as finer.scores
    rng = random.Random(0)
    tags = ["O", "O", "O", "B-A", "I-A", "B-B", "I-B"]
    gold = [[rng.choice(tags) for _ in range(rng.randint(1, 12))] for _ in range(300)]
    gold = [[t if not (t.startswith("I-") and (i == 0 or g[i - 1][2:] != t[2:])) else "O"
             for i, t in enumerate(g)] for g in gold]            # make gold valid IOB2
    pred = [[rng.choice(tags) for _ in g] for g in gold]
    tp, ng, npred = counts(gold, pred).sum(0)
    ref = scores(gold, pred)
    assert round(tp / npred, 4) == ref["micro_P"]
    assert round(tp / ng, 4) == ref["micro_R"]
    assert round(2 * tp / (ng + npred), 4) == ref["micro_F1"]


def test_latency_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(latency, "RESULTS", str(tmp_path))
    table = pd.DataFrame([dict(model="HMM + Viterbi", key="hmm", device="cpu", hardware="CPU (test)",
                               sentences=10, words=100, sentences_per_s=5.0, ms_per_sentence=200.0,
                               parameters=7, weights="fitted on train")])
    latency.write_markdown(table)
    text = (tmp_path / "LATENCY.md").read_text()
    assert "| HMM + Viterbi |" in text and "CPU (test)" in text
