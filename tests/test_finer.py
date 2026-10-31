"""Checks for the shared evaluator and the HMM.   Run:  python -m pytest tests -q"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from finer import ID2LABEL, norm_token, scores  # noqa: E402
from baseline_hmm import HMM  # noqa: E402


def test_label_set():
    assert len(ID2LABEL) == 279 and ID2LABEL[0] == "O"
    b = {t[2:] for t in ID2LABEL if t.startswith("B-")}
    i = {t[2:] for t in ID2LABEL if t.startswith("I-")}
    assert len(b) == 139 and b == i


def test_lecture_example():
    # Week 6, slides 20-21: gold Ada (PER), New York (LOC); prediction Ada (PER), York (LOC).
    gold = [["B-PER", "O", "B-LOC", "I-LOC", "O"]]
    pred = [["B-PER", "O", "O", "B-LOC", "O"]]
    r = scores(gold, pred)
    assert r["micro_P"] == 0.5 and r["micro_R"] == 0.5 and r["micro_F1"] == 0.5


def test_all_O_has_zero_recall():
    # Week 6, slide 17: predicting O everywhere gives high accuracy but zero entity recall.
    gold = [["O"] * 9 + ["B-PER"]] * 10
    pred = [["O"] * 10] * 10
    assert scores(gold, pred)["micro_R"] == 0.0


def test_truncated_prediction_counts_as_missed():
    # A prediction shorter than the sentence must not hide the entity at the end.
    gold = [["O", "O", "B-PER"]]
    assert scores(gold, [["O", "O"]])["micro_R"] == 0.0
    assert scores(gold, [["O", "O", "B-PER"]])["micro_F1"] == 1.0


def test_wrong_sentence_count_is_an_error():
    with pytest.raises(ValueError):
        scores([["O"], ["O"]], [["O"]])


def test_norm_token():
    assert norm_token("645.1") == "XXX.X"
    assert norm_token("1,250") == "X,XXX"
    assert norm_token("12%") == "XX%"
    assert norm_token("Revenue") == "revenue"


def test_hmm_viterbi_matches_lecture():
    # Week 6, slides 26-29: "can fish", best path V N with joint weight 0.1344.
    h = HMM()
    h.tags = ["N", "V"]
    h.wi, h.V = {"can": 0, "fish": 1}, 2
    h.log_start = np.log([0.6, 0.4])
    h.log_trans = np.log([[0.3, 0.7], [0.8, 0.2]])
    h.log_emit = np.log([[0.4, 0.6, 1e-9, 1e-9, 1e-9], [0.7, 0.3, 1e-9, 1e-9, 1e-9]])
    assert h.predict(["can", "fish"]) == ["V", "N"]


def test_hmm_learns_simple_sequence():
    df = pd.DataFrame({"tokens": [["rate", "of", "5.0%"]] * 5 + [["the", "year"]] * 5,
                       "tags": [["O", "O", "B-RATE"]] * 5 + [["O", "O"]] * 5})
    h = HMM().fit(df)
    assert h.predict(["rate", "of", "7.5%"]) == ["O", "O", "B-RATE"]
