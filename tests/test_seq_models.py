"""Checks that the CRF and the BiLSTM-CRF learn from context and keep the shared prediction
format (one row per sentence, input order, full length, global tag ids).
Run:  python -m pytest tests -q"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from baseline_crf import Batches as CRFBatches, FeatureCRF  # noqa: E402
from bilstm_crf import Batches as LSTMBatches, BiLSTMCRF, Vocab  # noqa: E402
from crf_features import CRFFeatures  # noqa: E402
from finer import ID2LABEL, LABEL2ID  # noqa: E402
from seq_utils import TagSet, batches, check_predictions, fit, predict, set_seed  # noqa: E402

RATE = "B-DebtInstrumentInterestRateStatedPercentage"
# The same number is an entity in one context and O in the other: only the words before it decide.
TRAIN = pd.DataFrame({"tokens": [["rate", "of", "5.0"]] * 5 + [["the", "year", "5.0"]] * 5,
                      "tags": [["O", "O", RATE]] * 5 + [["O", "O", "O"]] * 5})
TEST = pd.DataFrame({"tokens": [["the", "year", "8.5"], ["rate", "of", "7.5"], ["rate"]]})
GOLD = [["O", "O", "O"], ["O", "O", RATE], ["O"]]


def train_crf(seed=0):
    set_seed(seed)
    tags = TagSet(TRAIN.tags)
    feats = CRFFeatures(window=2).fit(TRAIN.tokens)
    f, s = feats.transform(TRAIN.tokens)
    keep = feats.keep_columns(2)
    model = FeatureCRF(feats.n_features, tags.names)
    fit(model, torch.optim.Adam(model.parameters(), lr=0.1),
        CRFBatches(f, s, keep, "cpu", [np.array(tags.encode(t)) for t in TRAIN.tags]), np.diff(s),
        CRFBatches(f, s, keep, "cpu"), list(TRAIN.tags), tags, epochs=30, patience=30, batch_tokens=1000,
        log=lambda *_: None)
    ft, st = feats.transform(TEST.tokens)
    return [tags.decode(p) for p in predict(model, CRFBatches(ft, st, keep, "cpu"), np.diff(st), 1000)]


def train_bilstm(seed=0):
    set_seed(seed)
    tags = TagSet(TRAIN.tags)
    vocab = Vocab(TRAIN.tokens)
    model = BiLSTMCRF(len(vocab.words), len(vocab.chars), len(vocab.cases), tags.names,
                      word_dim=8, char_dim=4, char_filters=4, hidden=16, dropout=0.0, word_dropout=0.0)
    enc = vocab.encode(TRAIN.tokens)
    fit(model, torch.optim.Adam(model.parameters(), lr=0.02),
        LSTMBatches(enc, "cpu", [np.array(tags.encode(t)) for t in TRAIN.tags]), np.diff(enc["starts"]),
        LSTMBatches(enc, "cpu"), list(TRAIN.tags), tags, epochs=40, patience=40, batch_tokens=1000,
        log=lambda *_: None)
    test = vocab.encode(TEST.tokens)
    return [tags.decode(p) for p in predict(model, LSTMBatches(test, "cpu"), np.diff(test["starts"]), 1000)]


def test_crf_uses_context():
    assert train_crf() == GOLD


def test_bilstm_crf_uses_context():
    assert train_bilstm() == GOLD


def test_same_seed_same_predictions():
    assert train_crf(1) == train_crf(1)
    assert train_bilstm(1) == train_bilstm(1)


def test_prediction_format():
    pred = train_crf()
    check_predictions(TEST, pred)  # one row per sentence, same length, known tags
    ids = [[LABEL2ID[t] for t in p] for p in pred]  # what results/<key>_test_predictions.parquet stores
    assert [[ID2LABEL[i] for i in r] for r in ids] == pred


def test_batches_cover_every_sentence_once():
    lengths = [5, 1, 40, 7, 7, 3, 100, 2]
    for shuffle in (False, True):
        out = batches(lengths, 20, shuffle, np.random.default_rng(0))
        assert sorted(i for b in out for i in b) == list(range(len(lengths)))
        assert all(len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 20 for b in out)
