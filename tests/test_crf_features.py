"""Checks for the CRF features (word shape, number detection, surrounding words).   Run:  python -m pytest tests -q"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from crf_features import CRFFeatures, attributes, casing, number_kind, short_shape  # noqa: E402

SENTS = [["The", "rate", "of", "5.0%", "."], ["Revenue", "was", "645.1", "million"]] * 2


def features(sentence, window=2):
    f = CRFFeatures(window=window).fit(SENTS)
    rows, starts = f.transform([sentence])
    return f, rows, starts


def as_dict(f, row):
    return dict(x.split("=", 1) for x in f.explain(row) if not x.startswith("bag"))


def test_word_attributes():
    assert attributes("645.1")[:5] == ("XXX.X", "d.d", "number", "decimal", "4")
    assert attributes("12%")[:4] == ("XX%", "d%", "number", "percent")
    assert attributes("1,250")[:4] == ("X,XXX", "d,d", "number", "thousands")
    assert number_kind("2019") == "year" and number_kind("250") == "integer"
    assert attributes("Revenue") == ("revenue", "Xx", "title", "-", "0", "rev", "nue")
    assert casing("(") == "punct" and number_kind("(") == "-"
    assert short_shape("U.S.") == "X.X."


def test_surrounding_words_and_sentence_edges():
    f, rows, starts = features(["The", "rate", "of", "5.0%", "."])
    assert starts.tolist() == [0, 5] and rows.shape == (5, len(f.columns))
    assert [x.split("=")[0] for x in f.explain(rows[0])] == f.columns
    first = as_dict(f, rows[0])
    assert first["norm[-1]"] == first["norm[-2]"] == "<BOS>" and first["norm[+1]"] == "rate"
    num = as_dict(f, rows[3])
    assert num["norm[0]"] == "X.X%" and num["shape[0]"] == "d.d%" and num["numkind[0]"] == "percent"
    assert (num["norm[-2]"], num["norm[-1]"], num["norm[+1]"], num["norm[+2]"]) == ("rate", "of", ".", "<EOS>")
    assert num["norm[-2]|norm[-1]"] == "rate|of" and num["norm[-1]|shape[0]"] == "of|d.d%"
    bag = [x.split("=", 1)[1] for x in f.explain(rows[3]) if x.startswith("bag")]
    assert bag == ["of", "rate", "the"] + ["<BOS>"] * 5


def test_unknown_words():
    f, rows, _ = features(["Zebra", "7,777,777", "xyz"])
    assert [as_dict(f, r)["norm[0]"] for r in rows] == ["<CAP>", "<NUM>", "<UNK>"]
    assert as_dict(f, rows[1])["norm[-2]|norm[-1]"] == "<unseen>"  # pair <BOS>|<CAP> never seen in train


def test_vocabulary_is_deterministic():
    a = CRFFeatures().fit(SENTS)
    b = CRFFeatures().fit(SENTS[::-1])
    assert a.itos == b.itos and a.n_features == b.n_features
    assert all((a.pairs[k] == b.pairs[k]).all() for k in a.pairs)
    assert (a.transform(SENTS)[0] == b.transform(SENTS)[0]).all()


def test_smaller_window_drops_far_words():
    f = CRFFeatures(window=3).fit(SENTS)
    kept = [f.columns[i] for i in f.keep_columns(2)]
    assert "norm[-3]" not in kept and "norm[+3]" not in kept and "norm[-2]" in kept
    assert len(kept) == len(f.columns) - 2
