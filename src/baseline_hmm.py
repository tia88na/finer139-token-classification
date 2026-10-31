"""Model 0: most-frequent-tag baseline.   Model 1: first-order HMM decoded with Viterbi.

Both models see the same normalised tokens: numbers -> digit shape (645.1 -> XXX.X),
words -> lowercase. Neither model sees the words *around* a number except through the
previous tag, which is exactly what this experiment is meant to show.

Run (CPU, about 7 minutes):   python src/baseline_hmm.py
Writes results/{baseline_mft,hmm}_metrics.json, _per_class.csv, _test_predictions.parquet
and results/hmm_analysis.txt.
"""
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finer import (LABEL2ID, RESULTS, confusions, evaluate, is_number,  # noqa: E402
                   load_splits, norm_token, save_results)


# ------------------------------------------------------------------ model 0
def most_frequent_tag(train):
    """Each normalised token gets the tag it had most often in train; unseen -> O."""
    cnt = defaultdict(Counter)
    for toks, tags in zip(train.tokens, train.tags):
        for w, t in zip(toks, tags):
            cnt[norm_token(w)][t] += 1
    lex = {w: c.most_common(1)[0][0] for w, c in cnt.items()}
    return lambda toks: [lex.get(norm_token(w), "O") for w in toks]


# ------------------------------------------------------------------ model 1
class HMM:
    """First-order HMM with add-k smoothing.

    P(tags, words) = pi(t1) b(t1, w1) prod_i a(t_{i-1}, t_i) b(t_i, w_i)

    Words seen fewer than twice in train are mapped to one of three unknown-word
    classes (<NUM>, <CAP>, <UNK>), so the model learns how likely each tag is to emit
    an unseen number / capitalised word / other word.
    """

    UNK = ["<NUM>", "<CAP>", "<UNK>"]

    def __init__(self, k=0.01):
        self.k = k

    @classmethod
    def unk_class(cls, w):
        if is_number(w):
            return "<NUM>"
        return "<CAP>" if w[:1].isupper() else "<UNK>"

    def _col(self, w):
        col = self.wi.get(norm_token(w))
        return col if col is not None else self.V + self.UNK.index(self.unk_class(w))

    def fit(self, train):
        self.tags = sorted({t for ts in train.tags for t in ts})   # states seen in train
        ti = {t: i for i, t in enumerate(self.tags)}
        wc = Counter(norm_token(w) for toks in train.tokens for w in toks)
        self.wi = {w: i for i, w in enumerate(w for w, c in wc.items() if c >= 2)}
        self.V = len(self.wi)
        S = len(self.tags)
        start = np.full(S, self.k)
        trans = np.full((S, S), self.k)
        emit = np.full((S, self.V + len(self.UNK)), self.k)
        for toks, ts in zip(train.tokens, train.tags):
            prev = None
            for w, t in zip(toks, ts):
                j = ti[t]
                emit[j, self._col(w)] += 1
                if prev is None:
                    start[j] += 1
                else:
                    trans[prev, j] += 1
                prev = j
        self.log_start = np.log(start / start.sum())
        self.log_trans = np.log(trans / trans.sum(1, keepdims=True))
        self.log_emit = np.log(emit / emit.sum(1, keepdims=True))
        return self

    def predict(self, toks):
        """Viterbi: keep the best path ending in each tag, then backtrace."""
        cols = [self._col(w) for w in toks]
        n, S = len(cols), len(self.tags)
        back = np.zeros((n, S), dtype=np.int32)
        score = self.log_start + self.log_emit[:, cols[0]]
        for i in range(1, n):
            cand = score[:, None] + self.log_trans          # [previous tag, current tag]
            back[i] = cand.argmax(0)
            score = cand.max(0) + self.log_emit[:, cols[i]]
        path = [int(score.argmax())]
        for i in range(n - 1, 0, -1):
            path.append(int(back[i, path[-1]]))
        return [self.tags[j] for j in reversed(path)]


# ------------------------------------------------------------------ analysis
def analysis(train, test, pred, hmm):
    """Numbers for the slides: why a token-only model cannot solve FiNER."""
    lines = []
    # 1) How ambiguous is a number shape?  (tags per shape among labelled numbers)
    by_shape = defaultdict(Counter)
    for toks, tags in zip(train.tokens, train.tags):
        for w, t in zip(toks, tags):
            if is_number(w):
                by_shape[norm_token(w)][t] += 1
    top = sorted(by_shape.items(), key=lambda kv: -sum(kv[1].values()))[:8]
    lines.append("Most frequent number shapes in train: share tagged O and number of distinct entity types")
    for s, c in top:
        tot = sum(c.values())
        types = len({t[2:] for t in c if t != "O"})
        lines.append(f"  {s:10s} {tot:>9,} occurrences   O = {c['O']/tot:6.1%}   entity types = {types}")
    # 2) Where does the HMM lose?  gold entity words predicted O
    ent = miss = 0
    for g, p in zip(test.tags, pred):
        for a, b in zip(g, p):
            if a != "O":
                ent += 1
                miss += b == "O"
    lines.append(f"\nGold entity words in test: {ent:,}; HMM predicts O for {miss:,} ({miss/ent:.1%}).")
    # 3) The transition out of O is tiny, so Viterbi rarely leaves O
    o = hmm.tags.index("O")
    p_oo = math.exp(hmm.log_trans[o, o])
    lines.append(f"P(O -> O) = {p_oo:.4f}; probability of leaving O at any word = {1 - p_oo:.4f}.")
    lines.append("The only evidence for an entity is the number's shape, and the same shape is O most of the time,\n"
                 "so the most likely path stays in O. The words that decide the tag ('interest rate of', 'matures in')\n"
                 "are not visible to an HMM emission b(tag, word).")
    lines.append("\nTop confusions on gold entity words (gold type -> predicted), whole test set:")
    for (a, b), c in confusions(list(test.tags), pred, 10):
        lines.append(f"  {c:6d}  {a[:60]:60s} -> {b[:40]}")
    txt = "\n".join(lines)
    with open(os.path.join(RESULTS, "hmm_analysis.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


def save_predictions(pred, key):
    """Word-level predicted tag ids; row i = row i of data/cleaned/test.parquet (same format as
    the transformer predictions), so the analysis person can load every model the same way."""
    pd.DataFrame({"predicted_tags": [[LABEL2ID[t] for t in p] for p in pred]}).to_parquet(
        os.path.join(RESULTS, f"{key}_test_predictions.parquet"))


def main():
    d = load_splits()
    train, test = d["train"], d["test"]
    print(f"train {len(train):,} sentences | test {len(test):,} "
          f"({int(test.in_validation.sum()):,} also in validation -> main view {int((~test.in_validation).sum()):,})")

    t0 = time.time()
    mft = most_frequent_tag(train)
    pred = [mft(t) for t in test.tokens]
    r = evaluate(test, pred, "Most-frequent tag")
    save_results(r, "baseline_mft")
    save_predictions(pred, "baseline_mft")
    print(f"  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    hmm = HMM().fit(train)
    print(f"HMM: {len(hmm.tags)} states, vocabulary {hmm.V:,} + {len(HMM.UNK)} unknown classes")
    pred = [hmm.predict(t) for t in test.tokens]
    r = evaluate(test, pred, "HMM + Viterbi")
    save_results(r, "hmm")
    save_predictions(pred, "hmm")
    print(f"  ({time.time() - t0:.0f}s)")
    analysis(train, test, pred, hmm)


if __name__ == "__main__":
    main()
