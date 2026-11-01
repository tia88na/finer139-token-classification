"""Hand-made features for the CRF baseline: word shape, number detection, surrounding words.

Every word is described by a few attributes (computed once per distinct word):

    norm     finer.norm_token: numbers -> digit shape (645.1 -> XXX.X), words -> lowercase
    shape    short word shape: X upper, x lower, d digit, repeats collapsed (Revenue -> Xx, 645.1 -> d.d)
    case     number / lower / title / upper / alnum / mixed / punct
    numkind  number detection: integer / thousands (1,250) / decimal / percent / year / - (not a number)
    ndigits  number of digits (capped at 9)
    prefix3, suffix3 of the lowercased word (<num> for numbers)

A feature template reads one attribute at a position relative to the current word,
e.g. norm[-1] = the previous word. Each template has its own weight table
(values x tags), so the CRF emission score of tag y at word i is

    e_i[y] = sum over templates f of W_f[value of f at i, y]

which is the usual log-linear CRF with binary indicator features. Templates:

    current word        all seven attributes, plus a bias
    surrounding words   norm at -W..-1 and +1..+W (W = window), shape at -2, -1, +1, +2
    word pairs          norm[-2]|norm[-1], norm[+1]|norm[+2], norm[-1]|shape[0]
    left context        bag of the 8 previous words (position ignored), so the deciding words
                        in "the interest rate of approximately 5.0%" are seen from the number

Positions outside the sentence read <BOS> / <EOS>. Attribute values (and word pairs) seen
fewer than `min_count` times in train are unknown; for `norm` the unknown value is one of
the HMM's classes <NUM> / <CAP> / <UNK>, so the model learns weights for unseen words.
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_hmm import HMM  # noqa: E402  (unknown-word classes <NUM> / <CAP> / <UNK>)
from finer import is_number, norm_token  # noqa: E402

ATTRS = ("norm", "shape", "case", "numkind", "ndigits", "prefix3", "suffix3")
SPECIAL = ["<PAD>", "<BOS>", "<EOS>"]
BOS, EOS = 1, 2
SHAPE_OFFSETS = (-2, -1, 1, 2)
PAIRS = (("norm[-2]|norm[-1]", ("norm", -2), ("norm", -1)),
         ("norm[+1]|norm[+2]", ("norm", 1), ("norm", 2)),
         ("norm[-1]|shape[0]", ("norm", -1), ("shape", 0)))
_YEAR = re.compile(r"(19|20)\d\d")


def short_shape(word: str) -> str:
    s = re.sub(r"\d", "d", re.sub("[a-z]", "x", re.sub("[A-Z]", "X", word)))
    return re.sub(r"(.)\1+", r"\1", s)


def casing(word: str) -> str:
    if is_number(word):
        return "number"
    if not any(ch.isalnum() for ch in word):
        return "punct"
    if word.islower():
        return "lower"
    if word.isupper():
        return "upper"
    if word.istitle():
        return "title"
    return "alnum" if any(ch.isdigit() for ch in word) else "mixed"


def number_kind(word: str) -> str:
    if not is_number(word):
        return "-"
    if word.endswith("%"):
        return "percent"
    if _YEAR.fullmatch(word):
        return "year"
    if "." in word:
        return "decimal"
    return "thousands" if "," in word else "integer"


def attributes(word: str) -> tuple[str, ...]:
    """The seven attribute values of a word, in the order of ATTRS."""
    num, low = is_number(word), word.lower()
    digits = sum(ch.isdigit() for ch in word)
    return (norm_token(word), short_shape(word), casing(word), number_kind(word), str(min(digits, 9)),
            "<num>" if num else low[:3], "<num>" if num else low[-3:])


def _unknown(attr: str, word: str) -> str:
    return HMM.unk_class(word) if attr == "norm" else "<UNK>"


def _shift(col: np.ndarray, k: int, pos: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Value of `col` k words away from every word; <BOS> / <EOS> outside the sentence."""
    if k == 0:
        return col
    inside = (pos + k >= 0) & (pos + k < length)
    src = np.clip(np.arange(len(col)) + k, 0, max(len(col) - 1, 0))
    return np.where(inside, col[src], BOS if k < 0 else EOS)


class CRFFeatures:
    """Builds the vocabularies on train (`fit`) and turns sentences into feature-id rows
    (`transform`): one row per word, one column per template use, ids index one shared
    weight matrix where template f owns rows base[f] .. base[f] + size[f] - 1."""

    def __init__(self, window: int = 3, min_count: int = 2, max_pairs: int = 200_000, left_bag: int = 8):
        self.window, self.min_count, self.max_pairs, self.left_bag = window, min_count, max_pairs, left_bag
        self.offsets = [k for k in range(-window, window + 1) if k != 0]
        self.BAG = f"bag(norm[-{left_bag}..-1])"

    # ------------------------------------------------------------------ vocabularies
    def fit(self, tokens: list[list[str]]) -> "CRFFeatures":
        counts = Counter(w for sent in tokens for w in sent)
        values = [Counter() for _ in ATTRS]
        for w, n in counts.items():
            for c, v in zip(values, attributes(w)):
                c[v] += n
        self.itos = {}
        for a, c in zip(ATTRS, values):
            unknown = HMM.UNK if a == "norm" else ["<UNK>"]
            kept = sorted((v for v, n in c.items() if n >= self.min_count), key=lambda v: (-c[v], v))
            self.itos[a] = SPECIAL + unknown + kept
        self.stoi = {a: {v: i for i, v in enumerate(s)} for a, s in self.itos.items()}
        self._word_ids = {}

        atoms, _, pos, length = self._atoms(tokens)
        self.pairs = {}
        for name, first, second in PAIRS:
            keys, cnt = np.unique(self._pair_keys(atoms, first, second, pos, length), return_counts=True)
            keys, cnt = keys[cnt >= self.min_count], cnt[cnt >= self.min_count]
            if len(keys) > self.max_pairs:  # keep the most frequent pairs (ties: smaller key)
                keys = np.sort(keys[np.lexsort((keys, -cnt))[: self.max_pairs]])
            self.pairs[name] = keys

        sizes = [(f"{a}[0]", len(self.itos[a])) for a in ATTRS] + [("bias", 1)]
        sizes += [(f"norm[{k:+d}]", len(self.itos["norm"])) for k in self.offsets]
        sizes += [(f"shape[{k:+d}]", len(self.itos["shape"])) for k in SHAPE_OFFSETS]
        sizes += [(name, len(self.pairs[name]) + 1) for name, _, _ in PAIRS]  # id 0 = unseen pair
        sizes += [(self.BAG, len(self.itos["norm"]))]
        self.size, self.base, total = dict(sizes), {}, 0
        for name, n in sizes:
            self.base[name] = total
            total += n
        self.n_features = total
        self.columns = ([f"{a}[0]" for a in ATTRS] + ["bias"] + [f"norm[{k:+d}]" for k in self.offsets]
                        + [f"shape[{k:+d}]" for k in SHAPE_OFFSETS] + [name for name, _, _ in PAIRS]
                        + [self.BAG] * self.left_bag)
        return self

    def _ids(self, word: str) -> tuple[int, ...]:
        ids = self._word_ids.get(word)
        if ids is None:
            ids = tuple(self.stoi[a].get(v, self.stoi[a][_unknown(a, word)]) for a, v in zip(ATTRS, attributes(word)))
            self._word_ids[word] = ids
        return ids

    def _atoms(self, tokens):
        """Attribute ids of every word [N, len(ATTRS)], sentence starts, position in sentence, sentence length."""
        lens = np.fromiter(map(len, tokens), dtype=np.int64, count=len(tokens))
        starts = np.zeros(len(tokens) + 1, dtype=np.int64)
        np.cumsum(lens, out=starts[1:])
        types = {}
        flat = np.fromiter((types.setdefault(w, len(types)) for sent in tokens for w in sent),
                           dtype=np.int64, count=int(starts[-1]))
        table = np.array([self._ids(w) for w in types], dtype=np.int64).reshape(-1, len(ATTRS))
        pos = np.arange(starts[-1]) - np.repeat(starts[:-1], lens)
        return table[flat], starts, pos, np.repeat(lens, lens)

    def _pair_keys(self, atoms, first, second, pos, length):
        (a1, k1), (a2, k2) = first, second
        x = _shift(atoms[:, ATTRS.index(a1)], k1, pos, length)
        y = _shift(atoms[:, ATTRS.index(a2)], k2, pos, length)
        return x * len(self.itos[a2]) + y

    # ------------------------------------------------------------------ features
    def transform(self, tokens: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
        """Feature ids [N words, n columns] (int32) and sentence starts [n sentences + 1]."""
        atoms, starts, pos, length = self._atoms(tokens)
        out = np.empty((len(atoms), len(self.columns)), dtype=np.int32)
        col = {a: atoms[:, i] for i, a in enumerate(ATTRS)}
        j = 0

        def put(name, ids):
            nonlocal j
            out[:, j] = self.base[name] + ids
            j += 1

        for a in ATTRS:
            put(f"{a}[0]", col[a])
        put("bias", 0)
        for k in self.offsets:
            put(f"norm[{k:+d}]", _shift(col["norm"], k, pos, length))
        for k in SHAPE_OFFSETS:
            put(f"shape[{k:+d}]", _shift(col["shape"], k, pos, length))
        for name, first, second in PAIRS:
            keys, vocab = self._pair_keys(atoms, first, second, pos, length), self.pairs[name]
            at = np.clip(np.searchsorted(vocab, keys), 0, max(len(vocab) - 1, 0))
            found = vocab[at] == keys if len(vocab) else np.zeros(len(keys), dtype=bool)
            put(name, np.where(found, at + 1, 0))
        for k in range(1, self.left_bag + 1):
            put(self.BAG, _shift(col["norm"], -k, pos, length))
        return out, starts

    def keep_columns(self, window: int) -> np.ndarray:
        """Columns to use for a smaller surrounding-word window than the one extracted."""
        drop = {f"norm[{k:+d}]" for k in self.offsets if abs(k) > window}
        return np.array([i for i, c in enumerate(self.columns) if c not in drop])

    def explain(self, row) -> list[str]:
        """Readable 'template=value' strings for one row of `transform` (tests, inspection)."""
        names = sorted(self.base, key=self.base.get)
        bases = np.array([self.base[n] for n in names])
        out = []
        for f in row:
            name = names[int(np.searchsorted(bases, f, side="right")) - 1]
            v = int(f) - self.base[name]
            pair = next((p for p in PAIRS if p[0] == name), None)
            if name == "bias":
                value = "1"
            elif pair:
                (a1, _), (a2, _) = pair[1], pair[2]
                value = "<unseen>" if v == 0 else "|".join(
                    self.itos[a][x] for a, x in zip((a1, a2), divmod(int(self.pairs[name][v - 1]), len(self.itos[a2]))))
            else:
                attr = "norm" if name == self.BAG else name.split("[")[0]
                value = self.itos[attr][v]
            out.append(f"{name}={value}")
        return out
