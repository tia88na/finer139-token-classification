"""Model 3: BiLSTM-CRF (Lample et al. 2016; Ma & Hovy 2016).

Each word is represented by
    * a word embedding of finer.norm_token (numbers -> digit shape, words -> lowercase);
      train words seen at least twice, the others -> <NUM> / <CAP> / <UNK> as in the HMM,
    * a character CNN over the raw word (keeps capitals, digits, %, $),
    * a small casing embedding (number / lower / title / upper / ..., crf_features.casing).
A bidirectional LSTM reads the whole sentence, so each word's vector depends on all the
words around it. A linear layer turns it into emission scores, and the same CRF layer as
the feature CRF (src/crf_layer.py) adds tag-transition scores: training minimises the CRF
negative log-likelihood of the gold tag sequence and prediction is Viterbi with the IOB2
constraints. There is no per-word softmax anywhere. Embeddings are learned from scratch on
the cleaned train set (no pre-trained vectors, nothing is downloaded).

The epoch and the settings in GRID are chosen on the 20,000-sentence validation sample;
the test set is decoded once, with the selected model.

Run:
    python src/bilstm_crf.py --limit 2000     # quick check: nothing saved, test not scored
    python src/bilstm_crf.py --grid           # choose settings on validation, then test
    python src/bilstm_crf.py --dropout 0.5 --layers 1   # one setting, then test
Writes results/bilstm_crf_{test_predictions.parquet,eval.json,per_class.csv,metrics.json,search.csv}
and models/bilstm_crf.pt.
"""
import argparse
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_hmm import HMM  # noqa: E402  (unknown-word classes <NUM> / <CAP> / <UNK>)
from crf_features import casing  # noqa: E402
from crf_layer import CRF  # noqa: E402
from finer import norm_token  # noqa: E402
from seq_utils import add_common_args, finish, load_data, pad, search, set_seed  # noqa: E402

KEY, NAME = "bilstm_crf", "BiLSTM-CRF"
# Fixed before the full runs; every point is scored on validation only.
GRID = [dict(dropout=dr, layers=n) for n in (1, 2) for dr in (0.33, 0.5)]
CASES = ["number", "lower", "title", "upper", "alnum", "mixed", "punct"]


class Vocab:
    """Word, character and casing ids, built on train. Id 0 is padding everywhere."""

    def __init__(self, tokens, min_count: int = 2, max_chars: int = 20):
        words = Counter(norm_token(w) for sent in tokens for w in sent)
        chars = Counter(ch for sent in tokens for w in sent for ch in w)
        self.words = ["<PAD>"] + HMM.UNK + sorted((w for w, c in words.items() if c >= min_count),
                                                  key=lambda w: (-words[w], w))
        self.chars = ["<PAD>", "<UNK>"] + sorted(ch for ch, c in chars.items() if c >= min_count)
        self.cases = ["<PAD>"] + CASES
        self.word_id = {w: i for i, w in enumerate(self.words)}
        self.char_id = {ch: i for i, ch in enumerate(self.chars)}
        self.max_chars = max_chars

    def _encode_word(self, w):
        unk = self.word_id[HMM.unk_class(w)]
        chars = [self.char_id.get(ch, 1) for ch in w[: self.max_chars]]
        return (self.word_id.get(norm_token(w), unk), unk, self.cases.index(casing(w)),
                chars + [0] * (self.max_chars - len(chars)))

    def encode(self, tokens) -> dict:
        """Per-word arrays of a split (computed once per distinct word) + sentence starts."""
        lens = np.fromiter(map(len, tokens), dtype=np.int64, count=len(tokens))
        starts = np.zeros(len(tokens) + 1, dtype=np.int64)
        np.cumsum(lens, out=starts[1:])
        types = {}
        flat = np.fromiter((types.setdefault(w, len(types)) for sent in tokens for w in sent),
                           dtype=np.int64, count=int(starts[-1]))
        enc = [self._encode_word(w) for w in types]
        return dict(words=np.array([e[0] for e in enc], dtype=np.int32)[flat],
                    unk=np.array([e[1] for e in enc], dtype=np.int32)[flat],
                    cases=np.array([e[2] for e in enc], dtype=np.uint8)[flat],
                    chars=np.array([e[3] for e in enc], dtype=np.uint8 if len(self.chars) < 256 else np.int32)[flat],
                    starts=starts)


class BiLSTMCRF(nn.Module):
    def __init__(self, n_words, n_chars, n_cases, tags, *, word_dim=100, char_dim=30, char_filters=50,
                 case_dim=8, hidden=256, layers=1, dropout=0.5, word_dropout=0.05, constrain_training=True):
        super().__init__()
        self.word = nn.Embedding(n_words, word_dim, padding_idx=0)
        self.char = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.case = nn.Embedding(n_cases, case_dim, padding_idx=0)
        for emb in (self.word, self.char, self.case):  # uniform init as in Ma & Hovy (2016)
            bound = math.sqrt(3.0 / emb.embedding_dim)
            nn.init.uniform_(emb.weight, -bound, bound)
            with torch.no_grad():
                emb.weight[0].zero_()
        self.conv = nn.Conv1d(char_dim, char_filters, kernel_size=3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(word_dim + char_filters + case_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(2 * hidden, len(tags))
        self.crf = CRF(tags, constrain_training)
        self.word_dropout = word_dropout

    def emissions(self, batch):
        words, chars, mask = batch["words"], batch["chars"], batch["mask"]
        if self.training and self.word_dropout:  # so the model learns what to do with unknown words
            drop = (torch.rand(words.shape, device=words.device) < self.word_dropout) & mask
            words = torch.where(drop, batch["unk"], words)
        b, t, c = chars.shape
        ch = self.conv(self.char(chars.reshape(b * t, c)).transpose(1, 2))  # [B*T, filters, chars]
        ch = ch.max(dim=2).values.view(b, t, -1)                               # max-pool over characters
        x = self.drop(torch.cat([self.word(words), ch, self.case(batch["cases"])], dim=-1))
        packed = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        h, _ = nn.utils.rnn.pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=t)
        return self.out(self.drop(h))

    def loss(self, batch):
        return self.crf.nll(self.emissions(batch), batch["tags"], batch["mask"]).sum()

    def decode(self, batch):
        return self.crf.decode(self.emissions(batch), batch["mask"])


class Batches:
    """Encoded words of one split; called with sentence indices it returns a padded batch."""

    def __init__(self, enc, device, tags=None):
        self.enc, self.device, self.tags = enc, device, tags

    def __call__(self, idx):
        s = self.enc["starts"]
        spans = [slice(s[i], s[i + 1]) for i in idx]
        batch = {k: pad([self.enc[k][sp] for sp in spans]).to(self.device) for k in ("words", "unk", "cases", "chars")}
        batch["mask"] = pad([np.ones(sp.stop - sp.start) for sp in spans], dtype=torch.bool).to(self.device)
        if self.tags is not None:
            batch["tags"] = pad([self.tags[i] for i in idx]).to(self.device)
        return batch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 8192 words + lr 2e-3 matched 4096 words + lr 1e-3 on validation (20k-sentence check) at half the steps
    add_common_args(p, epochs=12, patience=3, batch_tokens=8192, lr=2e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--hidden", type=int, default=256, help="LSTM units per direction")
    p.add_argument("--word-dim", type=int, default=100)
    p.add_argument("--word-dropout", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=5.0, help="gradient-norm clipping")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.perf_counter()
    train, val, test, tagset, constrain = load_data(args)
    vocab = Vocab(train.tokens)
    enc_train, enc_val = vocab.encode(train.tokens), vocab.encode(val.tokens)
    train_tags = [np.array(tagset.encode(t)) for t in train.tags]
    print(f"device {device} | vocabulary: {len(vocab.words):,} words, {len(vocab.chars)} characters "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    configs = GRID if args.grid else [dict(dropout=args.dropout, layers=args.layers)]
    configs = [dict(c, hidden=args.hidden, word_dim=args.word_dim, word_dropout=args.word_dropout, clip=args.clip)
               for c in configs]

    def make(cfg):
        return BiLSTMCRF(len(vocab.words), len(vocab.chars), len(vocab.cases), tagset.names,
                         word_dim=cfg["word_dim"], hidden=cfg["hidden"], layers=cfg["layers"],
                         dropout=cfg["dropout"], word_dropout=cfg["word_dropout"],
                         constrain_training=constrain).to(device)

    def build(cfg):
        model = make(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        return model, optimizer, Batches(enc_train, device, train_tags), Batches(enc_val, device)

    best, trials = search(configs, build, np.diff(enc_train["starts"]), list(val.tags), tagset, args, device)
    cfg, _, state = best
    model = make(cfg)
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    finish(KEY, NAME, args, model, Batches(vocab.encode(test.tokens), device), test, tagset, best, trials,
           metrics=dict(n_train=len(train), n_val=len(val), n_test=len(test), tags=len(tagset),
                        constrained_training=constrain, optimizer="Adam", parameters=n_params,
                        vocabulary=dict(words=len(vocab.words), chars=len(vocab.chars), min_count=2)),
           checkpoint=dict(vocab=dict(words=vocab.words, chars=vocab.chars, cases=vocab.cases,
                                      max_chars=vocab.max_chars)),
           device=device, t0=t0)


if __name__ == "__main__":
    main()
