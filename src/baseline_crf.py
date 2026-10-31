"""Model 2: linear-chain CRF with hand-made features (word shape, number detection, surrounding words).

Unlike the HMM, the CRF scores each tag with features of the words AROUND the current
word (src/crf_features.py) and normalises over whole tag sequences (src/crf_layer.py),
so "interest rate of 5.0%" and "matures in 2025" can decide the tag of a number.

Training maximises the L2-regularised conditional log-likelihood on the whole cleaned
train set with mini-batch gradients on the GPU (the objective of CRFsuite's `l2sgd`
trainer; a CPU CRFsuite with 168 tags x 13.1M words would need hours per pass).
Settings and the epoch are chosen on the 20,000-sentence validation sample; the test
set is decoded once, with the selected model.

Run:
    python src/baseline_crf.py --limit 2000      # quick check: nothing saved, test not scored
    python src/baseline_crf.py --grid            # choose settings on validation, then test
    python src/baseline_crf.py --window 3 --weight-decay 0.01   # one setting, then test
Writes results/crf_{test_predictions.parquet,eval.json,per_class.csv,metrics.json,search.csv}
and models/crf.pt.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crf_features import CRFFeatures  # noqa: E402
from crf_layer import CRF  # noqa: E402
from seq_utils import add_common_args, finish, load_data, pad, search, set_seed  # noqa: E402

KEY, NAME = "crf", "CRF"
# Fixed before the full runs; every point is scored on validation only.
GRID = [dict(window=w, weight_decay=wd) for w in (2, 3) for wd in (0.0, 0.01, 0.1)]


class FeatureCRF(nn.Module):
    """Emission score of tag y at word i = sum of the weights W[f, y] of the word's active
    features f (an EmbeddingBag in 'sum' mode is exactly that), plus the shared CRF layer."""

    def __init__(self, n_features: int, tags: list[str], constrain_training: bool = True):
        super().__init__()
        self.weights = nn.EmbeddingBag(n_features, len(tags), mode="sum")
        nn.init.zeros_(self.weights.weight)  # a CRF starts from all-zero weights
        self.crf = CRF(tags, constrain_training)

    def emissions(self, feats):
        b, t, c = feats.shape
        return self.weights(feats.reshape(b * t, c)).view(b, t, -1)

    def loss(self, batch):
        return self.crf.nll(self.emissions(batch["feats"]), batch["tags"], batch["mask"]).sum()

    def decode(self, batch):
        return self.crf.decode(self.emissions(batch["feats"]), batch["mask"])


class Batches:
    """Feature rows of one split; called with sentence indices it returns a padded batch."""

    def __init__(self, feats, starts, keep, device, tags=None):
        self.feats, self.starts, self.device, self.tags = feats, starts, device, tags
        self.keep = torch.as_tensor(keep)

    def __call__(self, idx):
        rows = [self.feats[self.starts[i]:self.starts[i + 1]] for i in idx]
        batch = {"feats": pad(rows)[:, :, self.keep].to(self.device),
                 "mask": pad([np.ones(len(r)) for r in rows], dtype=torch.bool).to(self.device)}
        if self.tags is not None:
            batch["tags"] = pad([self.tags[i] for i in idx]).to(self.device)
        return batch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, epochs=8, patience=2, batch_tokens=8192, lr=0.01)
    p.add_argument("--window", type=int, default=3, help="surrounding words on each side (norm[-W..+W])")
    p.add_argument("--weight-decay", type=float, default=0.01, help="L2 (decoupled, AdamW)")
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--max-pairs", type=int, default=200_000, help="word pairs kept per pair template")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.perf_counter()
    train, val, test, tagset, constrain = load_data(args)

    configs = GRID if args.grid else [dict(window=args.window, weight_decay=args.weight_decay)]
    feats = CRFFeatures(window=max(c["window"] for c in configs), min_count=args.min_count,
                        max_pairs=args.max_pairs).fit(train.tokens)
    f_train, s_train = feats.transform(train.tokens)
    f_val, s_val = feats.transform(val.tokens)
    train_tags = [np.array(tagset.encode(t)) for t in train.tags]
    mb = feats.n_features * len(tagset) * 4 / 2**20
    print(f"device {device} | features: {feats.n_features:,} rows x {len(tagset)} tags = {mb:,.0f} MB of weights; "
          f"{len(feats.columns)} columns per word; vocab norm {len(feats.itos['norm']):,} shape "
          f"{len(feats.itos['shape']):,} pairs {[len(v) for v in feats.pairs.values()]} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    def build(cfg):
        keep = feats.keep_columns(cfg["window"])
        model = FeatureCRF(feats.n_features, tagset.names, constrain).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg["weight_decay"])
        return (model, optimizer, Batches(f_train, s_train, keep, device, train_tags),
                Batches(f_val, s_val, keep, device))

    best, trials = search(configs, build, np.diff(s_train), list(val.tags), tagset, args, device)
    cfg, _, state = best
    model = FeatureCRF(feats.n_features, tagset.names, constrain).to(device)
    model.load_state_dict(state)
    f_test, s_test = feats.transform(test.tokens)
    finish(KEY, NAME, args, model, Batches(f_test, s_test, feats.keep_columns(cfg["window"]), device),
           test, tagset, best, trials,
           metrics=dict(n_train=len(train), n_val=len(val), n_test=len(test), tags=len(tagset),
                        constrained_training=constrain, optimizer="AdamW (decoupled weight decay = L2)",
                        features=dict(n_features=feats.n_features, columns=feats.columns, sizes=feats.size,
                                      min_count=args.min_count, max_pairs=args.max_pairs)),
           checkpoint=dict(features={k: v for k, v in vars(feats).items() if k != "_word_ids"}),
           device=device, t0=t0)


if __name__ == "__main__":
    main()
