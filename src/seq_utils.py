"""Training and output helpers shared by the CRF (baseline_crf.py) and the BiLSTM-CRF (bilstm_crf.py).

A model only has to provide
    model.loss(batch)   -> summed -log P(gold tags) over the batch (a 0-d tensor)
    model.decode(batch) -> list of internal tag-id lists, one per sentence
and a function that turns a list of sentence indices into a `batch` dict with a boolean
`mask` [B, T]. Everything else (batching, early stopping on validation, test prediction,
saving in the shared format) lives here, so both models are trained and scored the same way.
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_hmm import save_predictions  # noqa: E402  (same prediction file format as the HMM)
from crf_layer import iob2_violations  # noqa: E402
from finer import LABEL2ID, RESULTS, ROOT, evaluate, load_splits, save_results, scores  # noqa: E402

SEED = 42
VAL_LIMIT = 20000  # validation sentences used for model selection, as in train_transformer22.py


def set_seed(seed: int = SEED) -> None:
    """Seed everything. CPU runs repeat exactly; on the GPU some CUDA kernels may still
    differ in the last floating-point bits (PyTorch warns once when it uses one)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class TagSet:
    """Tags seen in train (O first, then sorted). Like the HMM, a model can only predict these.
    Internal id <-> tag name; tag names map to the global ids of finer.LABEL2ID when saved."""

    def __init__(self, tag_seqs):
        seen = {t for tags in tag_seqs for t in tags}
        self.names = ["O"] + sorted(seen - {"O"})
        self.index = {t: i for i, t in enumerate(self.names)}

    def __len__(self):
        return len(self.names)

    def encode(self, tags):
        return [self.index[t] for t in tags]

    def decode(self, ids):
        return [self.names[i] for i in ids]


def val_sample(val: pd.DataFrame, limit: int | None = VAL_LIMIT, seed: int = SEED) -> pd.DataFrame:
    """The validation sentences used to choose settings and epochs: the same random 20,000
    rows as train_transformer22.py (`frame.sample(n=20000, random_state=42)`)."""
    if limit and limit < len(val):
        val = val.sample(n=limit, random_state=seed)
    return val.reset_index(drop=True)


def batches(lengths, max_tokens: int, shuffle: bool = False, rng=None) -> list[list[int]]:
    """Sentence indices grouped by similar length; a batch holds at most `max_tokens` words
    including padding (a longer sentence gets a batch of its own). Sentences are never cut.
    With `shuffle`, lengths get +-10% noise so batches change between epochs, and the batch
    order is shuffled."""
    lengths = np.asarray(lengths)
    key = lengths * np.exp(rng.uniform(-0.1, 0.1, len(lengths))) if shuffle else lengths
    out, cur, longest = [], [], 0
    for i in np.argsort(key, kind="stable"):
        n = int(lengths[i])
        if cur and max(longest, n) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, longest = [], 0
        cur.append(int(i))
        longest = max(longest, n)
    if cur:
        out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


def pad(rows, dtype=torch.long, value=0) -> torch.Tensor:
    """List of equal-rank arrays with different first dimensions -> padded tensor [B, T, ...]."""
    longest = max(len(r) for r in rows)
    out = np.full((len(rows), longest) + np.shape(rows[0])[1:], value, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return torch.as_tensor(out, dtype=dtype)


@torch.no_grad()
def predict(model, make_batch, lengths, max_tokens: int) -> list[list[int]]:
    """Internal tag ids for every sentence, in the original order, full length."""
    model.eval()
    out = [None] * len(lengths)
    for idx in batches(lengths, max_tokens):
        for i, p in zip(idx, model.decode(make_batch(idx))):
            out[i] = p
    return out


def fit(model, optimizer, train_batch, train_lengths, val_batch, val_gold, tagset, *,
        epochs: int, patience: int, batch_tokens: int, clip: float | None = None,
        seed: int = SEED, log=print) -> dict:
    """Minimise the CRF negative log-likelihood (per word) on train; after every epoch
    decode the validation sample and keep the weights with the best validation micro-F1
    (strict IOB2, finer.scores). Stops after `patience` epochs without improvement.
    The model ends with the best weights. Returns the selected epoch and the history."""
    rng = np.random.default_rng(seed)
    val_lengths = [len(g) for g in val_gold]
    history, best, best_epoch, best_state, wait = [], -1.0, 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        start, total, words = time.perf_counter(), 0.0, 0
        order = batches(train_lengths, batch_tokens, shuffle=True, rng=rng)
        for step, idx in enumerate(order, 1):
            batch = train_batch(idx)
            n = int(batch["mask"].sum())
            loss = model.loss(batch)
            optimizer.zero_grad(set_to_none=True)
            (loss / n).backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total, words = total + loss.item(), words + n
            if not np.isfinite(total):
                raise RuntimeError(f"loss is not finite at epoch {epoch} step {step}")
            if step % 500 == 0:
                left = (time.perf_counter() - start) / step * (len(order) - step) / 60
                log(f"  epoch {epoch} step {step}/{len(order)} loss/word {total / words:.4f} (~{left:.1f} min left)")
        train_min = (time.perf_counter() - start) / 60
        pred = predict(model, val_batch, val_lengths, 4 * batch_tokens)
        val = scores(val_gold, [tagset.decode(p) for p in pred])
        history.append(dict(epoch=epoch, loss_per_word=round(total / words, 5), train_minutes=round(train_min, 2),
                            minutes=round((time.perf_counter() - start) / 60, 2), **val))
        log(f"  epoch {epoch}: loss/word {total / words:.4f}  validation micro-F1 {val['micro_F1']:.4f} "
            f"macro-F1 {val['macro_F1']:.4f} (P {val['micro_P']:.4f} R {val['micro_R']:.4f})  "
            f"[{train_min:.1f} min train, {history[-1]['minutes']:.1f} min total]")
        if val["micro_F1"] > best:
            best, best_epoch, wait = val["micro_F1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                log(f"  no improvement for {patience} epochs, stop")
                break
    model.load_state_dict(best_state)  # epoch chosen on validation, never on test
    return dict(selected_epoch=best_epoch, val_micro_F1=best, history=history)


def environment(device: torch.device) -> dict:
    import seqeval
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                numpy=np.__version__, pandas=pd.__version__, seqeval=getattr(seqeval, "__version__", "?"),
                device=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor())


# ---------------------------------------------------------------- the steps both scripts share
def add_common_args(parser, *, epochs: int, patience: int, batch_tokens: int, lr: float) -> None:
    parser.add_argument("--grid", action="store_true", help="try every setting in GRID, keep the best on validation")
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--epochs", type=int, default=epochs, help="maximum; the epoch is chosen on validation")
    parser.add_argument("--patience", type=int, default=patience)
    parser.add_argument("--batch-tokens", type=int, default=batch_tokens, help="words per batch, padding included")
    parser.add_argument("--val-limit", type=int, default=VAL_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model-dir", default=os.path.join(ROOT, "models"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, help="quick check: this many sentences per split; nothing is "
                                                  "saved and test predictions are only checked for format")


def load_data(args):
    """Cleaned train, the validation sample and the whole test set (with --limit: samples)."""
    d = load_splits()
    train, test = d["train"], d["test"]
    val = val_sample(d["validation"], min(args.limit or args.val_limit, args.val_limit), args.seed)
    if args.limit:
        train = train.sample(n=min(args.limit, len(train)), random_state=args.seed).reset_index(drop=True)
        test = test.sample(n=min(args.limit, len(test)), random_state=args.seed).reset_index(drop=True)
    tagset = TagSet(train.tags)
    bad = iob2_violations(train.tags)
    print(f"train {len(train):,} sentences | validation sample {len(val):,} | test {len(test):,} | "
          f"{len(tagset)} tags seen in train | IOB2 violations in train: {bad} -> constraints in training "
          f"{'on' if bad == 0 else 'off'} (always on in decoding)", flush=True)
    return train, val, test, tagset, bad == 0


def search(configs, build, train_lengths, val_gold, tagset, args, device):
    """Train one model per setting, each with its epoch chosen on validation; keep the setting
    with the best validation micro-F1. `build(cfg)` -> (model, optimizer, train_batch, val_batch).
    The test set is not used here. Returns (cfg, fit result, weights) of the best and all trials."""
    trials, best = [], None
    for cfg in configs:
        set_seed(args.seed)
        print(f"\nsetting {cfg}", flush=True)
        model, optimizer, train_batch, val_batch = build(cfg)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        out = fit(model, optimizer, train_batch, train_lengths, val_batch, val_gold, tagset,
                  epochs=args.epochs, patience=args.patience, batch_tokens=args.batch_tokens,
                  clip=cfg.get("clip"), seed=args.seed)
        gpu = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        trials.append(dict(**cfg, selected_epoch=out["selected_epoch"], val_micro_F1=out["val_micro_F1"],
                           val_macro_F1=out["history"][out["selected_epoch"] - 1]["macro_F1"],
                           epochs_run=len(out["history"]), minutes=round((time.perf_counter() - start) / 60, 1),
                           peak_gpu_gb=round(gpu, 2)))
        print(f"setting done: validation micro-F1 {out['val_micro_F1']:.4f} (epoch {out['selected_epoch']}), "
              f"peak GPU memory {gpu:.1f} GB", flush=True)
        if best is None or out["val_micro_F1"] > best[1]["val_micro_F1"]:
            best = (cfg, out, {k: v.cpu() for k, v in model.state_dict().items()})
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    cfg, out, _ = best
    print(f"\nselected on validation: {cfg} epoch {out['selected_epoch']} micro-F1 {out['val_micro_F1']:.4f}", flush=True)
    return best, trials


def check_predictions(test: pd.DataFrame, pred: list[list[str]]) -> None:
    """One prediction per test sentence, in order, as long as the sentence, with known tags."""
    assert len(pred) == len(test), (len(pred), len(test))
    for toks, p in zip(test.tokens, pred):
        assert len(p) == len(toks)
    assert all(t in LABEL2ID for p in pred for t in p)


def finish(key, name, args, model, test_batch, test, tagset, best, trials, metrics, checkpoint, device, t0):
    """Decode the test set once with the selected model. With --limit only check the format.
    Otherwise score with the shared evaluator and write, like the other models:
    results/<key>_test_predictions.parquet (global tag ids, row i = test row i),
    results/<key>_eval.json and _per_class.csv (finer.save_results), results/<key>_metrics.json
    (settings, selection history, versions), results/<key>_search.csv, models/<key>.pt."""
    cfg, out, state = best
    pred = predict(model, test_batch, [len(t) for t in test.tokens], 4 * args.batch_tokens)
    pred = [tagset.decode(p) for p in pred]
    check_predictions(test, pred)
    if args.limit:
        print(f"Quick check finished in {(time.perf_counter() - t0) / 60:.1f} min: test predictions have "
              f"the right format; test was not scored and nothing was saved.")
        return None
    res = evaluate(test, pred, name)
    save_results(res, key)
    save_predictions(pred, key)
    pd.DataFrame(trials).to_csv(os.path.join(RESULTS, f"{key}_search.csv"), index=False)
    metrics = dict(model=name, seed=args.seed, selected=cfg, selected_epoch=out["selected_epoch"],
                   val_micro_F1=out["val_micro_F1"], history=out["history"], grid=trials, **metrics,
                   training=dict(lr=args.lr, max_epochs=args.epochs, patience=args.patience,
                                 batch_tokens=args.batch_tokens, val_limit=args.val_limit),
                   minutes=round((time.perf_counter() - t0) / 60, 1), environment=environment(device),
                   test={"main": res["main"], "reference": res["reference"]})
    with open(os.path.join(RESULTS, f"{key}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    os.makedirs(args.model_dir, exist_ok=True)
    path = os.path.join(args.model_dir, f"{key}.pt")
    torch.save(dict(state_dict=state, config=cfg, tags=tagset.names, **checkpoint), path)
    print(f"Saved results/{key}_test_predictions.parquet, _eval.json, _per_class.csv, _metrics.json, "
          f"_search.csv and {path} (load with torch.load(path, weights_only=False))")
    return res
