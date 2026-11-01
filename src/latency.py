"""Inference speed (latency / throughput) of every model on the same machine.

What is measured: wall-clock time to turn raw test sentences into word-level tags, i.e.
input preparation (normalising, feature extraction or sub-word tokenisation) + the model
+ decoding (Viterbi for HMM / CRF / BiLSTM-CRF, arg-max for the transformers). Training
time is not included. Every model labels the SAME fixed sample of main-view test
sentences (default 2,000, seed 42), after a warm-up run that is not timed. Each model is
timed `--repeats` times and the median is reported.

Speed depends on the size of a model, not on the values of its weights. When a trained
checkpoint is available (models/<key>.pt for CRF / BiLSTM-CRF, a fine-tuned directory for
the transformers) it is loaded; otherwise a model of exactly the same architecture and
size is built (CRF / BiLSTM-CRF: from the train vocabulary and the selected settings in
results/<key>_metrics.json; transformers: the pre-trained encoder with a 279-tag head).
The output says which one was used ("weights" column).

Run (Colab, so that every model runs on the same hardware):
    python src/latency.py --device cpu
    python src/latency.py --device cuda
    python src/latency.py --device cpu --models baseline_mft hmm      # a subset
    python src/latency.py --device cuda --models sec-bert-shape sec-bert-shape-crf --tag shape_crf
Writes/updates results/latency.csv (one row per model x device) and results/LATENCY.md.
"""
import argparse
import json
import os
import platform
import re
import statistics
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finer import ID2LABEL, LABEL2ID, RESULTS, ROOT, load_splits  # noqa: E402

MODELS = ["baseline_mft", "hmm", "crf", "bilstm_crf", "bert-base-cased", "sec-bert-shape", "sec-bert-shape-crf"]
NAMES = {"baseline_mft": "Most-frequent tag", "hmm": "HMM + Viterbi", "crf": "CRF",
         "bilstm_crf": "BiLSTM-CRF", "bert-base-cased": "BERT-base-cased", "sec-bert-shape": "SEC-BERT-SHAPE",
         "sec-bert-shape-crf": "SEC-BERT-SHAPE + CRF"}
HF_IDS = {"bert-base-cased": "bert-base-cased", "sec-bert-shape": "nlpaueb/sec-bert-shape",
          "sec-bert-shape-crf": "nlpaueb/sec-bert-shape"}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------ model wrappers
# Each builder returns (predict_fn, n_parameters, weights_description).
# predict_fn(list of token lists) -> list of tag lists (one tag per word).

def build_baseline(train, device):
    from baseline_hmm import most_frequent_tag
    f = most_frequent_tag(train)
    return (lambda sents: [f(s) for s in sents]), 0, "fitted on train"


def build_hmm(train, device):
    from baseline_hmm import HMM
    h = HMM().fit(train)
    n = h.log_start.size + h.log_trans.size + h.log_emit.size
    return (lambda sents: [h.predict(s) for s in sents]), n, "fitted on train"


def _checkpoint(key):
    path = os.path.join(ROOT, "models", f"{key}.pt")
    return torch.load(path, map_location="cpu", weights_only=False) if os.path.exists(path) else None


def build_crf(train, device):
    from baseline_crf import Batches, FeatureCRF
    from crf_features import CRFFeatures
    from seq_utils import TagSet, iob2_violations, predict
    meta = json.load(open(os.path.join(RESULTS, "crf_metrics.json")))
    cfg, feat_cfg = meta["selected"], meta["features"]
    ckpt = _checkpoint("crf")
    tagset = TagSet(train.tags)
    feats = CRFFeatures(window=cfg["window"], min_count=feat_cfg["min_count"],
                        max_pairs=feat_cfg["max_pairs"]).fit(train.tokens)
    model = FeatureCRF(feats.n_features, tagset.names, iob2_violations(train.tags) == 0).to(device)
    weights = "untrained, same size"
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        weights = "trained (models/crf.pt)"
    keep = feats.keep_columns(cfg["window"])
    bt = 4 * meta["training"]["batch_tokens"]

    def run(sents):
        f, s = feats.transform(sents)
        out = predict(model, Batches(f, s, keep, device), [len(t) for t in sents], bt)
        return [tagset.decode(p) for p in out]
    return run, sum(p.numel() for p in model.parameters()), weights


def build_bilstm_crf(train, device):
    from bilstm_crf import Batches, BiLSTMCRF, Vocab
    from seq_utils import TagSet, iob2_violations, predict
    meta = json.load(open(os.path.join(RESULTS, "bilstm_crf_metrics.json")))
    cfg = meta["selected"]
    ckpt = _checkpoint("bilstm_crf")
    tagset = TagSet(train.tags)
    vocab = Vocab(train.tokens)
    model = BiLSTMCRF(len(vocab.words), len(vocab.chars), len(vocab.cases), tagset.names,
                      word_dim=cfg["word_dim"], hidden=cfg["hidden"], layers=cfg["layers"],
                      dropout=cfg["dropout"], word_dropout=cfg["word_dropout"],
                      constrain_training=iob2_violations(train.tags) == 0).to(device)
    weights = "untrained, same size"
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        weights = "trained (models/bilstm_crf.pt)"
    bt = 4 * meta["training"]["batch_tokens"]

    def run(sents):
        out = predict(model, Batches(vocab.encode(sents), device), [len(t) for t in sents], bt)
        return [tagset.decode(p) for p in out]
    return run, sum(p.numel() for p in model.parameters()), weights


def build_transformer(key, device, batch_size=64, max_length=512):
    """BERT / SEC-BERT-SHAPE with a per-piece arg-max head, or (key ending in "-crf") the same
    encoder with the team's CRF layer over the first-piece scores, decoded with Viterbi exactly
    as `Tagger.decode` in train_transformer22.py."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    local = os.path.join(ROOT, "models", key)
    src = local if os.path.isdir(local) else HF_IDS[key]
    use_crf = key.endswith("-crf")
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForTokenClassification.from_pretrained(
        src, num_labels=len(ID2LABEL), id2label=dict(enumerate(ID2LABEL)), label2id=LABEL2ID).to(device).eval()
    weights = f"fine-tuned ({local})" if src == local else "pre-trained encoder + new 279-tag head, same size"
    crf = None
    if use_crf:
        from crf_layer import CRF
        sys.path.insert(0, ROOT)
        from train_transformer22 import first_pieces  # same word-level gather as training
        crf = CRF(list(ID2LABEL), constrain_training=True).to(device).eval()
        weights += " + untrained CRF layer (same size)"
    shapes = "shape" in key
    vocab = set(tok.get_vocab())
    number = re.compile(r"(\d+[\d,.]*)|([,.]\d+)")  # same rule as train_transformer22.py

    def prep(words):
        if not shapes:
            return words
        out = []
        for w in words:
            if number.fullmatch(w):
                s = "[" + re.sub(r"\d", "X", w) + "]"
                out.append(s if s in vocab else "[NUM]")
            else:
                out.append(w)
        return out

    @torch.inference_mode()
    def run(sents):
        order = sorted(range(len(sents)), key=lambda i: len(sents[i]))
        out = [None] * len(sents)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([prep(sents[i]) for i in idx], is_split_into_words=True, truncation=True,
                      max_length=max_length, padding=True, return_tensors="pt")
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            if crf is not None:
                # labels only mark each word's first piece (value 0); the CRF never reads them
                first = torch.full(logits.shape[:2], -100, dtype=torch.long)
                kept = []                      # word index of each first piece, in order
                for row in range(len(idx)):
                    prev, words = None, []
                    for j, w in enumerate(enc.word_ids(row)):
                        if w is not None and w != prev:
                            first[row, j] = 0
                            words.append(w)
                        prev = w
                    kept.append(words)
                emissions, _, mask, counts = first_pieces(logits, first.to(logits.device))
                with torch.autocast(device.type, enabled=False):   # as in Tagger.decode
                    paths = crf.decode(emissions.float(), mask)
                for row, i in enumerate(idx):
                    tags = ["O"] * len(sents[i])   # words cut by truncation stay O
                    for w, t in zip(kept[row], paths[row][: len(kept[row])]):
                        tags[w] = ID2LABEL[int(t)]
                    out[i] = tags
                continue
            best = logits.argmax(-1).cpu()
            for row, i in enumerate(idx):
                tags, prev = ["O"] * len(sents[i]), None
                for j, w in enumerate(enc.word_ids(row)):
                    if w is not None and w != prev:
                        tags[w] = ID2LABEL[int(best[row, j])]
                    prev = w
                out[i] = tags
        return out
    n_params = sum(p.numel() for p in model.parameters()) + (sum(p.numel() for p in crf.parameters()) if crf else 0)
    return run, n_params, weights


def build(key, train, device):
    if key == "baseline_mft":
        return build_baseline(train, device)
    if key == "hmm":
        return build_hmm(train, device)
    if key == "crf":
        return build_crf(train, device)
    if key == "bilstm_crf":
        return build_bilstm_crf(train, device)
    return build_transformer(key, device)


# ------------------------------------------------------------------ main
def hardware(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"CPU ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--n", type=int, default=2000, help="test sentences in the timed sample")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="write results/latency_<tag>.csv and LATENCY_<TAG>.md instead")
    args = ap.parse_args()
    device = torch.device(args.device)

    d = load_splits()
    train = d["train"]
    main_test = d["test"][~d["test"]["in_validation"]].reset_index(drop=True)
    sample = main_test.sample(n=min(args.n, len(main_test)), random_state=args.seed)
    sents = list(sample["tokens"])
    n_words = sum(map(len, sents))
    print(f"timed sample: {len(sents):,} main-view test sentences, {n_words:,} words | device: {hardware(device)}",
          flush=True)

    rows = []
    for key in args.models:
        t0 = time.perf_counter()
        run, n_params, weights = build(key, train, device)
        setup = time.perf_counter() - t0
        warm = run(sents[:200])                                   # warm-up, not timed
        assert all(len(p) == len(s) for p, s in zip(warm, sents[:200]))
        times = []
        for _ in range(args.repeats):
            sync(device)
            start = time.perf_counter()
            pred = run(sents)
            sync(device)
            times.append(time.perf_counter() - start)
        assert len(pred) == len(sents) and all(len(p) == len(s) for p, s in zip(pred, sents))
        t = statistics.median(times)
        row = dict(model=NAMES[key], key=key, device=args.device, hardware=hardware(device),
                   sentences=len(sents), words=n_words, seconds=round(t, 3),
                   sentences_per_s=round(len(sents) / t, 1), words_per_s=round(n_words / t),
                   ms_per_sentence=round(1000 * t / len(sents), 3), parameters=int(n_params),
                   weights=weights, setup_s=round(setup, 1), repeats=args.repeats)
        rows.append(row)
        print(f"{row['model']:20s} {row['sentences_per_s']:>10,.1f} sent/s  {row['ms_per_sentence']:>9.3f} ms/sent  "
              f"{row['parameters']:>12,} params  [{weights}]", flush=True)
        del run
        if device.type == "cuda":
            torch.cuda.empty_cache()

    suffix = f"_{args.tag}" if args.tag else ""
    path = os.path.join(RESULTS, f"latency{suffix}.csv")
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old.set_index(["key", "device"]).index.isin(new.set_index(["key", "device"]).index)]
    table = pd.concat([old, new], ignore_index=True)
    table["order"] = table["key"].map({k: i for i, k in enumerate(MODELS)})
    table = table.sort_values(["device", "order"]).drop(columns="order")
    table.to_csv(path, index=False)
    write_markdown(table, f"LATENCY{suffix.upper()}.md")


def write_markdown(table, filename="LATENCY.md"):
    scores = {}
    for key in MODELS:
        p = os.path.join(RESULTS, f"{key}_eval.json")
        if os.path.exists(p):
            scores[key] = json.load(open(p))["main"]["micro_F1"]
    lines = ["# Inference speed\n",
             "Time to label the same sample of main-view test sentences, from raw words to tags "
             "(input preparation + model + decoding), median of the repeats. Higher sentences/s is faster. "
             "Produced by `python src/latency.py`.\n"]
    for dev, part in table.groupby("device", sort=False):
        lines += [f"## {dev.upper()}: {part['hardware'].iloc[0]}\n",
                  f"Sample: {int(part['sentences'].iloc[0]):,} sentences, {int(part['words'].iloc[0]):,} words.\n",
                  "| Model | micro-F1 | sentences / s | ms / sentence | parameters | weights |",
                  "|---|---:|---:|---:|---:|---|"]
        for _, r in part.iterrows():
            f1 = f"{100 * scores[r['key']]:.1f}" if r["key"] in scores else "-"
            lines.append(f"| {r['model']} | {f1} | {r['sentences_per_s']:,.1f} | {r['ms_per_sentence']:.3f} | "
                         f"{int(r['parameters']):,} | {r['weights']} |")
        lines.append("")
    with open(os.path.join(RESULTS, filename), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
