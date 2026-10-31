"""Fine-tune a pretrained transformer for FiNER-139 token classification.

Reads the cleaned word-level data in data/cleaned/ (one row per sentence: tokens,
ner_tags), so every model gets the same sentences and is scored on the same words.
Each word's label goes on its first sub-word piece; the other pieces get -100 and
are ignored by the loss. Test sentences are encoded up to --eval-max-length pieces
so long sentences are not cut; words still beyond the limit are predicted as O.

Metric: entity-level micro and macro F1 with seqeval (token accuracy is useless
here because about 99% of tokens are O).

Models:
    bert-base-cased          baseline
    nlpaueb/sec-bert-shape   new model: BERT pre-trained on SEC filings, numbers are
                             replaced by their shape ("645.1" -> "[XXX.X]")

Needs a GPU (Colab: Runtime > Change runtime type > T4 GPU):
    !pip install -q seqeval
    !python train_transformer22.py --model bert-base-cased --limit 2000      # quick check
    !python train_transformer22.py --model bert-base-cased
    !python train_transformer22.py --model nlpaueb/sec-bert-shape
"""
import argparse
import csv
import json
import platform
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import transformers
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, get_linear_schedule_with_warmup)

SEED = 42
ROOT = Path(__file__).resolve().parent
NUMBER = re.compile(r"(\d+[\d,.]*)|([,.]\d+)")  # same rule as the SEC-BERT model card


def label_names(path):
    """The 279 BIO tag names stored in the parquet metadata by the datasets library."""
    meta = pq.read_schema(path).metadata or {}
    if b"huggingface" in meta:
        features = json.loads(meta[b"huggingface"])["info"]["features"]
        return features["ner_tags"]["feature"]["names"]
    from datasets import load_dataset_builder
    return load_dataset_builder("nlpaueb/finer-139").info.features["ner_tags"].feature.names


def load_split(path, limit=None):
    frame = pd.read_parquet(path, columns=["tokens", "ner_tags"])
    if limit and limit < len(frame):
        frame = frame.sample(n=limit, random_state=SEED)
    return ([[str(w) for w in t] for t in frame["tokens"]],
            [[int(x) for x in t] for t in frame["ner_tags"]])


def to_shapes(words, special_tokens):
    """SEC-BERT-SHAPE input: each number becomes its shape token, or [NUM] if unknown."""
    out = []
    for word in words:
        if NUMBER.fullmatch(word):
            shape = "[" + re.sub(r"\d", "X", word) + "]"
            out.append(shape if shape in special_tokens else "[NUM]")
        else:
            out.append(word)
    return out


def encode(tokenizer, sentences, tags, max_length, shapes):
    """Sub-word encoding with word labels on the first piece of each word.

    Returns the model inputs and, per sentence, the indices of the words that kept
    a first piece after truncation (used to map predictions back to words).
    """
    if shapes:
        special = set(tokenizer.get_vocab())  # includes the added shape tokens
        if "[NUM]" not in special:
            raise SystemExit(f"{tokenizer.name_or_path} has no [NUM] token; it is not SEC-BERT-SHAPE.")
        sentences = [to_shapes(words, special) for words in sentences]
    items, word_index = [], []
    for start in range(0, len(sentences), 2048):
        chunk = sentences[start:start + 2048]
        encoded = tokenizer(chunk, is_split_into_words=True, truncation=True,
                            max_length=max_length)
        for i in range(len(chunk)):
            labels, kept, previous = [], [], None
            for word in encoded.word_ids(i):
                if word is None or word == previous:
                    labels.append(-100)
                else:
                    labels.append(tags[start + i][word])
                    kept.append(word)
                previous = word
            item = {key: values[i] for key, values in encoded.items()}
            item["labels"] = labels
            items.append(item)
            word_index.append(kept)
    return items, word_index


@torch.inference_mode()
def predict(model, items, word_index, n_words, collate, batch_size, device, amp):
    """Word-level predicted tag ids for every sentence (O for words cut by truncation)."""
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
    predictions = [None] * len(items)
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        batch = collate([items[i] for i in ids])
        labels = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(**batch).logits
        best = logits.argmax(dim=-1).cpu()
        for row, i in enumerate(ids):
            words = np.zeros(n_words[i], dtype=np.int64)  # 0 = O
            words[word_index[i]] = best[row][labels[row] != -100].numpy()
            predictions[i] = words
    return predictions


def scores(gold_ids, predicted_ids, names, report=False):
    gold = [[names[t] for t in sentence] for sentence in gold_ids]
    predicted = [[names[t] for t in sentence] for sentence in predicted_ids]
    result = dict(micro_f1=f1_score(gold, predicted, average="micro"),
                  macro_f1=f1_score(gold, predicted, average="macro"),
                  precision=precision_score(gold, predicted, average="micro"),
                  recall=recall_score(gold, predicted, average="micro"))
    if report:
        result["report"] = classification_report(gold, predicted, digits=4, zero_division=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="bert-base-cased")
    parser.add_argument("--data-dir", default="data/cleaned")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128, help="sub-word pieces in training")
    parser.add_argument("--eval-max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--val-limit", type=int, default=20000,
                        help="validation sentences used to pick the epoch (test is always full)")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable mixed precision (slower; use if the loss is not finite)")
    parser.add_argument("--limit", type=int,
                        help="Quick check: this many sentences per split, results not saved")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = args.amp and device.type == "cuda"
    if device.type == "cpu" and not args.limit:
        raise SystemExit("No GPU found. In Colab choose Runtime > Change runtime type > T4 GPU, "
                         "or pass --limit 200 for a quick CPU check.")
    shapes = "shape" in args.model.lower()
    data_dir = ROOT / args.data_dir
    names = label_names(data_dir / "test.parquet")
    data = {"train": load_split(data_dir / "train.parquet", args.limit),
            "val": load_split(data_dir / "validation.parquet", args.limit or args.val_limit),
            "test": load_split(data_dir / "test.parquet", args.limit)}
    print({split: len(sentences) for split, (sentences, _) in data.items()},
          f"labels={len(names)} shapes={shapes}", flush=True)

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collate = DataCollatorForTokenClassification(tokenizer)
    encoded = {}
    for split, (sentences, tags) in data.items():
        length = args.max_length if split == "train" else args.eval_max_length
        encoded[split] = encode(tokenizer, sentences, tags, length, shapes)
    cut = sum(len(kept) < len(words) for kept, words in zip(encoded["test"][1], data["test"][0]))
    print(f"Encoded in {time.perf_counter() - start:.0f}s; test sentences still truncated: {cut}",
          flush=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(names), id2label=dict(enumerate(names)),
        label2id={name: i for i, name in enumerate(names)})
    model = model.float().to(device)
    loader = DataLoader(encoded["train"][0], batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    def evaluate(split, report=False):
        items, word_index = encoded[split]
        n_words = [len(words) for words in data[split][0]]
        predicted = predict(model, items, word_index, n_words, collate,
                            4 * args.batch_size, device, args.amp)
        return scores(data[split][1], predicted, names, report), predicted

    best_score, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, epoch_start = 0.0, time.perf_counter()
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=args.amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Loss is not finite. Run again with --no-amp.")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            if step % 500 == 0 or step == len(loader):
                minutes_left = (time.perf_counter() - epoch_start) / step * (len(loader) - step) / 60
                print(f"  epoch {epoch} step {step}/{len(loader)} loss={total_loss / step:.4f} "
                      f"(~{minutes_left:.0f} min left in epoch)", flush=True)
        validation, _ = evaluate("val")
        history.append(dict(epoch=epoch, loss=total_loss / len(loader), **validation))
        print(f"  epoch {epoch}: validation micro-F1={validation['micro_f1']:.4f} "
              f"macro-F1={validation['macro_f1']:.4f}", flush=True)
        if validation["micro_f1"] > best_score:
            best_score, best_epoch = validation["micro_f1"], epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)  # epoch chosen on validation, never on test
    test, test_predictions = evaluate("test", report=True)
    seconds = time.perf_counter() - start
    name = args.model.split("/")[-1]
    print(f"{name} test: micro-F1={test['micro_f1']:.4f} macro-F1={test['macro_f1']:.4f} "
          f"P={test['precision']:.4f} R={test['recall']:.4f} ({seconds / 60:.1f} min)", flush=True)
    if args.limit:
        print("Quick check finished; results were not saved.")
        return

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    row = dict(model=name, n_train=len(data["train"][0]), n_val=len(data["val"][0]),
               n_test=len(data["test"][0]), selected_epoch=best_epoch,
               val_micro_f1=round(best_score, 4), test_micro_f1=round(test["micro_f1"], 4),
               test_macro_f1=round(test["macro_f1"], 4), test_precision=round(test["precision"], 4),
               test_recall=round(test["recall"], 4), minutes=round(seconds / 60, 1))
    table = out / "transformer_benchmark.csv"
    rows = [r for r in csv.DictReader(table.open(encoding="utf-8"))] if table.exists() else []
    rows = [r for r in rows if r["model"] != name] + [row]
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    (out / f"{name}_report.txt").write_text(test.pop("report"), encoding="utf-8")
    details = dict(seed=SEED, model=args.model, shapes=shapes, epochs=args.epochs,
                   max_length=args.max_length, eval_max_length=args.eval_max_length,
                   batch_size=args.batch_size, learning_rate=args.lr, val_limit=args.val_limit,
                   mixed_precision=args.amp, test_sentences_truncated=cut,
                   device=torch.cuda.get_device_name(0), python=platform.python_version(),
                   torch=torch.__version__, transformers=transformers.__version__,
                   selected_epoch=best_epoch, history=history, test=test)
    (out / f"{name}_metrics.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    pd.DataFrame({"predicted_tags": [p.tolist() for p in test_predictions]}).to_parquet(
        out / f"{name}_test_predictions.parquet")
    print(f"Saved results/transformer_benchmark.csv, results/{name}_report.txt, "
          f"results/{name}_metrics.json, results/{name}_test_predictions.parquet")


if __name__ == "__main__":
    main()
