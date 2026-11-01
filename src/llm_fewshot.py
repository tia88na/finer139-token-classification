"""LLM comparison: an open instruction-tuned LLM tags FiNER-139 numbers with few-shot prompting.

No training. For each test sentence the LLM sees the 139 entity types, three labelled
examples from the TRAIN split and the sentence with its numbers marked, and answers with
one entity type (or O) per marked number. Its answers become word-level IOB2 tags and are
scored with the team's shared evaluator, next to every saved model on the SAME sentences.

Design choices (and what they mean for the comparison)
* Sample: a fixed random sample of main-view test sentences (not in validation), seed 42:
  --with-entities sentences that contain at least one entity + --without-entities that
  contain none. Running an LLM on all 92,608 sentences would take days on a free GPU.
  All models are re-scored on exactly this sample, so the table is a like-for-like view.
* Candidates: only number-like words (finer.is_number) are asked about; every other word is
  O. 94.7% of entity words are numbers (EDA), so the LLM cannot get the remaining ones
  (e.g. month names inside dates). The share of gold entity words that are candidates is
  reported, so this ceiling is visible.
* Multi-word entities: a candidate gets B-<type>; numbers directly following it with the
  same answer become I-<type>.
* Invalid answers (not one of the 139 types) become O and are counted.
* Greedy decoding (deterministic), 4-bit weights so a 7B model fits a free Colab T4.

Run (Colab, T4 GPU, about 30-40 min for the default 300 sentences):
    !pip install -q seqeval bitsandbytes accelerate
    !python src/llm_fewshot.py
    !python src/llm_fewshot.py --limit 10            # quick check (10 sentences)
    !python src/llm_fewshot.py --model Qwen/Qwen2.5-3B-Instruct   # smaller / faster
Check without a GPU or a model (the "LLM" answers with the gold types, so the pipeline
must score 100% on candidate words):
    python src/llm_fewshot.py --oracle --limit 50
Writes results/llm/{LLM.md, scores.csv, sample.csv, predictions.parquet, raw_outputs.jsonl}.
"""
import argparse
import json
import os
import random
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finer import ID2LABEL, RESULTS, is_number, load_splits, scores  # noqa: E402

SEED = 42
OUT = os.path.join(RESULTS, "llm")
TYPES = sorted({t[2:] for t in ID2LABEL if t != "O"})
NAMES = {"baseline_mft": "Most-frequent tag", "hmm": "HMM + Viterbi", "crf": "CRF",
         "bilstm_crf": "BiLSTM-CRF", "bert-base-cased": "BERT-base-cased",
         "sec-bert-shape": "SEC-BERT-SHAPE", "sec-bert-shape-crf": "SEC-BERT-SHAPE + CRF"}

SYSTEM = (
    "You are an expert in US GAAP financial reporting and XBRL tagging. "
    "You read one sentence from a company's SEC filing (10-K / 10-Q). Some numbers in it "
    "are marked like [[1: 5.5]]. For every marked number, decide which XBRL entity type it "
    "reports, using the words around it, or answer O if it is not one of the types below "
    "(for example years, dates of events, note numbers, or amounts of other concepts).\n\n"
    "Allowed entity types:\n" + "\n".join(TYPES) + "\n\n"
    "Answer with JSON only, one key per marked number, e.g. {\"1\": \"DebtInstrumentFaceAmount\", \"2\": \"O\"}. "
    "Use the type names exactly as listed.")


def candidates(tokens):
    return [i for i, w in enumerate(tokens) if is_number(w)]


def marked(tokens, cand):
    pos = {i: k + 1 for k, i in enumerate(cand)}
    return " ".join(f"[[{pos[i]}: {w}]]" if i in pos else w for i, w in enumerate(tokens))


def gold_answer(tokens, tags, cand):
    return {str(k + 1): (tags[i][2:] if tags[i] != "O" else "O") for k, i in enumerate(cand)}


def pick_examples(train, n, rng):
    """n short train sentences with 1-3 entities and a few untagged numbers (fixed seed)."""
    pool = [j for j, (t, g) in enumerate(zip(train.tokens, train.tags))
            if 12 <= len(t) <= 45 and 1 <= sum(x.startswith("B-") for x in g) <= 3
            and len(candidates(t)) > sum(x != "O" for x in g)]
    return [pool[j] for j in rng.sample(range(len(pool)), n)]


def build_messages(tokens, examples):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex_tokens, ex_tags in examples:
        cand = candidates(ex_tokens)
        msgs.append({"role": "user", "content": marked(ex_tokens, cand)})
        msgs.append({"role": "assistant", "content": json.dumps(gold_answer(ex_tokens, ex_tags, cand))})
    msgs.append({"role": "user", "content": marked(tokens, candidates(tokens))})
    return msgs


def parse(text, n):
    """{index: type} from the model's answer; anything unusable -> O. Returns (answers, invalid)."""
    found = {}
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            found = {str(k): str(v).strip() for k, v in json.loads(m.group(0)).items()}
        except (json.JSONDecodeError, AttributeError):
            found = {}
    if not found:  # fall back to "1": "Type" pairs
        found = dict(re.findall(r'"?(\d+)"?\s*:\s*"([A-Za-z0-9]+)"', text))
    answers, invalid = {}, 0
    valid = set(TYPES)
    for k in range(1, n + 1):
        t = found.get(str(k), "O")
        if t != "O" and t not in valid:
            invalid += 1
            t = "O"
        answers[k] = t
    return answers, invalid


def to_tags(tokens, cand, answers):
    tags = ["O"] * len(tokens)
    prev_i, prev_t = None, None
    for k, i in enumerate(cand, start=1):
        t = answers[k]
        if t != "O":
            tags[i] = (f"I-{t}" if prev_t == t and prev_i == i - 1 else f"B-{t}")
        prev_i, prev_t = i, t
    return tags


def load_llm(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant, device_map="auto")
    model.eval()
    return tok, model


def generate(tok, model, batch_msgs, max_new_tokens):
    import torch
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--with-entities", type=int, default=200)
    ap.add_argument("--without-entities", type=int, default=100)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, help="quick check: only the first N sampled sentences")
    ap.add_argument("--oracle", action="store_true", help="pipeline check: answer with the gold types")
    args = ap.parse_args()

    d = load_splits()
    train, test = d["train"], d["test"]
    rng = random.Random(SEED)
    main_rows = [i for i in range(len(test)) if not test.in_validation.iloc[i]]
    with_ent = [i for i in main_rows if any(t != "O" for t in test.tags.iloc[i])]
    without = [i for i in main_rows if all(t == "O" for t in test.tags.iloc[i])]
    sample = sorted(rng.sample(with_ent, args.with_entities) + rng.sample(without, args.without_entities))
    if args.limit:
        sample = sample[:args.limit]
    examples = [(train.tokens.iloc[j], train.tags.iloc[j]) for j in pick_examples(train, args.shots, rng)]
    gold = [test.tags.iloc[i] for i in sample]
    ent_words = sum(t != "O" for g in gold for t in g)
    cand_ent = sum(test.tags.iloc[i][j] != "O" for i in sample for j in candidates(test.tokens.iloc[i]))
    print(f"{len(sample)} sentences ({sum(any(t != 'O' for t in g) for g in gold)} with entities); "
          f"{cand_ent}/{ent_words} gold entity words are numbers the LLM is asked about", flush=True)

    os.makedirs(OUT, exist_ok=True)
    if not args.oracle:
        tok, model = load_llm(args.model)
    preds, raw, invalid_total, asked, t0 = [], [], 0, 0, time.time()
    todo = [(k, i) for k, i in enumerate(sample)]
    results = {}
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start:start + args.batch_size]
        need = [(k, i) for k, i in chunk if candidates(test.tokens.iloc[i])]
        if args.oracle:
            texts = [json.dumps(gold_answer(test.tokens.iloc[i], test.tags.iloc[i],
                                            candidates(test.tokens.iloc[i]))) for _, i in need]
        elif need:
            n_max = max(len(candidates(test.tokens.iloc[i])) for _, i in need)
            texts = generate(tok, model, [build_messages(test.tokens.iloc[i], examples) for _, i in need],
                             max_new_tokens=16 + 24 * n_max)
        else:
            texts = []
        for (k, i), text in zip(need, texts):
            cand = candidates(test.tokens.iloc[i])
            answers, bad = parse(text, len(cand))
            invalid_total += bad
            asked += len(cand)
            results[k] = to_tags(test.tokens.iloc[i], cand, answers)
            raw.append({"test_row": i, "answer": text})
        for k, i in chunk:
            results.setdefault(k, ["O"] * len(test.tokens.iloc[i]))
        done = min(start + args.batch_size, len(todo))
        if done % 20 < args.batch_size or done == len(todo):
            el = time.time() - t0
            print(f"  {done}/{len(todo)} sentences, {el / 60:.1f} min, "
                  f"~{el / done * (len(todo) - done) / 60:.0f} min left", flush=True)
    preds = [results[k] for k in range(len(sample))]
    minutes = (time.time() - t0) / 60

    name = "Oracle check" if args.oracle else args.model.split("/")[-1] + f" ({args.shots}-shot)"
    rows = [dict(model=name, **{k: round(100 * v, 1) for k, v in scores(gold, preds).items()})]
    # Ceiling of this setup: the gold type for every asked number, O elsewhere
    ceiling = [to_tags(test.tokens.iloc[i], candidates(test.tokens.iloc[i]),
                       {k + 1: v for k, v in enumerate(gold_answer(test.tokens.iloc[i], test.tags.iloc[i],
                                                                   candidates(test.tokens.iloc[i])).values())})
               for i in sample]
    rows.append(dict(model="Ceiling of the LLM setup (gold answer for every asked number)",
                     **{k: round(100 * v, 1) for k, v in scores(gold, ceiling).items()}))
    from finer import LABEL2ID  # noqa: E402
    for path in sorted(os.listdir(RESULTS)):
        if not path.endswith("_test_predictions.parquet"):
            continue
        key = path[: -len("_test_predictions.parquet")]
        p = pd.read_parquet(os.path.join(RESULTS, path))["predicted_tags"]
        if len(p) != len(test):
            continue
        mp = [[ID2LABEL[int(x)] for x in p.iloc[i]] for i in sample]
        rows.append(dict(model=NAMES.get(key, key), **{k: round(100 * v, 1) for k, v in scores(gold, mp).items()}))
    table = pd.DataFrame(rows).sort_values("micro_F1")
    print(table.to_string(index=False))
    if args.limit and not args.oracle:
        print("Quick check finished; results were not saved.")
        return

    suffix = "_oracle" if args.oracle else ""
    table.to_csv(os.path.join(OUT, f"scores{suffix}.csv"), index=False)
    pd.DataFrame({"test_row": sample}).to_csv(os.path.join(OUT, f"sample{suffix}.csv"), index=False)
    pd.DataFrame({"test_row": sample, "predicted_tags": [[LABEL2ID[t] for t in p] for p in preds]}).to_parquet(
        os.path.join(OUT, f"predictions{suffix}.parquet"))
    with open(os.path.join(OUT, f"raw_outputs{suffix}.jsonl"), "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    if args.oracle:
        print("Oracle check saved (files end in _oracle).")
        return
    md = ["# LLM few-shot comparison\n",
          f"Model: `{args.model}` (4-bit, greedy decoding), {args.shots} labelled train examples in the prompt, "
          f"no training. Produced by `python src/llm_fewshot.py`.\n",
          f"Sample: {len(sample)} main-view test sentences, seed {SEED} "
          f"({args.with_entities} with at least one entity, {args.without_entities} without). "
          f"Every model below is scored on exactly these sentences with the shared evaluator "
          f"(strict IOB2), so the numbers are not the full-test numbers of `results/RESULTS.md`.\n",
          f"- The LLM was asked about {asked:,} numbers; {invalid_total} answers were not a valid type (counted as O).",
          f"- {cand_ent} of {ent_words} gold entity words ({100 * cand_ent / max(ent_words, 1):.1f}%) are numbers, "
          f"so the LLM cannot reach the other {ent_words - cand_ent}. The ceiling row shows the best "
          f"score this setup allows (every asked number answered correctly).",
          f"- Run time: {minutes:.0f} min on {('a ' + __import__('torch').cuda.get_device_name(0)) if not args.oracle else 'CPU'}.\n",
          "| Model | P | R | micro-F1 | macro-F1 |", "|---|---:|---:|---:|---:|"]
    for r in rows[:2] + sorted(rows[2:], key=lambda r: r["micro_F1"]):
        md.append(f"| {r['model']} | {r['micro_P']:.1f} | {r['micro_R']:.1f} | **{r['micro_F1']:.1f}** | {r['macro_F1']:.1f} |")
    md += ["", "Limitations: one prompt, one model and one sample; macro-F1 on a few hundred sentences "
           "covers only the types that occur in it and is noisy. The LLM is not fine-tuned, while every "
           "other model was trained on 261,610 sentences."]
    with open(os.path.join(OUT, "LLM.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved {OUT}/LLM.md, scores.csv, sample.csv, predictions.parquet, raw_outputs.jsonl")


if __name__ == "__main__":
    main()
