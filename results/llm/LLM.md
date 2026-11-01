# LLM few-shot comparison

Model: `Qwen/Qwen2.5-7B-Instruct` (4-bit, greedy decoding), 3 labelled train examples in the prompt, no training. Produced by `python src/llm_fewshot.py`.

Sample: 300 main-view test sentences, seed 42 (200 with at least one entity, 100 without). Every model below is scored on exactly these sentences with the shared evaluator (strict IOB2), so the numbers are not the full-test numbers of `results/RESULTS.md`.

- The LLM was asked about 1,168 numbers; 33 answers were not a valid type (counted as O).
- 292 of 325 gold entity words (89.8%) are numbers, so the LLM cannot reach the other 33. The ceiling row shows the best score this setup allows (every asked number answered correctly).
- Run time: 32 min on a Tesla T4.

| Model | P | R | micro-F1 | macro-F1 |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct (3-shot) | 36.6 | 42.7 | **39.4** | 27.9 |
| Ceiling of the LLM setup (gold answer for every asked number) | 99.3 | 92.4 | **95.7** | 90.7 |
| Most-frequent tag | 37.5 | 1.0 | **1.9** | 0.3 |
| HMM + Viterbi | 10.0 | 1.9 | **3.2** | 1.3 |
| CRF | 68.6 | 59.9 | **63.9** | 44.5 |
| BiLSTM-CRF | 79.7 | 79.0 | **79.4** | 67.7 |
| BERT-base-cased | 78.5 | 81.5 | **80.0** | 67.7 |
| SEC-BERT-SHAPE + CRF | 78.5 | 82.5 | **80.4** | 66.6 |
| SEC-BERT-SHAPE | 79.7 | 82.5 | **81.1** | 68.9 |

Limitations: one prompt, one model and one sample; macro-F1 on a few hundred sentences covers only the types that occur in it and is noisy. The LLM is not fine-tuned, while every other model was trained on 261,610 sentences.
