# Inference speed

Time to label the same sample of main-view test sentences, from raw words to tags (input preparation + model + decoding), median of the repeats. Higher sentences/s is faster. Produced by `python src/latency.py`.

## CPU: CPU (x86_64, 1 threads)

Sample: 2,000 sentences, 96,002 words.

| Model | micro-F1 | sentences / s | ms / sentence | parameters | weights |
|---|---:|---:|---:|---:|---|
| Most-frequent tag | 3.1 | 35,750.7 | 0.028 | 0 | fitted on train |
| HMM + Viterbi | 3.5 | 213.8 | 4.678 | 4,519,368 | fitted on train |
| CRF | 59.0 | 202.8 | 4.931 | 116,236,848 | untrained, same size |
| BiLSTM-CRF | 71.5 | 142.1 | 7.037 | 3,649,276 | untrained, same size |
| BERT-base-cased | 70.4 | 5.7 | 176.461 | 107,934,231 | pre-trained encoder + new 279-tag head, same size |
| SEC-BERT-SHAPE | 73.0 | 9.2 | 109.190 | 108,705,303 | pre-trained encoder + new 279-tag head, same size |

## CUDA: Tesla T4

Sample: 2,000 sentences, 96,002 words.

| Model | micro-F1 | sentences / s | ms / sentence | parameters | weights |
|---|---:|---:|---:|---:|---|
| Most-frequent tag | 3.1 | 33,046.0 | 0.030 | 0 | fitted on train |
| HMM + Viterbi | 3.5 | 223.2 | 4.480 | 4,519,368 | fitted on train |
| CRF | 59.0 | 5,344.4 | 0.187 | 116,236,848 | untrained, same size |
| BiLSTM-CRF | 71.5 | 4,703.7 | 0.213 | 3,649,276 | untrained, same size |
| BERT-base-cased | 70.4 | 679.4 | 1.472 | 107,934,231 | pre-trained encoder + new 279-tag head, same size |
| SEC-BERT-SHAPE | 73.0 | 818.8 | 1.221 | 108,705,303 | pre-trained encoder + new 279-tag head, same size |
