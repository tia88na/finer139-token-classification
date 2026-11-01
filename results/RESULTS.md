# Test results

Entity-level, strict IOB2 (seqeval), word level, full-length sentences.
**Main** = cleaned test without the sentences that also occur in validation (92,608 sentences). **Full test** = all 95,030 cleaned test sentences.
95% CI: 10,000 paired sentence-bootstrap replicates of the main view. Speed: sentences/s on a Tesla T4 (`results/LATENCY.md`); † same architecture as the 2-epoch run (not timed separately); ‡ runs on the CPU; - not measured.

| Model | P | R | micro-F1 | 95% CI | macro-F1 | micro-F1 (full test) | sent./s (T4) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Most-frequent tag | 37.1 | 1.6 | **3.1** | [2.7, 3.5] | 0.6 | 3.1 | 33,046 |
| HMM + Viterbi | 5.8 | 2.5 | **3.5** | [3.2, 3.8] | 1.0 | 3.5 | 223‡ |
| CRF | 57.5 | 60.6 | **59.0** | [58.4, 59.6] | 52.9 | 59.0 | 5,344 |
| BiLSTM-CRF | 66.8 | 76.8 | **71.5** | [70.9, 72.0] | 67.7 | 71.4 | 4,704 |
| BERT-base-cased | 63.1 | 79.8 | **70.4** | [69.9, 71.0] | 66.0 | 70.4 | 679 |
| SEC-BERT-SHAPE | 66.2 | 81.2 | **73.0** | [72.4, 73.5] | 69.2 | 72.9 | 819 |
| SEC-BERT-SHAPE + CRF | 66.4 | 82.3 | **73.5** | [73.0, 74.1] | 70.0 | 73.5 | - |
| BERT-base-cased (4 epochs) | 66.7 | 81.5 | **73.3** | [72.8, 73.9] | 70.4 | 73.3 | 679† |
| SEC-BERT-SHAPE (4 epochs) | 70.1 | 82.8 | **75.9** | [75.4, 76.5] | 73.2 | 75.9 | 819† |
