# Test results

Entity-level, strict IOB2 (seqeval), word level, full-length sentences.
**Main** = cleaned test without the sentences that also occur in validation (92,608 sentences). **Full test** = all 95,030 cleaned test sentences.

| Model | P | R | micro-F1 | macro-F1 | micro-F1 (full test) |
|---|---:|---:|---:|---:|---:|
| Most-frequent tag | 37.1 | 1.6 | **3.1** | 0.6 | 3.1 |
| HMM + Viterbi | 5.8 | 2.5 | **3.5** | 1.0 | 3.5 |
| CRF | 57.5 | 60.6 | **59.0** | 52.9 | 59.0 |
| BiLSTM-CRF | 66.8 | 76.8 | **71.5** | 67.7 | 71.4 |
| BERT-base-cased | 63.0 | 79.8 | **70.4** | 66.0 | 70.4 |
| SEC-BERT-SHAPE | 66.2 | 81.2 | **73.0** | 69.2 | 72.9 |
