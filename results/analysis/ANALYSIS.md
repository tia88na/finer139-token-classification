# Error analysis

Main test view only: exact token duplicates of validation are excluded. Strict IOB2 entity spans, original words, full sentences. Prediction row counts, word counts and tag IDs are checked; aggregate P/R/F1 must reproduce the saved common evaluator results.

## Overall results

| Model | P | R | F1 | Macro-F1 | TP | Gold entities | Predicted entities |
|---|---:|---:|---:|---:|---:|---:|---:|
| bilstm_crf | 66.80 | 76.79 | 71.45 | 67.68 | 21646 | 28189 | 32405 |
| bert-base-cased | 63.05 | 79.80 | 70.44 | 66.01 | 22494 | 28189 | 35674 |
| sec-bert-shape | 66.23 | 81.19 | 72.95 | 69.15 | 22887 | 28189 | 34556 |

## Findings

- SEC-BERT-SHAPE improves on BERT by 2.51 F1 points. It has 1,195 fewer spurious predictions and 319 fewer exact-boundary wrong-type errors.
- For every model, spurious entity predictions outnumber completely missed gold entities. This supports the observed precision/recall gap; it does not establish why overprediction occurs.
- Boundary-overlap errors are much less frequent than wrong-type errors and spurious predictions.
- Training frequency here counts complete entity types after merging B-/I- tags. Rare BIO continuation tags in the EDA must not be confused with rare complete entity types. Absent frequency buckets have no evaluated gold entities or predictions.
- The length breakdown does not show a uniform decline as sentences grow longer. Do not claim that long sentences are the principal failure mode from these results alone.

## Error partitions

Gold-side categories partition false negatives; prediction-side categories partition false positives. A wrong type at the correct boundaries causes both a false negative and a false positive. Do not sum the two sides as unique errors. Boundary overlap may also involve a wrong type. Orphan I tags do not form entities under strict IOB2.

| Model | Missed (gold) | Wrong type (gold) | Boundary overlap (gold) | Spurious (prediction) | Wrong type (prediction) | Boundary overlap (prediction) |
|---|---:|---:|---:|---:|---:|---:|
| bilstm_crf | 3337 | 3173 | 33 | 7553 | 3173 | 33 |
| bert-base-cased | 1934 | 3683 | 78 | 9417 | 3683 | 80 |
| sec-bert-shape | 1857 | 3364 | 81 | 8222 | 3364 | 83 |

## Sentence length (original words)

| Model | Bucket | Gold entities | Predicted entities | F1 |
|---|---|---:|---:|---:|
| bilstm_crf | 1_1-32 | 7808 | 9499 | 68.38 |
| bilstm_crf | 2_33-64 | 12358 | 14003 | 72.36 |
| bilstm_crf | 3_65-128 | 6274 | 7021 | 73.25 |
| bilstm_crf | 4_129+ | 1749 | 1882 | 72.82 |
| bert-base-cased | 1_1-32 | 7808 | 10437 | 67.94 |
| bert-base-cased | 2_33-64 | 12358 | 15547 | 71.36 |
| bert-base-cased | 3_65-128 | 6274 | 7766 | 71.94 |
| bert-base-cased | 4_129+ | 1749 | 1924 | 70.19 |
| sec-bert-shape | 1_1-32 | 7808 | 10019 | 70.02 |
| sec-bert-shape | 2_33-64 | 12358 | 15076 | 73.72 |
| sec-bert-shape | 3_65-128 | 6274 | 7455 | 74.91 |
| sec-bert-shape | 4_129+ | 1749 | 2006 | 74.14 |

## Training entity frequency

| Model | Bucket | Gold entities | Predicted entities | F1 |
|---|---|---:|---:|---:|
| bilstm_crf | 3_100-999 | 5752 | 6474 | 64.57 |
| bilstm_crf | 4_1000+ | 22437 | 25931 | 73.18 |
| bert-base-cased | 3_100-999 | 5752 | 7330 | 63.38 |
| bert-base-cased | 4_1000+ | 22437 | 28344 | 72.26 |
| sec-bert-shape | 3_100-999 | 5752 | 6986 | 66.51 |
| sec-bert-shape | 4_1000+ | 22437 | 27570 | 74.59 |

Frequency buckets use strict entity occurrences in cleaned training data, not test frequency. Predictions are assigned to the bucket of their predicted type, preserving false positives. These are pooled micro-F1 scores. Word-length buckets do not identify tokenizer truncation; a long-sentence association alone does not establish its cause.

## Most common type confusions

Word-level diagnostics, with BIO prefixes removed; these counts are not entity-level errors. O-related errors are in token_confusions.csv but excluded from this table.

| Model | Gold type | Predicted type | Words |
|---|---|---|---:|
| bilstm_crf | EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions | EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized | 114 |
| bilstm_crf | ShareBasedCompensation | AllocatedShareBasedCompensationExpense | 107 |
| bilstm_crf | DebtInstrumentFaceAmount | LineOfCreditFacilityMaximumBorrowingCapacity | 89 |
| bilstm_crf | SaleOfStockNumberOfSharesIssuedInTransaction | StockIssuedDuringPeriodSharesNewIssues | 84 |
| bilstm_crf | LineOfCreditFacilityCurrentBorrowingCapacity | LineOfCreditFacilityMaximumBorrowingCapacity | 80 |
| bert-base-cased | ShareBasedCompensation | AllocatedShareBasedCompensationExpense | 121 |
| bert-base-cased | StockRepurchasedDuringPeriodShares | TreasuryStockSharesAcquired | 97 |
| bert-base-cased | LineOfCreditFacilityCurrentBorrowingCapacity | LineOfCreditFacilityMaximumBorrowingCapacity | 95 |
| bert-base-cased | EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions | EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized | 92 |
| bert-base-cased | LineOfCreditFacilityMaximumBorrowingCapacity | DebtInstrumentFaceAmount | 91 |
| sec-bert-shape | ShareBasedCompensation | AllocatedShareBasedCompensationExpense | 118 |
| sec-bert-shape | LineOfCreditFacilityMaximumBorrowingCapacity | DebtInstrumentFaceAmount | 83 |
| sec-bert-shape | EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions | EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized | 80 |
| sec-bert-shape | LineOfCreditFacilityCurrentBorrowingCapacity | LineOfCreditFacilityMaximumBorrowingCapacity | 79 |
| sec-bert-shape | SaleOfStockNumberOfSharesIssuedInTransaction | StockIssuedDuringPeriodSharesNewIssues | 75 |

## BERT versus SEC-BERT-SHAPE

SEC-BERT-SHAPE gets 1,412 gold spans correct that BERT misses; BERT gets 1,019 correct that SEC-BERT-SHAPE misses. These are exact gold-span recoveries, not net F1 changes; false positives also matter.

Both domain pretraining and number representation differ between these models, so the difference cannot be attributed to shape replacement alone.

## Examples

Deterministic illustrative cases: first two main-test sentences per model/category. Examples are not a random sample and do not establish prevalence. Spans below use zero-based word indices with an exclusive end. Full tokens and all spans are in examples.json.

### bilstm_crf: gold_missed, test row 35

Sentence: At December 31 , 2019 , the UPB related to 24 loans with five different borrowers represented 13 % of total assets .

Gold: [[17, 18, "ConcentrationRiskPercentage1"]]

Prediction: []

### bilstm_crf: pred_spurious, test row 57

Sentence: At March 31 , 2020 and December 31 , 2019 , our weighted average servicing fee was 43.6 basis points and 43.8 basis points , respectively .

Gold: []

Prediction: [[17, 18, "LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage"]]

### bilstm_crf: pred_spurious, test row 63

Sentence: The weighted average effective interest rate was 10.91 % and 10.85 % at March 31 , 2020 and December 31 , 2019 , respectively , including the accretion of a portion of the discount deemed collectible .

Gold: []

Prediction: [[7, 8, "DebtInstrumentInterestRateEffectivePercentage"], [10, 11, "DebtInstrumentInterestRateEffectivePercentage"]]

### bilstm_crf: gold_missed, test row 76

Sentence: The debt carrying value for the Agency Business at March 31 , 2020 and December 31 , 2019 was net of unamortized deferred finance costs of $ 1.4 million and $ 0.2 million , respectively . 21 Table of Contents ARBOR REALTY TRUST , INC . AND SUBSIDIARIES NOTES TO CONSOLIDATED FINANCIAL STATEMENTS ( Unaudited ) March 31 , 2020 ( 2 ) This repurchase facility includes six - month extension options into perpetuity .

Gold: [[27, 28, "DeferredFinanceCostsNet"], [31, 32, "DeferredFinanceCostsNet"]]

Prediction: []

### bilstm_crf: gold_wrong_type, test row 115

Sentence: In March 2018 , we issued $ 100.0 million aggregate principal amount of 5.625 % senior unsecured notes due in May 2023 ( the " Initial Notes " ) in a private placement , and , in May 2018 , we issued an additional $ 25.0 million ( the " Reopened Notes " and , together with the Initial Notes , the " 5.625 % Notes , " ) which brought the aggregate outstanding principal amount to $ 125.0 million .

Gold: [[7, 8, "DebtInstrumentFaceAmount"], [13, 14, "DebtInstrumentInterestRateStatedPercentage"], [78, 79, "DebtInstrumentFaceAmount"]]

Prediction: [[7, 8, "DebtInstrumentFaceAmount"], [13, 14, "DebtInstrumentInterestRateStatedPercentage"], [63, 64, "DebtInstrumentInterestRateStatedPercentage"], [78, 79, "DebtInstrumentCarryingAmount"]]

### bilstm_crf: gold_wrong_type, test row 136

Sentence: Including the amortization of the deferred financing fees and debt discount , our weighted average total cost of the notes was 6.75 % and 6.80 % at March 31 , 2020 and December 31 , 2019 , respectively .

Gold: [[21, 22, "DebtInstrumentInterestRateEffectivePercentage"], [24, 25, "DebtInstrumentInterestRateEffectivePercentage"]]

Prediction: [[21, 22, "DebtWeightedAverageInterestRate"], [24, 25, "DebtWeightedAverageInterestRate"]]

### bilstm_crf: gold_boundary_overlap, test row 6070

Sentence: The intangible asset lives range from 5 to 20 years and have a weighted - average remaining life of 16.9 years as of March 31 , 2020 .

Gold: [[6, 7, "FiniteLivedIntangibleAssetUsefulLife"], [8, 10, "FiniteLivedIntangibleAssetUsefulLife"], [19, 20, "FiniteLivedIntangibleAssetUsefulLife"]]

Prediction: [[6, 7, "FiniteLivedIntangibleAssetUsefulLife"], [8, 9, "FiniteLivedIntangibleAssetUsefulLife"], [19, 20, "FiniteLivedIntangibleAssetUsefulLife"]]

### bilstm_crf: gold_boundary_overlap, test row 11887

Sentence: The identified core deposit intangibles for each acquisition are being amortized using a non - pro rata basis over an estimated life of 10 to 15 years .

Gold: [[23, 24, "AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife"], [25, 27, "AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife"]]

Prediction: [[23, 24, "FiniteLivedIntangibleAssetUsefulLife"], [25, 26, "FiniteLivedIntangibleAssetUsefulLife"]]

### bert-base-cased: pred_spurious, test row 61

Sentence: As of March 31 , 2020 , we retained 49 % , or $ 106.2 million initial face value , of seven B Piece bonds , which were purchased at a discount for $ 74.7 million , and sold the remaining 51 % to a third - party at par .

Gold: []

Prediction: [[34, 35, "DebtInstrumentUnamortizedDiscount"]]

### bert-base-cased: pred_spurious, test row 63

Sentence: The weighted average effective interest rate was 10.91 % and 10.85 % at March 31 , 2020 and December 31 , 2019 , respectively , including the accretion of a portion of the discount deemed collectible .

Gold: []

Prediction: [[7, 8, "DebtInstrumentInterestRateEffectivePercentage"], [10, 11, "DebtInstrumentInterestRateEffectivePercentage"]]

### bert-base-cased: gold_missed, test row 81

Sentence: During the three months ended March 31 , 2020 , we amended our joint repurchase facility shared between the Structured Business and the Agency Business to increase the total committed amount by $ 400.0 million to $ 1.10 billion , of which $ 600.0 million matures in March 2021 and $ 500.0 million matures in March 2022 , with each maturity eligible for a one - year extension option .

Gold: [[43, 44, "DebtInstrumentBasisSpreadOnVariableRate1"]]

Prediction: []

### bert-base-cased: gold_missed, test row 85

Sentence: In March 2020 , we amended a $ 300.0 million repurchase agreement , increasing the committed amount to $ 400.0 million .

Gold: [[8, 9, "LineOfCreditFacilityMaximumBorrowingCapacity"], [19, 20, "LineOfCreditFacilityMaximumBorrowingCapacity"]]

Prediction: [[8, 9, "StockRepurchaseProgramAuthorizedAmount1"]]

### bert-base-cased: gold_wrong_type, test row 85

Sentence: In March 2020 , we amended a $ 300.0 million repurchase agreement , increasing the committed amount to $ 400.0 million .

Gold: [[8, 9, "LineOfCreditFacilityMaximumBorrowingCapacity"], [19, 20, "LineOfCreditFacilityMaximumBorrowingCapacity"]]

Prediction: [[8, 9, "StockRepurchaseProgramAuthorizedAmount1"]]

### bert-base-cased: gold_wrong_type, test row 96

Sentence: In March 2020 , we completed the unwind of CLO VIII , redeeming $ 282.9 million of outstanding notes , which were repaid primarily from the refinancing of the remaining assets primarily within CLO XIII , as well as with cash held 23 Table of Contents ARBOR REALTY TRUST , INC . AND SUBSIDIARIES NOTES TO CONSOLIDATED FINANCIAL STATEMENTS ( Unaudited ) March 31 , 2020 by CLO VIII , and expensed $ 1.5 million of deferred financing fees into loss on extinguishment of debt on the consolidated statements of operations .

Gold: [[73, 74, "AmortizationOfFinancingCosts"]]

Prediction: [[14, 15, "RepaymentsOfDebt"], [73, 74, "GainsLossesOnExtinguishmentOfDebt"]]

### bert-base-cased: gold_boundary_overlap, test row 492

Sentence: The weighted average remaining lease term of the Company ’ s operating leases as of March 31 , 2020 was approximately 6 years and the weighted average incremental borrowing rate was 4.2 % as of March 31 , 2020 .

Gold: [[21, 22, "OperatingLeaseWeightedAverageRemainingLeaseTerm1"], [31, 32, "OperatingLeaseWeightedAverageDiscountRatePercent"]]

Prediction: [[21, 23, "OperatingLeaseWeightedAverageRemainingLeaseTerm1"], [31, 32, "OperatingLeaseWeightedAverageDiscountRatePercent"]]

### bert-base-cased: gold_boundary_overlap, test row 3079

Sentence: The Taxable Revenue Bonds ’ aggregate principal amount outstanding as of December 31 , 2019 of $ 24.6 million matured on January 1 , 2020 and was repaid in full on January 2 , 2020 , using the funds held in a certificate of deposit by the Company .

Gold: [[17, 18, "LongTermDebt"], [21, 25, "DebtInstrumentMaturityDate"]]

Prediction: [[17, 18, "DebtInstrumentCarryingAmount"], [21, 22, "DebtInstrumentMaturityDate"]]

### sec-bert-shape: gold_missed, test row 35

Sentence: At December 31 , 2019 , the UPB related to 24 loans with five different borrowers represented 13 % of total assets .

Gold: [[17, 18, "ConcentrationRiskPercentage1"]]

Prediction: []

### sec-bert-shape: gold_missed, test row 39

Sentence: As of December 31 , 2019 , 18 % and 12 % of the outstanding balance of our loan and investment portfolio had underlying properties in New York and Texas , respectively .

Gold: [[7, 8, "ConcentrationRiskPercentage1"], [10, 11, "ConcentrationRiskPercentage1"]]

Prediction: []

### sec-bert-shape: pred_spurious, test row 62

Sentence: These securities are collateralized by a pool of multifamily mortgage loans , bear interest at an initial weighted average variable rate of 3.74 % and have an estimated weighted average remaining maturity of 5.6 years .

Gold: []

Prediction: [[22, 23, "DebtWeightedAverageInterestRate"]]

### sec-bert-shape: pred_spurious, test row 63

Sentence: The weighted average effective interest rate was 10.91 % and 10.85 % at March 31 , 2020 and December 31 , 2019 , respectively , including the accretion of a portion of the discount deemed collectible .

Gold: []

Prediction: [[7, 8, "DebtInstrumentInterestRateEffectivePercentage"], [10, 11, "DebtInstrumentInterestRateEffectivePercentage"]]

### sec-bert-shape: gold_wrong_type, test row 81

Sentence: During the three months ended March 31 , 2020 , we amended our joint repurchase facility shared between the Structured Business and the Agency Business to increase the total committed amount by $ 400.0 million to $ 1.10 billion , of which $ 600.0 million matures in March 2021 and $ 500.0 million matures in March 2022 , with each maturity eligible for a one - year extension option .

Gold: [[43, 44, "DebtInstrumentBasisSpreadOnVariableRate1"]]

Prediction: [[37, 38, "LineOfCreditFacilityMaximumBorrowingCapacity"], [43, 44, "LineOfCreditFacilityMaximumBorrowingCapacity"], [51, 52, "LineOfCreditFacilityMaximumBorrowingCapacity"]]

### sec-bert-shape: gold_wrong_type, test row 96

Sentence: In March 2020 , we completed the unwind of CLO VIII , redeeming $ 282.9 million of outstanding notes , which were repaid primarily from the refinancing of the remaining assets primarily within CLO XIII , as well as with cash held 23 Table of Contents ARBOR REALTY TRUST , INC . AND SUBSIDIARIES NOTES TO CONSOLIDATED FINANCIAL STATEMENTS ( Unaudited ) March 31 , 2020 by CLO VIII , and expensed $ 1.5 million of deferred financing fees into loss on extinguishment of debt on the consolidated statements of operations .

Gold: [[73, 74, "AmortizationOfFinancingCosts"]]

Prediction: [[14, 15, "RepaymentsOfDebt"], [73, 74, "GainsLossesOnExtinguishmentOfDebt"]]

### sec-bert-shape: gold_boundary_overlap, test row 907

Sentence: The SRT Loan matures on January 9 , 2023 .

Gold: [[5, 9, "DebtInstrumentMaturityDate"]]

Prediction: [[5, 6, "DebtInstrumentMaturityDate"]]

### sec-bert-shape: gold_boundary_overlap, test row 3884

Sentence: In July 2014 , Summit Holdings and its 100 % owned finance subsidiary , Finance Corp. ( together with Summit Holdings , the " Co - Issuers " ) co - issued $ 300.0 million of 5.5 % senior unsecured notes maturing August 15 , 2022 ( the " 5.5 % Senior Notes " and , together with the 5.75 % Senior Notes ( defined below ) , the “ Senior Notes ” ) as described in the 2019 Annual Report .

Gold: [[36, 37, "DebtInstrumentInterestRateStatedPercentage"], [42, 46, "DebtInstrumentMaturityDate"], [59, 60, "DebtInstrumentInterestRateStatedPercentage"]]

Prediction: [[33, 34, "DebtInstrumentFaceAmount"], [36, 37, "DebtInstrumentInterestRateStatedPercentage"], [42, 44, "DebtInstrumentMaturityDate"], [59, 60, "DebtInstrumentInterestRateStatedPercentage"]]

## Limitations and scope

- Observational analysis of saved predictions from one training seed; no new training or latency claims.
- O-only downsampling changes the training distribution. Its effect on precision/recall is a hypothesis, not a demonstrated cause.
- No tokenizer-level truncation attribution is possible from saved labels alone.
- Test-set analysis is descriptive; selecting changes against these errors would require fresh validation and an untouched final evaluation.
- No presentation slides or speech notes are stored here.
