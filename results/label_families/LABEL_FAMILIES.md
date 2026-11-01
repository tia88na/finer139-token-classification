# Label family analysis

The 139 entity types are grouped into 11 financial families. Scores reuse `results/analysis/per_class.csv` (main test view, 92,608 sentences, strict IOB2 entity spans, saved predictions); no model was re-run. Summed over all families, the counts reproduce each model's saved micro P/R/F1, and the mean of its 139 per-label F1 values reproduces its saved macro-F1.

## Two family scores

- **Pooled F1** (the main family score): the TP, gold and predicted entity counts of all labels in the family are summed first; then P = TP / predicted, R = TP / gold, F1 = 2 TP / (gold + predicted). It equals the mean of the per-label F1 values weighted by gold + predicted entities, so frequent labels dominate it, as in micro-F1.
- **Label-mean F1**: the unweighted mean of the per-label F1 values in the family. Every label counts equally, as in macro-F1. It is not the F1 of the pooled counts and the two should not be compared with each other as if they were the same measure.
- A span is a true positive only if its boundaries and exact type match. Predicting another type of the same family is a false negative for the gold type and a false positive for the predicted type. These are not family-relaxed scores.

## Pooled F1 by family

| Family | Types | Gold entities | BiLSTM-CRF | BERT-base-cased | SEC-BERT-SHAPE | SHAPE − BERT | SHAPE − BiLSTM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Debt & interest | 21 | 6,248 | 67.61 | 66.88 | **69.40** | +2.52 | +1.79 |
| Credit facilities | 8 | 2,705 | 67.96 | 66.61 | **69.29** | +2.68 | +1.33 |
| Equity & share capital | 21 | 3,366 | 72.90 | 72.28 | **73.81** | +1.53 | +0.91 |
| Share-based compensation | 18 | 4,048 | 76.25 | 76.22 | **77.02** | +0.80 | +0.78 |
| Revenue & customer contracts | 8 | 1,261 | **74.41** | 69.83 | 73.62 | +3.79 | -0.79 |
| Income tax | 6 | 1,438 | 82.10 | 81.93 | **84.03** | +2.10 | +1.93 |
| Leases | 11 | 1,272 | 67.54 | 67.41 | **69.59** | +2.18 | +2.06 |
| Acquisitions & investments | 15 | 2,245 | 63.79 | 62.02 | **66.52** | +4.50 | +2.73 |
| Long-lived assets | 8 | 1,935 | 77.83 | 78.19 | **80.69** | +2.51 | +2.86 |
| Contingencies & guarantees | 6 | 549 | **65.19** | 59.32 | 61.84 | +2.51 | -3.35 |
| Other | 17 | 3,122 | 72.69 | 71.74 | **75.33** | +3.59 | +2.64 |
| **All types (micro-F1)** | 139 | 28,189 | 71.45 | 70.44 | **72.95** | +2.51 | +1.51 |

## Label-mean F1 by family

| Family | Types | BiLSTM-CRF | BERT-base-cased | SEC-BERT-SHAPE | SHAPE − BERT | SHAPE − BiLSTM |
|---|---:|---:|---:|---:|---:|---:|
| Debt & interest | 21 | 62.88 | 60.63 | **63.34** | +2.72 | +0.47 |
| Credit facilities | 8 | **62.84** | 57.95 | 62.06 | +4.10 | -0.79 |
| Equity & share capital | 21 | 67.20 | 66.37 | **68.90** | +2.53 | +1.70 |
| Share-based compensation | 18 | 72.23 | 72.27 | **74.21** | +1.94 | +1.97 |
| Revenue & customer contracts | 8 | **72.39** | 67.37 | 70.74 | +3.37 | -1.65 |
| Income tax | 6 | 78.04 | 79.22 | **81.55** | +2.33 | +3.51 |
| Leases | 11 | 66.15 | 65.84 | **68.16** | +2.31 | +2.01 |
| Acquisitions & investments | 15 | 61.83 | 59.72 | **64.73** | +5.01 | +2.91 |
| Long-lived assets | 8 | 73.45 | 74.34 | **77.09** | +2.75 | +3.64 |
| Contingencies & guarantees | 6 | **63.92** | 59.00 | 61.60 | +2.60 | -2.32 |
| Other | 17 | 70.56 | 68.31 | **72.99** | +4.68 | +2.43 |
| **All types (macro-F1)** | 139 | 67.68 | 66.01 | **69.15** | +3.14 | +1.47 |

## Pooled precision and recall by family

| Family | BiLSTM-CRF P | BiLSTM-CRF R | BERT-base-cased P | BERT-base-cased R | SEC-BERT-SHAPE P | SEC-BERT-SHAPE R |
|---|---:|---:|---:|---:|---:|---:|
| Debt & interest | 62.40 | 73.77 | 59.77 | 75.91 | 62.96 | 77.30 |
| Credit facilities | 61.73 | 75.60 | 59.08 | 76.34 | 62.55 | 77.67 |
| Equity & share capital | 71.09 | 74.81 | 67.22 | 78.16 | 69.63 | 78.52 |
| Share-based compensation | 70.81 | 82.58 | 69.24 | 84.78 | 70.58 | 84.76 |
| Revenue & customer contracts | 72.69 | 76.21 | 65.07 | 75.34 | 69.61 | 78.11 |
| Income tax | 76.80 | 88.18 | 74.11 | 91.59 | 76.42 | 93.32 |
| Leases | 61.16 | 75.39 | 57.90 | 80.66 | 61.02 | 80.97 |
| Acquisitions & investments | 61.83 | 65.88 | 53.91 | 73.01 | 58.99 | 76.26 |
| Long-lived assets | 73.27 | 83.00 | 71.39 | 86.41 | 73.67 | 89.20 |
| Contingencies & guarantees | 59.34 | 72.31 | 49.69 | 73.59 | 52.41 | 75.41 |
| Other | 67.44 | 78.83 | 62.74 | 83.76 | 67.25 | 85.62 |

## Observations

- Highest pooled F1: BiLSTM-CRF in 2, BERT-base-cased in 0, SEC-BERT-SHAPE in 9 of 11 families.
- SEC-BERT-SHAPE minus BERT-base-cased is positive in 11 of 11 families; SEC-BERT-SHAPE minus BiLSTM-CRF is positive in 9.
- BiLSTM-CRF: highest pooled F1 in Income tax (82.10), lowest in Acquisitions & investments (63.79).
- BERT-base-cased: highest pooled F1 in Income tax (81.93), lowest in Contingencies & guarantees (59.32).
- SEC-BERT-SHAPE: highest pooled F1 in Income tax (84.03), lowest in Contingencies & guarantees (61.84).
- Pooled and label-mean F1 differ most for Credit facilities with BERT-base-cased (pooled 66.61, label-mean 57.95). A label-mean below the pooled score means the types with fewer gold + predicted entities score lower; above it, the reverse.
- Families with fewer than 1,000 gold entities (Contingencies & guarantees) rest on the fewest entities; differences between models there are the most likely to be sampling noise.

## Families and their types

Gold = main-view test entities; train = strict entity occurrences in cleaned train. Types marked * are close calls, explained below.

### Debt & interest (21 types, 6,248 gold entities)

- `AmortizationOfFinancingCosts`* (gold 78, train 872)
- `DebtInstrumentBasisSpreadOnVariableRate1` (gold 1259, train 6503)
- `DebtInstrumentCarryingAmount` (gold 312, train 2457)
- `DebtInstrumentConvertibleConversionPrice1` (gold 124, train 602)
- `DebtInstrumentFaceAmount` (gold 998, train 7083)
- `DebtInstrumentFairValue` (gold 109, train 1111)
- `DebtInstrumentInterestRateEffectivePercentage` (gold 256, train 1540)
- `DebtInstrumentInterestRateStatedPercentage` (gold 1272, train 9828)
- `DebtInstrumentMaturityDate` (gold 149, train 752)
- `DebtInstrumentRedemptionPricePercentage` (gold 143, train 1132)
- `DebtInstrumentTerm` (gold 150, train 795)
- `DebtInstrumentUnamortizedDiscount` (gold 126, train 814)
- `DebtWeightedAverageInterestRate` (gold 134, train 982)
- `DeferredFinanceCostsGross` (gold 113, train 595)
- `DeferredFinanceCostsNet` (gold 176, train 1265)
- `GainsLossesOnExtinguishmentOfDebt` (gold 116, train 1057)
- `InterestExpense`* (gold 82, train 795)
- `InterestExpenseDebt` (gold 109, train 1138)
- `LongTermDebt` (gold 295, train 2406)
- `LongTermDebtFairValue` (gold 160, train 1591)
- `RepaymentsOfDebt` (gold 87, train 723)

### Credit facilities (8 types, 2,705 gold entities)

- `LettersOfCreditOutstandingAmount`* (gold 288, train 2745)
- `LineOfCredit` (gold 379, train 3111)
- `LineOfCreditFacilityCommitmentFeePercentage` (gold 104, train 500)
- `LineOfCreditFacilityCurrentBorrowingCapacity` (gold 161, train 1256)
- `LineOfCreditFacilityInterestRateAtPeriodEnd` (gold 95, train 733)
- `LineOfCreditFacilityMaximumBorrowingCapacity` (gold 1228, train 7376)
- `LineOfCreditFacilityRemainingBorrowingCapacity` (gold 265, train 2336)
- `LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage` (gold 185, train 763)

### Equity & share capital (21 types, 3,366 gold entities)

- `AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount`* (gold 642, train 7227)
- `ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1` (gold 153, train 589)
- `CommonStockCapitalSharesReservedForFutureIssuance` (gold 120, train 641)
- `CommonStockDividendsPerShareDeclared` (gold 136, train 1681)
- `CommonStockParOrStatedValuePerShare` (gold 186, train 742)
- `CommonStockSharesAuthorized` (gold 128, train 775)
- `CommonStockSharesOutstanding` (gold 127, train 669)
- `PreferredStockDividendRatePercentage`* (gold 106, train 494)
- `PreferredStockSharesAuthorized` (gold 113, train 569)
- `ProceedsFromIssuanceOfCommonStock` (gold 75, train 503)
- `SaleOfStockNumberOfSharesIssuedInTransaction` (gold 177, train 780)
- `SaleOfStockPricePerShare` (gold 128, train 524)
- `SharePrice` (gold 138, train 750)
- `StockIssuedDuringPeriodSharesNewIssues` (gold 215, train 1292)
- `StockRepurchaseProgramAuthorizedAmount1` (gold 258, train 1862)
- `StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1` (gold 130, train 1309)
- `StockRepurchasedAndRetiredDuringPeriodShares` (gold 89, train 762)
- `StockRepurchasedDuringPeriodShares` (gold 117, train 1021)
- `TreasuryStockAcquiredAverageCostPerShare` (gold 83, train 908)
- `TreasuryStockSharesAcquired` (gold 151, train 1785)
- `TreasuryStockValueAcquiredCostMethod` (gold 94, train 1010)

### Share-based compensation (18 types, 4,048 gold entities)

- `AllocatedShareBasedCompensationExpense` (gold 676, train 8370)
- `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized` (gold 327, train 2897)
- `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1` (gold 406, train 3263)
- `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions` (gold 155, train 1225)
- `EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense`* (gold 64, train 991)
- `ShareBasedCompensation` (gold 127, train 1446)
- `ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1` (gold 338, train 1568)
- `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod` (gold 368, train 3965)
- `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodWeightedAverageGrantDateFairValue` (gold 172, train 1861)
- `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber` (gold 133, train 784)
- `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue` (gold 121, train 1393)
- `ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized` (gold 231, train 1182)
- `ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAvailableForGrant` (gold 225, train 1707)
- `ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsExercisesInPeriodTotalIntrinsicValue` (gold 129, train 1326)
- `ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross` (gold 138, train 1009)
- `ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodWeightedAverageGrantDateFairValue` (gold 95, train 1151)
- `SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage` (gold 237, train 1032)
- `SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod` (gold 106, train 428)

### Revenue & customer contracts (8 types, 1,261 gold entities)

- `CapitalizedContractCostAmortization`* (gold 81, train 1074)
- `ContractWithCustomerLiability` (gold 180, train 1374)
- `ContractWithCustomerLiabilityRevenueRecognized` (gold 274, train 3003)
- `RevenueFromContractWithCustomerExcludingAssessedTax` (gold 272, train 3103)
- `RevenueFromContractWithCustomerIncludingAssessedTax` (gold 46, train 743)
- `RevenueFromRelatedParties`* (gold 107, train 1833)
- `RevenueRemainingPerformanceObligation` (gold 200, train 2072)
- `Revenues` (gold 101, train 1917)

### Income tax (6 types, 1,438 gold entities)

- `EffectiveIncomeTaxRateContinuingOperations` (gold 523, train 7083)
- `EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate` (gold 140, train 1836)
- `IncomeTaxExpenseBenefit` (gold 422, train 4292)
- `OperatingLossCarryforwards`* (gold 104, train 919)
- `UnrecognizedTaxBenefits` (gold 150, train 1540)
- `UnrecognizedTaxBenefitsThatWouldImpactEffectiveTaxRate` (gold 99, train 1049)

### Leases (11 types, 1,272 gold entities)

- `LeaseAndRentalExpense`* (gold 72, train 1500)
- `LesseeOperatingLeaseRenewalTerm` (gold 129, train 669)
- `LesseeOperatingLeaseTermOfContract` (gold 195, train 1346)
- `OperatingLeaseCost` (gold 74, train 1059)
- `OperatingLeaseExpense` (gold 137, train 952)
- `OperatingLeaseLiability` (gold 166, train 1970)
- `OperatingLeasePayments` (gold 80, train 796)
- `OperatingLeaseRightOfUseAsset` (gold 203, train 2509)
- `OperatingLeaseWeightedAverageDiscountRatePercent` (gold 88, train 902)
- `OperatingLeaseWeightedAverageRemainingLeaseTerm1` (gold 96, train 1027)
- `OperatingLeasesRentExpenseNet` (gold 32, train 1179)

### Acquisitions & investments (15 types, 2,245 gold entities)

- `BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued` (gold 123, train 645)
- `BusinessAcquisitionPercentageOfVotingInterestsAcquired` (gold 139, train 1125)
- `BusinessCombinationAcquisitionRelatedCosts` (gold 143, train 1894)
- `BusinessCombinationConsiderationTransferred1` (gold 237, train 2133)
- `BusinessCombinationContingentConsiderationLiability` (gold 152, train 1155)
- `BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill`* (gold 101, train 681)
- `BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles`* (gold 67, train 628)
- `DisposalGroupIncludingDiscontinuedOperationConsideration`* (gold 82, train 743)
- `EquityMethodInvestmentOwnershipPercentage` (gold 390, train 2668)
- `EquityMethodInvestments` (gold 160, train 1448)
- `IncomeLossFromEquityMethodInvestments` (gold 79, train 1225)
- `MinorityInterestOwnershipPercentageByNoncontrollingOwners`* (gold 154, train 833)
- `MinorityInterestOwnershipPercentageByParent`* (gold 122, train 903)
- `PaymentsToAcquireBusinessesGross` (gold 232, train 1855)
- `PaymentsToAcquireBusinessesNetOfCashAcquired` (gold 64, train 715)

### Long-lived assets (8 types, 1,935 gold entities)

- `AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife`* (gold 151, train 1289)
- `AmortizationOfIntangibleAssets` (gold 459, train 6192)
- `AssetImpairmentCharges` (gold 159, train 1194)
- `Depreciation` (gold 248, train 2935)
- `FiniteLivedIntangibleAssetUsefulLife` (gold 184, train 1391)
- `Goodwill` (gold 312, train 2324)
- `GoodwillImpairmentLoss` (gold 248, train 880)
- `PropertyPlantAndEquipmentUsefulLife` (gold 174, train 1016)

### Contingencies & guarantees (6 types, 549 gold entities)

- `AccrualForEnvironmentalLossContingencies` (gold 54, train 1023)
- `GuaranteeObligationsMaximumExposure` (gold 107, train 1054)
- `LossContingencyAccrualAtCarryingValue` (gold 128, train 1050)
- `LossContingencyDamagesSoughtValue` (gold 112, train 611)
- `LossContingencyEstimateOfPossibleLoss` (gold 108, train 1016)
- `LossContingencyPendingClaimsNumber` (gold 40, train 581)

### Other (17 types, 3,122 gold entities)

- `AreaOfRealEstateProperty` (gold 59, train 619)
- `CashAndCashEquivalentsFairValueDisclosure`* (gold 16, train 320)
- `ConcentrationRiskPercentage1`* (gold 611, train 5000)
- `CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption` (gold 75, train 720)
- `DefinedBenefitPlanContributionsByEmployer`* (gold 109, train 1455)
- `DefinedContributionPlanCostRecognized`* (gold 94, train 1593)
- `DerivativeFixedInterestRate`* (gold 108, train 524)
- `DerivativeNotionalAmount`* (gold 559, train 2469)
- `NumberOfOperatingSegments` (gold 212, train 829)
- `NumberOfRealEstateProperties` (gold 227, train 1118)
- `NumberOfReportableSegments` (gold 321, train 1489)
- `PublicUtilitiesRequestedRateIncreaseDecreaseAmount` (gold 36, train 280)
- `RelatedPartyTransactionAmountsOfTransaction` (gold 128, train 804)
- `RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty` (gold 152, train 1943)
- `RestructuringAndRelatedCostExpectedCost1`* (gold 68, train 728)
- `RestructuringCharges`* (gold 319, train 4062)
- `SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense` (gold 28, train 1145)

## Ambiguous assignments

Each type is assigned to the family of its main accounting topic, judged from the XBRL concept name. These types could reasonably belong elsewhere:

| Type | Assigned family and reason |
|---|---|
| `AmortizationOfFinancingCosts` | Debt & interest, not Long-lived assets: amortises debt issuance costs. |
| `InterestExpense` | Debt & interest: a cost of borrowing even when no debt instrument is named. |
| `LettersOfCreditOutstandingAmount` | Credit facilities: letters of credit are usually issued under a credit facility. |
| `AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount` | Equity & share capital: a number of potential shares in the EPS calculation; the securities may be options, warrants or convertible debt. |
| `PreferredStockDividendRatePercentage` | Equity & share capital, not Debt & interest: a rate, but on preferred stock. |
| `EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense` | Share-based compensation, not Income tax: the tax benefit reported with share-based compensation expense. |
| `CapitalizedContractCostAmortization` | Revenue & customer contracts, not Long-lived assets: costs of obtaining or fulfilling customer contracts. |
| `RevenueFromRelatedParties` | Revenue & customer contracts, not Other (related party): a revenue amount. |
| `OperatingLossCarryforwards` | Income tax: tax loss carryforwards. |
| `BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill` | Acquisitions & investments, not Long-lived assets: part of a purchase price allocation. |
| `BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles` | Acquisitions & investments, not Long-lived assets: part of a purchase price allocation. |
| `AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife` | Long-lived assets, not Acquisitions & investments: a useful life of intangible assets. |
| `DisposalGroupIncludingDiscontinuedOperationConsideration` | Acquisitions & investments: consideration for a divestiture, the counterpart of an acquisition. |
| `MinorityInterestOwnershipPercentageByNoncontrollingOwners` | Acquisitions & investments, not Equity & share capital: an ownership interest in a subsidiary. |
| `MinorityInterestOwnershipPercentageByParent` | Acquisitions & investments, not Equity & share capital: an ownership interest in a subsidiary. |
| `LeaseAndRentalExpense` | Leases: also covers rentals that are not operating leases. |
| `DerivativeFixedInterestRate` | Other, not Debt & interest: a derivative (usually a hedge); too few types for a family of their own. |
| `DerivativeNotionalAmount` | Other, not Debt & interest: a derivative (usually a hedge); too few types for a family of their own. |
| `ConcentrationRiskPercentage1` | Other: the concentration may concern revenue, receivables or suppliers. |
| `CashAndCashEquivalentsFairValueDisclosure` | Other: no family for cash or fair value disclosures. |
| `RestructuringAndRelatedCostExpectedCost1` | Other: restructuring, too few types for a family of their own. |
| `RestructuringCharges` | Other: restructuring, too few types for a family of their own. |
| `DefinedBenefitPlanContributionsByEmployer` | Other: retirement plans, too few types for a family of their own. |
| `DefinedContributionPlanCostRecognized` | Other: retirement plans, too few types for a family of their own. |

## Limitations

- The taxonomy is a design choice made from concept names before any family score was computed, and it was not changed afterwards. Other groupings would give other family scores.
- Other is a heterogeneous remainder; its score has no single financial interpretation.
- Descriptive test-set analysis of one training seed per model. No confidence intervals are computed per family; differences in small families may be within sentence-sampling noise (overall intervals: `results/assessment/BOOTSTRAP.md`).
- Family scores do not show why a family is harder; they locate where errors concentrate.

Reproduce: `python src/label_families.py`
