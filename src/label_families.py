"""Group the 139 FiNER-139 entity types into financial families and score each family.

Run from the repository root: python src/label_families.py
Reads results/analysis/per_class.csv (written by src/analyze_errors.py); no prediction is
re-read and no model is re-run. All outputs go to results/label_families/.

Family scores pool the per-label TP, gold and predicted entity counts (strict IOB2, main
test view). A span is a true positive only if its exact type matches, so predicting another
type of the same family is still an error. The unweighted mean of the per-label F1 values
of a family is reported separately and is not the F1 of the pooled counts.
"""
import json
from pathlib import Path

import pandas as pd

from analyze_errors import MODELS, metrics
from evaluate_predictions import NAMES
from finer import ID2LABEL

ROOT = Path(__file__).resolve().parents[1]

# Assigned from the XBRL concept names alone, before any family score was computed.
# Each type goes to the family of its main accounting topic; close calls are in AMBIGUOUS.
FAMILIES = {
    'Debt & interest': (
        'AmortizationOfFinancingCosts',
        'DebtInstrumentBasisSpreadOnVariableRate1',
        'DebtInstrumentCarryingAmount',
        'DebtInstrumentConvertibleConversionPrice1',
        'DebtInstrumentFaceAmount',
        'DebtInstrumentFairValue',
        'DebtInstrumentInterestRateEffectivePercentage',
        'DebtInstrumentInterestRateStatedPercentage',
        'DebtInstrumentMaturityDate',
        'DebtInstrumentRedemptionPricePercentage',
        'DebtInstrumentTerm',
        'DebtInstrumentUnamortizedDiscount',
        'DebtWeightedAverageInterestRate',
        'DeferredFinanceCostsGross',
        'DeferredFinanceCostsNet',
        'GainsLossesOnExtinguishmentOfDebt',
        'InterestExpense',
        'InterestExpenseDebt',
        'LongTermDebt',
        'LongTermDebtFairValue',
        'RepaymentsOfDebt',
    ),
    'Credit facilities': (
        'LettersOfCreditOutstandingAmount',
        'LineOfCredit',
        'LineOfCreditFacilityCommitmentFeePercentage',
        'LineOfCreditFacilityCurrentBorrowingCapacity',
        'LineOfCreditFacilityInterestRateAtPeriodEnd',
        'LineOfCreditFacilityMaximumBorrowingCapacity',
        'LineOfCreditFacilityRemainingBorrowingCapacity',
        'LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage',
    ),
    'Equity & share capital': (
        'AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount',
        'ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1',
        'CommonStockCapitalSharesReservedForFutureIssuance',
        'CommonStockDividendsPerShareDeclared',
        'CommonStockParOrStatedValuePerShare',
        'CommonStockSharesAuthorized',
        'CommonStockSharesOutstanding',
        'PreferredStockDividendRatePercentage',
        'PreferredStockSharesAuthorized',
        'ProceedsFromIssuanceOfCommonStock',
        'SaleOfStockNumberOfSharesIssuedInTransaction',
        'SaleOfStockPricePerShare',
        'SharePrice',
        'StockIssuedDuringPeriodSharesNewIssues',
        'StockRepurchaseProgramAuthorizedAmount1',
        'StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1',
        'StockRepurchasedAndRetiredDuringPeriodShares',
        'StockRepurchasedDuringPeriodShares',
        'TreasuryStockAcquiredAverageCostPerShare',
        'TreasuryStockSharesAcquired',
        'TreasuryStockValueAcquiredCostMethod',
    ),
    'Share-based compensation': (
        'AllocatedShareBasedCompensationExpense',
        'EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized',
        'EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1',
        'EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions',
        'EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense',
        'ShareBasedCompensation',
        'ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1',
        'ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod',
        'ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodWeightedAverageGrantDateFairValue',
        'ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber',
        'ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue',
        'ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized',
        'ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAvailableForGrant',
        'ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsExercisesInPeriodTotalIntrinsicValue',
        'ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross',
        'ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodWeightedAverageGrantDateFairValue',
        'SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage',
        'SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod',
    ),
    'Revenue & customer contracts': (
        'CapitalizedContractCostAmortization',
        'ContractWithCustomerLiability',
        'ContractWithCustomerLiabilityRevenueRecognized',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'RevenueFromRelatedParties',
        'RevenueRemainingPerformanceObligation',
        'Revenues',
    ),
    'Income tax': (
        'EffectiveIncomeTaxRateContinuingOperations',
        'EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate',
        'IncomeTaxExpenseBenefit',
        'OperatingLossCarryforwards',
        'UnrecognizedTaxBenefits',
        'UnrecognizedTaxBenefitsThatWouldImpactEffectiveTaxRate',
    ),
    'Leases': (
        'LeaseAndRentalExpense',
        'LesseeOperatingLeaseRenewalTerm',
        'LesseeOperatingLeaseTermOfContract',
        'OperatingLeaseCost',
        'OperatingLeaseExpense',
        'OperatingLeaseLiability',
        'OperatingLeasePayments',
        'OperatingLeaseRightOfUseAsset',
        'OperatingLeaseWeightedAverageDiscountRatePercent',
        'OperatingLeaseWeightedAverageRemainingLeaseTerm1',
        'OperatingLeasesRentExpenseNet',
    ),
    'Acquisitions & investments': (
        'BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued',
        'BusinessAcquisitionPercentageOfVotingInterestsAcquired',
        'BusinessCombinationAcquisitionRelatedCosts',
        'BusinessCombinationConsiderationTransferred1',
        'BusinessCombinationContingentConsiderationLiability',
        'BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill',
        'BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles',
        'DisposalGroupIncludingDiscontinuedOperationConsideration',
        'EquityMethodInvestmentOwnershipPercentage',
        'EquityMethodInvestments',
        'IncomeLossFromEquityMethodInvestments',
        'MinorityInterestOwnershipPercentageByNoncontrollingOwners',
        'MinorityInterestOwnershipPercentageByParent',
        'PaymentsToAcquireBusinessesGross',
        'PaymentsToAcquireBusinessesNetOfCashAcquired',
    ),
    'Long-lived assets': (
        'AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife',
        'AmortizationOfIntangibleAssets',
        'AssetImpairmentCharges',
        'Depreciation',
        'FiniteLivedIntangibleAssetUsefulLife',
        'Goodwill',
        'GoodwillImpairmentLoss',
        'PropertyPlantAndEquipmentUsefulLife',
    ),
    'Contingencies & guarantees': (
        'AccrualForEnvironmentalLossContingencies',
        'GuaranteeObligationsMaximumExposure',
        'LossContingencyAccrualAtCarryingValue',
        'LossContingencyDamagesSoughtValue',
        'LossContingencyEstimateOfPossibleLoss',
        'LossContingencyPendingClaimsNumber',
    ),
    'Other': (
        'AreaOfRealEstateProperty',
        'CashAndCashEquivalentsFairValueDisclosure',
        'ConcentrationRiskPercentage1',
        'CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption',
        'DefinedBenefitPlanContributionsByEmployer',
        'DefinedContributionPlanCostRecognized',
        'DerivativeFixedInterestRate',
        'DerivativeNotionalAmount',
        'NumberOfOperatingSegments',
        'NumberOfRealEstateProperties',
        'NumberOfReportableSegments',
        'PublicUtilitiesRequestedRateIncreaseDecreaseAmount',
        'RelatedPartyTransactionAmountsOfTransaction',
        'RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty',
        'RestructuringAndRelatedCostExpectedCost1',
        'RestructuringCharges',
        'SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense',
    ),
}

_TOO_FEW = 'too few types for a family of their own'
AMBIGUOUS = {
    'AmortizationOfFinancingCosts': 'Debt & interest, not Long-lived assets: amortises debt issuance costs.',
    'InterestExpense': 'Debt & interest: a cost of borrowing even when no debt instrument is named.',
    'LettersOfCreditOutstandingAmount': 'Credit facilities: letters of credit are usually issued under a credit facility.',
    'AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount':
        'Equity & share capital: a number of potential shares in the EPS calculation; '
        'the securities may be options, warrants or convertible debt.',
    'PreferredStockDividendRatePercentage': 'Equity & share capital, not Debt & interest: a rate, but on preferred stock.',
    'EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense':
        'Share-based compensation, not Income tax: the tax benefit reported with share-based compensation expense.',
    'CapitalizedContractCostAmortization':
        'Revenue & customer contracts, not Long-lived assets: costs of obtaining or fulfilling customer contracts.',
    'RevenueFromRelatedParties': 'Revenue & customer contracts, not Other (related party): a revenue amount.',
    'OperatingLossCarryforwards': 'Income tax: tax loss carryforwards.',
    'BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill':
        'Acquisitions & investments, not Long-lived assets: part of a purchase price allocation.',
    'BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles':
        'Acquisitions & investments, not Long-lived assets: part of a purchase price allocation.',
    'AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife':
        'Long-lived assets, not Acquisitions & investments: a useful life of intangible assets.',
    'DisposalGroupIncludingDiscontinuedOperationConsideration':
        'Acquisitions & investments: consideration for a divestiture, the counterpart of an acquisition.',
    'MinorityInterestOwnershipPercentageByNoncontrollingOwners':
        'Acquisitions & investments, not Equity & share capital: an ownership interest in a subsidiary.',
    'MinorityInterestOwnershipPercentageByParent':
        'Acquisitions & investments, not Equity & share capital: an ownership interest in a subsidiary.',
    'LeaseAndRentalExpense': 'Leases: also covers rentals that are not operating leases.',
    'DerivativeFixedInterestRate': f'Other, not Debt & interest: a derivative (usually a hedge); {_TOO_FEW}.',
    'DerivativeNotionalAmount': f'Other, not Debt & interest: a derivative (usually a hedge); {_TOO_FEW}.',
    'ConcentrationRiskPercentage1': 'Other: the concentration may concern revenue, receivables or suppliers.',
    'CashAndCashEquivalentsFairValueDisclosure': 'Other: no family for cash or fair value disclosures.',
    'RestructuringAndRelatedCostExpectedCost1': f'Other: restructuring, {_TOO_FEW}.',
    'RestructuringCharges': f'Other: restructuring, {_TOO_FEW}.',
    'DefinedBenefitPlanContributionsByEmployer': f'Other: retirement plans, {_TOO_FEW}.',
    'DefinedContributionPlanCostRecognized': f'Other: retirement plans, {_TOO_FEW}.',
}


def partition(families, types):
    """Label -> family. Raises unless every type is in exactly one nonempty family."""
    mapping = {}
    for family, labels in families.items():
        if not labels:
            raise ValueError(f'{family}: empty family')
        for label in labels:
            if label in mapping:
                raise ValueError(f'{label}: in both {mapping[label]} and {family}')
            mapping[label] = family
    missing, unknown = set(types) - set(mapping), set(mapping) - set(types)
    if missing or unknown:
        raise ValueError(f'types without a family: {sorted(missing)}; unknown types: {sorted(unknown)}')
    return mapping


def label_to_family():
    """The FAMILIES mapping, checked against the 139 entity types of the dataset."""
    types = {tag[2:] for tag in ID2LABEL if tag != 'O'}
    mapping = partition(FAMILIES, types)
    if set(AMBIGUOUS) - types:
        raise ValueError(f'unknown types in AMBIGUOUS: {sorted(set(AMBIGUOUS) - types)}')
    return mapping


def family_scores(per_class, mapping):
    """Pooled and label-mean scores per (model, family).

    per_class has one row per (model, label) with integer-valued train_entities, gold,
    predicted and tp. Pooled P/R/F1 come from the summed counts; label_mean_f1 is the
    unweighted mean of the per-label F1 values.
    """
    frame = per_class.assign(family=per_class.label.map(mapping))
    if frame.family.isna().any():
        raise ValueError(f'labels without a family: {sorted(frame.label[frame.family.isna()])}')
    rows = []
    for model in frame.model.unique():
        for family in dict.fromkeys(mapping.values()):
            group = frame[(frame.model == model) & (frame.family == family)]
            if group.empty:
                continue
            tp, gold, predicted = (int(group[k].sum()) for k in ('tp', 'gold', 'predicted'))
            label_f1 = [metrics(int(r.tp), int(r.gold), int(r.predicted))['f1'] for r in group.itertuples()]
            rows.append(dict(model=model, family=family, labels=len(group),
                             train_entities=int(group.train_entities.sum()), gold=gold, predicted=predicted,
                             tp=tp, **metrics(tp, gold, predicted), label_mean_f1=sum(label_f1) / len(label_f1)))
    return pd.DataFrame(rows)


def main():
    output = ROOT / 'results' / 'label_families'
    output.mkdir(exist_ok=True)
    mapping = label_to_family()
    per_class = pd.read_csv(ROOT / 'results' / 'analysis' / 'per_class.csv')
    summary = pd.read_csv(ROOT / 'results' / 'analysis' / 'summary.csv').set_index('model')
    for model in MODELS:
        if set(per_class.label[per_class.model == model]) != set(mapping):
            raise ValueError(f'{model}: per_class.csv does not list each of the {len(mapping)} types once')
    gold = per_class.pivot(index='label', columns='model', values='gold')
    if (gold.nunique(axis=1) != 1).any():
        raise ValueError('gold entity counts differ between models')

    scores = family_scores(per_class, mapping)
    overall = {}
    for model in MODELS:
        rows = scores[scores.model == model]
        tp, gold_n, predicted = (int(rows[k].sum()) for k in ('tp', 'gold', 'predicted'))
        if (tp, gold_n, predicted) != tuple(int(summary.loc[model, k]) for k in ('tp', 'gold', 'predicted')):
            raise AssertionError(f'{model}: family counts do not add up to summary.csv')
        own = metrics(tp, gold_n, predicted)
        labels = per_class[per_class.model == model]
        own['macro_f1'] = sum(metrics(int(r.tp), int(r.gold), int(r.predicted))['f1']
                              for r in labels.itertuples()) / len(labels)
        saved = json.loads((ROOT / 'results' / f'{model}_eval.json').read_text())['main']
        for k, ref in (('precision', 'micro_P'), ('recall', 'micro_R'), ('f1', 'micro_F1'), ('macro_f1', 'macro_F1')):
            if round(own[k], 4) != saved[ref]:
                raise AssertionError(f'{model}: {k} disagrees with the saved main result')
        overall[model] = dict(tp=tp, gold=gold_n, predicted=predicted, **own)
        print(f'{model}: families reproduce main micro-F1 {own["f1"]:.4f} and macro-F1 {own["macro_f1"]:.4f}')

    first = per_class[per_class.model == MODELS[0]].set_index('label')
    pd.DataFrame([dict(label=label, family=family, train_entities=int(first.loc[label, 'train_entities']),
                       test_gold_entities=int(first.loc[label, 'gold']), ambiguous_note=AMBIGUOUS.get(label, ''))
                  for family, labels in FAMILIES.items() for label in labels]
                 ).to_csv(output / 'label_to_family.csv', index=False)
    scores.to_csv(output / 'family_scores.csv', index=False)
    write_report(output, scores, overall, first)


def write_report(output, scores, overall, labels):
    families = list(FAMILIES)
    table = {col: scores.pivot(index='family', columns='model', values=col).loc[families, list(MODELS)]
             for col in ('precision', 'recall', 'f1', 'label_mean_f1')}
    size = scores[scores.model == MODELS[0]].set_index('family')
    names = [NAMES[m] for m in MODELS]
    bilstm, bert, shape = MODELS

    def row(values):
        best = max(values)
        return ' | '.join(f'**{100*v:.2f}**' if v == best else f'{100*v:.2f}' for v in values)

    lines = ['# Label family analysis', '',
             'The 139 entity types are grouped into 11 financial families. Scores reuse '
             '`results/analysis/per_class.csv` (main test view, 92,608 sentences, strict IOB2 entity spans, '
             'saved predictions); no model was re-run. Summed over all families, the counts reproduce each '
             "model's saved micro P/R/F1, and the mean of its 139 per-label F1 values reproduces its saved macro-F1.", '',
             '## Two family scores', '',
             '- **Pooled F1** (the main family score): the TP, gold and predicted entity counts of all labels in '
             'the family are summed first; then P = TP / predicted, R = TP / gold, F1 = 2 TP / (gold + predicted). '
             'It equals the mean of the per-label F1 values weighted by gold + predicted entities, so frequent '
             'labels dominate it, as in micro-F1.',
             '- **Label-mean F1**: the unweighted mean of the per-label F1 values in the family. Every label counts '
             'equally, as in macro-F1. It is not the F1 of the pooled counts and the two should not be compared '
             'with each other as if they were the same measure.',
             '- A span is a true positive only if its boundaries and exact type match. Predicting another type of '
             'the same family is a false negative for the gold type and a false positive for the predicted type. '
             'These are not family-relaxed scores.', '',
             '## Pooled F1 by family', '',
             f'| Family | Types | Gold entities | {" | ".join(names)} | SHAPE − BERT | SHAPE − BiLSTM |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    f1 = table['f1']
    for family in families:
        v = f1.loc[family]
        lines.append(f'| {family} | {size.loc[family, "labels"]} | {size.loc[family, "gold"]:,} | {row(v.tolist())} | '
                     f'{100*(v[shape]-v[bert]):+.2f} | {100*(v[shape]-v[bilstm]):+.2f} |')
    v = {m: overall[m]['f1'] for m in MODELS}
    lines.append(f'| **All types (micro-F1)** | 139 | {overall[shape]["gold"]:,} | {row(list(v.values()))} | '
                 f'{100*(v[shape]-v[bert]):+.2f} | {100*(v[shape]-v[bilstm]):+.2f} |')
    lines += ['', '## Label-mean F1 by family', '',
              f'| Family | Types | {" | ".join(names)} | SHAPE − BERT | SHAPE − BiLSTM |',
              '|---|---:|---:|---:|---:|---:|---:|']
    mean = table['label_mean_f1']
    for family in families:
        v = mean.loc[family]
        lines.append(f'| {family} | {size.loc[family, "labels"]} | {row(v.tolist())} | '
                     f'{100*(v[shape]-v[bert]):+.2f} | {100*(v[shape]-v[bilstm]):+.2f} |')
    v = {m: overall[m]['macro_f1'] for m in MODELS}
    lines.append(f'| **All types (macro-F1)** | 139 | {row(list(v.values()))} | '
                 f'{100*(v[shape]-v[bert]):+.2f} | {100*(v[shape]-v[bilstm]):+.2f} |')
    lines += ['', '## Pooled precision and recall by family', '',
              '| Family | ' + ' | '.join(f'{n} P | {n} R' for n in names) + ' |',
              '|---|' + '---:|' * (2 * len(MODELS))]
    for family in families:
        lines.append(f'| {family} | ' + ' | '.join(f'{100*table["precision"].loc[family, m]:.2f} | '
                                                   f'{100*table["recall"].loc[family, m]:.2f}' for m in MODELS) + ' |')

    winners = f1.eq(f1.max(axis=1), axis=0).sum()
    gap = (mean - f1).stack()
    widest = gap.abs().idxmax()
    small = [f for f in families if size.loc[f, 'gold'] < 1000]
    lines += ['', '## Observations', '',
              '- Highest pooled F1: ' + ', '.join(f'{NAMES[m]} in {winners[m]}' for m in MODELS)
              + f' of {len(families)} families.',
              f'- SEC-BERT-SHAPE minus BERT-base-cased is positive in {int((f1[shape] > f1[bert]).sum())} of '
              f'{len(families)} families; SEC-BERT-SHAPE minus BiLSTM-CRF is positive in '
              f'{int((f1[shape] > f1[bilstm]).sum())}.']
    for m in MODELS:
        lines.append(f'- {NAMES[m]}: highest pooled F1 in {f1[m].idxmax()} ({100*f1[m].max():.2f}), '
                     f'lowest in {f1[m].idxmin()} ({100*f1[m].min():.2f}).')
    lines += [f'- Pooled and label-mean F1 differ most for {widest[0]} with {NAMES[widest[1]]} '
              f'(pooled {100*f1.loc[widest]:.2f}, label-mean {100*mean.loc[widest]:.2f}). '
              'A label-mean below the pooled score means the types with fewer gold + predicted entities score lower; '
              'above it, the reverse.',
              f'- Families with fewer than 1,000 gold entities ({", ".join(small) or "none"}) rest on the fewest '
              'entities; differences between models there are the most likely to be sampling noise.']
    lines += ['', '## Families and their types', '',
              'Gold = main-view test entities; train = strict entity occurrences in cleaned train. '
              'Types marked * are close calls, explained below.']
    for family, members in FAMILIES.items():
        lines += ['', f'### {family} ({len(members)} types, {size.loc[family, "gold"]:,} gold entities)', '']
        for label in members:
            mark = '*' if label in AMBIGUOUS else ''
            lines.append(f'- `{label}`{mark} (gold {labels.loc[label, "gold"]}, '
                         f'train {labels.loc[label, "train_entities"]})')
    lines += ['', '## Ambiguous assignments', '',
              'Each type is assigned to the family of its main accounting topic, judged from the XBRL concept name. '
              'These types could reasonably belong elsewhere:', '',
              '| Type | Assigned family and reason |', '|---|---|']
    lines += [f'| `{label}` | {note} |' for label, note in AMBIGUOUS.items()]
    lines += ['', '## Limitations', '',
              '- The taxonomy is a design choice made from concept names before any family score was computed, '
              'and it was not changed afterwards. Other groupings would give other family scores.',
              '- Other is a heterogeneous remainder; its score has no single financial interpretation.',
              '- Descriptive test-set analysis of one training seed per model. No confidence intervals are computed '
              'per family; differences in small families may be within sentence-sampling noise '
              '(overall intervals: `results/assessment/BOOTSTRAP.md`).',
              '- Family scores do not show why a family is harder; they locate where errors concentrate.', '',
              'Reproduce: `python src/label_families.py`', '']
    (output / 'LABEL_FAMILIES.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
