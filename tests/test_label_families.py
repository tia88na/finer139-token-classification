"""Check that the label families partition the 139 types and that family scores pool counts."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from finer import ID2LABEL
from label_families import AMBIGUOUS, FAMILIES, family_scores, label_to_family, partition


def test_every_type_is_in_exactly_one_family():
    types = {tag[2:] for tag in ID2LABEL if tag != 'O'}
    mapping = label_to_family()
    assert len(types) == 139
    assert set(mapping) == types
    assert sum(len(labels) for labels in FAMILIES.values()) == len(types)
    assert len(FAMILIES) == 11 and 'Other' in FAMILIES
    assert set(AMBIGUOUS) <= types


@pytest.mark.parametrize('families, types', [
    ({'A': ('x',), 'B': ('x', 'y')}, ['x', 'y']),  # duplicate
    ({'A': ('x',)}, ['x', 'y']),                    # missing
    ({'A': ('x', 'z')}, ['x']),                     # unknown
    ({'A': ('x',), 'B': ()}, ['x']),                # empty family
])
def test_partition_rejects_invalid_families(families, types):
    with pytest.raises(ValueError):
        partition(families, types)


def test_family_f1_pools_counts_instead_of_averaging_label_f1():
    rows = pd.DataFrame([dict(model='m', label='x', train_entities=5, gold=1, predicted=1, tp=1),
                         dict(model='m', label='y', train_entities=7, gold=9, predicted=9, tp=0),
                         dict(model='m', label='z', train_entities=1, gold=2, predicted=0, tp=0)])
    out = family_scores(rows, {'x': 'F', 'y': 'F', 'z': 'G'}).set_index('family')
    assert out.loc['F', ['labels', 'train_entities', 'tp', 'gold', 'predicted']].tolist() == [2, 12, 1, 10, 10]
    assert out.loc['F', 'f1'] == pytest.approx(.1)
    assert out.loc['F', 'label_mean_f1'] == pytest.approx(.5)
    assert out.loc['G', ['precision', 'recall', 'f1']].tolist() == [0, 0, 0]


def test_pooled_f1_is_mean_label_f1_weighted_by_gold_plus_predicted():
    rows = pd.DataFrame([dict(model='m', label='x', train_entities=0, gold=4, predicted=6, tp=3),
                         dict(model='m', label='y', train_entities=0, gold=5, predicted=1, tp=1)])
    out = family_scores(rows, {'x': 'F', 'y': 'F'}).iloc[0]
    weighted = (10 * (2 * 3 / 10) + 6 * (2 * 1 / 6)) / 16
    assert out.f1 == pytest.approx(weighted)


def test_label_without_family_is_rejected():
    rows = pd.DataFrame([dict(model='m', label='x', train_entities=0, gold=1, predicted=1, tp=1)])
    with pytest.raises(ValueError):
        family_scores(rows, {})
