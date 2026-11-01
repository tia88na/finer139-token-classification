import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from compare_models_bootstrap import comparison_rows


def test_named_contrasts_survive_reordered_columns_and_identical_predictions():
    samples = np.array([[.2, .2, .7], [.4, .4, .6], [.6, .6, .9]])
    point = samples.mean(axis=0)
    models = ('a', 'b', 'c')
    pairs = (('c', 'a'), ('a', 'b'))
    expected = comparison_rows(samples, point, models, pairs)
    actual = comparison_rows(samples[:, [2, 0, 1]], point[[2, 0, 1]], ('c', 'a', 'b'), pairs)
    assert expected == actual
    assert expected[0]['delta_f1_points'] > 0
    assert all(expected[1][k] == 0 for k in ('delta_f1_points', 'ci95_low', 'ci95_high', 'family95_low', 'family95_high'))


def test_eight_comparison_adjustment_uses_wider_tail_quantiles():
    x = np.linspace(0, 1, 10001)
    samples = np.column_stack([x, np.zeros_like(x), np.ones_like(x)])
    pairs = (('a', 'b'), ('c', 'a'), ('b', 'a'), ('a', 'c'),
             ('b', 'c'), ('c', 'b'), ('a', 'a'), ('b', 'b'))
    row = comparison_rows(samples, samples.mean(axis=0), ('a', 'b', 'c'), pairs)[0]
    assert np.isclose(row['ci95_low'], 2.5)
    assert np.isclose(row['ci95_high'], 97.5)
    assert np.isclose(row['family95_low'], .3125)
    assert np.isclose(row['family95_high'], 99.6875)
