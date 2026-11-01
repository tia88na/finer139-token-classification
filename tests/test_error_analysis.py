"""Check error accounting and strict span semantics against seqeval."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analyze_errors import entities, error_counts, metrics
from finer import scores


def test_strict_orphan_and_type_change():
    assert entities(['I-A', 'I-A', 'B-B', 'I-A', 'B-C', 'I-C']) == {(2, 3, 'B'), (4, 6, 'C')}


def test_errors_partition_false_negatives_and_positives():
    g = {(0, 1, 'A'), (2, 4, 'B'), (5, 6, 'C'), (8, 9, 'D')}
    p = {(0, 1, 'X'), (2, 3, 'B'), (7, 8, 'C'), (8, 9, 'D')}
    c = error_counts(g, p)
    assert c['tp'] == 1
    assert c['gold_wrong_type'] == c['gold_boundary_overlap'] == c['gold_missed'] == 1
    assert c['pred_wrong_type'] == c['pred_boundary_overlap'] == c['pred_spurious'] == 1
    assert sum(c[k] for k in ('gold_wrong_type', 'gold_boundary_overlap', 'gold_missed')) == c['gold'] - c['tp']
    assert sum(c[k] for k in ('pred_wrong_type', 'pred_boundary_overlap', 'pred_spurious')) == c['predicted'] - c['tp']


def test_span_metrics_match_strict_seqeval():
    import itertools
    gold = ['B-A', 'I-A', 'O']
    for pred in itertools.product(['O', 'B-A', 'I-A', 'B-B', 'I-B'], repeat=3):
        gs, ps = entities(gold), entities(pred)
        own = metrics(len(gs & ps), len(gs), len(ps))
        expected = scores([gold], [list(pred)])
        assert round(own['f1'], 4) == expected['micro_F1']
        assert round(own['precision'], 4) == expected['micro_P']
        assert round(own['recall'], 4) == expected['micro_R']
