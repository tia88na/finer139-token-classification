"""Check the majority-vote rule, the BIO handling and the validity of every ensemble output."""
import itertools
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ensemble import KEY, VOTERS, canonical, combine, is_valid, repair, run, vote, write_outputs
from finer import scores

TAGS = ['O', 'B-A', 'I-A', 'B-B', 'I-B']
SEQS2 = [list(s) for s in itertools.product(TAGS, repeat=2)]
SEQS3 = [list(s) for s in itertools.product(TAGS, repeat=3)]


def test_canonical_turns_orphan_i_tags_into_o():
    assert canonical(['I-A', 'I-A', 'B-B', 'I-A', 'B-C', 'I-C']) == ['O', 'O', 'B-B', 'O', 'B-C', 'I-C']
    assert canonical(['B-A', 'I-A', 'B-A', 'O', 'B-B']) == ['B-A', 'I-A', 'B-A', 'O', 'B-B']


def test_canonical_keeps_strict_scores():
    for gold in (['B-A', 'I-A', 'O'], ['O', 'B-B', 'B-A'], ['B-B', 'I-B', 'I-B']):
        for pred in SEQS3:
            assert scores([gold], [pred]) == scores([gold], [canonical(pred)])


def test_validity_matches_the_crf_layer_definition():
    pytest.importorskip('torch')
    from crf_layer import iob2_violations
    for seq in SEQS3:
        assert is_valid(seq) == (iob2_violations([seq]) == 0)


def test_majority_wins_and_o_is_a_vote():
    assert vote([['B-A', 'O'], ['B-A', 'O'], ['B-B', 'B-A']], 2) == ['B-A', 'O']
    assert vote([['O'], ['O'], ['B-A']], 2) == ['O']


def test_three_way_tie_goes_to_the_tie_breaker():
    assert vote([['B-A'], ['B-B'], ['O']], 2) == ['O']
    assert vote([['B-A'], ['B-B'], ['O']], 0) == ['B-A']


def test_different_sentence_lengths_are_rejected():
    with pytest.raises(ValueError):
        vote([['O'], ['O', 'O'], ['O']], 2)


def test_repair_starts_an_entity_at_every_orphan_i_tag():
    assert repair(['I-A', 'I-A', 'O', 'I-B', 'B-A', 'I-B']) == ['B-A', 'I-A', 'O', 'B-B', 'B-A', 'B-B']
    for seq in SEQS3:
        if is_valid(seq):
            assert repair(seq) == seq


def test_unanimous_voters_give_their_strict_reading():
    for seq in SEQS3:
        assert combine([seq, seq, seq], 2) == canonical(seq)


def test_every_combination_gives_valid_iob2():
    for seqs in itertools.product(SEQS2, repeat=3):
        out = combine(list(seqs), 2)
        assert len(out) == 2 and is_valid(out)
    rng = random.Random(0)
    for _ in range(3000):
        seqs = [[rng.choice(TAGS) for _ in range(7)] for _ in range(3)]
        assert is_valid(combine(seqs, 2))


def test_only_the_tie_breaker_position_matters():
    for a, b, c in itertools.product(SEQS2, repeat=3):
        assert combine([a, b, c], 2) == combine([b, a, c], 2)


def test_output_leaves_the_tie_breaker_only_where_the_others_agree():
    for a, b, c in itertools.product(SEQS2, repeat=3):
        x, y, z = canonical(a), canonical(b), canonical(c)
        for i, tag in enumerate(vote([x, y, z], 2)):
            assert tag == z[i] or x[i] == y[i] == tag


def test_run_and_outputs_on_synthetic_data(tmp_path):
    assert VOTERS == ('bilstm_crf', 'bert-base-cased', 'sec-bert-shape')
    a, b = 'Revenues', 'Goodwill'
    gold = [[f'B-{a}', 'O', f'B-{b}'], ['O', f'B-{a}', f'I-{a}'], [f'B-{b}', 'O'], ['O', 'O']]
    test = pd.DataFrame({'tokens': [['w'] * len(g) for g in gold], 'tags': gold,
                         'in_validation': [False, False, False, True]})
    predictions = {
        'bilstm_crf': [[f'B-{a}', 'O', f'B-{b}'], ['O', f'B-{a}', f'I-{a}'], ['O', 'O'], ['O', 'O']],
        'bert-base-cased': [[f'B-{a}', 'O', 'O'], ['O', f'I-{a}', f'I-{a}'], [f'B-{b}', 'O'], ['O', f'B-{a}']],
        'sec-bert-shape': [[f'B-{b}', 'O', f'B-{b}'], ['O', f'B-{a}', f'B-{a}'], [f'B-{a}', 'O'], ['O', 'O']],
    }
    r = run(test, predictions, replicates=200, seed=1, verbose=False)
    assert r['predictions'][0] == [f'B-{a}', 'O', f'B-{b}']   # BiLSTM-CRF and BERT outvote the tie-breaker
    assert r['predictions'][2] == [f'B-{a}', 'O']             # three-way tie: the tie-breaker decides
    assert all(is_valid(p) for p in r['predictions'])
    assert r['cleaning'][1]['orphan_tags_to_O'] == 2           # BERT's entity that starts with I-
    assert r['predictions'][1] == ['O', f'B-{a}', f'B-{a}']
    assert r['stats']['sentences'] == 3
    write_outputs(tmp_path, r, dict(git=None, file_sha256={}))
    for name in (f'{KEY}_test_predictions.parquet', f'{KEY}_eval.json', f'{KEY}_per_class.csv', 'scores.csv',
                 'cleaning_check.csv', 'vote_stats.csv', 'bootstrap.csv', 'method.json', 'ENSEMBLE.md'):
        assert (tmp_path / name).exists()
    assert len(pd.read_parquet(tmp_path / f'{KEY}_test_predictions.parquet')) == len(test)
    assert 'exploratory' in (tmp_path / 'ENSEMBLE.md').read_text(encoding='utf-8').lower()
