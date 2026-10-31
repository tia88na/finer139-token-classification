import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from assess_results import paired_bootstrap, pooled_f1, sample_candidates


def test_pooled_f1_is_not_mean_sentence_f1():
    counts=np.array([[1,1,1],[0,9,9]])
    assert pooled_f1(counts.sum(axis=0)) == .1
    assert pooled_f1(counts).mean() == .5
    assert pooled_f1([0,0,0]) == 0


def test_pairing_and_seed_are_preserved():
    counts=np.array([[[1,1,2],[1,1,2]],[[0,2,0],[0,2,0]],[[0,0,0],[0,0,0]]])
    a,k=paired_bootstrap(counts,100,42)
    b,_=paired_bootstrap(counts,100,42)
    assert k==3
    assert np.array_equal(a,b)
    assert np.all(a[:,0]-a[:,1]==0)


def test_grouped_draws_match_known_two_sentence_distribution():
    # Each replicate has 0, 1 or 2 copies of the successful sentence,
    # with exact probabilities 1/4, 1/2, 1/4 (two fair draws).
    counts=np.array([[[1,1,1]],[[0,1,1]]])
    draws,_=paired_bootstrap(counts,20000,8)
    vals,frequencies=np.unique(draws,return_counts=True)
    assert np.array_equal(vals,[0,.5,1])
    assert np.allclose(frequencies/20000,[.25,.5,.25],atol=.02)


def test_audit_sample_without_replacement_is_reproducible():
    pop=list(range(100))
    s=sample_candidates(pop,50,42)
    assert len(set(s))==50
    assert s==sample_candidates(pop,50,42)
    with pytest.raises(ValueError):
        sample_candidates(pop,101,42)


def test_invalid_counts_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([[[2,1,1]]]),10)
