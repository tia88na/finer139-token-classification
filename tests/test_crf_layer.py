"""Checks for the shared CRF layer against brute-force enumeration.   Run:  python -m pytest tests -q"""
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from crf_layer import CRF, iob2_allowed, iob2_violations  # noqa: E402

TAGS = ["O", "B-A", "I-A", "B-B", "I-B"]


def random_crf(seed=0):
    torch.manual_seed(seed)
    crf = CRF(TAGS)
    for p in (crf.start, crf.end, crf.trans):
        torch.nn.init.normal_(p)
    lengths = [4, 2, 1]
    em = torch.randn(len(lengths), max(lengths), len(TAGS))
    mask = torch.tensor([[i < n for i in range(max(lengths))] for n in lengths])
    return crf, em, mask, lengths


def brute_force(crf, em, n, constrained):
    """Score of every tag sequence of length n for one sentence: {sequence: score}."""
    start, trans = crf._scores(constrained)
    out = {}
    for y in itertools.product(range(len(TAGS)), repeat=n):
        s = start[y[0]] + em[0, y[0]] + crf.end[y[-1]]
        for i in range(1, n):
            s = s + trans[y[i - 1], y[i]] + em[i, y[i]]
        out[y] = s
    return out


def test_iob2_allowed():
    start, trans = iob2_allowed(TAGS)
    assert start.tolist() == [True, True, False, True, False]
    assert trans[TAGS.index("B-A"), TAGS.index("I-A")] and trans[TAGS.index("I-A"), TAGS.index("I-A")]
    assert not trans[TAGS.index("O"), TAGS.index("I-A")]
    assert not trans[TAGS.index("B-B"), TAGS.index("I-A")]
    assert trans[TAGS.index("I-A"), TAGS.index("B-B")] and trans[TAGS.index("I-A"), TAGS.index("O")]


def test_iob2_violations():
    assert iob2_violations([["O", "B-A", "I-A", "O"], ["B-B"]]) == 0
    assert iob2_violations([["I-A"], ["O", "I-A"], ["B-B", "I-A"]]) == 3


def test_partition_and_nll_match_brute_force():
    crf, em, mask, lengths = random_crf()
    tags = torch.randint(0, len(TAGS), mask.shape)
    for constrained in (False, True):
        crf.constrain_training = constrained
        nll = crf.nll(em, tags, mask)
        for b, n in enumerate(lengths):
            scores = brute_force(crf, em[b], n, constrained)
            log_z = torch.logsumexp(torch.stack(list(scores.values())), 0)
            gold = scores[tuple(tags[b, :n].tolist())]
            assert torch.allclose(nll[b], log_z - gold, atol=1e-4)
            assert nll[b] >= -1e-5


def test_gradients_match_brute_force():
    # The hand-written backward pass (marginals) must equal autograd through brute-force enumeration.
    for constrained in (False, True):
        crf, em, mask, lengths = random_crf(2)
        crf.constrain_training = constrained
        em.requires_grad_(True)
        tags = torch.randint(0, len(TAGS), mask.shape)
        weights = torch.tensor([1.0, 0.5, 2.0])  # different upstream gradient per sentence
        (crf.nll(em, tags, mask) * weights).sum().backward()
        ours = [em.grad.clone()] + [p.grad.clone() for p in (crf.start, crf.trans, crf.end)]
        for p in (em, crf.start, crf.trans, crf.end):
            p.grad = None
        total = 0.0
        for b, n in enumerate(lengths):
            scores = brute_force(crf, em[b], n, constrained)
            log_z = torch.logsumexp(torch.stack(list(scores.values())), 0)
            total = total + weights[b] * (log_z - scores[tuple(tags[b, :n].tolist())])
        total.backward()
        reference = [em.grad] + [p.grad for p in (crf.start, crf.trans, crf.end)]
        for a, r in zip(ours, reference):
            assert torch.allclose(a, r, atol=1e-5), (a - r).abs().max()


def test_partition_is_stable_for_large_scores():
    # Scores of a few hundred (a confident model) must not underflow the exp-space forward pass.
    crf, em, mask, lengths = random_crf(1)
    em = em * 300.0
    tags = torch.randint(0, len(TAGS), mask.shape)
    nll = crf.nll(em, tags, mask)
    for b, n in enumerate(lengths):
        scores = brute_force(crf, em[b].double(), n, constrained=True)
        log_z = torch.logsumexp(torch.stack(list(scores.values())), 0)
        assert torch.isfinite(nll[b]) and abs(nll[b].item() - (log_z - scores[tuple(tags[b, :n].tolist())]).item()) < 1e-2


def test_viterbi_matches_brute_force():
    for seed in range(5):
        crf, em, mask, lengths = random_crf(seed)
        pred = crf.decode(em, mask)
        for b, n in enumerate(lengths):
            scores = brute_force(crf, em[b], n, constrained=True)
            best = max(scores, key=lambda y: scores[y].item())
            assert pred[b] == list(best)


def test_decode_never_breaks_iob2():
    crf, em, mask, _ = random_crf()
    em[:, :, TAGS.index("I-A")] += 10.0   # emissions push hard for I-A everywhere
    pred = crf.decode(em, mask)
    assert iob2_violations([[TAGS[i] for i in p] for p in pred]) == 0


def test_batched_decode_equals_single():
    crf, em, mask, lengths = random_crf(3)
    batched = crf.decode(em, mask)
    for b, n in enumerate(lengths):
        assert crf.decode(em[b:b + 1, :n], mask[b:b + 1, :n]) == [batched[b]]
