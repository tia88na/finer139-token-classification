"""Linear-chain CRF layer shared by the feature CRF (baseline_crf.py) and the BiLSTM-CRF (bilstm_crf.py).

For a sentence with words 1..n and a tag sequence y, the CRF score is

    s(y) = start[y_1] + sum_i e_i[y_i] + sum_{i>1} trans[y_{i-1}, y_i] + end[y_n]

and P(y | sentence) = exp s(y) / Z, where Z sums exp s over every tag sequence.
The two models differ only in the emission scores e_i: a sum of hand-made feature
weights (CRF) or the output of a BiLSTM (BiLSTM-CRF).

* Training minimises -log P(gold) = log Z - s(gold). log Z comes from the forward
  algorithm, computed in log space.
* Decoding is Viterbi: the tag sequence with the highest score.
* IOB2 constraints: I-X may not open a sentence or follow anything other than B-X / I-X.
  Strict seqeval never counts such a tag as an entity, so the decoder never produces one.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

FORBIDDEN = -1e4  # added to the score of a transition that IOB2 does not allow


def iob2_allowed(tags: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean masks start[L] and trans[L, L] (previous tag, next tag) of the allowed moves."""
    n = len(tags)
    start = torch.ones(n, dtype=torch.bool)
    trans = torch.ones(n, n, dtype=torch.bool)
    for j, t in enumerate(tags):
        if t.startswith("I-"):
            start[j] = False
            for i, s in enumerate(tags):
                trans[i, j] = s in ("B-" + t[2:], t)
    return start, trans


def iob2_violations(tag_seqs) -> int:
    """Number of I-X tags that open a sentence or follow a tag other than B-X / I-X."""
    bad = 0
    for tags in tag_seqs:
        prev = "O"
        for t in tags:
            bad += t.startswith("I-") and prev[2:] != t[2:]
            prev = t
    return bad


class LogPartition(torch.autograd.Function):
    """log Z of every sentence (forward algorithm) with its exact gradient (backward algorithm).

    Forward:  alpha_t[j] = sum_i alpha_{t-1}[i] exp(trans[i, j]) exp(e_t[j]),  Z = sum_j alpha_n[j] exp(end[j]),
    computed in exp space as one matrix product per word and rescaled to sum 1 at every word
    (log Z = sum of the log scales), in float64 so nothing underflows. Forbidden IOB2 moves
    (-1e4) become exact zeros. The gradient of log Z is the expected feature count:
        d log Z / d e_t[j]       = P(y_t = j)               = alpha_t[j] beta_t[j] / Z
        d log Z / d trans[i, j]  = sum_t P(y_t-1 = i, y_t = j)
    with beta from the backward algorithm (same scaling). Writing the gradient out, instead of
    letting autograd unroll the loop over words, needs ~7 small GPU operations per word instead
    of ~45, and the loop over words (not the arithmetic) is what costs time. Sentences are
    sorted by length so the ones still running at word t are always the first k_t rows.
    The tests check log Z and every gradient against brute-force enumeration.
    """

    @staticmethod
    def forward(ctx, em, lengths, start, trans, end):
        b, n, k = em.shape
        dev = em.device
        lengths, order = lengths.sort(descending=True, stable=True)
        active = (lengths.numpy()[None, :] > np.arange(n)[:, None]).sum(1).tolist()   # k_t
        order = order.to(dev)
        x = em.detach().double()[order].transpose(0, 1).contiguous()                  # [T, B, L]
        top = x.max(2, keepdim=True).values
        e = (x - top).exp_()
        shift = [p.detach().max() for p in (start, trans, end)]
        s_exp, m_exp, end_exp = [(p.detach().double() - c).exp() for p, c in zip((start, trans, end), shift)]
        alpha = torch.zeros(n, b, k, dtype=torch.float64, device=dev)
        scale = torch.ones(n, b, dtype=torch.float64, device=dev)
        v = s_exp * e[0]
        torch.sum(v, 1, out=scale[0]).clamp_(min=1e-300)
        torch.div(v, scale[0].unsqueeze(1), out=alpha[0])
        for t in range(1, n):
            a = active[t]
            v = torch.mm(alpha[t - 1, :a], m_exp).mul_(e[t, :a])
            torch.sum(v, 1, out=scale[t, :a]).clamp_(min=1e-300)
            torch.div(v, scale[t, :a].unsqueeze(1), out=alpha[t, :a])
        last, rows = (lengths - 1).to(dev), torch.arange(b, device=dev)
        z = alpha[last, rows] @ end_exp
        words = torch.arange(n, device=dev).unsqueeze(1) <= last.unsqueeze(0)          # [T, B]
        log_z = (scale.log() + top.squeeze(2) * words).sum(0) + z.log() + last * shift[1].double() + shift[0] + shift[2]
        ctx.save_for_backward(e, alpha, scale, m_exp, end_exp, z, order, last)
        ctx.active = active
        out = torch.empty_like(log_z)
        out[order] = log_z
        return out.to(em.dtype)

    @staticmethod
    def backward(ctx, grad):
        e, alpha, scale, m_exp, end_exp, z, order, last = ctx.saved_tensors
        n, b, k = e.shape
        rows = torch.arange(b, device=e.device)
        g = grad.double()[order]
        beta = torch.zeros_like(alpha)
        beta[last, rows] = end_exp / z.unsqueeze(1)          # each sentence starts at its last word
        flow = torch.zeros_like(alpha)                        # e_t beta_t / scale_t
        for t in range(n - 1, 0, -1):
            a = ctx.active[t]
            torch.mul(e[t, :a], beta[t, :a], out=flow[t, :a]).div_(scale[t, :a].unsqueeze(1))
            torch.mm(flow[t, :a], m_exp.T, out=beta[t - 1, :a])
        marg = alpha * beta * g.view(1, b, 1)                 # P(y_t = j), weighted per sentence
        d_em = torch.empty(b, n, k, dtype=torch.float64, device=e.device)
        d_em[order] = marg.transpose(0, 1)
        d_trans = m_exp * ((alpha[:-1] * g.view(1, b, 1)).reshape(-1, k).T @ flow[1:].reshape(-1, k))
        d_start, d_end = marg[0].sum(0), marg[last, rows].sum(0)
        return d_em.to(grad.dtype), None, d_start.float(), d_trans.float(), d_end.float()


class CRF(nn.Module):
    """Transition, start and end scores over `tags`, with the IOB2 constraints built in.

    Inputs are batches: emissions [B, T, L], tags [B, T] (internal ids), mask [B, T]
    (True on real words, padding only at the end, every sentence at least one word).
    """

    def __init__(self, tags: list[str], constrain_training: bool = True):
        super().__init__()
        n = len(tags)
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))
        self.trans = nn.Parameter(torch.zeros(n, n))
        start_ok, trans_ok = iob2_allowed(tags)
        self.register_buffer("start_penalty", (~start_ok).float() * FORBIDDEN)
        self.register_buffer("trans_penalty", (~trans_ok).float() * FORBIDDEN)
        self.constrain_training = constrain_training

    def _scores(self, constrained: bool):
        if constrained:
            return self.start + self.start_penalty, self.trans + self.trans_penalty
        return self.start, self.trans

    def nll(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """-log P(tags | sentence) for every sentence of the batch, shape [B]."""
        emissions = emissions.float()
        start, trans = self._scores(self.constrain_training)
        return self._log_partition(emissions, mask, start, trans) - self._score(emissions, tags, mask, start, trans)

    def _score(self, em, tags, mask, start, trans):
        """s(tags): start + emissions + transitions + end, padding ignored."""
        m = mask.float()
        s = start[tags[:, 0]] + (em.gather(2, tags.unsqueeze(2)).squeeze(2) * m).sum(1)
        if tags.shape[1] > 1:
            s = s + (trans[tags[:, :-1], tags[:, 1:]] * m[:, 1:]).sum(1)
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return s + self.end[last]

    def _log_partition(self, em, mask, start, trans):
        return LogPartition.apply(em, mask.sum(1).cpu(), start, trans, self.end)

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi (always with the IOB2 constraints): best tag ids, one list per sentence."""
        em = emissions.float()
        start, trans = self._scores(True)
        b, n, k = em.shape
        keep = torch.arange(k, device=em.device).expand(b, k)  # identity backpointer on padding
        score = start + em[:, 0]
        back = []
        for t in range(1, n):
            best, prev = (score.unsqueeze(2) + trans).max(dim=1)
            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, best + em[:, t], score)
            back.append(torch.where(m, prev, keep))
        last = (score + self.end).argmax(1)
        path = [last]
        for bp in reversed(back):
            last = bp.gather(1, last.unsqueeze(1)).squeeze(1)
            path.append(last)
        path = torch.stack(path[::-1], 1).cpu()
        return [path[i, :int(length)].tolist() for i, length in enumerate(mask.sum(1).cpu())]
