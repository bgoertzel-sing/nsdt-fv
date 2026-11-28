"""Python reference of the PLN fragment of Appendix E (plan v0.1), for Stage S and P0.

Simple truth values (s, c): strength and confidence; confidence and evidence count are related by
c = n / (n + k), n = k c / (1 - c), with k = 10. A kernel-accepted proof is (1, 1) and a counterexample
(0, 1): binary facts with infinite count, never revised. A batch of m passed random tests is strength 1 with
count n = w * min(m, m_max), w = 0.05, m_max = 200.

Revision (independent evidence): s = (n1 s1 + n2 s2) / (n1 + n2), n = n1 + n2. With premise sets, the smaller
count is first discounted by the overlap of the two premise sets: n_small *= 1 - |P1 & P2| / |P_small| (our
reading of "discounted by the overlap"; with disjoint premise sets this is plain revision).
Deduction (A->B, B->C, term strengths sB, sC): s = sAB sBC + (1 - sAB)(sC - sB sBC)/(1 - sB), n = d min(nAB, nBC),
d = 0.9, clipped to [0, 1]. Inversion: sBA = sAB sA / sB. Induction (A->B, A->C |- B->C): invert the first
premise, then deduction. Abduction (A->C, B->C |- A->B): invert the second premise, then deduction.
Similarity of two functions: weighted overlap of their proved shapes, weight -log sT, over the weighted union.
Transfer: deduction with Similar(f, g) read as inheritance f->g with strength sigma and HasProperty(g, T).
P(q true): the revised strength shrunk toward the prior sT by (1 - c): c s + (1 - c) sT.

Premise sets are Python ints used as bitsets (cell indices), so union and overlap are | and &.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

K = 10.0
W_TEST = 0.05
M_MAX = 200
D_DED = 0.9
INF = math.inf


def clip01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def conf_of(n, k=K):
    if n == INF:
        return 1.0
    return n / (n + k)


def count_of(c, k=K):
    if c >= 1.0:
        return INF
    return k * c / (1.0 - c)


@dataclass(frozen=True)
class TV:
    s: float
    n: float                 # evidence count (INF for a kernel fact)
    premises: int = 0        # bitset of premise ids
    fact: bool = False       # kernel proof or counterexample: never revised

    @property
    def c(self):
        return conf_of(self.n)

    def check(self):
        if not (0.0 <= self.s <= 1.0) or not (0.0 <= self.c <= 1.0) or self.n < 0:
            raise ValueError(f"truth value out of range: {self}")
        return self


def kernel_proof(pid=0):
    return TV(1.0, INF, pid, True)


def counterexample(pid=0):
    return TV(0.0, INF, pid, True)


def tests_passed(m, pid=0, w=W_TEST, m_max=M_MAX):
    return TV(1.0, w * min(m, m_max), pid)


def _popcount(x):
    return bin(x).count("1")


def revise(a: TV, b: TV) -> TV:
    """Revision of two estimates of the same relation. Facts are never revised: a fact wins (two facts must agree)."""
    if a.fact or b.fact:
        if a.fact and b.fact and a.s != b.s:
            raise ValueError("contradictory kernel facts")
        return a if a.fact else b
    na, nb = a.n, b.n
    if a.premises and b.premises:
        ov = _popcount(a.premises & b.premises)
        if ov:
            if na <= nb:
                na *= 1.0 - ov / _popcount(a.premises)
            else:
                nb *= 1.0 - ov / _popcount(b.premises)
    n = na + nb
    s = (na * a.s + nb * b.s) / n if n > 0 else (a.s + b.s) / 2.0
    return TV(clip01(s), n, a.premises | b.premises)


def revise_all(tvs):
    """Revise a list of estimates in a fixed order: decreasing count, then decreasing strength (deterministic)."""
    tvs = sorted(tvs, key=lambda t: (-t.n, -t.s, t.premises))
    out = None
    for t in tvs:
        out = t if out is None else revise(out, t)
    return out


def deduction_strength(sAB, sBC, sB, sC):
    if sB >= 1.0 - 1e-12:            # B is everything: A->C carries C's strength (limit of the formula)
        return clip01(sC)
    return clip01(sAB * sBC + (1.0 - sAB) * (sC - sB * sBC) / (1.0 - sB))


def deduction(ab: TV, bc: TV, sB, sC, d=D_DED) -> TV:
    return TV(deduction_strength(ab.s, bc.s, sB, sC), d * min(ab.n, bc.n), ab.premises | bc.premises)


def inversion_strength(sAB, sA, sB):
    if sB <= 0.0:
        return 0.0
    return clip01(sAB * sA / sB)


def inversion(ab: TV, sA, sB) -> TV:
    return TV(inversion_strength(ab.s, sA, sB), ab.n, ab.premises)


def induction(ab: TV, ac: TV, sA, sB, sC, d=D_DED) -> TV:
    """A->B, A->C |- B->C: invert A->B to B->A, then deduction B->A->C (middle term A)."""
    return deduction(inversion(ab, sA, sB), ac, sA, sC, d)


def abduction(ac: TV, bc: TV, sA, sB, sC, d=D_DED) -> TV:
    """A->C, B->C |- A->B: invert B->C to C->B, then deduction A->C->B (middle term C)."""
    return deduction(ac, inversion(bc, sB, sC), sC, sB, d)


def shape_weights(freq):
    """{shape: -log sT} for sT in (0, 1]."""
    return {t: (-math.log(s) if s > 0 else 0.0) for t, s in freq.items()}


def similarity(shapes_f, shapes_g, weight):
    """Weighted overlap of two sets of proved shapes over their weighted union."""
    union = shapes_f | shapes_g
    den = sum(weight[t] for t in union)
    if den <= 0:
        return 0.0
    return sum(weight[t] for t in shapes_f & shapes_g) / den


def transfer(sim: TV, g_has: TV, sB, sT, d=D_DED) -> TV:
    """Similar(f, g) as inheritance f->g (strength sigma), HasProperty(g, T): deduction f->g->T."""
    return deduction(sim, g_has, sB, sT, d)


def p_true(tv: TV | None, prior):
    """Strength shrunk toward the prior by (1 - c)."""
    if tv is None:
        return clip01(prior)
    c = tv.c
    return clip01(c * tv.s + (1.0 - c) * prior)
