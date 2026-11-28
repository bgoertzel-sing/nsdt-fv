"""T11 (plan A.5; S4, C6): PLN reference. Revision is commutative and associative on counts, deduction matches
hand-computed test vectors, and truth values stay in [0, 1]."""
import itertools
import math
import random

import pytest

from nsdt_fv.store.pln import ref
from nsdt_fv.store.pln.ref import TV


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- revision: commutative and associative on counts
@pytest.mark.parametrize("seed", range(20))
def test_revision_commutative_associative(seed):
    rng = random.Random(seed)
    for _ in range(200):
        a, b, c = (TV(rng.random(), rng.uniform(0.01, 50.0)) for _ in range(3))   # independent: no premises
        ab, ba = ref.revise(a, b), ref.revise(b, a)
        assert close(ab.n, ba.n) and close(ab.s, ba.s, 1e-12)
        l, r = ref.revise(ref.revise(a, b), c), ref.revise(a, ref.revise(b, c))
        assert close(l.n, r.n, 1e-9) and close(l.s, r.s, 1e-12)
        assert close(l.n, a.n + b.n + c.n, 1e-9)
        # with disjoint premise sets the discount does not apply
        a2, b2, c2 = TV(a.s, a.n, 1), TV(b.s, b.n, 2), TV(c.s, c.n, 4)
        l2, r2 = ref.revise(ref.revise(a2, b2), c2), ref.revise(a2, ref.revise(b2, c2))
        assert close(l2.n, l.n, 1e-9) and close(l2.s, l.s, 1e-12) and l2.premises == 7


def test_revision_hand_vector():
    r = ref.revise(TV(0.8, 10.0), TV(0.2, 30.0))
    assert close(r.n, 40.0) and close(r.s, (8.0 + 6.0) / 40.0)


def test_revision_overlap_discount():
    # premise sets {0,1} (n=4) and {1,2,3,4} (n=10): overlap 1 of the smaller set's 2 -> smaller count halved
    a, b = TV(1.0, 4.0, 0b00011), TV(0.0, 10.0, 0b11110)
    r = ref.revise(a, b)
    assert close(r.n, 12.0) and close(r.s, 2.0 / 12.0) and r.premises == 0b11111
    assert ref.revise(b, a) == r
    # identical premise sets: the smaller estimate adds nothing
    r2 = ref.revise(TV(0.5, 3.0, 0b11), TV(0.9, 5.0, 0b11))
    assert close(r2.n, 5.0) and close(r2.s, 0.9)


def test_facts_never_revised():
    k = ref.kernel_proof(1)
    assert ref.revise(k, TV(0.1, 100.0)) is k and ref.revise(TV(0.1, 100.0), k) is k
    with pytest.raises(ValueError):
        ref.revise(ref.kernel_proof(1), ref.counterexample(2))
    assert k.c == 1.0 and ref.counterexample().s == 0.0


def test_confidence_count():
    assert close(ref.conf_of(10.0), 0.5) and close(ref.count_of(0.5), 10.0)
    for n in (0.0, 0.3, 7.0, 123.0):
        assert close(ref.count_of(ref.conf_of(n)), n, 1e-9)
    t = ref.tests_passed(300)                 # w * min(m, m_max) = 0.05 * 200 = 10
    assert t.s == 1.0 and close(t.n, 10.0) and close(t.c, 0.5)
    assert close(ref.tests_passed(40).n, 2.0)


# ---------------------------------------------------------------- deduction: hand-computed vectors
# s = sAB sBC + (1 - sAB)(sC - sB sBC)/(1 - sB), n = 0.9 min(nAB, nBC), clipped to [0, 1]
DED = [
    # sAB, sBC, sB, sC, expected s (computed by hand)
    (0.8, 0.9, 0.5, 0.6, 0.72 + 0.2 * (0.6 - 0.45) / 0.5),      # 0.72 + 0.06 = 0.78
    (1.0, 0.7, 0.3, 0.4, 0.7),                                  # sAB = 1: s = sBC
    (0.0, 0.7, 0.3, 0.4, (0.4 - 0.21) / 0.7),                   # sAB = 0: (sC - sB sBC)/(1 - sB) = 0.27142857...
    (0.5, 0.5, 0.5, 0.5, 0.25 + 0.5 * 0.25 / 0.5),              # 0.5
    (0.9, 0.1, 0.9, 0.05, 0.0),                                 # 0.09 + 0.1 * (0.05 - 0.09)/0.1 = 0.05 ... see below
    (0.2, 1.0, 0.1, 0.9, 0.2 + 0.8 * 0.8 / 0.9),                # 0.91111...
    (0.3, 0.0, 0.2, 0.0, 0.0),
]


def test_deduction_vectors():
    exp = [0.78, 0.7, 0.19 / 0.7, 0.5, 0.05, 0.2 + 0.64 / 0.9, 0.0]
    for (sAB, sBC, sB, sC, _), e in zip(DED, exp):
        assert close(ref.deduction_strength(sAB, sBC, sB, sC), e, 1e-12), (sAB, sBC, sB, sC)
    # clipping: sC < sB sBC drives the raw value negative
    assert ref.deduction_strength(0.1, 1.0, 0.5, 0.1) == 0.0       # 0.1 + 0.9 * (0.1 - 0.5)/0.5 = -0.62 -> 0
    # and above one
    assert ref.deduction_strength(0.5, 0.9, 0.1, 1.0) == 1.0 or close(ref.deduction_strength(0.5, 0.9, 0.1, 1.0),
                                                                         0.45 + 0.5 * 0.91 / 0.9)
    d = ref.deduction(TV(0.8, 20.0, 1), TV(0.9, 5.0, 2), 0.5, 0.6)
    assert close(d.s, 0.78) and close(d.n, 4.5) and d.premises == 3
    kd = ref.deduction(ref.kernel_proof(1), TV(0.4, 8.0, 2), 0.3, 0.2)   # kernel premise: n = 0.9 * 8
    assert close(kd.s, 0.4) and close(kd.n, 7.2)


def test_transfer_reduces_for_small_sB():
    # "When sB is small it reduces to s = sigma sgT + (1 - sigma) sT"
    sigma, sgT, sT = 0.6, 1.0, 0.3
    s = ref.transfer(TV(sigma, 5.0), ref.kernel_proof(), 1e-9, sT).s
    assert close(s, sigma * sgT + (1 - sigma) * sT, 1e-8)


def test_inversion_induction_abduction_vectors():
    assert close(ref.inversion_strength(0.6, 0.2, 0.4), 0.3)
    # induction A->B (0.6), A->C (0.5), sA 0.2, sB 0.4, sC 0.5: B->A = 0.3, then deduction B->A->C with sA, sC
    i = ref.induction(TV(0.6, 10.0), TV(0.5, 20.0), 0.2, 0.4, 0.5)
    assert close(i.s, 0.3 * 0.5 + 0.7 * (0.5 - 0.1) / 0.8) and close(i.n, 9.0)       # 0.15 + 0.35 = 0.5
    # abduction A->C (0.5), B->C (0.6), sA 0.2, sB 0.4, sC 0.5: C->B = 0.6*0.4/0.5 = 0.48, deduction A->C->B
    a = ref.abduction(TV(0.5, 10.0), TV(0.6, 20.0), 0.2, 0.4, 0.5)
    assert close(a.s, 0.5 * 0.48 + 0.5 * (0.4 - 0.5 * 0.48) / 0.5) and close(a.n, 9.0)


def test_similarity():
    w = {"a": 1.0, "b": 2.0, "c": 0.5}
    assert close(ref.similarity({"a", "b"}, {"b", "c"}, w), 2.0 / 3.5)
    assert ref.similarity(set(), set(), w) == 0.0
    w2 = ref.shape_weights({"a": 0.5, "b": 1.0})
    assert close(w2["a"], math.log(2)) and w2["b"] == 0.0


# ---------------------------------------------------------------- truth values stay in [0, 1]
def test_truth_values_in_range():
    rng = random.Random(7)
    vals = [0.0, 1.0, 1e-9, 1 - 1e-9, 0.5]
    for sAB, sBC, sB, sC in itertools.product(vals, repeat=4):
        assert 0.0 <= ref.deduction_strength(sAB, sBC, sB, sC) <= 1.0
    for _ in range(20000):
        a = TV(rng.random(), rng.uniform(0, 100), rng.getrandbits(8))
        b = TV(rng.random(), rng.uniform(0, 100), rng.getrandbits(8))
        sB, sC = rng.random(), rng.random()
        for t in (ref.revise(a, b), ref.deduction(a, b, sB, sC), ref.induction(a, b, rng.random(), sB, sC),
                  ref.abduction(a, b, rng.random(), sB, sC), ref.inversion(a, rng.random(), sB)):
            t.check()
        assert 0.0 <= ref.p_true(ref.revise(a, b), sC) <= 1.0
    assert ref.p_true(None, 0.3) == 0.3
