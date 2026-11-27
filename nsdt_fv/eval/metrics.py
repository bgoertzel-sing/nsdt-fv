"""Metrics of plan section 7, computed from raw verdict logs (rule R3).

E_j is the in-window test set of stage j (j = 1..T). a[t, j] is the pass@k rate on E_j of the model
obtained after training through stage t (t = 0 is the base model). For a stream of T stages:

  LA  = 1/T     sum_{j=1..T}   a[j, j]
  FWD = 1/T     sum_{j=1..T}   a[j-1, j]
  ACC = 1/T     sum_{j=1..T}   a[T, j]
  F   = 1/(T-1) sum_{j=1..T-1} ( max_{j<=t<=T-1} a[t, j]  -  a[T, j] )
  BWT = 1/(T-1) sum_{j=1..T-1} ( a[T, j] - a[j, j] )
  GC  = (ACC_arm - ACC_B2) / (ACC_B6 - ACC_B2)

pass@k uses the unbiased estimator 1 - C(n-c, k) / C(n, k) over the n >= k samples of an
obligation (with n = k this is "any of the k samples accepted").
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict


class MetricsError(ValueError):
    pass


def read_log(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pass_at_k(n: int, c: int, k: int) -> float:
    if k < 1 or n < k:
        raise MetricsError(f"pass@{k} needs at least {k} samples, got {n}")
    if c < 0 or c > n:
        raise MetricsError("accepted count out of range")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def accuracy_matrix(lines, test_sets: dict, k: int, run: str | None = None, phase: str = "eval"):
    """{(t, j): a[t, j]} from verdict lines. test_sets: {j: [obligation ids]}.

    A cell exists for (t, j) if any line of stage t targets E_j; then every obligation of E_j must
    have at least k samples at stage t (structural check), and sample indices must be unique."""
    where = {}
    for j, ids in test_sets.items():
        for i in ids:
            if i in where:
                raise MetricsError(f"obligation {i} is in two test sets")
            where[i] = j
    samples = defaultdict(dict)          # (t, id) -> {sample: accepted}
    for ln in lines:
        if ln["phase"] != phase or (run is not None and ln["run"] != run):
            continue
        if ln["target_id"] not in where:
            continue
        key = (int(ln["stage"]), ln["target_id"])
        s = int(ln["sample"])
        if s in samples[key]:
            raise MetricsError(f"duplicate sample {s} for {key}")
        samples[key][s] = bool(ln["accepted"])
    cells = defaultdict(list)
    touched = set()
    for (t, i) in samples:
        touched.add((t, where[i]))
    out = {}
    for (t, j) in sorted(touched):
        vals = []
        for i in test_sets[j]:
            got = samples.get((t, i))
            if not got:
                raise MetricsError(f"stage {t}: obligation {i} of E_{j} has no samples")
            vals.append(pass_at_k(len(got), sum(got.values()), k))
        out[(t, j)] = math.fsum(vals) / len(vals)
    return out


def _a(A, t, j):
    if (t, j) not in A:
        raise MetricsError(f"missing a[{t},{j}]")
    return A[(t, j)]


def stream_metrics(A: dict, T: int) -> dict:
    if T < 2:
        raise MetricsError("F and BWT need T >= 2")
    js = range(1, T + 1)
    LA = math.fsum(_a(A, j, j) for j in js) / T
    FWD = math.fsum(_a(A, j - 1, j) for j in js) / T
    ACC = math.fsum(_a(A, T, j) for j in js) / T
    F = math.fsum(max(_a(A, t, j) for t in range(j, T)) - _a(A, T, j) for j in range(1, T)) / (T - 1)
    BWT = math.fsum(_a(A, T, j) - _a(A, j, j) for j in range(1, T)) / (T - 1)
    return {"LA": LA, "FWD": FWD, "ACC": ACC, "F": F, "BWT": BWT}


def gap_closed(acc_arm: float, acc_b2: float, acc_b6: float) -> float:
    if acc_b6 == acc_b2:
        raise MetricsError("GC undefined: ACC_B6 == ACC_B2")
    return (acc_arm - acc_b2) / (acc_b6 - acc_b2)


def metrics_from_log(lines, test_sets, k, run=None):
    A = accuracy_matrix(lines, test_sets, k, run=run)
    return stream_metrics(A, len(test_sets)), A


# ---------------------------------------------------------------- counts for reports (R3)
def count(lines, **where) -> int:
    return sum(1 for ln in lines if all(ln.get(k) == v for k, v in where.items()))


def tally(lines, field, **where) -> dict:
    return dict(Counter(ln.get(field) for ln in lines if all(ln.get(k) == v for k, v in where.items())))


def distinct(lines, field, **where) -> int:
    return len({ln.get(field) for ln in lines if all(ln.get(k) == v for k, v in where.items())})
