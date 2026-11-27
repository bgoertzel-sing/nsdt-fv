"""Paired bootstrap over obligations (plan section 7, Statistics) for the stream metrics.

Per-obligation pass@k values p[t][id] are read from verdict logs. A resample draws, independently
within each in-window test set E_j, n_j obligations with replacement; the same draw is applied to
every stage t and every arm (paired). Each resample recomputes LA, FWD, ACC, F, BWT exactly as
eval/metrics.py does, on resampled cell means.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .metrics import MetricsError, pass_at_k


def per_obligation(lines, test_sets, k, run=None, phase="eval"):
    """{(t, j): np.array of pass@k per obligation of E_j, in test-set order}."""
    where = {i: (j, n) for j, ids in test_sets.items() for n, i in enumerate(ids)}
    cnt = defaultdict(lambda: [0, 0])
    for ln in lines:
        if ln["phase"] != phase or (run is not None and ln["run"] != run) or ln["target_id"] not in where:
            continue
        c = cnt[(int(ln["stage"]), ln["target_id"])]
        c[0] += 1
        c[1] += int(bool(ln["accepted"]))
    cells = defaultdict(dict)
    for (t, i), (n, c) in cnt.items():
        j, pos = where[i]
        cells[(t, j)][pos] = pass_at_k(n, c, k)
    out = {}
    for (t, j), d in cells.items():
        if len(d) != len(test_sets[j]):
            raise MetricsError(f"cell ({t},{j}) incomplete")
        out[(t, j)] = np.array([d[p] for p in range(len(test_sets[j]))], dtype=float)
    return out


def draws(test_sets, B, seed):
    rng = np.random.default_rng(seed)
    return {j: rng.integers(0, len(ids), size=(B, len(ids))) for j, ids in sorted(test_sets.items())}


def resampled_metrics(P, T, R):
    """P: per_obligation table; R: draws. Returns dict metric -> array of B resampled values."""
    A = {cell: v[R[cell[1]]].mean(axis=1) for cell, v in P.items()}
    js = range(1, T + 1)
    LA = sum(A[(j, j)] for j in js) / T
    FWD = sum(A[(j - 1, j)] for j in js) / T
    ACC = sum(A[(T, j)] for j in js) / T
    F = sum(np.max(np.stack([A[(t, j)] for t in range(j, T)]), axis=0) - A[(T, j)] for j in range(1, T)) / (T - 1)
    BWT = sum(A[(T, j)] - A[(j, j)] for j in range(1, T)) / (T - 1)
    return {"LA": LA, "FWD": FWD, "ACC": ACC, "F": F, "BWT": BWT}


def interval(x, level=0.95):
    lo, hi = np.quantile(x, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
