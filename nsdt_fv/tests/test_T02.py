"""T2 (A.5, S0): on synthetic verdict logs with known answers, metrics.py reproduces LA, FWD, ACC,
F, BWT and GC to 1e-9. The expected values are hand-derived literals (nsdt_fv/eval/t2_cases.py)."""
import itertools
import json

import pytest

from nsdt_fv.eval import t2_cases
from nsdt_fv.eval.metrics import (MetricsError, accuracy_matrix, gap_closed, pass_at_k, read_log,
                                  stream_metrics)
from nsdt_fv.eval.s0 import run_t2, t2_results

TOL = 1e-9


def test_case_a():
    lines, ts = t2_cases.build_case_a()
    m = stream_metrics(accuracy_matrix(lines, ts, 1, run="T2/A"), 3)
    for k, v in t2_cases.CASE_A["expected"].items():
        assert abs(m[k] - v) <= TOL, (k, m[k], v)


def test_case_b_pass_at_k():
    lines, ts = t2_cases.build_case_b()
    m = stream_metrics(accuracy_matrix(lines, ts, 2, run="T2/B"), 2)
    for k, v in t2_cases.CASE_B["expected"].items():
        assert abs(m[k] - v) <= TOL, (k, m[k], v)


def test_case_c_gap_closed():
    la, ts = t2_cases.build_case_a()
    acc_arm = stream_metrics(accuracy_matrix(la, ts, 1, run="T2/A"), 3)["ACC"]
    l2, ts2 = t2_cases.build_quarter_log("B2", t2_cases.CASE_C["B2"])
    l6, _ = t2_cases.build_quarter_log("B6", t2_cases.CASE_C["B6"])
    acc2 = stream_metrics(accuracy_matrix(l2, ts2, 1), 3)["ACC"]
    acc6 = stream_metrics(accuracy_matrix(l6, ts2, 1), 3)["ACC"]
    ex = t2_cases.CASE_C["expected"]
    assert abs(acc2 - ex["ACC_B2"]) <= TOL and abs(acc6 - ex["ACC_B6"]) <= TOL
    assert abs(gap_closed(acc_arm, acc2, acc6) - ex["GC"]) <= TOL


def test_through_log_files(tmp_path):
    """Same numbers when the logs go through disk, as the report does."""
    run_t2(str(tmp_path))
    lines = read_log(str(tmp_path / "t2_synthetic.jsonl"))
    assert len(lines) > 0
    for case, metric, expected, got in t2_results(run_t2(str(tmp_path))):
        assert abs(expected - got) <= TOL, (case, metric)


def test_pass_at_k_against_brute_force():
    """Independent check of the estimator: average over all k-subsets of n samples."""
    for n in range(1, 7):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                samples = [1] * c + [0] * (n - c)
                subs = list(itertools.combinations(range(n), k))
                brute = sum(any(samples[i] for i in s) for s in subs) / len(subs)
                assert abs(pass_at_k(n, c, k) - brute) <= TOL


def test_malformed_logs_are_refused():
    lines, ts = t2_cases.build_case_b()
    with pytest.raises(MetricsError):
        accuracy_matrix(lines + [dict(lines[0])], ts, 2)          # duplicate sample
    with pytest.raises(MetricsError):
        accuracy_matrix([l for l in lines if l["target_id"] != "b1_0"], ts, 2)   # missing obligation
    with pytest.raises(MetricsError):
        accuracy_matrix(lines, ts, 5)                              # fewer samples than k
    A = accuracy_matrix(lines, ts, 2)
    del A[(1, 1)]
    with pytest.raises(MetricsError):
        stream_metrics(A, 2)
    with pytest.raises(MetricsError):
        gap_closed(0.5, 0.3, 0.3)
