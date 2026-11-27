"""Acceptance test T4 (S1): evaluation sets are identical across arms, by hash.

Two parts: (1) the T4 machinery on synthetic logs (equal sets pass, a changed set fails);
(2) the committed S1 run, if present: every arm's recorded eval-set hashes and the target-id sets read
back from its verdict log agree.
"""
import json
import os

import pytest

from nsdt_fv.eval.report_s1 import t4
from nsdt_fv.eval.s1 import eval_set_hash

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(HERE, "runs", "S1")


def _arm(sets, extra=None):
    lines = []
    for (t, j), ids in sets.items():
        for i in ids:
            lines.append({"stage": t, "eval_set": j, "target_id": i})
    if extra:
        lines.append(extra)
    hashes = {f"{t},{j}": eval_set_hash([{"id": i, "snapshot_id": f"s{j}", "statement": i} for i in ids])
              for (t, j), ids in sets.items()}
    return {"lines": lines, "evalsets": hashes}


def test_t4_synthetic():
    test_sets = {1: ["a", "b"], 2: ["c"]}
    cells = {(0, 1): ["a", "b"], (1, 1): ["a", "b"], (1, 2): ["c"]}
    good = {"B0": _arm(cells), "B2": _arm(cells)}
    assert t4(good, test_sets)["passed"]
    bad_cells = dict(cells)
    bad_cells[(1, 1)] = ["a"]
    bad = {"B0": _arm(cells), "B2": _arm(bad_cells)}
    r = t4(bad, test_sets)
    assert not r["passed"] and not r["recorded_hashes_equal"] and not r["logged_id_sets_equal"]


def test_eval_set_hash_ignores_order_and_proofs():
    a = [{"id": "x", "snapshot_id": "s", "statement": "p"}, {"id": "y", "snapshot_id": "s", "statement": "q"}]
    b = [dict(a[1], proof="zzz"), a[0]]
    assert eval_set_hash(a) == eval_set_hash(b)
    assert eval_set_hash(a) != eval_set_hash(a[:1])


@pytest.mark.skipif(not os.path.exists(os.path.join(RUN, "meta.json")), reason="no S1 run committed")
def test_t4_on_s1_run():
    from nsdt_fv.eval.report_s1 import load
    meta, stages, test_sets, obl, arms = load(RUN)
    assert set(arms) == {"B0", "B2", "B5-10", "B6"}
    r = t4(arms, test_sets)
    assert r["passed"], r
