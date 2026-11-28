"""G0 test (S5): replacing the ToyEq backend by a Lean backend requires no change to the trainer or
the evaluator (plan 8.1, gate G0, last condition).

A stub Lean backend (backends/lean_stub.py, canned verdicts, no Lean) runs through the unchanged trainer
(train/lora.Sequential, train/joint.Joint, train/model) and evaluator (eval/runner, eval/metrics) in a
fresh interpreter (nsdt_fv/gate/g0_stub_run.py), so that pytest's conftest (which builds the ToyEq stream)
cannot hide a ToyEq import: the run records sys.modules and must load no ToyEq module. This file imports
no ToyEq code either (checked below). That train/ and eval/ are byte-identical to S4's head is checked by
nsdt_fv/gate/g0_check.sh (git diff), not here, because a test cannot know its own base commit.
"""
import ast
import json
import os
import subprocess
import sys

import pytest

from nsdt_fv.backends.base import CheckerBackend
from nsdt_fv.backends.lean_stub import StubLeanBackend

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PKG)
FORBIDDEN = ("toyeq", "data.stream", "data.holdout", "backends.mock", "conftest")


def _imports(path):
    tree = ast.parse(open(path).read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            out.add((node.module or "") + ":" + ",".join(a.name for a in node.names))
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


@pytest.mark.parametrize("rel", ["tests/test_G0_stub_lean.py", "gate/g0_stub_run.py", "backends/lean_stub.py"])
def test_no_toyeq_imports(rel):
    imps = _imports(os.path.join(PKG, rel))
    bad = [i for i in imps if any(f in i for f in FORBIDDEN)]
    assert not bad, (rel, bad)


OBL = {"id": "repo-x@abc:Foo.bar", "snapshot_id": "repo-x@w1", "decl": "Foo.bar",
       "statement": "theorem bar (n : Nat) : n + 0 = n", "proof": ":= by simp"}


def test_stub_backend_verdicts():
    b = StubLeanBackend([OBL], canned={("repo-x@w1", OBL["id"], "by slow_tactic"): "timeout"})
    assert isinstance(b, CheckerBackend)
    assert b.snapshots() == ["repo-x@w1"]
    s, i = "repo-x@w1", OBL["id"]
    assert b.check(s, i, "by simp", 60).category == "ok"                  # leading ':=' ignored
    assert b.check(s, i, "by  simp", 60).accepted                          # whitespace-insensitive
    assert b.check(s, i, "by omega", 60).category == "error"
    assert b.check(s, i, "by sorry", 60).category == "sorry"
    assert b.check(s, i, "axiom cheat : False\nby simp", 60).category == "axiom"
    assert b.check(s, i, "theorem bar (n : Nat) : n = n := by simp", 60).category == "changed_statement"
    assert b.check(s, i, "theorem bar (n : Nat) : n + 0 = n := by simp", 60).category == "ok"
    assert b.check(s, i, "by slow_tactic", 60).category == "timeout"       # canned verdict wins
    with pytest.raises(KeyError):
        b.check(s, "nope", "by simp", 60)
    assert not b.evaluate_term(s, "", "f 1", 10).ok


@pytest.fixture(scope="module")
def stub_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("g0")
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, "-m", "nsdt_fv.gate.g0_stub_run", "--out", str(out)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.load(open(out / "result.json")), out


def test_stub_run_loads_no_toyeq(stub_run):
    res, _ = stub_run
    assert res["toy_modules_before"] == [] and res["toy_modules_after"] == []
    loaded = set(res["nsdt_modules_loaded"])
    # the unchanged trainer and evaluator modules were the ones exercised
    for m in ("nsdt_fv.train.lora", "nsdt_fv.train.joint", "nsdt_fv.train.common", "nsdt_fv.train.model",
              "nsdt_fv.eval.runner", "nsdt_fv.eval.metrics", "nsdt_fv.backends.lean_stub"):
        assert m in loaded, m


def test_stub_run_through_trainer_and_evaluator(stub_run):
    res, out = stub_run
    T, k = res["T"], res["k"]
    assert T == 4 and res["test"] > 0
    for arm in res["arms"]:
        n_test = res["test"]
        assert arm["verdict_lines"] == (T + 1) * n_test * k            # every stage scored every test set
        assert sum(arm["backend_calls"].values()) == arm["cache_misses"] > 0
        assert arm["guard_checked"] > 0                                  # training data passed the guard callable
        assert [s["stage"] for s in arm["train"]] == list(range(1, T + 1))
        assert all(s["steps"] > 0 for s in arm["train"])
        assert set(arm["metrics"]) == {"LA", "FWD", "ACC", "F", "BWT"}
        assert all(0.0 <= v <= 1.0 for kk, v in arm["metrics"].items() if kk != "BWT")
        assert set(arm["categories"]) <= {"ok", "error", "timeout", "sorry", "axiom", "changed_statement"}
        lines = [json.loads(l) for l in open(out / f"{arm['arm']}.verdicts.jsonl")]
        assert len(lines) == arm["verdict_lines"] and sum(l["accepted"] for l in lines) == arm["accepted_lines"]
    assert any(arm["accepted_lines"] > 0 for arm in res["arms"])       # the trained model gets some proofs past the stub


def test_stub_run_canaries(stub_run):
    res, _ = stub_run
    c = res["canary"]
    assert c["truth"]["metrics"]["ACC"] == 1.0 and c["truth"]["metrics"]["F"] == 0.0 and c["truth"]["categories"] == ["ok"]
    assert c["sorry"]["metrics"]["ACC"] == 0.0 and c["sorry"]["categories"] == ["sorry"]
    assert res["guard_canary_raised"] is True
