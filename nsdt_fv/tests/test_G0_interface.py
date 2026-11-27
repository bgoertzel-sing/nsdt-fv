"""Gate G0, last condition: replacing the ToyEq backend by a Lean backend requires no change to the
trainer or the evaluator. The same runner and metrics code runs over ToyEqBackend and over
MockLeanBackend (a backend that knows nothing about ToyEq), and the evaluator imports only the
CheckerBackend protocol. Also checks verdict caching (R5) and sampling seeds (R6)."""
import ast
import os

import pytest

from nsdt_fv.backends.base import CheckerBackend, Verdict
from nsdt_fv.backends.mock import MockLeanBackend
from nsdt_fv.eval import runner
from nsdt_fv.eval.metrics import accuracy_matrix, stream_metrics
from nsdt_fv.data.stream import test_view as to_test_view
from nsdt_fv.data.schemas import validate_verdict_line

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _imports(path):
    tree = ast.parse(open(path).read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            out.add(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def test_evaluator_imports_only_the_protocol():
    for rel in ("eval/runner.py", "eval/metrics.py"):
        imps = _imports(os.path.join(PKG, rel))
        assert not any("toyeq" in i or "mock" in i for i in imps), (rel, imps)


def _sampler_from(proofs):
    def sampler(o, seeds):
        return [proofs[o["id"]] if (s % 3) else "simp" for s in seeds]
    return sampler


def _run(backend, obligations, proofs, stages, k=2):
    cache = runner.VerdictCache()
    ch = runner.CachedChecker(backend, cache)
    lines = []
    for t in range(stages + 1):
        lines += runner.evaluate(ch, "G0/demo", t, obligations, _sampler_from(proofs), k)
    return lines, cache


def test_same_harness_over_toyeq_and_mock(stream):
    tests = [to_test_view(o) for o in stream.obligations if o["split"] == "test"][:12]
    proofs = {o["id"]: o["proof"] for o in stream.obligations}
    ts = {1: [o["id"] for o in tests[:6]], 2: [o["id"] for o in tests[6:]]}
    toy = stream.backend()
    lean_like = MockLeanBackend({(o["snapshot_id"], o["id"]): {proofs[o["id"]]} for o in tests})
    results = {}
    for name, b in (("toyeq", toy), ("mock", lean_like)):
        assert isinstance(b, CheckerBackend)
        lines, cache = _run(b, tests, proofs, 2)
        for ln in lines:
            validate_verdict_line(ln)
        assert cache.hits > 0          # stages repeat the same candidates
        results[name] = stream_metrics(accuracy_matrix(lines, ts, 2), 2)
    # both backends accept exactly the ground-truth proofs here, so the metrics agree
    assert results["toyeq"] == results["mock"]


def test_verdict_cache_r5():
    b = MockLeanBackend({("s@1", "o1"): {"by simp"}})
    cache = runner.VerdictCache()
    ch = runner.CachedChecker(b, cache)
    v1, hit1 = ch.check("s@1", "o1", "by simp", 120)
    v2, hit2 = ch.check("s@1", "o1", "by simp", 300)      # other timeout: cached verdict, not recomputed
    assert v1.accepted and v2.accepted and (hit1, hit2) == (False, True)
    assert cache.timeout_mismatches == 1 and b.calls == 1
    with pytest.raises(RuntimeError):
        cache.put(runner.VerdictCache.key("s@1", "o1", "by simp"), v1, 120)


def test_cache_persists(tmp_path):
    p = str(tmp_path / "cache.jsonl")
    b = MockLeanBackend({("s@1", "o1"): {"by simp"}})
    runner.CachedChecker(b, runner.VerdictCache(p)).check("s@1", "o1", "by omega", 120)
    c2 = runner.VerdictCache(p)
    v, hit = runner.CachedChecker(b, c2).check("s@1", "o1", "by omega", 120)
    assert hit and v.category == "error" and b.calls == 1


def test_sampling_seeds_r6():
    a = runner.sample_seed("S1/B2/toy/seed0", 3, "x@y:z", 2)
    assert a == runner.sample_seed("S1/B2/toy/seed0", 3, "x@y:z", 2)
    assert a != runner.sample_seed("S1/B2/toy/seed0", 3, "x@y:z", 3)


def test_mock_rules():
    b = MockLeanBackend({("s", "o"): {"by simp"}})
    assert b.check("s", "o", "by sorry", 10).category == "sorry"
    assert b.check("s", "o", "axiom foo : False", 10).category == "axiom"
    with pytest.raises(ValueError):
        Verdict(True, "error")
