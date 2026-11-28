"""S3 unit tests: the A.3 Channel interface, the hold-out guard on every emitted channel item, the per-stage
cap, the channel canaries, and T10 (channel B: every emitted statement has a checker-accepted machine
proof at emission; channel C: no emitted seed has a recorded counterexample)."""
import pytest

from nsdt_fv.backends.toyeq import statement_key
from nsdt_fv.backends.toyeq_oracle import counterexample
from nsdt_fv.data.holdout import HoldoutGuard
from nsdt_fv.data.stream import build_stream
from nsdt_fv.store.channels import make_channel
from nsdt_fv.store.graph import ToyStore

CFG = {"A": {"cap_nextstate": 30, "cap_premise": 30, "cap_unfold": 20, "cap_statements": 20},
       "B": {"max_candidates": 80, "cap_statements": 25},
       "C": {"inputs": 200, "max_size": 5, "max_terms": 600, "cap_statements": 20},
       "D": {"per_proof": 1, "cap_mutants": 20, "cap_repairs": 20, "cap_priorities": 10},
       "E": {"min_count": 3, "cap_plans": 20, "max_targets": 10, "tries": 4, "cap_priorities": 10}}


@pytest.fixture(scope="module")
def world():
    s = build_stream(seed=0, n_projects=2)
    ws = dict(s.worlds)
    ws["toy-prelude"] = s.prelude.world
    g = HoldoutGuard(s.obligations, ws)
    return s, g


def _run(world, cid, stage=1):
    s, g = world
    store = ToyStore(s, g)
    sid = s.stages[stage - 1][0]
    store.add_snapshot(sid, stage)
    keys = {statement_key(s.worlds[e["snapshot_id"]].statement(e["statement"])) for e in store.entries.values()}
    calls = []
    real = g.check

    def counting(ex):
        calls.append(ex)
        return real(ex)
    g.check = counting
    try:
        ch = make_channel(cid, CFG[cid], s, g, s.backend())
        recs = ch.run_stage(stage, sid, keys, store=store)
    finally:
        g.check = real
    return s, sid, ch, recs, calls


@pytest.mark.parametrize("cid", list("ABCDE"))
def test_interface_guard_and_cap(world, cid):
    s, sid, ch, recs, calls = _run(world, cid)
    assert ch.id == cid
    for m in ("wake_examples", "nrem_targets", "rem_seeds", "annotate"):
        assert callable(getattr(ch, m))
    offered = [r for r in recs if r.get("_ex")]
    assert offered, "channel produced nothing"
    assert len(calls) >= len(offered)                  # every offered item went through HoldoutGuard.check
    for r in recs:
        if r["emitted"]:
            assert r["guard"] == "pass"
    for kind, key in (("statement", "cap_statements"), ("repair", "cap_repairs"), ("plan", "cap_plans"),
                      ("priority", "cap_priorities"), ("nextstate", "cap_nextstate")):
        if key in CFG[cid]:
            assert sum(1 for r in recs if r["what"] == kind and r["emitted"]) <= CFG[cid][key]
    emitted_ex = ch.wake_examples(1)
    for ex in emitted_ex:                              # wake examples are channel-sourced and from this window
        assert ex["source"] == "channel"
        assert all(s.by_id()[i]["snapshot_id"] == sid for i in ex["source_ids"])


def test_T10_channel_B(world):
    s, sid, ch, recs, _ = _run(world, "B")
    em = [r for r in recs if r["what"] == "statement" and r["emitted"]]
    assert em
    be = s.backend()
    for r in em:
        assert r["proof"] and be.check_statement(sid, "", r["statement"], r["proof"], 120).accepted
    assert all(r["emitted"] is False for r in recs if r["what"] == "unproved")


def test_T10_channel_C(world):
    s, sid, ch, recs, _ = _run(world, "C")
    em = [r for r in recs if r["what"] == "statement" and r["emitted"]]
    assert em
    refuted = {r["statement"] for r in recs if r["what"] == "refuted"}
    for r in em:
        assert r.get("counterexample") is None and r["statement"] not in refuted
    w = s.worlds[sid]
    # survivors were tested on 400 inputs in all; a further independent test should rarely refute them
    bad = sum(1 for r in em if counterexample(w, w.statement(r["statement"]), n=200, seed=12345) not in (None,))
    assert bad <= len(em) // 10


def test_channel_canaries_raise(world):
    from nsdt_fv.eval.s3 import channel_canaries
    s, g = world
    out = channel_canaries(s, g, s.backend(), {"channels": CFG}, None, 192, 0)
    assert len(out) == 5
    for c in out:
        assert c["items"] > 0 and c["refused"] == c["items"], c

