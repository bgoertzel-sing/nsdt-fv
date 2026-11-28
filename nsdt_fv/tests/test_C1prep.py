"""C1-prep (order t-2026-11-28-nsdt-fv-c1-prep): the Lean-side stream builder and hold-out guard, run on the
committed source-level history metadata (runs/C1prep/history), without Lean.

- the history files are complete and traceable to commits (23 repositories, pinned commit last);
- window, wake/test and forward definitions of 5.2 on a hand-made history with known answers;
- the stream builder on the metadata of real repositories: schema, benchmark hold-out, field names;
- T3 adapted to Lean obligations: the five injected leaks raise for every held-out obligation (test, benchmark,
  forward), with and without the identifier; legitimate examples do not; L2 removes exactly the held-out edges;
- the Lean backends are typed skeletons that raise NotImplementedError;
- the committed tables and report regenerate byte-identically.
"""
import json
import os

import pytest

from nsdt_fv.backends.base import CheckerBackend
from nsdt_fv.backends.lean_repl import LeanReplBackend, agreement
from nsdt_fv.backends.lean_vsb import LeanVSBBackend
from nsdt_fv.data import stream as S
from nsdt_fv.data.holdout import LEAN_HELD_OUT, LeakError, LeanHoldoutGuard, lean_statement_key
from nsdt_fv.data.schemas import to_benchmark_names, to_internal_names, validate_lean_obligation
from nsdt_fv.eval import c1prep as C

REPOS_JSON = os.path.join(C.VSB, "repos.json")
GUARD_REPOS = ["capless-lean", "lean-formal-reasoning-program", "clean"]


# ----------------------------------------------------------------------------- history metadata
def test_history_files_complete_and_traceable():
    repos = json.load(open(REPOS_JSON))["repos"]
    hists = C.load_histories()
    assert sorted(h["repo"] for h in hists) == sorted(repos)
    for h in hists:
        nh = h["n_history_commits"]
        assert h["commits"][nh - 1][0] == repos[h["repo"]]["commit"]          # c* closes the history
        assert len(h["commits"]) == nh + h["n_forward_commits"]
        assert "estimate" in h and "NOT the Lean extractor" in h["estimate"]
        n_intervals = sum(len(d[4]) for d in h["decls"])
        assert sum(c[3] for c in h["commits"]) == n_intervals                # per-commit 'new' = interval starts
        live = S._counts_live(h)
        assert [c[5] for c in h["commits"]] == live                          # per-commit 'live' consistent
        assert h["totals"]["live_at_pinned"] == live[nh - 1]


# ----------------------------------------------------------------------------- 5.2 definitions, known answers
def _toy_history():
    # commits 0..5 to c* (index 5), forward 6..7. keys a..h with presence intervals.
    commits = [[f"{i:040x}", f"2025-01-{i + 1:02d}T00:00:00Z", 1, 0, 0, 0, 0] for i in range(8)]
    decls = [
        ["a" * 16, "1" * 16, "p" * 16, 0, [[0, None]]],        # from the start
        ["b" * 16, "1" * 16, "p" * 16, 0, [[1, None]]],
        ["c" * 16, "1" * 16, "p" * 16, 0, [[1, 4]]],           # deleted in window 2: new in w1 only
        ["d" * 16, "1" * 16, "p" * 16, 0, [[2, None]]],
        ["e" * 16, "1" * 16, "p" * 16, 0, [[3, None]]],
        ["f" * 16, "1" * 16, "p" * 16, 0, [[4, None]]],
        ["0" * 16, "1" * 16, "p" * 16, 0, [[5, None]]],
        ["9" * 16, "1" * 16, "p" * 16, 0, [[0, 1], [4, None]]],   # removed and re-added: new in w2
        ["8" * 16, "1" * 16, "p" * 16, 0, [[6, None]]],         # forward
        ["7" * 16, "1" * 16, "p" * 16, 0, [[5, 7]]],            # gone again before HEAD: not forward
    ]
    for d in decls:
        for a, b in d[4]:
            commits[a][3] += 1
            if b is not None:
                commits[b][4] += 1
    h = {"repo": "toyrepo", "n_history_commits": 6, "n_forward_commits": 2, "commits": commits, "decls": decls,
         "pinned_commit": commits[5][0]}
    live = S._counts_live(h)
    for i, c in enumerate(commits):
        c[5] = live[i]
    return h


def test_windows_known_answer():
    h = _toy_history()
    cuts = S.window_cuts(h, 2)
    # S_0 = {a,9} (2); rest = {b,d,e,f,0,9,7} (7): 5
    # S_1 = {a,b,c} (3 new); rest = live(c*) \ S_1 = {d,e,f,0,9,7} (6): |3-6| = 3
    # S_2 = {a,b,c,d} (4);   rest = {e,f,0,9,7} (5): 1   <- best
    # S_3 = {a,b,c,d,e} (5); rest = {f,0,9,7} (4): 1 (tie, earliest wins)
    assert cuts == [2, 5]
    mem = S.window_members(h, cuts)
    names = [sorted(h["decls"][i][0][0] for i, _ in m) for m in mem]
    assert names == [["a", "b", "c", "d"], ["0", "7", "9", "e", "f"]]
    entry9 = [e for i, e in mem[1] if h["decls"][i][0][0] == "9"]
    assert entry9 == [4]                                    # entry = start of the interval containing c*
    fw = S.forward_members(h)
    assert [h["decls"][i][0][0] for i, _ in fw] == ["8"]
    wake, test = S.split_window(h, mem[1])
    assert len(test) == 1 and h["decls"][test[0][0]][0][0] == "7"   # last by entry date (index 5), then key
    summ = S.repo_windows(h, 2)
    assert [w["n_new"] for w in summ["windows"]] == [4, 5]
    assert summ["windows"][1]["end_commit"] == h["pinned_commit"]
    assert summ["forward"]["n_new"] == 1


def test_merge_order_ties_by_repo_name():
    a = {"repo": "zeta", "windows": [{"window": 1, "end_date": "2025-01-01T00:00:00Z"},
                                     {"window": 2, "end_date": "2025-03-01T00:00:00Z"}]}
    b = {"repo": "alpha", "windows": [{"window": 1, "end_date": "2025-01-01T00:00:00Z"},
                                      {"window": 2, "end_date": "2025-02-01T00:00:00Z"}]}
    assert [s for s, _ in S.merge_stages([a, b])] == ["alpha@w1", "zeta@w1", "alpha@w2", "zeta@w2"]


# ----------------------------------------------------------------------------- stream builder on real metadata
@pytest.fixture(scope="module")
def lstream():
    return C.stream_for(GUARD_REPOS)


def test_stream_on_metadata(lstream):
    s = lstream
    tasks = [t for t in C.load_tasks() if t["lean_root"] in GUARD_REPOS]
    assert len(s.stages) == 2 * len(GUARD_REPOS)
    assert [x for x, _ in s.stages] == [x for x, _ in sorted(s.stages, key=lambda x: (x[1], x[0].split("@")[0]))]
    ids = [o["id"] for o in s.obligations]
    assert len(ids) == len(set(ids))
    bench = [o for o in s.obligations if o["split"] == "benchmark"]
    assert len(bench) == len(tasks)
    bench_names = {S.name_hash(t["thm_name"]) for t in tasks}
    for o in s.obligations:
        if o["materializer"] == "placeholder" and o["split"] in ("wake", "test"):
            assert o["decl"].split(".d_")[-1][:16] not in bench_names    # benchmark-named keys never in wake/test
    for j, ids_j in s.test_sets().items():
        assert ids_j, f"stage {j} has no test obligations"
    for o in s.obligations:
        stored = to_benchmark_names(o)
        validate_lean_obligation(stored)
        assert "thm_stmt" in stored and "ground_truth_proof" in stored and "statement" not in stored
        assert to_internal_names(stored) == o
    # counts agree with the window summaries, up to the statement-collision resolution
    c = s.config["collisions"]
    n_ww = sum(1 for o in s.obligations if o["split"] in ("wake", "test") and o["window"] > 0)
    assert n_ww == sum(x["n_wake"] + x["n_test"] for w in s.windows for x in w["windows"]) - c["wake_dropped"]
    # no statement key is both trained and held out
    from nsdt_fv.data.holdout import lean_statement_key
    wake_keys = {lean_statement_key(o["statement"]) for o in s.obligations if o["split"] == "wake"}
    assert not any(lean_statement_key(o["statement"]) in wake_keys for o in s.obligations if o["split"] in LEAN_HELD_OUT)


def test_backends_are_typed_skeletons(lstream):
    for b in (lstream.backend("vsb"), lstream.backend("repl")):
        assert isinstance(b, (LeanVSBBackend, LeanReplBackend))
        assert isinstance(b, CheckerBackend)
        assert b.snapshots() == sorted(lstream.snapshots)
        o = lstream.obligations[0]
        with pytest.raises(NotImplementedError):
            b.check(o["snapshot_id"], o["id"], o["proof"], 60)
        with pytest.raises(NotImplementedError):
            b.check_statement(o["snapshot_id"], "", o["statement"], ":= by simp", 60)
        with pytest.raises(NotImplementedError):
            b.evaluate_term(o["snapshot_id"], "", "1 + 1", 60)
    assert "theorem" in lstream.backend("vsb").check_input(lstream.obligations[0]["id"], "by simp")
    a = agreement(["ok", "error", "timeout"], ["ok", "error", "error"])
    assert a["agree"] == 2 and not a["all_agree"] and a["confusion"] == {"timeout->error": 1}


# ----------------------------------------------------------------------------- T3 on Lean obligations
def test_statement_key_normalisation():
    s = "theorem foo (n m : Nat) (h : n ≤ m) : f (n + 1) = g m"
    assert lean_statement_key(s) == lean_statement_key("lemma bar (a b : Nat) (k : a ≤ b) : g b = f (a + 1)")
    assert lean_statement_key(s) == lean_statement_key(C.alpha_variant(s))
    assert lean_statement_key(s) == lean_statement_key("@[simp] private theorem foo\n  (n m : Nat)\n (h : n ≤ m) :\n f (n + 1) = g m :=")
    assert lean_statement_key(s) != lean_statement_key("theorem foo (n m : Nat) (h : n ≤ m) : f (n + 1) = g n")
    assert lean_statement_key(s) != lean_statement_key("theorem foo (n m : Nat) (h : n < m) : f (n + 1) = g m")


@pytest.fixture(scope="module")
def t3(lstream):
    return C.run_t3(lstream, LeanHoldoutGuard(lstream.obligations))


def test_t3_lean_each_injected_leak_raises(t3, lstream):
    lines, _ = t3
    held = [o for o in lstream.obligations if o["split"] in LEAN_HELD_OUT]
    assert {o["split"] for o in held} == set(LEAN_HELD_OUT)           # test, benchmark and forward all present
    leaks = [l for l in lines if l["part"] == "leak"]
    assert {l["leak"] for l in leaks} == set(C.LEAK_TYPES)
    for lt in C.LEAK_TYPES:
        for variant in ("stealth", "with_id"):
            sub = [l for l in leaks if l["leak"] == lt and l["variant"] == variant]
            assert len(sub) == len(held)
            assert all(l["raised"] for l in sub), (lt, variant, [l for l in sub if not l["raised"]][:3])
    decoys = [l for l in lines if l["part"] == "decoy"]
    assert decoys and all(l["raised"] for l in decoys)


def test_t3_lean_no_false_positives(t3):
    lines, _ = t3
    controls = [l for l in lines if l["part"] == "control"]
    assert len(controls) > 1000
    assert [l for l in controls if l["raised"]] == []


def test_t3_lean_l2_l3_l4_l6_l7(t3, lstream):
    lines, g = t3
    l2 = [l for l in lines if l["part"] == "L2"][0]
    assert l2["edges_removed"] > 0 and l2["removed_exactly_held_out"]
    o = next(x for x in lstream.obligations if x["split"] == "test")
    assert g.filter_dreams([{"statement": C.alpha_variant(o["statement"])}]) == [] and g.dropped_dreams == 1
    with pytest.raises(LeakError) as e:
        g.check({"kind": "wake", "statement": "theorem t : True", "source": "frontier"})
    assert e.value.rule == "L7"
    with pytest.raises(LeakError):
        LeanHoldoutGuard.check_eval_order([{"type": "train", "stage": 1}, {"type": "train", "stage": 2},
                                           {"type": "eval", "model_stage": 1}])
    wake = next(x for x in lstream.obligations if x["split"] == "wake" and x["repo"] != "clean")
    g6 = LeanHoldoutGuard(lstream.obligations, dev_projects={"clean"})
    with pytest.raises(LeakError) as e:
        g6.check({"kind": "wake", "statement": wake["statement"], "source_ids": [wake["id"]]})
    assert e.value.rule == "L6"
    with pytest.raises(LeakError):
        g.check({"kind": "wake", "input": "x"})        # proof-derived data must carry its statement
    # the id of a held-out obligation in free text
    with pytest.raises(LeakError):
        g.check({"kind": "aux", "statement": wake["statement"], "input": f"see {o['id']}."})


def test_t3_committed_summaries():
    # placeholder texts on the seven shortlisted repositories: every injection raises, no control does
    t = json.load(open(os.path.join(C.RUN, "t3_placeholder_summary.json")))
    assert set(t["summary"]["leaks"]) == set(C.LEAK_TYPES)
    for lt, v in t["summary"]["leaks"].items():
        assert v["raised"] == v["n"] > 0, lt
    assert t["summary"]["controls"]["raised"] == 0
    assert t["summary"]["decoys"]["raised"] == t["summary"]["decoys"]["n"] > 0
    assert t["summary"]["L2"]["removed_exactly_held_out"]
    # real source texts (regex-cut, ISSUES 66): residuals stay below 0.1% and are reported
    r = json.load(open(os.path.join(C.RUN, "t3_real_summary.json")))
    assert set(r["summary"]["leaks"]) == set(C.LEAK_TYPES)
    for lt, v in r["summary"]["leaks"].items():
        assert v["n"] > 0 and v["n"] - v["raised"] <= 0.001 * v["n"], lt
    assert r["summary"]["controls"]["raised"] <= 0.001 * r["summary"]["controls"]["n"]
    assert r["summary"]["decoys"]["raised"] == r["summary"]["decoys"]["n"] > 0


# ----------------------------------------------------------------------------- tables regenerate
def test_tables_regenerate_byte_identically():
    files, _, _ = C.compute_tables()
    for rel, text in files.items():
        with open(os.path.join(C.PKG, rel)) as f:
            assert f.read() == text, rel
