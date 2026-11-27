"""T1 (A.5, S0): ToyEq checker. All ground-truth proofs are accepted. For 1,000 randomly mutated
proofs, any accepted proof whose statement fails on one of 200 random ground instances is a
soundness bug; there must be none. Plus unit tests of the checker rules of C.2."""
import pytest

from nsdt_fv.backends.toyeq import (World, ToyError, check_proof, parse_statement, make_fundef,
                                    parse_term, ToyEqBackend)
from nsdt_fv.eval import s0


@pytest.fixture(scope="module")
def t1(stream, tmp_path_factory):
    return s0.run_t1(stream, str(tmp_path_factory.mktemp("t1")), n_mutants=1000, n_stmt_only=300,
                     n_instances=200, seed=0)


def test_ground_truth_all_accepted(t1, stream):
    gt = [l for l in t1 if l["part"] == "gt"]
    assert len(gt) == len(stream.obligations) + len(stream.prelude.lemmas)
    assert all(l["accepted"] for l in gt), [l["target_id"] for l in gt if not l["accepted"]][:5]
    # known-answer canary for the independent evaluator: every library statement is true
    assert all(l["statement_true"] for l in gt)
    assert {l["instances"] for l in gt} == {200}


def test_no_soundness_bug_in_1000_mutants(t1):
    mut = [l for l in t1 if l["part"] in ("proof_only", "proof_and_statement")]
    assert len(mut) == 1000
    assert {l["instances"] for l in mut} == {200}
    bugs = [l for l in t1 if l.get("soundness_bug")]
    assert bugs == [], bugs[:3]


def test_mutants_are_not_vacuous(t1):
    """Useless-if clause of the order: the mutants must sometimes be accepted, including with a
    changed statement, and many must have false statements that the checker rejects."""
    mut = [l for l in t1 if l["part"] in ("proof_only", "proof_and_statement")]
    assert sum(l["accepted"] for l in mut) >= 20
    assert sum(l["accepted"] and l["statement_mutated"] for l in mut) >= 5
    assert sum((not l["statement_true"]) and (not l["accepted"]) for l in mut) >= 100


def test_canary_planted_bug_is_detected(t1):
    """The soundness procedure must catch a checker that accepts too much."""
    allm = [l for l in t1 if l["part"] != "gt"]
    assert sum(l["canary_flag"] for l in allm) >= 1


# ------------------------------------------------------------------ checker rules (C.2)
def _w():
    w = World("unit")
    w.define("add", ["Nat", "Nat"], "Nat", ["add(Z, m) = m", "add(S(n), m) = S(add(n, m))"])
    w.define("app", ["List", "List"], "List", ["app(Nil, ys) = ys", "app(Cons(h, t), ys) = Cons(h, app(t, ys))"])
    w.define("neg", ["Bool"], "Bool", ["neg(T) = F", "neg(F) = T"])
    return w


def test_plan_example():
    w = _w()
    st = w.statement("forall xs ys zs. app(app(xs,ys),zs) = app(xs,app(ys,zs))")
    assert check_proof(w, st, "induct xs { simp refl } { simp rw IH refl }", 0).accepted


def test_rules():
    w = _w()
    z = w.statement("forall n. add(n, Z) = n")
    w.add_lemma("add_z_r", z, "induct n { simp refl } { simp rw IH refl }")
    st = w.statement("forall n. add(add(n, Z), Z) = n")
    assert check_proof(w, st, "rw add_z_r rw add_z_r refl", 1).accepted
    v = check_proof(w, st, "rw add_z_r rw add_z_r refl", 0)          # lemma does not precede
    assert not v.accepted and "not available" in v.messages[0]["text"]
    v = check_proof(w, st, "rw add_z_r rw add_z_r refl simp", 1)     # step after closed goal
    assert not v.accepted and "after the goal was closed" in v.messages[0]["text"]
    assert not check_proof(w, st, "rw add_z_r", 1).accepted           # unsolved
    assert check_proof(w, st, "sorry", 1).category == "sorry"
    assert check_proof(w, st, "axiom", 1).category == "axiom"
    assert check_proof(w, st, " ".join(["simp"] * 41) + " refl", 1).category == "error"   # > 40 steps
    assert check_proof(w, z, "induct n { simp refl } { simp rw IH refl }", 0, timeout_s=0).category == "timeout"
    b = w.statement("forall b. neg(neg(b)) = b")
    assert check_proof(w, b, "cases b { simp refl } { simp refl }", 1).accepted
    assert not check_proof(w, b, "cases b { simp refl }", 1).accepted  # one block per constructor
    false = w.statement("forall n m. add(n, m) = add(n, n)")
    assert not check_proof(w, false, "induct n { simp refl } { simp rw IH refl }", 1).accepted


def test_rw_respects_types():
    w = _w()
    w.add_lemma("app_nil_r", w.statement("forall xs. app(xs, Nil) = xs"))
    w.add_lemma("add_z_r", w.statement("forall n. add(n, Z) = n"))
    st = w.statement("forall xs. app(xs, Nil) = app(xs, Nil)")
    # rw<- add_z_r would turn the List term app(xs, Nil) into add(app(xs, Nil), Z) if pattern
    # variables ignored types; there is no Nat subterm, so nothing may match
    v = check_proof(w, st, "rw<- add_z_r refl", 2)
    assert not v.accepted and "no subterm matches" in v.messages[0]["text"]


def test_induct_refused_on_hypothesis_variable():
    w = _w()
    st = w.statement("forall n m. add(n, m) = add(n, m)")
    # in the S case the IH mentions the recursive child n free, so a second induction on n is refused
    v = check_proof(w, st, "induct n { refl } { induct n { refl } { refl } }", 0)
    assert not v.accepted and "occurs in hypothesis" in v.messages[0]["text"]
    # a case split on n is still allowed, and induction on a variable the IH quantifies over is too
    assert check_proof(w, st, "induct n { refl } { cases n { refl } { refl } }", 0).accepted
    assert check_proof(w, st, "induct n { refl } { induct m { refl } { refl } }", 0).accepted


def test_definition_validation():
    funcs = {}
    eq = lambda a, b: (parse_term(a), parse_term(b))
    with pytest.raises(ToyError):      # not exhaustive
        make_fundef("f", ["Nat"], "Nat", [eq("f(Z)", "Z")], funcs)
    with pytest.raises(ToyError):      # not structural
        make_fundef("g", ["Nat"], "Nat", [eq("g(Z)", "Z"), eq("g(S(n))", "g(S(n))")], funcs)
    with pytest.raises(ToyError):      # calls an unknown (later) function
        make_fundef("h", ["Nat"], "Nat", [eq("h(Z)", "Z"), eq("h(S(n))", "k(n)")], funcs)
    with pytest.raises(ToyError):      # matches on two arguments
        make_fundef("e", ["Nat", "Nat"], "Nat", [eq("e(Z, Z)", "Z"), eq("e(S(n), S(m))", "Z")], funcs)


def test_backend_categories(stream):
    b = stream.backend()
    o = next(x for x in stream.obligations if x["split"] == "wake")
    assert b.check(o["snapshot_id"], o["id"], o["proof"], 120).accepted
    assert b.check(o["snapshot_id"], o["id"], f"theorem {o['statement']} := {o['proof']}", 120).accepted
    other = next(x for x in stream.obligations if x["repo"] == o["repo"] and x["id"] != o["id"])
    v = b.check(o["snapshot_id"], o["id"], f"theorem {other['statement']} := {o['proof']}", 120)
    assert v.category == "changed_statement"
    # determinism (R5/T6 analogue): the same candidate gives the same verdict
    for x in stream.obligations[:50]:
        v1 = b.check(x["snapshot_id"], x["id"], "simp refl", 120)
        v2 = b.check(x["snapshot_id"], x["id"], "simp refl", 120)
        assert (v1.category, v1.messages) == (v2.category, v2.messages)
