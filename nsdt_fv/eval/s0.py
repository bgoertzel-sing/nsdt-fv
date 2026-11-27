"""Experiment S0 (plan 8.1): ToyEq, checker, generators, harness, metrics, hold-out guard.

Procedures for acceptance tests T1-T3 (A.5). Each writes raw logs under the run directory; the
report (reports/S0.md) is generated from those logs only (rule R3), by report_s0.py.
"""
from __future__ import annotations

import json
import os
import platform
import random
import shutil
import subprocess
import time

from ..backends import toyeq as tq
from ..backends.toyeq import (CTORS, TYPES, Statement, ToyError, World, check_proof, parse_proof,
                              parse_statement, show, statement_key, canonical, term_vars, typecheck,
                              _positions, _replace_at)
from ..backends.toyeq_oracle import counterexample
from ..data.holdout import HoldoutGuard, LeakError
from ..data.stream import Stream, build_stream, snapshot_world, test_view, write_stream, proof_lemmas
from .metrics import accuracy_matrix, gap_closed, stream_metrics
from .runner import candidate_sha, VerdictCache, CachedChecker, verdict_line
from . import t2_cases

N_INSTANCES = 200


# ----------------------------------------------------------------------------- environment (8.1)
def env_probe() -> dict:
    mem_kb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_kb = int(line.split()[1])
    except OSError:
        pass
    gpu = shutil.which("nvidia-smi") is not None
    try:
        import yaml
        yv = yaml.__version__
    except Exception:
        yv = None
    return {"cpu_count": os.cpu_count(), "mem_gb": round(mem_kb / 1048576, 1) if mem_kb else None,
            "nvidia_smi": gpu, "python": platform.python_version(), "platform": platform.platform(),
            "pyyaml": yv}


def git_commit(path) -> str:
    try:
        rev = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--", "nsdt_fv"],
                               capture_output=True, text=True).stdout
        dirty = [l for l in dirty.splitlines() if "/runs/" not in l and "/reports/" not in l]
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------- T1 mutants
def _lists(steps):
    """All step lists (mutable): the top list and every block."""
    out = [steps]
    for s in steps:
        if s[2]:
            for b in s[2]:
                out.extend(_lists(b))
    return out


def _mutable(steps):
    return [[k, a, [_mutable(b) for b in bl] if bl else None] for k, a, bl, _ in steps]


def _render(steps):
    out = []
    for k, a, bl in steps:
        if k in ("simp", "refl"):
            out.append(k)
        elif k in ("rw", "rw<-"):
            out.append(f"{k} {a}")
        else:
            out.append(f"{k} {a} " + " ".join("{ " + _render(b) + " }" for b in bl))
    return " ".join(out)


PROOF_OPS = ["delete", "swap", "replace", "flip", "swap_lemma", "future_lemma", "self_ref", "split_var",
             "induct_to_cases", "drop_block", "dup_block", "insert_refl", "insert_simp", "transplant"]
STMT_OPS = ["swap_args", "replace_fn", "base_ctor", "var_swap", "wrap", "flip_sides"]


def mutate_proof(proof, rng, ctx):
    steps = _mutable(parse_proof(proof))
    op = rng.choice(PROOF_OPS)
    lists = [l for l in _lists(steps)]
    rw_steps = [s for l in lists for s in l if s[0] in ("rw", "rw<-")]
    split_steps = [s for l in lists for s in l if s[0] in ("cases", "induct")]
    avail = ctx["avail_names"]
    if op == "transplant":
        return rng.choice(ctx["other_proofs"]), op
    if op == "delete":
        l = rng.choice([l for l in lists if l] or [steps])
        if not l:
            return None, op
        del l[rng.randrange(len(l))]
    elif op == "swap":
        ls = [l for l in lists if len(l) >= 2]
        if not ls:
            return None, op
        l = rng.choice(ls)
        i = rng.randrange(len(l) - 1)
        l[i], l[i + 1] = l[i + 1], l[i]
    elif op == "replace":
        l = rng.choice([l for l in lists if l] or [steps])
        if not l:
            return None, op
        r = rng.random()
        names = avail + ["IH", "IH1", "IH2"]
        if r < 0.2:
            new = ["simp", None, None]
        elif r < 0.35:
            new = ["refl", None, None]
        elif r < 0.85 and names:
            new = [rng.choice(["rw", "rw<-"]), rng.choice(names), None]
        else:
            v, ty = rng.choice(ctx["vars"]) if ctx["vars"] else ("x", "Nat")
            n = len(TYPES.get(ty, ("a", "b")))
            new = [rng.choice(["cases", "induct"]), v, [[["simp", None, None], ["refl", None, None]] for _ in range(n)]]
        l[rng.randrange(len(l))] = new
    elif op == "flip":
        if not rw_steps:
            return None, op
        s = rng.choice(rw_steps)
        s[0] = "rw<-" if s[0] == "rw" else "rw"
    elif op == "swap_lemma":
        if not rw_steps or not avail:
            return None, op
        rng.choice(rw_steps)[1] = rng.choice(avail)
    elif op == "future_lemma":
        if not rw_steps or not ctx["future_names"]:
            return None, op
        rng.choice(rw_steps)[1] = rng.choice(ctx["future_names"])
    elif op == "self_ref":
        if rw_steps:
            rng.choice(rw_steps)[1] = ctx["self"]
        else:
            steps.insert(0, ["rw", ctx["self"], None])
    elif op == "split_var":
        if not split_steps or len(ctx["vars"]) < 2:
            return None, op
        s = rng.choice(split_steps)
        others = [v for v, _ in ctx["vars"] if v != s[1]]
        s[1] = rng.choice(others)
    elif op == "induct_to_cases":
        ind = [s for s in split_steps if s[0] == "induct"]
        if not ind:
            return None, op
        rng.choice(ind)[0] = "cases"
    elif op == "drop_block":
        if not split_steps:
            return None, op
        s = rng.choice(split_steps)
        del s[2][rng.randrange(len(s[2]))]
        if not s[2]:
            return None, op
    elif op == "dup_block":
        if not split_steps:
            return None, op
        s = rng.choice(split_steps)
        i = rng.randrange(len(s[2]))
        s[2][1 - i if len(s[2]) == 2 else i] = json.loads(json.dumps(s[2][i]))
    elif op == "insert_refl":
        l = rng.choice(lists)
        l.insert(rng.randrange(len(l) + 1), ["refl", None, None])
    elif op == "insert_simp":
        l = rng.choice(lists)
        l.insert(rng.randrange(len(l) + 1), ["simp", None, None])
    return _render(steps), op


def _subterms(t, path=()):
    return list(_positions(t))


def mutate_statement(st: Statement, world: World, rng):
    env = st.env()
    funcs = world.funcs
    op = rng.choice(STMT_OPS)
    side = rng.randint(0, 1)
    t = st.lhs if side == 0 else st.rhs
    other = st.rhs if side == 0 else st.lhs
    pos = _subterms(t)
    new = None
    if op == "flip_sides":
        return Statement(st.vars, st.rhs, st.lhs), op
    if op == "swap_args":
        cands = []
        for path, s in pos:
            if isinstance(s, tuple) and len(s) >= 3:
                ats = CTORS[s[0]][0] if s[0] in CTORS else funcs[s[0]].argtypes
                for i in range(len(ats)):
                    for j in range(i + 1, len(ats)):
                        if ats[i] == ats[j] and s[1 + i] != s[1 + j]:
                            cands.append((path, s, i, j))
        if cands:
            path, s, i, j = rng.choice(cands)
            a = list(s)
            a[1 + i], a[1 + j] = a[1 + j], a[1 + i]
            new = _replace_at(t, path, tuple(a))
    elif op == "replace_fn":
        cands = []
        for path, s in pos:
            if isinstance(s, tuple) and s[0] in funcs:
                f = funcs[s[0]]
                alts = [g for g in funcs.values() if g.name != f.name and g.argtypes == f.argtypes
                        and g.rettype == f.rettype]
                if alts:
                    cands.append((path, s, alts))
        if cands:
            path, s, alts = rng.choice(cands)
            new = _replace_at(t, path, (rng.choice(alts).name,) + s[1:])
    elif op == "base_ctor":
        path, s = rng.choice(pos)
        ty = tq._typeof(s, env, funcs)
        base = (TYPES[ty][0],)
        if s != base:
            new = _replace_at(t, path, base)
    elif op == "var_swap":
        cands = [(p, s) for p, s in pos if isinstance(s, str)
                 and any(v != s and ty == env[s] for v, ty in st.vars)]
        if cands:
            path, s = rng.choice(cands)
            new = _replace_at(t, path, rng.choice([v for v, ty in st.vars if v != s and ty == env[s]]))
    elif op == "wrap":
        path, s = rng.choice(pos)
        ty = tq._typeof(s, env, funcs)
        w = {"Nat": ("S", s), "List": ("Cons", ("Z",), s), "Tree": ("Node", ("Leaf",), ("Z",), s),
             "Bool": None}[ty]
        if w is not None:
            new = _replace_at(t, path, w)
    if new is None:
        return None, op
    lhs, rhs = (new, other) if side == 0 else (other, new)
    used = term_vars(lhs) + [v for v in term_vars(rhs) if v not in term_vars(lhs)]
    vars_ = tuple((v, ty) for v, ty in st.vars if v in used)
    out = Statement(vars_, lhs, rhs)
    typecheck(out.lhs, out.env(), funcs)
    if typecheck(out.lhs, out.env(), funcs) != typecheck(out.rhs, out.env(), funcs):
        return None, op
    return out, op


def reason_class(v) -> str:
    if v.accepted:
        return "accepted"
    if not v.messages:
        return v.category
    m = v.messages[0]["text"]
    for key, cls in (("refl: the two sides differ", "refl: sides differ"), ("no subterm matches", "rw: no match"),
                     ("unknown identifier", "unknown identifier"), ("is not available here", "lemma not available (order)"),
                     ("unsolved goal", "unsolved goal"), ("step after the goal was closed", "step after closed goal"),
                     ("blocks, got", "wrong number of blocks"), ("unknown variable", "unknown variable"),
                     ("occurs in hypothesis", "induct on a variable in a hypothesis"),
                     ("parse error", "parse error"), ("not fixed by the match", "rw: unbound variable"),
                     ("fuel limit", "simp fuel"), ("steps; the limit", "too many steps")):
        if key in m:
            return cls
    return "other: " + m[:40]


def _stable(text):
    import hashlib
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def buggy_check(world, st, proof, avail):
    """Planted bug for the T1 canary: also accepts when refl fails on two sides with the same head."""
    v = check_proof(world, st, proof, avail)
    if v.accepted or not v.messages or "refl: the two sides differ" not in v.messages[0]["text"]:
        return v
    trace = []
    check_proof(world, st, proof, avail, trace=trace)
    g = trace[-1][2] if trace else None
    if g is not None and isinstance(g.lhs, tuple) and isinstance(g.rhs, tuple) and g.lhs[0] == g.rhs[0]:
        from ..backends.base import Verdict
        return Verdict(True, "ok", [], v.elapsed_s)
    return v


def _contexts(stream: Stream):
    ctx = []
    for p in stream.projects:
        w2 = stream.worlds[f"{p.name}@w2"]
        for o in stream.obligations:
            if o["repo"] != p.name:
                continue
            w = stream.worlds[o["snapshot_id"]]
            pos = w.lemma_pos[o["decl"]]
            st = w.lemmas[pos][1]
            ctx.append({"o": o, "world": w, "pos": pos, "st": st,
                        "avail_names": [n for n, _ in w.lemmas[:pos]],
                        "future_names": [n for n, _ in w2.lemmas[w2.lemma_pos[o["decl"]] + 1:]],
                        "self": o["decl"], "vars": list(st.vars),
                        "other_proofs": sorted({x.proof for x in p.lemmas if x.name != o["decl"]})})
    return ctx


def run_t1(stream: Stream, out_dir: str, n_mutants=1000, n_stmt_only=300, n_instances=N_INSTANCES, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "t1_checks.jsonl")
    backend = stream.backend()
    cache = VerdictCache()
    checker = CachedChecker(backend, cache)
    lines = []
    # (a) ground truth: every stream obligation through the backend, plus the prelude
    for o in stream.obligations:
        v, _ = checker.check(o["snapshot_id"], o["id"], o["proof"], 120)
        w = stream.worlds[o["snapshot_id"]]
        st = w.lemmas[w.lemma_pos[o["decl"]]][1]
        cex = counterexample(w, st, n_instances, seed=_stable(o["id"]))
        lines.append(verdict_line("S0/T1/gt", 0, "eval", o["id"], 0, o["proof"], v, part="gt",
                                  instances=n_instances, statement_true=cex is None))
    pw = stream.prelude.world
    for i, (name, st) in enumerate(pw.lemmas):
        v = check_proof(pw, st, pw.proofs[name], i)
        cex = counterexample(pw, st, n_instances, seed=_stable(name))
        lines.append(verdict_line("S0/T1/gt-prelude", 0, "eval", f"toy-prelude:{name}", 0, pw.proofs[name], v,
                                  part="gt", instances=n_instances, statement_true=cex is None))
    # (b) mutants
    rng = random.Random(seed)
    ctxs = _contexts(stream)
    mutants = []
    seen = set()
    tries = 0
    while len(mutants) < n_mutants + n_stmt_only and tries < 50 * (n_mutants + n_stmt_only):
        tries += 1
        c = rng.choice(ctxs)
        kind = "proof_only" if len(mutants) < n_mutants // 2 else (
            "proof_and_statement" if len(mutants) < n_mutants else "statement_only")
        proof, pop = (c["o"]["proof"], None)
        if kind != "statement_only":
            proof, pop = mutate_proof(c["o"]["proof"], rng, c)
            if proof is None:
                continue
        st, sop = c["st"], None
        if kind != "proof_only":
            try:
                st, sop = mutate_statement(c["st"], c["world"], rng)
            except ToyError:
                continue
            if st is None:
                continue
        key = (c["o"]["id"], st.show(), proof)
        if key in seen or (st == c["st"] and proof == c["o"]["proof"]):
            continue
        seen.add(key)
        mutants.append((kind, c, st, proof, pop, sop))
    for idx, (kind, c, st, proof, pop, sop) in enumerate(mutants):
        if kind == "proof_only":
            v, _ = checker.check(c["o"]["snapshot_id"], c["o"]["id"], proof, 120)
        else:
            v = check_proof(c["world"], st, proof, c["pos"])
        cex = counterexample(c["world"], st, n_instances, seed=_stable(st.show()))
        canary = buggy_check(c["world"], st, proof, c["pos"])
        lines.append(verdict_line("S0/T1/mutants", 0, "eval", f"{c['o']['id']}#m{idx}", 0, proof, v,
                                  part=kind, proof_op=pop, statement_op=sop,
                                  statement_mutated=st != c["st"],
                                  statement_true=cex is None, instances=n_instances,
                                  reason=reason_class(v),
                                  soundness_bug=bool(v.accepted and cex is not None),
                                  canary_accepted=bool(canary.accepted),
                                  canary_flag=bool(canary.accepted and cex is not None)))
    with open(path, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln, sort_keys=True) + "\n")
    return lines


# ----------------------------------------------------------------------------- T3 leaks
LEAK_TYPES = ["proof_text", "premise_list", "dream_equals_test", "repair_pair", "annotation_with_proof"]


def _alpha_flip(st: Statement):
    c = canonical(st, flip=True)
    ren = {v: v + "_z" for v, _ in c.vars}
    from ..backends.toyeq import subst
    return Statement(tuple((ren[v], t) for v, t in c.vars), subst(c.lhs, ren), subst(c.rhs, ren))


def leak_examples(o, w: World, rng):
    pos = w.lemma_pos[o["decl"]]
    st = w.lemmas[pos][1]
    prem = proof_lemmas(o["proof"])
    broken, _ = mutate_proof(o["proof"], rng, {"avail_names": [n for n, _ in w.lemmas[:pos]], "future_names": [],
                                              "self": o["decl"], "vars": list(st.vars), "other_proofs": ["simp refl"]})
    return {
        "proof_text": {"kind": "wake", "statement": o["statement"],
                       "input": o["src_up_to_decl"] + "\nProve:\n" + o["statement"], "target": o["proof"]},
        "premise_list": {"kind": "premise", "statement": o["statement"],
                         "input": f"state: {st.body()}", "target": " ".join(prem) if prem else "(none)"},
        "dream_equals_test": {"kind": "dream", "statement": _alpha_flip(st).show(), "source": "model"},
        "repair_pair": {"kind": "repair", "statement": o["statement"],
                        "input": f"attempt: {broken or 'refl'} output: refl: the two sides differ",
                        "target": o["proof"]},
        "annotation_with_proof": {"kind": "annotation", "obligation_id": o["id"],
                                  "input": f"similar proof: {o['proof']}"},
    }


def control_examples(stream: Stream):
    """Legitimate training/mining data that must pass the guard."""
    out = []
    for o in stream.obligations:
        w = stream.worlds[o["snapshot_id"]]
        pos = w.lemma_pos[o["decl"]]
        st = w.lemmas[pos][1]
        if o["split"] == "test":
            defs = sorted(tq.symbols(st.lhs) | tq.symbols(st.rhs))
            out.append(("annotation_unfold", {"kind": "annotation", "obligation_id": o["id"],
                        "input": "\n".join(w.funcs[d].source() for d in defs if d in w.funcs)}))
            continue
        src = {"source_ids": [o["id"]]}
        out.append(("wake", dict(src, kind="wake", statement=o["statement"],
                                 input=o["src_up_to_decl"] + "\nProve:\n" + o["statement"], target=o["proof"])))
        trace = []
        check_proof(w, st, o["proof"], pos, trace=trace)
        for kind, arg, g in trace:
            if kind in ("rw", "rw<-") and not arg.startswith("IH"):
                out.append(("premise", dict(src, kind="premise", statement=o["statement"],
                                            input=f"state: {show(g.lhs)} = {show(g.rhs)}", target=arg)))
        out.append(("dream_rederive", {"kind": "dream", "statement": o["statement"], "source": "model"}))
        out.append(("repair", dict(src, kind="repair", statement=o["statement"],
                                   input="attempt: refl output: refl: the two sides differ", target=o["proof"])))
    return out


def run_t3(stream: Stream, out_dir: str, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    guard = HoldoutGuard(stream.obligations, stream.worlds)
    rng = random.Random(seed)
    lines = []
    tests = [o for o in stream.obligations if o["split"] == "test"]
    distinctive = {tag for lst in guard.proof_needles.by_first.values() for _, tag in lst}
    for o in tests:
        w = stream.worlds[o["snapshot_id"]]
        for lt, ex in leak_examples(o, w, rng).items():
            for variant, e in (("stealth", ex), ("with_id", dict(ex, source_ids=[o["id"]]))):
                try:
                    guard.check(e)
                    raised, rule = False, None
                except LeakError as err:
                    raised, rule = True, err.rule
                lines.append({"part": "leak", "leak": lt, "variant": variant, "target_id": o["id"],
                              "raised": raised, "rule": rule})
        # decoy: a distinctive test proof carried by an example about an unrelated wake statement
        if o["id"] in distinctive:
            e = {"kind": "wake", "statement": next(x["statement"] for x in stream.obligations if x["split"] == "wake"),
                 "input": "Prove: ...", "target": o["proof"]}
            try:
                guard.check(e)
                lines.append({"part": "decoy", "leak": "proof_text_under_other_statement", "variant": "stealth",
                              "target_id": o["id"], "raised": False, "rule": None})
            except LeakError as err:
                lines.append({"part": "decoy", "leak": "proof_text_under_other_statement", "variant": "stealth",
                              "target_id": o["id"], "raised": True, "rule": err.rule})
    for kind, e in control_examples(stream):
        try:
            guard.check(e)
            raised, rule, detail = False, None, None
        except LeakError as err:
            raised, rule, detail = True, err.rule, err.detail
        lines.append({"part": "control", "leak": kind, "variant": "legit", "target_id": "", "raised": raised,
                      "rule": rule, "detail": detail})
    # L2: proof_uses edges out of test statements are removed
    kept = guard.filter_edges(stream.edges)
    removed = [e for e in stream.edges if e not in kept]
    lines.append({"part": "L2", "leak": "proof_uses_edges_of_test_statements", "variant": "filter",
                  "target_id": "", "raised": False, "rule": "L2", "edges_in": len(stream.edges),
                  "edges_removed": len(removed),
                  "removed_all_test_proof_uses": all(e["type"] == "proof_uses" for e in removed)})
    with open(os.path.join(out_dir, "t3_guard.jsonl"), "w") as f:
        for ln in lines:
            f.write(json.dumps(ln, sort_keys=True) + "\n")
    return lines, guard


# ----------------------------------------------------------------------------- T2 log
def run_t2(out_dir: str):
    """Writes the synthetic verdict logs and the metrics computed from them."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    la, ts_a = t2_cases.build_case_a()
    lb, ts_b = t2_cases.build_case_b()
    lb2, ts_c = t2_cases.build_quarter_log("T2/C-B2", t2_cases.CASE_C["B2"])
    lb6, _ = t2_cases.build_quarter_log("T2/C-B6", t2_cases.CASE_C["B6"])
    with open(os.path.join(out_dir, "t2_synthetic.jsonl"), "w") as f:
        for ln in la + lb + lb2 + lb6:
            f.write(json.dumps(ln, sort_keys=True) + "\n")
    return {"A": (la, ts_a), "B": (lb, ts_b), "C-B2": (lb2, ts_c), "C-B6": (lb6, ts_c)}


def t2_results(logs: dict) -> list:
    """[(case, metric, expected, computed)] — expected values are the hand-derived literals."""
    out = []
    la, ts_a = logs["A"]
    ma = stream_metrics(accuracy_matrix(la, ts_a, 1, run="T2/A"), 3)
    for m, e in t2_cases.CASE_A["expected"].items():
        out.append(("A", m, e, ma[m]))
    lb, ts_b = logs["B"]
    mb = stream_metrics(accuracy_matrix(lb, ts_b, 2, run="T2/B"), 2)
    for m, e in t2_cases.CASE_B["expected"].items():
        out.append(("B", m, e, mb[m]))
    l2, ts = logs["C-B2"]
    l6, _ = logs["C-B6"]
    acc2 = stream_metrics(accuracy_matrix(l2, ts, 1), 3)["ACC"]
    acc6 = stream_metrics(accuracy_matrix(l6, ts, 1), 3)["ACC"]
    ex = t2_cases.CASE_C["expected"]
    out.append(("C", "ACC_B2", ex["ACC_B2"], acc2))
    out.append(("C", "ACC_B6", ex["ACC_B6"], acc6))
    out.append(("C", "GC", ex["GC"], gap_closed(ma["ACC"], acc2, acc6)))
    return out


# ----------------------------------------------------------------------------- driver
def run_all(cfg: dict, out_dir: str | None = None):
    from . import report_s0
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = out_dir or os.path.join(root, cfg.get("run_dir", "runs/S0"))
    os.makedirs(out_dir, exist_ok=True)
    timing = {}
    t0 = time.time()
    stream = build_stream(cfg["seed"])
    timing["build_stream_s"] = time.time() - t0
    info = write_stream(stream, os.path.join(out_dir, "data"))
    t = time.time()
    run_t1(stream, out_dir, cfg["t1"]["mutants"], cfg["t1"]["statement_only"], cfg["t1"]["instances"], cfg["seed"])
    timing["t1_s"] = time.time() - t
    t = time.time()
    run_t2(out_dir)
    timing["t2_s"] = time.time() - t
    t = time.time()
    run_t3(stream, out_dir, cfg["seed"])
    timing["t3_s"] = time.time() - t
    timing["total_s"] = time.time() - t0
    meta = {"experiment": "S0", "seed": cfg["seed"], "env": env_probe(), "timing": timing,
            "code_commit": git_commit(os.path.dirname(root)), "data_version": info["data_version"],
            "projects": [{"name": p.name, "template": p.template, "prefix": p.prefix, "lemmas": len(p.lemmas),
                          "window1": p.cut, "wrappers": sum(1 for n in p.world.funcs if n.startswith(p.prefix + "w")),
                          "functions": len(p.world.funcs), "seed_failed": p.stats["seed_failed"],
                          "gen_stats": {k: v for k, v in p.stats.items() if k != "seed_failed"}}
                         for p in stream.projects + [stream.prelude]],
            "stages": [[sid, d.isoformat()] for sid, d in stream.stages]}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    report_s0.write_report(out_dir, os.path.join(root, "reports", "S0.md"))
    return meta
