"""Gate G0, last condition: "replacing the ToyEq backend by a Lean backend requires no change to the
trainer or the evaluator" (plan 8.1). This driver runs a small Lean-flavoured stream through the
unchanged trainer (train/lora.Sequential = B2, train/joint.Joint = B6, train/model) and the unchanged
evaluator (eval/runner.evaluate + CachedChecker + VerdictCache, eval/metrics) with StubLeanBackend
(backends/lean_stub.py, canned verdicts, no Lean). It imports no ToyEq code, and it records which modules
were loaded so the test can show that none of ToyEq was.

    python3 -m nsdt_fv.gate.g0_stub_run --out nsdt_fv/runs/G0/stub_run

What is NOT exercised here: the hold-out guard (data/holdout.py) and the stream builder (data/stream.py)
are ToyEq-specific (they parse ToyEq statements and replay ToyEq proofs); the trainer takes the guard as a
callable, so this driver passes a minimal identifier/statement/proof-text guard (StubGuard below). A Lean
guard and a Lean stream builder are C1 work (reports/ISSUES.md).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

from ..backends.base import CheckerBackend
from ..backends.lean_stub import StubLeanBackend, lean_tokens
from ..data.schemas import validate_obligation, validate_verdict_line
from ..eval import runner
from ..eval.metrics import accuracy_matrix, stream_metrics
from ..train.joint import Joint
from ..train.lora import Sequential
from ..train.model import TinyLM, Tokenizer, base_state_hash, sample

LEAN_KEYWORDS = ["by", "simp", "omega", "rw", "exact", "intro", "induction", "with", "theorem", "lemma",
                 "Nat", "List", "length", "simp_all", "constructor", "rfl", "decide", "fun", "at"]

# statement family -> proof (ASCII Lean). Two repositories, two windows each; the window decides
# which families appear, so later windows need proofs the earlier ones did not teach.
FAMILIES = {
    "add_comm": ("(n m : Nat) : n + m + {c} = m + n + {c}", "by omega"),
    "app_nil": ("(xs : List Nat) : (xs ++ []).length + {c} = xs.length + {c}", "by simp"),
    "mul_comm": ("(a b : Nat) : a * b * {c} = b * a * {c}", "by rw [Nat.mul_comm a b]"),
    "and_intro": ("(p q : Prop) (hp : p) (hq : q) (h{c} : True) : p /\\ q", "by exact And.intro hp hq"),
    "le_succ": ("(n : Nat) : n <= n + {c}", "by omega"),
    "len_cons": ("(x : Nat) (xs : List Nat) : (x :: xs).length = xs.length + 1 + {c} - {c}", "by simp"),
}
PLAN = {("stub-alpha", 1): ["add_comm", "app_nil"], ("stub-beta", 1): ["mul_comm", "add_comm"],
        ("stub-alpha", 2): ["and_intro", "le_succ"], ("stub-beta", 2): ["len_cons", "mul_comm"]}
PER_WINDOW = 12          # obligations per window; the last 25% are the in-window test set


def build_stream(seed=0):
    """Appendix B obligation records (validated) and the stage order (window end, ties by repo name)."""
    rng = random.Random(seed)
    obls = []
    stages = sorted(PLAN, key=lambda rw: (rw[1], rw[0]))
    for t, (repo, w) in enumerate(stages, start=1):
        commit = f"w{w}stub{seed}"
        for i in range(PER_WINDOW):
            fam = PLAN[(repo, w)][i % 2]
            c = rng.randint(1, 99)
            sig, proof = FAMILIES[fam]
            decl = f"Stub.{fam}_{w}_{i}"
            o = {"id": f"{repo}@{commit}:{decl}", "repo": repo, "commit": commit, "snapshot_id": f"{repo}@w{w}",
                 "module": f"Stub.{repo.replace('-', '_')}.W{w}", "decl": decl, "kind": "theorem",
                 "statement": f"theorem {fam}_{w}_{i} " + sig.format(c=c), "src_up_to_decl": "", "scope": "",
                 "deps_project": [], "deps_library": [], "n_deps_direct": 0, "n_deps_transitive": 0,
                 "proof": proof, "first_seen": f"2025-0{w}-{10 + i:02d}", "window": w,
                 "split": "test" if i >= PER_WINDOW - PER_WINDOW // 4 else "wake",
                 "closed_by_baseline_tactics": False, "stage": t}
            validate_obligation({k: v for k, v in o.items() if k != "stage"})
            obls.append(o)
    return obls, len(stages)


def test_view(o):
    v = dict(o)
    v.pop("proof", None)
    return v


class StubGuard:
    """Minimal backend-neutral stand-in for the hold-out guard (L1 identifiers and statements, L3, L7):
    raises on a test obligation identifier, on a proof-derived example or dream about a test statement,
    on a distinctive test proof in the text, and on frontier output. Not a replacement for HoldoutGuard."""

    class Leak(Exception):
        pass

    def __init__(self, obligations):
        test = [o for o in obligations if o["split"] == "test"]
        self.test_ids = {o["id"] for o in test}
        self.test_statements = {" ".join(lean_tokens(o["statement"])) for o in test}
        generic = {" ".join(lean_tokens(o["proof"])) for o in obligations if o["split"] != "test"}
        self.test_proofs = {" ".join(lean_tokens(o["proof"])) for o in test} - generic
        self.checked = 0

    def check(self, ex):
        self.checked += 1
        if ex.get("source") == "frontier":
            raise self.Leak("L7 frontier output")
        for i in ex.get("source_ids", []):
            if i in self.test_ids:
                raise self.Leak(f"L1 test identifier {i}")
        if " ".join(lean_tokens(ex.get("statement", ""))) in self.test_statements:
            raise self.Leak("L1/L3 test statement")
        text = " ".join(lean_tokens(ex.get("target", "")))
        if any(p and p in text for p in self.test_proofs):
            raise self.Leak("L1 test proof text")


def wake_example(o):
    return {"kind": "wake", "statement": o["statement"], "prompt": o["statement"], "target": o["proof"],
            "source_ids": [o["id"]], "source": "human"}


def run_arm(arm, obls, T, seed, k, cfg, mcfg, log_path=None):
    tok = Tokenizer(LEAN_KEYWORDS)
    torch.manual_seed(seed)
    model = TinyLM(tok.vocab_size, **mcfg)
    guard = StubGuard(obls)
    learner = (Sequential if arm == "B2" else Joint)(model, tok, cfg, seed, guard.check)
    backend = StubLeanBackend([o for o in obls])            # the backend holds the ground truth, as Lean would
    assert isinstance(backend, CheckerBackend)
    checker = runner.CachedChecker(backend, runner.VerdictCache())
    tests = {j: [test_view(o) for o in obls if o["stage"] == j and o["split"] == "test"] for j in range(1, T + 1)}
    all_tests = [o for j in range(1, T + 1) for o in tests[j]]
    run = f"G0/{arm}/stub-lean/seed{seed}"

    def sampler(o, seeds):
        return sample(model, tok, [o["statement"]] * len(seeds), seeds, temperature=1.0, max_new=48)

    lines = runner.evaluate(checker, run, 0, all_tests, sampler, k)
    history, train_stats = [], []
    base_hash = base_state_hash(model)
    for t in range(1, T + 1):
        wake = [wake_example(o) for o in obls if o["stage"] == t and o["split"] == "wake"]
        st = learner.train_stage(t, wake, history)
        train_stats.append({"stage": t, **{k2: v for k2, v in st.items() if isinstance(v, (int, float, type(None)))}})
        history += wake
        lines += runner.evaluate(checker, run, t, all_tests, sampler, k)
    for ln in lines:
        validate_verdict_line({f: ln[f] for f in ("run", "stage", "phase", "target_id", "sample", "candidate_sha",
                                                   "accepted", "category", "elapsed_s", "repair_round")})
    if log_path:
        with open(log_path, "w") as f:
            for ln in lines:
                f.write(json.dumps(ln) + "\n")
    test_sets = {j: [o["id"] for o in tests[j]] for j in tests}
    A = accuracy_matrix(lines, test_sets, k, run=run)
    m = stream_metrics(A, T)
    cats = {}
    for ln in lines:
        cats[ln["category"]] = cats.get(ln["category"], 0) + 1
    return {"arm": arm, "run": run, "metrics": m, "A": {f"{t},{j}": v for (t, j), v in sorted(A.items())},
            "verdict_lines": len(lines), "categories": cats, "accepted_lines": sum(l["accepted"] for l in lines),
            "backend_calls": dict(backend.calls), "cache_hits": checker.cache.hits,
            "cache_misses": checker.cache.misses, "guard_checked": guard.checked,
            "base_hash_before_training": base_hash, "train": train_stats}


def canary(obls, T, k):
    """Known answer: ground-truth proofs through the same evaluate() and metrics give ACC 1, F 0; proofs with
    'sorry' give ACC 0 and category 'sorry'."""
    out = {}
    for name, fn in (("truth", lambda o: o["proof"]), ("sorry", lambda o: "by sorry")):
        backend = StubLeanBackend(obls)
        checker = runner.CachedChecker(backend, runner.VerdictCache())
        tests = [test_view(o) for o in obls if o["split"] == "test"]
        truth = {o["id"]: fn(o) for o in obls}
        lines = []
        for t in range(T + 1):
            lines += runner.evaluate(checker, f"G0/canary-{name}", t, tests, lambda o, s: [truth[o["id"]]] * len(s), k)
        ts = {j: [o["id"] for o in obls if o["stage"] == j and o["split"] == "test"] for j in range(1, T + 1)}
        out[name] = {"metrics": stream_metrics(accuracy_matrix(lines, ts, k), T),
                     "categories": sorted({l["category"] for l in lines})}
    return out


def guard_canary(obls):
    g = StubGuard(obls)
    t = next(o for o in obls if o["split"] == "test")
    try:
        g.check(wake_example(t))
        return False
    except StubGuard.Leak:
        return True


def toy_modules():
    return sorted(m for m in sys.modules if any(s in m for s in ("toyeq", "data.stream", "data.holdout",
                                                                  "backends.mock", "data.toyeq")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=40)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    obls, T = build_stream(a.seed)
    cfg = {"mode": "full", "epochs": a.epochs, "joint_epochs": a.epochs, "batch": 8, "lr": 3e-3}
    mcfg = {"d": 64, "layers": 2, "heads": 4, "ff": 128, "max_len": 160, "drop": 0.0}
    res = {"T": T, "k": a.k, "seed": a.seed, "obligations": len(obls),
           "wake": sum(o["split"] == "wake" for o in obls), "test": sum(o["split"] == "test" for o in obls),
           "snapshots": sorted({o["snapshot_id"] for o in obls}), "trainer_cfg": cfg, "model_cfg": mcfg,
           "toy_modules_before": toy_modules()}
    res["arms"] = [run_arm(arm, obls, T, a.seed, a.k, cfg, mcfg, os.path.join(a.out, f"{arm}.verdicts.jsonl"))
                   for arm in ("B2", "B6")]
    res["canary"] = canary(obls, T, a.k)
    res["guard_canary_raised"] = guard_canary(obls)
    res["toy_modules_after"] = toy_modules()
    res["nsdt_modules_loaded"] = sorted(m for m in sys.modules if m.startswith("nsdt_fv"))
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    for r in res["arms"]:
        print(r["arm"], {k: round(v, 3) for k, v in r["metrics"].items()}, "accepted lines", r["accepted_lines"],
              "of", r["verdict_lines"], "categories", r["categories"])
    print("canary", res["canary"], "guard canary raised", res["guard_canary_raised"])
    print("ToyEq-side modules loaded:", res["toy_modules_after"] or "none")


if __name__ == "__main__":
    main()
