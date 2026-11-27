"""Experiment S2 (plan 8.1): DT on the toy stream, arms D1-D4 with DT-ref (A.4).

    python -m nsdt_fv.eval.s2 --config nsdt_fv/configs/S2.yaml --arm D1 [--seed 1] [--out DIR]

Same stream, evaluation sets, sampling settings and base model as S1 (no recalibration). The base is
S1's base of the same seed (runs/S2/base/base.pt is a copy of runs/S1/base/base.pt; its hash is
checked against S1's meta.json). Writes raw logs to <out>/<arm>/ (eval verdict lines, dream verdict
lines, conjecture log, one line per training example, per-stage statistics, event order, eval-set
hashes, guard and canary results, meta). Every number of reports/S2.md is recomputed from these logs
by eval/report_s2.py (R3).

Arms (Table 3):  D1  wake, re-derivation sleep, conjecture sleep, kernel filter, negative phase
                 D2  D1 without the kernel filter: every generated dream is learned, verdicts are
                     logged for measurement but never reach the learner (no fading term, no
                     negatives, no dream library)
                 D3  D1 without conjecture sleep
                 D4  D1 without the negative phase
                 B2c supplementary: sequential fine-tuning of the same cap only (S1's B2 recipe)

Per stage t (after evaluating the stage-0 model): wake on window t, re-derivation sleep, conjecture
sleep; base hash check (T8); then evaluate E_j for j <= min(t+1, T) exactly as S1 does (L4).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import sys
import time

import numpy as np
import torch

from ..data.holdout import HoldoutGuard, LeakError
from ..data.stream import BASELINE_TACTICS, build_stream, test_view, write_stream
from ..dt.dt_ref import CapLM, DTRef, cap_hash
from ..store.graph import ToyStore
from ..train.common import epochs_to_steps, train_steps
from ..train.model import Tokenizer, base_state_hash
from .runner import CachedChecker, VerdictCache, sample_seed, verdict_line
from .s1 import (KEYWORDS, Logger, StructuralError, canary_oracle, cells_for_stage, check_stage_lines,
                 eval_set_hash, eval_stage, file_sha, git_commit, make_model, wake_example)
from . import metrics as M

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {
    "D1": {"filter": True, "rem": True, "neg": True},
    "D2": {"filter": False, "rem": True, "neg": False},
    "D3": {"filter": True, "rem": False, "neg": True},
    "D4": {"filter": True, "rem": True, "neg": False},
}
CONJ = "conj:"


def sha12(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def conj_prompt(stmts):
    return CONJ + " " + " ; ".join(stmts)


def fits(tok, prompt, target_len, max_len):
    return len(tok.encode(prompt)) + 2 + target_len <= max_len


def aux_examples(window_obls, tok, max_len, rng, guard):
    """Wake auxiliary task for conjecture sleep: three statements of the window -> a fourth.
    Built from the window's (non-test) statements only; no proof text."""
    stm = [o for o in window_obls if not guard.is_test_statement(o["statement"])]
    out = []
    if len(stm) < 4:
        return out
    for o in stm:
        tl = len(tok.encode(o["statement"])) + 1
        others = [x for x in stm if x["id"] != o["id"]]
        for _ in range(20):                     # rejection sampling for triples that fit max_len
            tri = rng.sample(others, 3)
            p = conj_prompt([x["statement"] for x in tri])
            if fits(tok, p, tl, max_len):
                out.append({"kind": "statement", "prompt": p, "target": o["statement"], "statement": o["statement"],
                            "source": "human", "source_ids": [o["id"]] + [x["id"] for x in tri]})
                break
    return out


def dream_example(statement, cand, vid, source_ids):
    return {"kind": "dream", "prompt": statement, "target": cand, "statement": statement, "source": "model",
            "source_ids": list(source_ids), "id": vid}


def guard_verdict(guard, ex):
    try:
        guard.check(ex)
        return "pass"
    except LeakError as e:
        return e.rule


class ArmRun:
    def __init__(self, arm, cfg, seed, stream, checker, guard, guard_fn, tok, base, base_hash, out, log):
        self.arm, self.cfg, self.seed, self.stream = arm, cfg, seed, stream
        self.checker, self.guard, self.guard_fn, self.tok = checker, guard, guard_fn, tok
        self.flags = ARMS.get(arm)
        self.run = f"S2/{arm}/tiny-{base_hash}/s{seed}"
        self.base, self.base_hash = base, base_hash
        cc = cfg["cap"]
        self.model = CapLM(base, cc["r"], cc.get("layers"), seed=seed * 7 + 3, kind=cc.get("kind", "mlp"))
        self.core = DTRef(self.model, tok, cc, seed, guard_fn)
        self.store = ToyStore(stream, guard)
        self.full = stream.by_id()
        self.out = out
        self.dlog = log("dreams.jsonl.gz")
        self.clog = log("conjectures.jsonl.gz")
        self.xlog = log("train_examples.jsonl.gz")
        self.backend = stream.backend()
        self.trivial = {}                     # target id -> closed by the baseline tactics (section 7)

    def log_examples(self, t, phase, exs, extra=None):
        for i, ex in enumerate(exs):
            rec = {"stage": t, "phase": phase, "kind": ex["kind"], "source": ex["source"],
                   "source_ids": ex.get("source_ids", []), "verdict_id": ex.get("id"),
                   "target_sha": sha12(ex["target"]), "statement_sha": sha12(ex["statement"])}
            if extra:
                rec.update(extra(i, ex))
            self.xlog.write(rec)

    # ---------------------------------------------------------------- wake
    def wake(self, t, window):
        dc = self.cfg["dream"]
        rng = random.Random(self.seed * 1_000_003 + t)
        sid = self.stream.stages[t - 1][0]
        wobls = [self.full[e["source_ids"][0]] for e in window]
        aux = aux_examples(wobls, self.tok, self.model.cfg["max_len"], rng, self.guard) if dc.get("aux_conj", True) else []
        exs = list(window) + aux
        st = self.core.wake(exs, seed=self.seed * 1000 + t, batch=self.cfg["cap"]["wake_batch"])
        self.log_examples(t, "wake", window)
        self.log_examples(t, "wake_aux", aux)
        st.update(n_human=len(window), n_aux=len(aux))
        return st

    # ---------------------------------------------------------------- sleep helpers
    def attempt(self, t, phase, items, n_cand, temperature, check):
        """items: list of (target_id, statement, source_ids, check_args). Generates n_cand candidates per
        item, runs the guard on every candidate, checks every guard-passing candidate with the kernel,
        logs one verdict line per candidate. Returns per item a list of (candidate, line)."""
        prompts, seeds, keys = [], [], []
        for (tid, stmt, _, _) in items:
            for s in range(n_cand):
                prompts.append(stmt)
                seeds.append(sample_seed(self.run, t, f"{phase}:{tid}", s))
                keys.append((tid, s))
        cands = self.core.generate(prompts, seeds, temperature, self.cfg["dream"]["max_new"]) if prompts else []
        res = []
        c_iter = iter(cands)
        for (tid, stmt, srcs, args) in items:
            got = []
            for s in range(n_cand):
                c = next(c_iter)
                vid = f"{self.run}|{t}|{phase}|{tid}|{s}"
                g = guard_verdict(self.guard, dream_example(stmt, c, vid, srcs))
                if g == "pass":
                    v, _ = check(args, c)
                else:                               # refused by the guard: never checked, never learned
                    from ..backends.base import Verdict
                    v = Verdict(False, "error", [{"severity": "error", "text": f"guard {g}", "line": 0,
                                                  "column": 0, "goal_state": ""}], 0.0)
                ln = verdict_line(self.run, t, phase, tid, s, c, v, statement=stmt, guard=g,
                                  verdict_id=vid, snapshot_id=args[0], trivial=self.trivial.get(tid))
                ln["candidate"] = c
                self.dlog.write(ln)
                got.append((c, ln))
            res.append(got)
        return res

    def select(self, t, phase, items, results):
        """Training examples from attempts. D1/D3/D4: accepted candidates (distinct texts) only, with an
        optional rejected partner for the negative phase. D2: every distinct guard-passing candidate."""
        acc, rej = [], {}
        for (tid, stmt, srcs, _), got in zip(items, results):
            seen = set()
            rejected = [(c, ln) for c, ln in got if not ln["accepted"] and ln["guard"] == "pass"]
            k = 0
            for c, ln in got:
                if ln["guard"] != "pass" or c in seen:
                    continue
                if self.flags["filter"] and not ln["accepted"]:
                    continue
                seen.add(c)
                ex = dream_example(stmt, c, ln["verdict_id"], srcs)
                if self.flags["neg"] and rejected:
                    rc, rln = rejected[k % len(rejected)]
                    k += 1
                    rej[len(acc)] = dream_example(stmt, rc, rln["verdict_id"], srcs)
                acc.append(ex)
        return acc, rej

    def learn(self, t, phase, acc, rej):
        st = self.core.learn(acc, rej, seed=self.seed * 1000 + t * 10 + (1 if phase == "nrem" else 2),
                             pref_weight=self.cfg["dream"]["pref_weight"])
        self.log_examples(t, phase, acc, extra=lambda i, ex: {"neg_verdict_id": rej[i]["id"] if i in rej else None})
        return st

    # ---------------------------------------------------------------- re-derivation sleep (NREM)
    def nrem(self, t):
        dc = self.cfg["dream"]
        sid = self.stream.stages[t - 1][0]
        pool = self.store.statements() if self.flags["filter"] else self.store.statements(kind="repo")
        ws = self.store.weights(pool, sid)
        n = min(dc["nrem_statements"], len(pool))
        rng = np.random.default_rng([self.seed, t, 1])
        w = np.array([x["w"] for x in ws], dtype=float)
        pick = sorted(rng.choice(len(pool), size=n, replace=False, p=w / w.sum()).tolist()) if n else []
        items = []
        for i in pick:
            e = pool[i]
            srcs = [e["obligation_id"]] if e["obligation_id"] else []
            args = (e["snapshot_id"], e["obligation_id"], e["statement"])
            tid = e["key"] if e["kind"] == "repo" else "dream:" + e["key"]
            self.trivial[tid] = e["trivial"]
            items.append((tid, e["statement"], srcs, args))

        def check(args, c):
            sid_, oid, stmt = args
            if oid is not None:
                return self.checker.check(sid_, oid, c, self.cfg["eval"]["timeout_s"])
            return self.checker.check_statement(sid_, "", stmt, c, self.cfg["eval"]["timeout_s"])
        res = self.attempt(t, "nrem", items, dc["candidates"], dc["temperature"], check)
        if self.flags["filter"]:                  # the learner sees verdicts (fading) only with the kernel
            for i, got in zip(pick, res):
                self.store.record_verdict(pool[i]["key"], any(ln["accepted"] for _, ln in got), t)
        acc, rej = self.select(t, "nrem", items, res)
        st = self.learn(t, "nrem", acc, rej)
        fad = sum(1 for i in pick if ws[i]["fading"] > 0)
        st.update(statements=len(items), attempts=sum(len(g) for g in res), pool=len(pool), fading_picked=fad,
                  examples=len(acc))
        return st

    # ---------------------------------------------------------------- conjecture sleep (REM)
    def rem(self, t):
        dc = self.cfg["dream"]
        rng = random.Random(self.seed * 7_000_003 + t)
        projects = sorted(self.store.latest_sid)
        max_len = self.model.cfg["max_len"]
        prompts, seeds, meta = [], [], []
        for q in range(dc["rem_raw"]):
            p = projects[rng.randrange(len(projects))]
            repo = self.store.statements(project=p, kind="repo")
            prompt = None
            for _ in range(50):
                tri = rng.sample(repo, 3) if len(repo) >= 3 else None
                if tri is None:
                    break
                cand = conj_prompt([e["statement"] for e in tri])
                if fits(self.tok, cand, dc["conj_max_new"], max_len):
                    prompt = cand
                    break
            if prompt is None:
                continue
            prompts.append(prompt)
            seeds.append(sample_seed(self.run, t, f"conj:{q}", 0))
            meta.append((q, p, [e["obligation_id"] for e in tri]))
        outs = self.core.generate(prompts, seeds, dc["temperature"], dc["conj_max_new"]) if prompts else []
        items, keys = [], {"conj:" + k for k in self.store.library}
        for (q, p, parents), text in zip(meta, outs):
            sid = self.store.latest_sid[p]
            w = self.stream.worlds[sid]
            rec = {"stage": t, "q": q, "project": p, "snapshot_id": sid, "text": text, "parents": parents}
            try:
                st = w.statement(text)
            except Exception as e:                 # ToyError and friends: not a statement
                rec["status"] = "parse_error"
                self.clog.write(rec)
                continue
            from ..backends.toyeq import statement_key
            key = statement_key(st)
            repo_keys = self._repo_keys(sid)
            if self.guard.is_test_statement(st.show()):
                rec["status"] = "test_statement"     # L3: dropped, never attempted (guard.filter_dreams)
                self.guard.filter_dreams([{"statement": st.show()}])
            elif key in repo_keys:
                rec["status"] = "in_repository"
            elif "conj:" + sha12(key) in keys:
                rec["status"] = "duplicate"
            elif len(items) >= dc["rem_conjectures"]:
                rec["status"] = "over_budget"
            else:
                rec["status"] = "attempted"
                tid = "conj:" + sha12(key)
                keys.add(tid)
                self.trivial[tid] = any(self.backend.check_statement(sid, "", st.show(), tac, 120).accepted
                                        for tac in BASELINE_TACTICS)
                items.append((tid, st.show(), [], (sid, None, st.show())))
            rec["statement"] = st.show()
            self.clog.write(rec)

        def check(args, c):
            sid_, _, stmt = args
            return self.checker.check_statement(sid_, "", stmt, c, self.cfg["eval"]["timeout_s"])
        res = self.attempt(t, "rem", items, dc["candidates"], dc["temperature"], check)
        acc, rej = self.select(t, "rem", items, res)
        n_lib = 0
        if self.flags["filter"]:
            for (tid, stmt, _, args), got in zip(items, res):
                if any(ln["accepted"] for _, ln in got):
                    n_lib += self.store.add_dream(tid[5:], stmt, args[0], t, self.trivial[tid])
        st = self.learn(t, "rem", acc, rej)
        st.update(raw=len(prompts), conjectures=len(items), attempts=sum(len(g) for g in res), examples=len(acc),
                  library_added=n_lib, library_size=len(self.store.library))
        return st

    def _repo_keys(self, sid):
        from ..backends.toyeq import statement_key
        if not hasattr(self, "_rk"):
            self._rk = {}
        if sid not in self._rk:
            w = self.stream.worlds[sid]
            self._rk[sid] = {statement_key(st) for _, st in w.lemmas}
        return self._rk[sid]


def run_arm(cfg, arm, out_dir=None):
    torch.set_num_threads(int(cfg.get("threads", 2)))
    seed = int(cfg["seed"])
    out_dir = out_dir or os.path.join(HERE, cfg["run_dir"] if seed == 0 else cfg["run_dir"] + f"_seed{seed}")
    adir = os.path.join(out_dir, arm)
    os.makedirs(adir, exist_ok=True)
    T0 = time.time()
    commit_at_start = git_commit()
    stream = build_stream(seed=cfg["stream"]["seed"], n_projects=cfg["stream"]["projects"])
    winfo = write_stream(stream, os.path.join(adir, "data"))
    T = len(stream.stages)
    full = stream.by_id()
    test_sets = stream.test_sets()
    views_by_j = {j: [test_view(full[i]) for i in ids] for j, ids in test_sets.items()}
    with open(os.path.join(adir, "test_sets.json"), "w") as f:
        json.dump({"stages": [sid for sid, _ in stream.stages], "test_sets": {str(j): ids for j, ids in test_sets.items()}},
                  f, indent=1)
    checker = CachedChecker(stream.backend(), VerdictCache(None))
    worlds = dict(stream.worlds)
    worlds["toy-prelude"] = stream.prelude.world
    guard = HoldoutGuard(stream.obligations, worlds, min_tokens=cfg.get("guard_min_tokens", 8))
    glog = {"checked": 0, "raised": 0}

    def guard_fn(ex):
        glog["checked"] += 1
        try:
            guard.check(ex)
        except LeakError:
            glog["raised"] += 1
            raise

    windows = {}
    for t, (sid, _) in enumerate(stream.stages, 1):
        windows[t] = [wake_example(o) for o in stream.obligations if o["snapshot_id"] == sid and o["split"] == "wake"]
    # ---- guard pre-checks and canaries (must raise)
    canaries = []
    for j in (1, T):
        o = full[test_sets[j][0]]
        canaries.append({"what": "test obligation as wake example", "id": o["id"],
                         "rule": guard_verdict(guard, wake_example(o))})
        canaries.append({"what": "dream whose statement is a test statement", "id": o["id"],
                         "rule": guard_verdict(guard, dream_example(o["statement"], "refl", "canary", []))})
    tags = {tag for lst in guard.proof_needles.by_first.values() for _, tag in lst}
    distinct = next(o for o in stream.obligations if o["id"] in tags)
    some_wake = next(o for o in stream.obligations if o["split"] == "wake")
    canaries.append({"what": "dream whose proof text is a distinctive test proof", "id": distinct["id"],
                     "rule": guard_verdict(guard, dream_example(some_wake["statement"], distinct["proof"], "canary", []))})
    if any(c["rule"] == "pass" for c in canaries):
        raise StructuralError(f"a guard canary did not raise: {canaries}")
    oracle = canary_oracle(checker, stream, views_by_j, T, cfg["eval"]["k"], adir)

    # ---- base: S1's base of this seed, frozen
    tok = Tokenizer(KEYWORDS)
    base = make_model(tok, cfg, seed)
    base_path = os.path.join(out_dir, "base", "base.pt")
    base.load_state_dict(torch.load(base_path))
    base_hash = base_state_hash(base)
    want = cfg["base_hash"][seed]
    if base_hash != want:
        raise StructuralError(f"base hash {base_hash} != S1 base hash {want}")

    def log(name):
        return Logger(os.path.join(adir, name))
    vlog, tlog, elog = log("verdicts.jsonl.gz"), log("train.jsonl"), log("events.jsonl")
    if arm == "B2c":
        run = f"S2/{arm}/tiny-{base_hash}/s{seed}"
        model = CapLM(base, cfg["cap"]["r"], cfg["cap"].get("layers"), seed=seed * 7 + 3,
                      kind=cfg["cap"].get("kind", "mlp"))
        runner = None
        xlog = log("train_examples.jsonl.gz")
    else:
        runner = ArmRun(arm, cfg, seed, stream, checker, guard, guard_fn, tok, base, base_hash, adir, log)
        model, run = runner.model, runner.run
    hashes, events = {}, []
    gen_total = train_total = sleep_total = 0.0
    last = min(T, int(cfg.get("pilot_stages") or T))     # pilot runs only (never for reported runs)
    for t in range(0, last + 1):
        if t >= 1:
            sid = stream.stages[t - 1][0]
            g0 = glog["checked"]
            t1 = time.time()
            st = {"stage": t, "snapshot_id": sid}
            if runner is None:                     # B2c: S1's B2 recipe on the cap parameters
                b2 = cfg["b2c"]
                steps = epochs_to_steps(len(windows[t]), b2["epochs"], b2["batch"])
                st["wake"] = train_steps(model, tok, model.cap_parameters(), windows[t], steps=steps, batch=b2["batch"],
                                         lr=b2["lr"], seed=seed * 1000 + t, guard=guard_fn)
                for ex in windows[t]:
                    xlog.write({"stage": t, "phase": "wake", "kind": ex["kind"], "source": ex["source"],
                                "source_ids": ex["source_ids"], "verdict_id": None,
                                "target_sha": sha12(ex["target"]), "statement_sha": sha12(ex["statement"])})
            else:
                st["store_added"] = runner.store.add_snapshot(sid, t)
                st["wake"] = runner.wake(t, windows[t])
                ts = time.time()
                st["nrem"] = runner.nrem(t)
                if runner.flags["rem"]:
                    st["rem"] = runner.rem(t)
                sleep_total += time.time() - ts
            train_total += time.time() - t1
            st.update(guard_checked=glog["checked"] - g0, guard_raised_total=glog["raised"],
                      base_hash=base_state_hash(base), cap_hash=cap_hash(model))
            if st["base_hash"] != base_hash:
                raise StructuralError(f"T8: {arm} stage {t}: base changed ({st['base_hash']} != {base_hash})")
            tlog.write(st)
            ev = {"type": "train", "stage": t}
            events.append(ev)
            elog.write(ev)
            if cfg.get("save_checkpoints", True):
                os.makedirs(os.path.join(adir, "ckpt"), exist_ok=True)
                torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("base.")},
                           os.path.join(adir, "ckpt", f"stage{t:02d}.pt"))          # R12
        cells = cells_for_stage(t, T)
        for j in cells:
            hashes[f"{t},{j}"] = eval_set_hash(views_by_j[j])
        lines, gs = eval_stage(model, tok, checker, run, t, cells, views_by_j, cfg, vlog)
        gen_total += gs
        check_stage_lines(lines, t, cells, test_sets, cfg["eval"]["k"])
        for j in cells:
            ev = {"type": "eval", "model_stage": t, "set": j}
            events.append(ev)
            elog.write(ev)
        HoldoutGuard.check_eval_order(events)
        acc = {j: round(v, 3) for (tt, j), v in M.accuracy_matrix(lines, {j: test_sets[j] for j in cells},
                                                                     cfg["eval"]["k"], run=run).items()}
        extra = ""
        if t >= 1 and runner is not None:
            extra = f" nrem {st['nrem']['examples']}/{st['nrem']['attempts']}"
            if "rem" in st:
                extra += f" rem {st['rem']['examples']}/{st['rem']['attempts']} conj {st['rem']['conjectures']}"
        print(f"[{arm} s{seed}] stage {t}: {acc}{extra} ({time.time() - T0:.0f}s)", flush=True)
    for lg in (vlog, tlog, elog):
        lg.close()
    if runner is not None:
        for lg in (runner.dlog, runner.clog, runner.xlog):
            lg.close()
    else:
        xlog.close()
    with open(os.path.join(adir, "evalsets.json"), "w") as f:
        json.dump(hashes, f, indent=1, sort_keys=True)
    import yaml
    meta = {
        "experiment": "S2", "arm": arm, "seed": seed, "run": run, "config": cfg, "code_commit": commit_at_start,
        "code_commit_at_end": git_commit(), "data_version": winfo["data_version"], "stream": winfo,
        "stages": [s for s, _ in stream.stages],
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                     "pyyaml": yaml.__version__, "platform": platform.platform()},
        "toolchain": {"toyeq_checker_sha": file_sha(os.path.join(HERE, "backends", "toyeq.py"))},
        "threads": torch.get_num_threads(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "base": {"hash": base_hash, "params": base.n_params(), "path": os.path.relpath(base_path, HERE)},
        "cap": {"layers": model.layers, "r": cfg["cap"]["r"], "kind": model.kind,
                "trainable_params": model.n_params(trainable_only=True)},
        "guard": {"training_checked": glog["checked"], "training_raised": glog["raised"], "canaries": canaries,
                  "l3_dropped_conjectures": guard.dropped_dreams},
        "canary": oracle,
        "wall_s": round(time.time() - T0, 1), "train_s": round(train_total, 1), "sleep_s": round(sleep_total, 1),
        "eval_generate_s": round(gen_total, 1), "dt_steps": runner.core.steps if runner else None,
        "dt_tokens": runner.core.tokens if runner else None,
    }
    with open(os.path.join(adir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True, default=str)
    if glog["raised"]:
        raise StructuralError(f"{arm}: {glog['raised']} guard raises during training")
    print(f"[{arm} s{seed}] finished in {time.time() - T0:.0f}s", flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pilot-stages", type=int, default=None, help="pilot only: stop after this stage")
    a = ap.parse_args(argv)
    import yaml
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.pilot_stages:
        cfg["pilot_stages"] = a.pilot_stages
        if not a.out:
            raise SystemExit("a pilot needs --out (it must not write into the reported run directories)")
    if cfg.get("experiment") != "S2":
        raise SystemExit("config is not an S2 config")
    if a.arm not in ARMS and a.arm != "B2c":
        raise SystemExit(f"unknown arm {a.arm}")
    run_arm(cfg, a.arm, out_dir=a.out)


if __name__ == "__main__":
    main()
