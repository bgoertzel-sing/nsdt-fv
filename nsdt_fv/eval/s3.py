"""Experiment S3 (plan 8.1): symbolic channels A-E on the toy, one at a time on top of D1.

    python -m nsdt_fv.eval.s3 --config nsdt_fv/configs/S3.yaml --arm N-B [--seed 0] [--out DIR]

Arms:  D1    S2's D1 re-run through this driver with no channel (determinism and equivalence check: its
             verdict log must equal runs/S2/D1's except for the run label)
       N-A .. N-E   D1 plus one channel (C5 of the plan, toy version). The channel runs at the start of
             every stage on the stage's snapshot (after the store has added it, before wake) and feeds
             D1 through the A.3 Channel interface:
               wake_examples -> appended to the wake set of the same window (A: NEXTSTATE, PREMISE,
                                UNFOLD; D: REPAIR; E: PLAN)
               nrem_targets  -> store statements whose re-derivation weight gets + prio_weight (D, E)
               rem_seeds     -> statements offered to conjecture sleep after the model's own
                                conjectures, in the channel's ranking, inside the same budget of at
                                most rem_conjectures statements per stage (A, B, C); a statement whose
                                prompt leaves less than conj_max_new tokens of the context is skipped
                                ('too_long', the rule D1 applies to its own conjecture prompts)
Everything else is S2's D1 unchanged (configs/S2.yaml values are copied into configs/S3.yaml): stream,
base, cap, wake, sleep budgets, evaluation sets and sampling. Sampling seeds of every S3 arm are those
of S2's D1 (seed_run = "S2/D1/tiny-<base hash>/s<seed>"), so arms differ only through what the channel
changed in training (common random numbers; the run label in the logs is "S3/<arm>/...").

Raw logs per arm (runs/S3/<arm>/): S2's set (verdicts, dreams, conjectures, train_examples, train,
events, evalsets, meta, canary) plus channels.jsonl.gz (one line per channel item, with guard verdict,
checker verdict, trivial and store-duplicate flags, machine proof where one exists, cap status) and
channel_canaries in meta.json. reports/S3.md is generated from these by eval/report_s3.py (R3).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time

import numpy as np
import torch

from ..backends.base import Verdict
from ..backends.toyeq import statement_key
from ..data.holdout import HoldoutGuard, LeakError
from ..data.stream import BASELINE_TACTICS, build_stream, test_view, write_stream
from ..dt.dt_ref import cap_hash
from ..store.channels import make_channel, public
from ..train.model import Tokenizer, base_state_hash, sample
from .runner import CachedChecker, VerdictCache, evaluate, sample_seed, verdict_line
from .s1 import (KEYWORDS, Logger, StructuralError, canary_oracle, cells_for_stage, check_stage_lines,
                 eval_set_hash, file_sha, git_commit, make_model, wake_example)
from .s2 import ArmRun, aux_examples, conj_prompt, dream_example, fits, guard_verdict, sha12
from . import metrics as M

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {"D1": None, "N-A": "A", "N-B": "B", "N-C": "C", "N-D": "D", "N-E": "E"}


def eval_stage(model, tok, checker, run, seed_run, t, cells, views_by_j, cfg, log):
    """s1.eval_stage with sampling seeds from seed_run (common random numbers) and lines labelled run."""
    ecfg = cfg["eval"]
    k = ecfg["k"]
    obls = [v for j in cells for v in views_by_j[j]]
    prompts, seeds = [], []
    for o in obls:
        for s in range(k):
            prompts.append(o["statement"])
            seeds.append(sample_seed(seed_run, t, o["id"], s))
    t0 = time.time()
    cands = sample(model, tok, prompts, seeds, temperature=ecfg["temperature"], max_new=ecfg["max_new"])
    gen_s = time.time() - t0
    table = {}
    for o_s, c in zip([(o["id"], s) for o in obls for s in range(k)], cands):
        table.setdefault(o_s[0], []).append(c)
    lines = evaluate(checker, run, t, obls, lambda o, sd: table[o["id"]], k, timeout_s=ecfg["timeout_s"])
    jmap = {v["id"]: j for j in cells for v in views_by_j[j]}
    for ln, c in zip(lines, [table[o["id"]][s] for o in obls for s in range(k)]):
        ln["candidate"] = c
        ln["eval_set"] = jmap[ln["target_id"]]
        log.write(ln)
    return lines, gen_s


class S3Run(ArmRun):
    def __init__(self, arm, cfg, seed, stream, checker, guard, guard_fn, tok, base, base_hash, out, log):
        super().__init__("D1", cfg, seed, stream, checker, guard, guard_fn, tok, base, base_hash, out, log)
        self.label = arm
        self.run = f"S3/{arm}/tiny-{base_hash}/s{seed}"
        self.seed_run = f"S2/D1/tiny-{base_hash}/s{seed}"
        cid = ARMS[arm]
        self.channel = None
        if cid is not None:
            self.channel = make_channel(cid, cfg["channels"][cid], stream, guard, self.backend, tok=tok,
                                        max_len=self.model.cfg["max_len"], seed=seed)
            self.chlog = log("channels.jsonl.gz")
        self._keys = {}
        self.ch_stats = {}

    def _store_keys(self):
        out = set()
        for e in self.store.entries.values():
            k = self._keys.get(e["key"])
            if k is None:
                k = statement_key(self.stream.worlds[e["snapshot_id"]].statement(e["statement"]))
                self._keys[e["key"]] = k
            out.add(k)
        return out

    def run_channel(self, t):
        if self.channel is None:
            return None
        sid = self.stream.stages[t - 1][0]
        t0 = time.time()
        recs = self.channel.run_stage(t, sid, self._store_keys(), store=self.store)
        for r in recs:
            self.chlog.write(public(r))
        st = {"items": len(recs), "emitted": sum(1 for r in recs if r["emitted"]),
              "guard_refused": sum(1 for r in recs if r["guard"] not in ("pass", "-")),
              "wake": len(self.channel.wake_examples(t)), "seeds": len(self.channel.rem_seeds(t, 10 ** 9)),
              "prio": len(self.channel.nrem_targets(t, 10 ** 9)), "seconds": round(time.time() - t0, 2)}
        self.ch_stats[t] = st
        return st

    # ---------------------------------------------------------------- wake (S2 D1 + channel examples)
    def wake(self, t, window):
        dc = self.cfg["dream"]
        rng = random.Random(self.seed * 1_000_003 + t)
        wobls = [self.full[e["source_ids"][0]] for e in window]
        aux = aux_examples(wobls, self.tok, self.model.cfg["max_len"], rng, self.guard) if dc.get("aux_conj", True) else []
        chx = self.channel.wake_examples(t) if self.channel is not None else []
        exs = list(window) + aux + chx
        st = self.core.wake(exs, seed=self.seed * 1000 + t, batch=self.cfg["cap"]["wake_batch"])
        self.log_examples(t, "wake", window)
        self.log_examples(t, "wake_aux", aux)
        self.log_examples(t, "wake_channel", chx)
        st.update(n_human=len(window), n_aux=len(aux), n_channel=len(chx))
        return st

    # ---------------------------------------------------------------- sampling with seed_run
    def attempt(self, t, phase, items, n_cand, temperature, check):
        prompts, seeds = [], []
        for (tid, stmt, _, _) in items:
            for s in range(n_cand):
                prompts.append(stmt)
                seeds.append(sample_seed(self.seed_run, t, f"{phase}:{tid}", s))
        cands = self.core.generate(prompts, seeds, temperature, self.cfg["dream"]["max_new"]) if prompts else []
        res, c_iter = [], iter(cands)
        for (tid, stmt, srcs, args) in items:
            got = []
            for s in range(n_cand):
                c = next(c_iter)
                vid = f"{self.run}|{t}|{phase}|{tid}|{s}"
                g = guard_verdict(self.guard, dream_example(stmt, c, vid, srcs))
                if g == "pass":
                    v, _ = check(args, c)
                else:
                    v = Verdict(False, "error", [{"severity": "error", "text": f"guard {g}", "line": 0,
                                                  "column": 0, "goal_state": ""}], 0.0)
                ln = verdict_line(self.run, t, phase, tid, s, c, v, statement=stmt, guard=g,
                                  verdict_id=vid, snapshot_id=args[0], trivial=self.trivial.get(tid))
                ln["candidate"] = c
                self.dlog.write(ln)
                got.append((c, ln))
            res.append(got)
        return res

    # ---------------------------------------------------------------- re-derivation sleep with priorities
    def nrem(self, t):
        dc = self.cfg["dream"]
        sid = self.stream.stages[t - 1][0]
        pool = self.store.statements()
        ws = self.store.weights(pool, sid)
        prio = set(self.channel.nrem_targets(t, 10 ** 9)) if self.channel is not None else set()
        pw = float(self.cfg.get("prio_weight", 4.0))
        n = min(dc["nrem_statements"], len(pool))
        rng = np.random.default_rng([self.seed, t, 1])
        w = np.array([x["w"] + (pw if e["key"] in prio else 0.0) for x, e in zip(ws, pool)], dtype=float)
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
        for i, got in zip(pick, res):
            self.store.record_verdict(pool[i]["key"], any(ln["accepted"] for _, ln in got), t)
        acc, rej = self.select(t, "nrem", items, res)
        st = self.learn(t, "nrem", acc, rej)
        fad = sum(1 for i in pick if ws[i]["fading"] > 0)
        st.update(statements=len(items), attempts=sum(len(g) for g in res), pool=len(pool), fading_picked=fad,
                  examples=len(acc), prio_pool=sum(1 for e in pool if e["key"] in prio),
                  prio_picked=sum(1 for i in pick if pool[i]["key"] in prio))
        return st

    # ---------------------------------------------------------------- conjecture sleep with channel seeds
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
            seeds.append(sample_seed(self.seed_run, t, f"conj:{q}", 0))
            meta.append((q, p, [e["obligation_id"] for e in tri]))
        outs = self.core.generate(prompts, seeds, dc["temperature"], dc["conj_max_new"]) if prompts else []
        items, keys = [], {"conj:" + k for k in self.store.library}
        cur_sid = self.stream.stages[t - 1][0]
        offered = [(q, p, parents, text, "model", None) for (q, p, parents), text in zip(meta, outs)]
        if self.channel is not None:
            cp = cur_sid.split("@")[0]
            for i, s in enumerate(self.channel.rem_seeds(t, 10 ** 9)):
                offered.append((f"ch{i}", cp, [], s["statement"], "channel", s["item"]))
        for (q, p, parents, text, origin, item) in offered:
            sid = self.store.latest_sid[p]
            w = self.stream.worlds[sid]
            rec = {"stage": t, "q": q, "project": p, "snapshot_id": sid, "text": text, "parents": parents,
                   "origin": origin, "item": item}
            try:
                st = w.statement(text)
            except Exception:
                rec["status"] = "parse_error"
                self.clog.write(rec)
                continue
            key = statement_key(st)
            if origin == "channel" and not fits(self.tok, st.show(), dc["conj_max_new"], max_len):
                rec["status"] = "too_long"           # prompt must leave room for a conj_max_new-token proof (as D1's prompts)
            elif self.guard.is_test_statement(st.show()):
                rec["status"] = "test_statement"
                self.guard.filter_dreams([{"statement": st.show()}])
            elif key in self._repo_keys(sid):
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
        for (tid, stmt, _, args), got in zip(items, res):
            if any(ln["accepted"] for _, ln in got):
                n_lib += self.store.add_dream(tid[5:], stmt, args[0], t, self.trivial[tid])
        st = self.learn(t, "rem", acc, rej)
        st.update(raw=len(prompts), conjectures=len(items), attempts=sum(len(g) for g in res), examples=len(acc),
                  library_added=n_lib, library_size=len(self.store.library))
        return st


def channel_canaries(stream, guard, backend, cfg, tok, max_len, seed):
    """Per channel: run the channel's own extraction on material that leaks a test obligation and
    require the guard to refuse every resulting item. A: mine a test proof (NEXTSTATE/PREMISE/UNFOLD and
    goal-state statements of it); B: a composition whose result is a test statement; C: a conjecture
    equal to a test statement; D: repair triples built from a test proof; E: a plan example and a
    method instantiation on a test proof. Returns a list of {channel, what, items, refused, rules}."""
    from ..store.channels import ChannelBase
    full = stream.by_id()
    test = [o for o in stream.obligations if o["split"] == "test" and not o["closed_by_baseline_tactics"]
            and len(o["proof"].split()) >= 8][:3]
    out = []
    for cid in "ABCDE":
        ch = make_channel(cid, cfg["channels"][cid], stream, guard, backend, tok=None, max_len=max_len, seed=seed)
        items = []
        for o in test:
            if cid in "ADE":                       # proof miners: pretend a test obligation is in the window
                ch.window = (lambda sid, o=o: [o])
                ch.store = None
                ch.run_stage(0, o["snapshot_id"], set())
                items += [r for r in ch.records if r.get("_ex") and r["what"] not in ("statement",) or
                          (r["what"] == "statement" and r["origin"].startswith("goal_state"))]
            if cid in "BC":                        # statement generators: an emitted test statement
                ex = {"kind": "dream", "statement": o["statement"], "source": "channel", "source_ids": [],
                      "proof": o["proof"] if cid == "B" else ""}
                items.append({"what": "statement", "guard": ch.guard_ok(ex)})
            if cid == "E":                         # a method instantiation that equals the test proof
                ex = {"kind": "dream", "statement": o["statement"], "proof": o["proof"], "source": "channel",
                      "source_ids": [o["id"]]}
                items.append({"what": "priority", "guard": ch.guard_ok(ex)})
        items = [r for r in items if r.get("what") not in ("neutral", "stats", "methods", "unproved", "refuted")]
        rules = {}
        for r in items:
            rules[r["guard"]] = rules.get(r["guard"], 0) + 1
        out.append({"channel": cid, "test_ids": [o["id"] for o in test], "items": len(items),
                    "refused": sum(1 for r in items if r["guard"] != "pass"), "rules": rules})
    return out


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
    tok = Tokenizer(KEYWORDS)
    ch_can = channel_canaries(stream, guard, stream.backend(), cfg, tok, cfg["model"]["max_len"], seed) \
        if ARMS[arm] is not None else []
    for cc in ch_can:
        if cc["items"] == 0 or cc["refused"] != cc["items"]:
            raise StructuralError(f"channel canary did not raise on every item: {cc}")
    oracle = canary_oracle(checker, stream, views_by_j, T, cfg["eval"]["k"], adir)

    base = make_model(tok, cfg, seed)
    base_path = os.path.join(out_dir, "base", "base.pt")
    base.load_state_dict(torch.load(base_path))
    base_hash = base_state_hash(base)
    if base_hash != cfg["base_hash"][seed]:
        raise StructuralError(f"base hash {base_hash} != S1 base hash {cfg['base_hash'][seed]}")

    def log(name):
        return Logger(os.path.join(adir, name))
    vlog, tlog, elog = log("verdicts.jsonl.gz"), log("train.jsonl"), log("events.jsonl")
    runner = S3Run(arm, cfg, seed, stream, checker, guard, guard_fn, tok, base, base_hash, adir, log)
    model, run = runner.model, runner.run
    hashes, events = {}, []
    gen_total = train_total = sleep_total = chan_total = 0.0
    last = min(T, int(cfg.get("pilot_stages") or T))
    for t in range(0, last + 1):
        if t >= 1:
            sid = stream.stages[t - 1][0]
            g0 = glog["checked"]
            t1 = time.time()
            st = {"stage": t, "snapshot_id": sid}
            st["store_added"] = runner.store.add_snapshot(sid, t)
            tc = time.time()
            st["channel"] = runner.run_channel(t)
            chan_total += time.time() - tc
            st["wake"] = runner.wake(t, windows[t])
            ts = time.time()
            st["nrem"] = runner.nrem(t)
            st["rem"] = runner.rem(t)
            sleep_total += time.time() - ts
            train_total += time.time() - t1
            st.update(guard_checked=glog["checked"] - g0, guard_raised_total=glog["raised"],
                      base_hash=base_state_hash(base), cap_hash=cap_hash(model))
            if st["base_hash"] != base_hash:
                raise StructuralError(f"T8: {arm} stage {t}: base changed")
            tlog.write(st)
            ev = {"type": "train", "stage": t}
            events.append(ev)
            elog.write(ev)
            if cfg.get("save_checkpoints", True):
                os.makedirs(os.path.join(adir, "ckpt"), exist_ok=True)
                torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("base.")},
                           os.path.join(adir, "ckpt", f"stage{t:02d}.pt"))
        cells = cells_for_stage(t, T)
        for j in cells:
            hashes[f"{t},{j}"] = eval_set_hash(views_by_j[j])
        lines, gs = eval_stage(model, tok, checker, run, runner.seed_run, t, cells, views_by_j, cfg, vlog)
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
        if t >= 1:
            extra = f" nrem {st['nrem']['examples']}/{st['nrem']['attempts']} rem {st['rem']['examples']}/{st['rem']['attempts']} conj {st['rem']['conjectures']}"
            if st.get("channel"):
                extra += f" ch {st['channel']}"
        print(f"[{arm} s{seed}] stage {t}: {acc}{extra} ({time.time() - T0:.0f}s)", flush=True)
    for lg in (vlog, tlog, elog, runner.dlog, runner.clog, runner.xlog):
        lg.close()
    if runner.channel is not None:
        runner.chlog.close()
    with open(os.path.join(adir, "evalsets.json"), "w") as f:
        json.dump(hashes, f, indent=1, sort_keys=True)
    import yaml
    meta = {
        "experiment": "S3", "arm": arm, "channel": ARMS[arm], "seed": seed, "run": run, "seed_run": runner.seed_run,
        "config": cfg, "code_commit": commit_at_start, "code_commit_at_end": git_commit(),
        "data_version": winfo["data_version"], "stream": winfo, "stages": [s for s, _ in stream.stages],
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                     "pyyaml": yaml.__version__, "platform": platform.platform()},
        "toolchain": {"toyeq_checker_sha": file_sha(os.path.join(HERE, "backends", "toyeq.py"))},
        "threads": torch.get_num_threads(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "base": {"hash": base_hash, "params": base.n_params(), "path": os.path.relpath(base_path, HERE)},
        "cap": {"layers": model.layers, "r": cfg["cap"]["r"], "kind": model.kind,
                "trainable_params": model.n_params(trainable_only=True)},
        "guard": {"training_checked": glog["checked"], "training_raised": glog["raised"], "canaries": canaries,
                  "l3_dropped_conjectures": guard.dropped_dreams, "channel_canaries": ch_can},
        "canary": oracle, "channel_stats": runner.ch_stats,
        "wall_s": round(time.time() - T0, 1), "train_s": round(train_total, 1), "sleep_s": round(sleep_total, 1),
        "channel_s": round(chan_total, 1), "eval_generate_s": round(gen_total, 1),
        "dt_steps": runner.core.steps, "dt_tokens": runner.core.tokens,
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
    if cfg.get("experiment") != "S3":
        raise SystemExit("config is not an S3 config")
    if a.arm not in ARMS:
        raise SystemExit(f"unknown arm {a.arm}")
    run_arm(cfg, a.arm, out_dir=a.out)


if __name__ == "__main__":
    main()

