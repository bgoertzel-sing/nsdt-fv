"""Experiment S1 (plan 8.1): baselines B0, B2, B5-10 and B6 on the twelve-stage ToyEq stream.

    python -m nsdt_fv.eval.s1 --config nsdt_fv/configs/S1.yaml [--out DIR] [--arms B0,B2]

Writes raw logs to runs/S1/ (verdict lines per arm, training statistics, event order, guard and
canary results, eval-set hashes, meta). Every number of reports/S1.md is recomputed from these logs
by eval/report_s1.py (rule R3).

Order of work for each arm (rule L4): evaluate the model at stage 0, then for t = 1..T train on
window t and evaluate on E_j for j <= min(t + 1, T). These are exactly the cells a[t, j] that LA, FWD,
ACC, F and BWT use.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import torch

from ..backends.base import CATEGORIES
from ..backends.toyeq import RESERVED
from ..data.holdout import HoldoutGuard, LeakError
from ..data.stream import build_stream, test_view, write_stream
from ..train.common import epochs_to_steps, train_steps
from ..train.joint import Joint
from ..train.lora import Sequential
from ..train.model import TinyLM, Tokenizer, adapter_hash, base_state_hash, sample
from ..train.replay import Replay
from .runner import CachedChecker, VerdictCache, evaluate, sample_seed
from . import metrics as M

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYWORDS = sorted(RESERVED | {"IH", "rw<-"})


class StructuralError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- helpers
def git_commit():
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=HERE,
                               capture_output=True, text=True).stdout.strip()
        return c + ("+dirty" if dirty else "")
    except Exception:                           # pragma: no cover
        return "unknown"


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def cells_for_stage(t, T):
    return list(range(1, min(t + 1, T) + 1))


def eval_set_hash(views):
    """T4: hash of an evaluation set (test views: id, snapshot and statement, no proofs)."""
    h = hashlib.sha256()
    for v in sorted(views, key=lambda v: v["id"]):
        h.update(json.dumps({"id": v["id"], "snapshot_id": v["snapshot_id"], "statement": v["statement"]},
                            sort_keys=True).encode())
    return h.hexdigest()


def wake_example(o):
    return {"kind": "wake", "prompt": o["statement"], "target": o["proof"], "statement": o["statement"],
            "source_ids": [o["id"]], "source": "human"}


def check_stage_lines(lines, t, cells, test_sets, k):
    """Per-unit structural check (sanity gate 2): refuse a stage whose verdict lines are malformed."""
    want = {i for j in cells for i in test_sets[j]}
    got = {}
    for ln in lines:
        if ln["stage"] != t or ln["phase"] != "eval":
            raise StructuralError(f"stage {t}: foreign line {ln['stage']}/{ln['phase']}")
        if ln["category"] not in CATEGORIES or ln["accepted"] != (ln["category"] == "ok"):
            raise StructuralError(f"stage {t}: bad verdict {ln}")
        if ln["target_id"] not in want:
            raise StructuralError(f"stage {t}: {ln['target_id']} is not in the cells {cells}")
        got.setdefault(ln["target_id"], set()).add(ln["sample"])
    if set(got) != want:
        raise StructuralError(f"stage {t}: {len(want - set(got))} obligations without samples")
    for i, ss in got.items():
        if ss != set(range(k)):
            raise StructuralError(f"stage {t}: {i} has samples {sorted(ss)}, want 0..{k - 1}")
    for ln in lines:
        if not isinstance(ln.get("candidate"), str):
            raise StructuralError(f"stage {t}: candidate text missing")


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w")

    def write(self, rec):
        self.f.write(json.dumps(rec, sort_keys=True) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ----------------------------------------------------------------------------- evaluation
def eval_stage(model, tok, checker, run, t, cells, views_by_j, cfg, log):
    ecfg = cfg["eval"]
    k = ecfg["k"]
    obls = [v for j in cells for v in views_by_j[j]]
    prompts, seeds, keys = [], [], []
    for o in obls:
        for s in range(k):
            prompts.append(o["statement"])
            seeds.append(sample_seed(run, t, o["id"], s))
            keys.append((o["id"], s))
    t0 = time.time()
    cands = sample(model, tok, prompts, seeds, temperature=ecfg["temperature"], max_new=ecfg["max_new"])
    gen_s = time.time() - t0
    table = {}
    for (oid, s), c in zip(keys, cands):
        table.setdefault(oid, []).append(c)
    lines = evaluate(checker, run, t, obls, lambda o, sd: table[o["id"]], k, timeout_s=ecfg["timeout_s"])
    for ln, c in zip(lines, [table[o["id"]][s] for o in obls for s in range(k)]):
        ln["candidate"] = c
        ln["eval_set"] = None
    jmap = {v["id"]: j for j in cells for v in views_by_j[j]}
    for ln in lines:
        ln["eval_set"] = jmap[ln["target_id"]]
        log.write(ln)
    return lines, gen_s


def canary_oracle(checker, stream, views_by_j, T, k, out_dir):
    """Known-answer canary (sanity gate 1): ground-truth proofs through the same evaluate() path and
    the same metrics code must give a[t, j] = 1 for every cell, hence LA = FWD = ACC = 1, F = BWT = 0."""
    full = stream.by_id()
    run = "S1/canary-oracle"
    log = Logger(os.path.join(out_dir, "canary", "verdicts.jsonl.gz"))
    lines = []
    for t in range(0, T + 1):
        cells = cells_for_stage(t, T)
        obls = [v for j in cells for v in views_by_j[j]]
        ls = evaluate(checker, run, t, obls, lambda o, sd: [full[o["id"]]["proof"]] * k, k)
        for ln in ls:
            ln["candidate"] = full[ln["target_id"]]["proof"]
            log.write(ln)
        check_stage_lines(ls, t, cells, {j: [v["id"] for v in views_by_j[j]] for j in views_by_j}, k)
        lines += ls
    log.close()
    test_sets = {j: [v["id"] for v in views_by_j[j]] for j in views_by_j}
    m, A = M.metrics_from_log(lines, test_sets, k, run=run)
    ok = all(abs(v - 1.0) < 1e-12 for v in A.values()) and abs(m["F"]) < 1e-12 and abs(m["BWT"]) < 1e-12 \
        and abs(m["ACC"] - 1.0) < 1e-12
    res = {"run": run, "cells": len(A), "metrics": m, "passed": ok}
    with open(os.path.join(out_dir, "canary", "result.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    if not ok:
        raise StructuralError(f"oracle canary failed: {m}")
    return res


# ----------------------------------------------------------------------------- main
def make_model(tok, cfg, seed):
    torch.manual_seed(seed)
    mc = cfg["model"]
    return TinyLM(tok.vocab_size, d=mc["d"], layers=mc["layers"], heads=mc["heads"], ff=mc["ff"],
                  max_len=mc["max_len"], drop=mc["dropout"])


def make_trainer(arm, model, tok, cfg, seed, guard):
    a = cfg["adapt"]
    if arm == "B2":
        return Sequential(model, tok, a, seed, guard)
    if arm.startswith("B5-"):
        return Replay(model, tok, a, seed, guard, percent=int(arm.split("-")[1]))
    if arm == "B6":
        return Joint(model, tok, a, seed, guard)
    raise ValueError(arm)


def run_all(cfg, out_dir=None, arms=None):
    torch.set_num_threads(int(cfg.get("threads", 2)))
    out_dir = out_dir or os.path.join(HERE, cfg["run_dir"])
    os.makedirs(out_dir, exist_ok=True)
    seed = int(cfg["seed"])
    T0 = time.time()
    sc = cfg["stream"]
    lo, hi = sc["lemmas_per_project"]
    sizes = None
    if (lo, hi) != (150, 250):
        import random
        r = random.Random(sc["seed"] * 7919 + 1)
        sizes = [r.randint(lo, hi) for _ in range(sc["projects"])]
    stream = build_stream(seed=sc["seed"], sizes=sizes, n_projects=sc["projects"])
    t_stream = time.time() - T0
    winfo = write_stream(stream, os.path.join(out_dir, "data"))
    T = len(stream.stages)
    full = stream.by_id()
    test_sets = stream.test_sets()
    views_by_j = {j: [test_view(full[i]) for i in ids] for j, ids in test_sets.items()}
    with open(os.path.join(out_dir, "test_obligations.jsonl"), "w") as f:
        for j in sorted(views_by_j):            # compact test views (no proofs, no file prefixes) for the report
            for v in views_by_j[j]:
                f.write(json.dumps({"id": v["id"], "set": j, "snapshot_id": v["snapshot_id"], "statement": v["statement"],
                                    "closed_by_baseline_tactics": v["closed_by_baseline_tactics"],
                                    "n_deps_transitive": v["n_deps_transitive"]}, sort_keys=True) + "\n")
    with open(os.path.join(out_dir, "test_sets.json"), "w") as f:
        json.dump({"stages": [sid for sid, _ in stream.stages], "test_sets": {str(j): ids for j, ids in test_sets.items()}},
                  f, indent=1)
    backend = stream.backend()
    checker = CachedChecker(backend, VerdictCache(None))

    # ---- hold-out guard (R4): built over the stream and the prelude world
    worlds = dict(stream.worlds)
    worlds["toy-prelude"] = stream.prelude.world
    guard = HoldoutGuard(stream.obligations, worlds, min_tokens=cfg.get("guard_min_tokens", 8))
    guard_log = {"checked": 0, "raised": 0, "by_stage": {}}

    def guard_fn(ex):
        guard_log["checked"] += 1
        try:
            guard.check(ex)
        except LeakError:
            guard_log["raised"] += 1
            raise

    windows = {}
    for t, (sid, _) in enumerate(stream.stages, 1):
        windows[t] = [wake_example(o) for o in stream.obligations if o["snapshot_id"] == sid and o["split"] == "wake"]
    train_ids = {i for exs in windows.values() for e in exs for i in e["source_ids"]}
    eval_ids = {i for ids in test_sets.values() for i in ids}
    train_test_statements = sum(1 for exs in windows.values() for e in exs if guard.is_test_statement(e["statement"]))
    leak_canaries = []
    for j in (1, T):                            # inject a test obligation as a wake example: must raise
        o = full[test_sets[j][0]]
        try:
            guard.check(wake_example(o))
            leak_canaries.append({"id": o["id"], "raised": False})
        except LeakError as e:
            leak_canaries.append({"id": o["id"], "raised": True, "rule": e.rule, "detail": e.detail})
    guard_static = {"train_ids": len(train_ids), "eval_ids": len(eval_ids),
                    "train_eval_id_overlap": len(train_ids & eval_ids),
                    "train_examples_with_test_statement": train_test_statements,
                    "leak_canaries": leak_canaries,
                    "distinctive_test_proofs": guard.n_distinctive_proofs}
    if guard_static["train_eval_id_overlap"] or train_test_statements or not all(c["raised"] for c in leak_canaries):
        raise StructuralError(f"hold-out guard pre-check failed: {guard_static}")

    canary = canary_oracle(checker, stream, views_by_j, T, cfg["eval"]["k"], out_dir)

    # ---- base model: pretrained from scratch on the prelude project only (not in the stream)
    tok = Tokenizer(KEYWORDS)
    bc = cfg["base"]
    base_path = os.path.join(out_dir, "base", "base.pt")
    pre = [{"kind": "wake", "prompt": r.statement.show(), "target": r.proof, "statement": r.statement.show(),
            "source_ids": [f"toy-prelude:{r.name}"], "source": "human"} for r in stream.prelude.lemmas]
    # Base material the guard rejects is excluded (conservative) and listed in meta.json. Stream wake
    # examples are never filtered: a guard raise there aborts the run.
    pre_excluded = []
    kept = []
    for ex in pre:
        try:
            guard.check(ex)
            kept.append(ex)
        except LeakError as e:
            pre_excluded.append({"source_ids": ex["source_ids"], "rule": e.rule, "detail": e.detail})
    pre = kept
    guard_static["base_examples_excluded"] = pre_excluded
    model = make_model(tok, cfg, seed)
    if os.path.exists(base_path) and cfg.get("reuse_base", True):
        model.load_state_dict(torch.load(base_path))
        base_stats = json.load(open(os.path.join(out_dir, "base", "train.json")))
        base_stats["reused"] = True
    else:
        steps = epochs_to_steps(len(pre), bc["epochs"], bc["batch"])
        base_stats = train_steps(model, tok, list(model.parameters()), pre, steps=steps, batch=bc["batch"],
                                 lr=bc["lr"], seed=seed, guard=guard_fn, weight_decay=bc.get("weight_decay", 0.0),
                                 log_every=200)
        base_stats.update(n_examples=len(pre), params=model.n_params(), base_hash=base_state_hash(model),
                          excluded=pre_excluded)
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        torch.save(model.state_dict(), base_path)
        with open(os.path.join(out_dir, "base", "train.json"), "w") as f:
            json.dump(base_stats, f, indent=1, sort_keys=True)
    base_state = copy.deepcopy(model.state_dict())
    base_hash = base_state_hash(model)
    print(f"base: {model.n_params()} params, hash {base_hash}, loss {base_stats.get('last_loss')}", flush=True)

    arms = arms or cfg["arms"]
    arm_meta = {}
    full_mode = cfg["adapt"].get("mode", "lora") == "full"

    def base_state_hash_of(state):
        tmp = make_model(tok, cfg, seed)
        tmp.load_state_dict(state)
        return base_state_hash(tmp)
    for arm in arms:
        ta = time.time()
        run = f"S1/{arm}/tiny-{base_hash}/s{seed}"
        adir = os.path.join(out_dir, arm)
        os.makedirs(adir, exist_ok=True)
        m = make_model(tok, cfg, seed)
        m.load_state_dict(base_state)
        trainer = None if arm == "B0" else make_trainer(arm, m, tok, cfg, seed, guard_fn)
        vlog = Logger(os.path.join(adir, "verdicts.jsonl.gz"))
        tlog = Logger(os.path.join(adir, "train.jsonl"))
        elog = Logger(os.path.join(adir, "events.jsonl"))
        hashes = {}
        events = []
        history = []
        gen_total = 0.0
        train_total = 0.0
        for t in range(0, T + 1):
            if t >= 1:
                g0 = guard_log["checked"]
                if trainer is not None:
                    t1 = time.time()
                    st = trainer.train_stage(t, windows[t], history)
                    train_total += time.time() - t1
                else:
                    st = {"steps": 0, "n_examples": 0, "n_window": len(windows[t]), "n_replay": 0,
                          "history_bytes_cum": 0, "train_tokens": 0}
                history = history + windows[t]
                st.update(stage=t, snapshot_id=stream.stages[t - 1][0], guard_checked=guard_log["checked"] - g0,
                          base_hash=base_state_hash(m), adapter_hash=adapter_hash(m))
                if (trainer is None or full_mode is False) and st["base_hash"] != base_hash:
                    raise StructuralError(f"{arm} stage {t}: the base changed ({st['base_hash']} != {base_hash})")
                if base_state_hash_of(base_state) != base_hash:
                    raise StructuralError(f"{arm} stage {t}: the stored base state was modified")
                if trainer is not None and st["guard_checked"] != st["n_examples"]:
                    raise StructuralError(f"{arm} stage {t}: {st['n_examples']} examples, {st['guard_checked']} guard checks")
                tlog.write(st)
                ev = {"type": "train", "stage": t}
                events.append(ev)
                elog.write(ev)
            cells = cells_for_stage(t, T)
            for j in cells:
                hashes[f"{t},{j}"] = eval_set_hash(views_by_j[j])
            lines, gs = eval_stage(m, tok, checker, run, t, cells, views_by_j, cfg, vlog)
            gen_total += gs
            check_stage_lines(lines, t, cells, test_sets, cfg["eval"]["k"])
            for j in cells:
                ev = {"type": "eval", "model_stage": t, "set": j}
                events.append(ev)
                elog.write(ev)
            HoldoutGuard.check_eval_order(events)          # L4
            acc = {j: round(v, 4) for (tt, j), v in M.accuracy_matrix(lines, {j: test_sets[j] for j in cells},
                                                                          cfg["eval"]["k"], run=run).items()}
            print(f"[{arm}] stage {t}: {acc} ({time.time() - ta:.0f}s)", flush=True)
            if trainer is not None and cfg.get("save_checkpoints", True):
                os.makedirs(os.path.join(adir, "ckpt"), exist_ok=True)
                sd = m.state_dict() if full_mode else \
                    {k: v for k, v in m.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
                torch.save(sd, os.path.join(adir, "ckpt", f"stage{t:02d}.pt"))          # R12
        vlog.close(); tlog.close(); elog.close()
        with open(os.path.join(adir, "evalsets.json"), "w") as f:
            json.dump(hashes, f, indent=1, sort_keys=True)
        arm_meta[arm] = {"run": run, "wall_s": round(time.time() - ta, 1), "train_s": round(train_total, 1),
                         "generate_s": round(gen_total, 1),
                         "history_bytes": getattr(trainer, "history_bytes", 0) if trainer else 0,
                         "trainable_params": m.n_params(trainable_only=True) if trainer else 0}
        print(f"[{arm}] done in {arm_meta[arm]['wall_s']}s", flush=True)

    meta_path = os.path.join(out_dir, "meta.json")
    old = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    old_arms = old.get("arms", {}) if old.get("data_version") == winfo["data_version"] else {}
    old_arms.update(arm_meta)
    import numpy, yaml
    meta = {
        "experiment": "S1", "seed": seed, "config": cfg, "code_commit": git_commit(),
        "data_version": winfo["data_version"], "stream": winfo, "stages": [s for s, _ in stream.stages],
        "stream_build_s": round(t_stream, 1),
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": numpy.__version__,
                     "pyyaml": yaml.__version__, "platform": platform.platform()},
        "toolchain": {"toyeq_checker_sha": file_sha(os.path.join(HERE, "backends", "toyeq.py"))},
        "threads": torch.get_num_threads(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "base": {"hash": base_hash, "params": model.n_params(), "stats": base_stats,
                 "pretraining": "toy-prelude project only (from scratch, no pretrained weights)"},
        "tokenizer": {"vocab": tok.vocab_size, "keywords": KEYWORDS},
        "guard": {**guard_static, "checked": guard_log["checked"], "raised": guard_log["raised"]},
        "canary": canary, "arms": old_arms,
        "guard_note": "training examples checked by HoldoutGuard.check before use; any raise aborts the run",
    }
    if old.get("guard") and old.get("data_version") == winfo["data_version"]:
        meta["guard"]["checked_previous_runs"] = old["guard"].get("checked", 0) + old["guard"].get("checked_previous_runs", 0)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True, default=str)
    print(f"S1 finished in {time.time() - T0:.0f}s", flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default=None)
    a = ap.parse_args(argv)
    import yaml
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("experiment") != "S1":
        raise SystemExit("config is not an S1 config")
    run_all(cfg, out_dir=a.out, arms=a.arms.split(",") if a.arms else None)


if __name__ == "__main__":
    main()
