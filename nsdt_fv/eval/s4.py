"""S4 driver (plan 8.1 S4, 9.4 P0 on the toy): build the property tables, label them with the checker, hide 20%
of the true pairs, rank the candidates with the Appendix E fragment and three baselines, and log everything the
report needs (rule R3: reports/S4.md is generated from runs/S4 by eval/report_s4.py).

    python3 -m nsdt_fv.eval.s4 [--stage table|label|rank|all]

Outputs in runs/S4/:
  meta.json                 versions, commits, config, timings, counts
  shapes.json               the shape vocabulary with C's support (functions it was seen for)
  cells.jsonl.gz            one line per cell: id, project, function, shape, statement, group (statement key)
  labels.jsonl.gz           one line per statement group: checker verdict of the found proof, the proof, the
                            descriptive counterexample column, prover time
  split_s<seed>/hidden.json the hidden true groups and their cells (seed recorded)
  split_s<seed>/scores.jsonl.gz   one line per candidate: label, scores of PLN (s, n, P), frequency, NN, random
  split_s<seed>/leak.json   the hidden-pairs-never-reach-PLN check
Checkpoint: labels are written per project and reused when present (R12).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml

from ..backends.toyeq import statement_key
from ..backends.toyeq_oracle import counterexample
from ..data.holdout import HoldoutGuard
from ..data.stream import BASELINE_TACTICS, build_stream
from ..data.toyeq_prover import RefProver
from ..store.pln import ref
from ..store.pln.table import Naming, base_defs, channel_c_shapes, instantiate, private_world, sha12

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--", "nsdt_fv"], cwd=HERE, text=True)
        return [l for l in out.splitlines() if "runs/S4" not in l]
    except Exception:
        return ["unknown"]


def wjsonl(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def rjsonl(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


# ---------------------------------------------------------------------------------------------- table
def build_table(cfg, out):
    s = build_stream(seed=cfg["stream"]["seed"], n_projects=cfg["stream"]["projects"])
    nm = Naming(s)
    bdefs = base_defs(s, nm)
    occ, n_stmt = channel_c_shapes(s, nm, os.path.join(HERE, cfg["shapes"]["source"]))
    vocab = sorted(t for t, fs in occ.items() if len(fs) >= int(cfg["shapes"]["min_support"]))
    worlds = dict(s.worlds)
    worlds["toy-prelude"] = s.prelude.world
    guard = HoldoutGuard(s.obligations, worlds, min_tokens=8)
    cells, dropped_test = [], 0
    for p in s.projects:
        w = s.worlds[f"{p.name}@w2"]
        for f in sorted(w.funcs):
            for T in vocab:
                text = instantiate(T, f, p.name, nm, w)
                if text is None:
                    continue
                if guard.is_test_statement(text):
                    dropped_test += 1
                    continue
                key = statement_key(w.statement(text))
                cells.append({"project": p.name, "function": f, "shape": T, "statement": text,
                              "group": p.name + "|" + key})
    for i, c in enumerate(cells):
        c["id"] = i
    wjsonl(os.path.join(out, "cells.jsonl.gz"), cells)
    with open(os.path.join(out, "shapes.json"), "w") as f:
        json.dump({"statements_read": n_stmt, "shapes_seen": len(occ), "min_support": cfg["shapes"]["min_support"],
                   "vocabulary": [{"shape": t, "support": len(occ[t]),
                                   "functions": sorted(f"{a}:{b}" for a, b in occ[t])} for t in vocab],
                   "template_functions": bdefs}, f, indent=1, sort_keys=True)
    info = {"stream_data_version": s.data_version(), "statements_read": n_stmt, "shapes_seen": len(occ),
            "vocabulary": len(vocab), "cells": len(cells), "groups": len({c["group"] for c in cells}),
            "functions": len({(c["project"], c["function"]) for c in cells}),
            "functions_in_projects": sum(len(s.worlds[f"{p.name}@w2"].funcs) for p in s.projects),
            "cells_dropped_test_statement": dropped_test}
    return info


# ---------------------------------------------------------------------------------------------- labels
_W = {}


def _init_worker(stream_seed, n_projects):
    s = build_stream(seed=stream_seed, n_projects=n_projects)
    test_decls = {(o["repo"], o["decl"]) for o in s.obligations if o["split"] == "test"}
    _W["s"] = s
    _W["be"] = s.backend()
    _W["pw"] = {p.name: private_world(s, p.name, test_decls) for p in s.projects}


def _label(job):
    """Run one labelling job in a thread with a 1 GiB stack: deep unary-number recursions (e.g. tri(tri(tri(n))))
    overflow the C stack of the main thread at ToyEq's recursion limit and kill the process (observed in the first
    S4 run; logs/aborted). A RecursionError is recorded, not raised."""
    import threading
    box = {}

    def run():
        try:
            box["r"] = _label_job(job)
        except BaseException as e:          # noqa: BLE001 - reported to the parent
            box["e"] = e

    threading.stack_size(1 << 30)
    th = threading.Thread(target=run)
    th.start()
    th.join()
    if "e" in box:
        raise box["e"]
    return box["r"]


def _label_job(job):
    group, proj, text, nodes, n_cex, cex_seed = job
    s, be = _W["s"], _W["be"]
    sid = f"{proj}@w2"
    w = s.worlds[sid]
    st = w.statement(text)
    t0 = time.time()
    try:
        cex = counterexample(w, st, n=n_cex, seed=cex_seed)
    except RecursionError:
        cex = {"_limit": True}
    proof, how = None, None
    for h in BASELINE_TACTICS:
        if be.check_statement(sid, "", text, h, 120).accepted:
            proof, how = h, "baseline"
            break
    if proof is None:
        pw = _W["pw"][proj]
        try:
            p = RefProver(pw, len(pw.lemmas), node_limit=nodes).prove(pw.statement(text))
        except RecursionError:
            p, how = None, "prover_recursion"
        if p is not None:
            if be.check_statement(sid, "", text, p, 120).accepted:
                proof, how = p, "prover"
            else:
                how = "prover_rejected"
    return {"group": group, "project": proj, "statement": text, "accepted": proof is not None, "proof": proof,
            "how": how, "counterexample": (None if cex is None else ("limit" if cex.get("_limit") else
                                                                       {k: repr(v) for k, v in cex.items()})),
            "elapsed_s": round(time.time() - t0, 4)}


def label_cells(cfg, out):
    cells = rjsonl(os.path.join(out, "cells.jsonl.gz"))
    groups = {}
    for c in cells:
        groups.setdefault(c["group"], (c["project"], c["statement"]))
    lab = cfg["label"]
    ldir = os.path.join(out, "labels")
    os.makedirs(ldir, exist_ok=True)
    projects = sorted({p for p, _ in groups.values()})
    t0 = time.time()
    with Pool(int(lab["workers"]), initializer=_init_worker,
              initargs=(cfg["stream"]["seed"], cfg["stream"]["projects"])) as pool:
        for proj in projects:
            path = os.path.join(ldir, f"{proj}.jsonl.gz")
            if os.path.exists(path):
                print(f"[S4] labels for {proj}: checkpoint present, reused", flush=True)
                continue
            jobs = [(g, p, t, int(lab["prover_nodes"]), int(lab["cex_inputs"]), int(lab["cex_seed"]))
                    for g, (p, t) in sorted(groups.items()) if p == proj]
            rows = pool.map(_label, jobs, chunksize=8)
            wjsonl(path + ".tmp", rows)
            os.replace(path + ".tmp", path)
            print(f"[S4] labels for {proj}: {len(rows)} groups, {sum(r['accepted'] for r in rows)} accepted, "
                  f"{time.time() - t0:.0f}s", flush=True)
    rows = []
    for proj in projects:
        rows += rjsonl(os.path.join(ldir, f"{proj}.jsonl.gz"))
    rows.sort(key=lambda r: r["group"])
    wjsonl(os.path.join(out, "labels.jsonl.gz"), rows)
    return {"label_wall_s": round(time.time() - t0, 1), "groups_labelled": len(rows),
            "label_core_s": round(sum(r["elapsed_s"] for r in rows), 1)}


# ---------------------------------------------------------------------------------------------- ranking
def hide(true_groups, frac, seed):
    """Hidden set: round(frac * n) of the sorted true groups, chosen by random.Random(seed)."""
    import random
    g = sorted(true_groups)
    rng = random.Random(seed)
    rng.shuffle(g)
    return sorted(g[:int(round(frac * len(g)))])


class VisibleTable:
    """The only object the rankers see: visible true cells (kernel facts) and the applicability of shapes.
    Built from the cell list and the set of visible true cell ids; it holds no label and no hidden cell."""

    def __init__(self, cells, visible_ids):
        self.funcs = sorted({(c["project"], c["function"]) for c in cells})
        self.applicable = {}                      # shape -> set of functions where its instance exists
        for c in cells:
            self.applicable.setdefault(c["shape"], set()).add((c["project"], c["function"]))
        self.rows = {f: {} for f in self.funcs}   # function -> {shape: visible cell id}
        for c in cells:
            if c["id"] in visible_ids:
                self.rows[(c["project"], c["function"])][c["shape"]] = c["id"]
        self.holders = {}
        for f, row in self.rows.items():
            for T in row:
                self.holders.setdefault(T, set()).add(f)
        self.freq = {T: len(self.holders.get(T, ())) / len(fs) for T, fs in self.applicable.items()}
        self.premise_ids = {cid for row in self.rows.values() for cid in row.values()}


def score_frequency(vt, cands):
    return [vt.freq[c["shape"]] for c in cands]


def score_nn(vt, cands):
    """Nearest neighbour by shared properties (1-NN vote): among the other functions where T applies, the nearest
    neighbours of f are those sharing the largest number of visible properties with f; the score is the share of
    them that have T visible (ties between equally near neighbours are averaged, so the score is a 1-NN estimate
    of P(HasProperty(f, T)) in [0, 1])."""
    out = []
    for c in cands:
        f = (c["project"], c["function"])
        Sf = set(vt.rows[f])
        hold = vt.holders.get(c["shape"], set())
        best, near = -1, []
        for g in sorted(vt.applicable[c["shape"]]):
            if g == f:
                continue
            k = len(Sf & set(vt.rows[g]))
            if k > best:
                best, near = k, [g]
            elif k == best:
                near.append(g)
        out.append(sum(1 for g in near if g in hold) / len(near) if near else 0.0)
    return out


def score_random(cands, seed):
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.random(len(cands))]


class PLNRanker:
    """Appendix E on the visible table: transfer (Similar(f, g), HasProperty(g, T)) and deduction through
    Implies(T1, T) induced over the table, merged by revision with the premise-overlap discount; P(q true) is the
    revised strength shrunk toward sT by (1 - c). Premise ids are visible cell ids (bits of a Python int)."""

    def __init__(self, vt, d=ref.D_DED):
        self.vt = vt
        self.d = d
        self.w = ref.shape_weights(vt.freq)
        self.N = len(vt.funcs)
        self._bits = {f: sum(1 << cid for cid in row.values()) for f, row in vt.rows.items()}
        self._imp = {}
        self.used = 0                             # union of all premise bits used (leak check)

    def implies(self, T1, T):
        """Implies(T1, T) by direct evaluation over the visible table: among functions where both shapes apply
        and T1 is visible, the share that has T visible; count = their number; premises = those cells."""
        k = (T1, T)
        if k not in self._imp:
            h1 = self.vt.holders.get(T1, set()) & self.vt.applicable[T]
            both = h1 & self.vt.holders.get(T, set())
            bits = 0
            for g in h1:
                bits |= 1 << self.vt.rows[g][T1]
            for g in both:
                bits |= 1 << self.vt.rows[g][T]
            self._imp[k] = (h1, len(both), bits)
        return self._imp[k]

    def estimate(self, f, T):
        vt = self.vt
        sT = vt.freq[T]
        Sf = vt.rows[f]
        concl = []
        # transfer from every function g that has T
        for g in sorted(vt.holders.get(T, ())):
            if g == f:
                continue
            sig = ref.similarity(set(Sf), set(vt.rows[g]), self.w)
            if sig <= 0.0:
                continue
            n_sim = len(set(Sf) | set(vt.rows[g]))
            sim = ref.TV(sig, float(n_sim), self._bits[f] | self._bits[g])
            ghas = ref.kernel_proof(1 << vt.rows[g][T])
            concl.append(ref.transfer(sim, ghas, 1.0 / self.N, sT, self.d))
        # deduction HasProperty(f, T1), Implies(T1, T) |- HasProperty(f, T)
        for T1 in sorted(Sf):
            if T1 == T:
                continue
            h1, n_both, bits = self.implies(T1, T)
            n1 = len(h1) - (1 if f in h1 else 0)        # f's own row is not evidence about f's T
            if n1 <= 0:
                continue
            imp = ref.TV(n_both / n1, float(n1), bits)
            fact = ref.kernel_proof(1 << Sf[T1])
            concl.append(ref.deduction(fact, imp, vt.freq[T1], sT, self.d))
        tv = ref.revise_all(concl) if concl else None
        if tv is not None:
            tv.check()
            self.used |= tv.premises
        return tv, len(concl), ref.p_true(tv, sT)


def rank_split(cfg, out, cells, labels, seed):
    acc = {g: r["accepted"] for g, r in labels.items()}
    true_groups = sorted(g for g, a in acc.items() if a)
    hidden = set(hide(true_groups, float(cfg["split"]["hide_fraction"]), seed))
    sdir = os.path.join(out, f"split_s{seed}")
    os.makedirs(sdir, exist_ok=True)
    visible_ids = {c["id"] for c in cells if acc[c["group"]] and c["group"] not in hidden}
    hidden_ids = {c["id"] for c in cells if c["group"] in hidden}
    with open(os.path.join(sdir, "hidden.json"), "w") as f:
        json.dump({"seed": seed, "hide_fraction": cfg["split"]["hide_fraction"], "true_groups": len(true_groups),
                   "hidden_groups": sorted(hidden), "hidden_cells": sorted(hidden_ids),
                   "sha": sha12(json.dumps(sorted(hidden)))}, f, indent=1)
    # the rankers' input: cells without labels + visible ids
    bare = [{k: c[k] for k in ("id", "project", "function", "shape")} for c in cells]
    vt = VisibleTable(bare, visible_ids)
    cands = [c for c in bare if c["id"] not in visible_ids]
    t0 = time.time()
    pln = PLNRanker(vt)
    rows = []
    s_freq = score_frequency(vt, cands)
    s_nn = score_nn(vt, cands)
    s_rand = score_random(cands, int(cfg["random_baseline_seed"]) + seed)
    for c, fq, nn, rd in zip(cands, s_freq, s_nn, s_rand):
        tv, nconc, p = pln.estimate((c["project"], c["function"]), c["shape"])
        rows.append({"id": c["id"], "function": c["project"] + ":" + c["function"], "shape": c["shape"],
                     "label": int(acc[cells[c["id"]]["group"]]), "hidden": int(c["id"] in hidden_ids),
                     "pln_p": round(p, 12), "pln_s": None if tv is None else round(tv.s, 12),
                     "pln_n": None if tv is None else round(tv.n, 9), "pln_conclusions": nconc,
                     "frequency": round(fq, 12), "nn": nn, "random": round(rd, 12)})
    t_rank = time.time() - t0
    wjsonl(os.path.join(sdir, "scores.jsonl.gz"), rows)
    # leak check: hidden cells vs everything the PLN ranker read
    used = set()
    u, i = pln.used, 0
    while u:
        if u & 1:
            used.add(i)
        u >>= 1
        i += 1
    groups_vis = {cells[i]["group"] for i in visible_ids}
    leak = {"seed": seed, "visible_true_cells": len(visible_ids), "hidden_cells": len(hidden_ids),
            "candidates": len(cands), "candidate_positives": sum(r["label"] for r in rows),
            "candidate_hidden": sum(r["hidden"] for r in rows),
            "premise_cells_used_by_pln": len(used),
            "premise_cells_outside_visible": len(used - visible_ids),
            "premise_cells_hidden": len(used & hidden_ids),
            "table_cells_hidden": len(vt.premise_ids & hidden_ids),
            "hidden_groups_with_a_visible_cell": len(hidden & groups_vis),
            "hidden_cells_not_candidates": len(hidden_ids - {c["id"] for c in cands}),
            "candidate_positives_not_hidden": sum(1 for r in rows if r["label"] and not r["hidden"]),
            "rank_s": round(t_rank, 1)}
    leak["passed"] = (leak["premise_cells_outside_visible"] == 0 and leak["premise_cells_hidden"] == 0
                      and leak["table_cells_hidden"] == 0 and leak["hidden_groups_with_a_visible_cell"] == 0
                      and leak["hidden_cells_not_candidates"] == 0 and leak["candidate_positives_not_hidden"] == 0)
    with open(os.path.join(sdir, "leak.json"), "w") as f:
        json.dump(leak, f, indent=1, sort_keys=True)
    if not leak["passed"]:
        raise RuntimeError(f"S4 leak check failed: {leak}")
    return leak


def canary(cfg):
    """Known-answer canary through the production ranking path: a synthetic table where f0 and f1 share every
    property but T*, which f1 has and f0's candidate is the hidden one; f2 shares nothing. PLN must rank (f0, T*)
    above (f2, T*), and the leak check machinery must see no hidden premise."""
    cells, i = [], 0
    funcs = ["f0", "f1", "f2", "f3"]
    rows = {"f0": ["A", "B", "C", "T*"], "f1": ["A", "B", "C", "T*"], "f2": ["D", "T*x"], "f3": ["D", "A"]}
    for f in funcs:
        for T in ["A", "B", "C", "D", "T*", "T*x"]:
            cells.append({"id": i, "project": "p", "function": f, "shape": T, "statement": f"{f}|{T}",
                          "group": f"p|{f}|{T}"})
            i += 1
    true_ids = {c["id"] for c in cells if c["shape"] in rows[c["function"]]}
    hidden = {c["id"] for c in cells if c["function"] == "f0" and c["shape"] == "T*"}
    vt = VisibleTable(cells, true_ids - hidden)
    pln = PLNRanker(vt)
    p0 = pln.estimate(("p", "f0"), "T*")[2]
    p2 = pln.estimate(("p", "f2"), "T*")[2]
    used_hidden = any((pln.used >> h) & 1 for h in hidden)
    return {"p_f0_Tstar": round(p0, 6), "p_f2_Tstar": round(p2, 6), "prior_Tstar": round(vt.freq["T*"], 6),
            "used_hidden": used_hidden, "passed": bool(p0 > p2 and p0 > vt.freq["T*"] and not used_hidden)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "S4.yaml"))
    ap.add_argument("--stage", default="all", choices=["table", "label", "rank", "all"])
    a = ap.parse_args(argv)
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    out = os.path.join(HERE, cfg["run_dir"])
    os.makedirs(out, exist_ok=True)
    mpath = os.path.join(out, "meta.json")
    meta = json.load(open(mpath)) if os.path.exists(mpath) else {}
    meta.setdefault("runs", [])
    run = {"stage": a.stage, "commit": git_commit(), "dirty": git_dirty(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "python": platform.python_version(), "numpy": np.__version__, "pyyaml": yaml.__version__,
           "platform": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0))}
    t0 = time.time()
    meta["config"] = cfg
    if a.stage in ("table", "all"):
        meta["table"] = build_table(cfg, out)
        print("[S4] table", meta["table"], flush=True)
    if a.stage in ("label", "all"):
        meta["label"] = label_cells(cfg, out)
        print("[S4] label", meta["label"], flush=True)
    if a.stage in ("rank", "all"):
        cells = rjsonl(os.path.join(out, "cells.jsonl.gz"))
        labels = {r["group"]: r for r in rjsonl(os.path.join(out, "labels.jsonl.gz"))}
        meta["canary"] = canary(cfg)
        print("[S4] canary", meta["canary"], flush=True)
        if not meta["canary"]["passed"]:
            raise RuntimeError("S4 canary failed")
        meta["splits"] = {}
        for seed in cfg["split"]["seeds"]:
            meta["splits"][str(seed)] = rank_split(cfg, out, cells, labels, int(seed))
            print("[S4] split", seed, meta["splits"][str(seed)], flush=True)
    run["wall_s"] = round(time.time() - t0, 1)
    meta["runs"].append(run)
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
