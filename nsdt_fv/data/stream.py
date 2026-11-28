"""The ToyEq stream (plan C.3 and 5.2): six projects with two windows each, plus a prelude project
that is not in the stream (base pretraining material). Produces obligations (Appendix B), store
edges, snapshots as ToyEq Worlds, and a ToyEqBackend over them."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass

from ..backends.toyeq import ToyEqBackend, World, check_proof, parse_proof, symbols
from .schemas import validate_obligation, validate_edge
from .toyeq_gen import Project, build_project
from .toyeq_templates import PRELUDE_TEMPLATE, STREAM_TEMPLATES

TEST_FRACTION = 0.2          # section 5.2: the last 20% of a window's new theorems are its test set
BASELINE_TACTICS = ("refl", "simp refl")   # toy analogue of the baseline closing-tactic list (section 7)


def proof_lemmas(proof: str) -> list:
    out = []
    for kind, arg, blocks, _ in _walk(parse_proof(proof)):
        if kind in ("rw", "rw<-") and not arg.startswith("IH") and arg not in out:
            out.append(arg)
    return out


def _walk(steps):
    for s in steps:
        yield s
        if s[2]:
            for b in s[2]:
                yield from _walk(b)


@dataclass
class Stream:
    projects: list                  # Project (stream projects, in template order)
    prelude: Project
    obligations: list               # dicts (full views, with proofs)
    edges: list
    worlds: dict                    # snapshot_id -> World
    stages: list                    # [(snapshot_id, window end date)] in stream order
    config: dict

    def backend(self) -> ToyEqBackend:
        return ToyEqBackend(self.worlds, {o["id"]: (o["snapshot_id"], o["decl"]) for o in self.obligations})

    def by_id(self):
        return {o["id"]: o for o in self.obligations}

    def test_sets(self) -> dict:
        """Stage index j (1-based) -> list of test obligation ids E_j."""
        out = {}
        for j, (sid, _) in enumerate(self.stages, 1):
            out[j] = [o["id"] for o in self.obligations if o["snapshot_id"] == sid and o["split"] == "test"]
        return out

    def data_version(self) -> str:
        h = hashlib.sha256()
        for o in self.obligations:
            h.update(json.dumps(o, sort_keys=True).encode())
        return h.hexdigest()[:16]


def test_view(o: dict) -> dict:
    v = dict(o)
    v.pop("proof", None)
    return v


def snapshot_world(p: Project, window: int) -> World:
    sid = f"{p.name}@w{window}"
    w = World(sid)
    for name, fd in p.world.funcs.items():
        if p.defs_phase[name] <= window:
            w.add_def(fd)
    upto = p.cut if window == 1 else len(p.lemmas)
    for rec in p.lemmas[:upto]:
        w.add_lemma(rec.name, rec.statement, rec.proof)
    return w


def _closure(start, graph):
    seen = set()
    todo = list(graph.get(start, ()))
    while todo:
        x = todo.pop()
        if x in seen:
            continue
        seen.add(x)
        todo.extend(graph.get(x, ()))
    return seen


def build_stream(seed: int = 0, sizes=None, prefixes=None, n_projects: int = 6) -> Stream:
    rng = random.Random(seed)
    templates = STREAM_TEMPLATES[:n_projects]
    projects = []
    for i, tpl in enumerate(templates):
        target = sizes[i] if sizes else rng.randint(150, 250)
        prefix = prefixes[i] if prefixes else f"{'abcdef'[i]}{tpl[0]}_"
        projects.append(build_project(f"toy-{tpl.replace('_', '-')}", tpl, prefix, seed * 1000 + i, target))
    prelude = build_project("toy-prelude", PRELUDE_TEMPLATE, "pre_", seed * 1000 + 99, 160)
    # window end dates: window 1 in the first half of 2025, window 2 in the second half
    base = dt.date(2025, 1, 1)
    ends = {}
    for p in projects:
        d1 = base + dt.timedelta(days=rng.randint(20, 170))
        d2 = base + dt.timedelta(days=rng.randint(200, 350))
        ends[p.name] = (d1, d2)
    stages = sorted([(f"{p.name}@w{k}", ends[p.name][k - 1]) for p in projects for k in (1, 2)],
                    key=lambda x: (x[1], x[0]))
    worlds = {}
    obligations = []
    edges = []
    for p in projects:
        for k in (1, 2):
            w = snapshot_world(p, k)
            worlds[w.snapshot_id] = w
        commit = {k: hashlib.sha256(worlds[f"{p.name}@w{k}"].source(with_proofs=True).encode()).hexdigest()[:12]
                  for k in (1, 2)}
        graph = {}
        for fname, fd in p.world.funcs.items():
            graph[fname] = sorted(set(symbols(fd.rhs) if fd.k is None else
                                      set().union(*[symbols(c[2]) for c in fd.cases.values()])) & set(p.world.funcs) - {fname})
        for rec in p.lemmas:
            graph[rec.name] = sorted((symbols(rec.statement.lhs) | symbols(rec.statement.rhs)) & set(p.world.funcs)) \
                + proof_lemmas(rec.proof)
        windows = {1: p.lemmas[:p.cut], 2: p.lemmas[p.cut:]}
        full2 = worlds[f"{p.name}@w2"]
        for k, recs in windows.items():
            sid = f"{p.name}@w{k}"
            wk = worlds[sid]
            n_test = max(1, int(round(TEST_FRACTION * len(recs))))
            d_start = base if k == 1 else ends[p.name][0]
            d_end = ends[p.name][k - 1]
            span = max(1, (d_end - d_start).days)
            test_names = {r.name for r in recs[len(recs) - n_test:]}
            for i, rec in enumerate(recs):
                pos = wk.lemma_pos[rec.name]
                funcs_used = sorted((symbols(rec.statement.lhs) | symbols(rec.statement.rhs)) & set(p.world.funcs))
                deps = funcs_used + proof_lemmas(rec.proof)
                trans = _closure(rec.name, graph)
                trivial = any(check_proof(wk, rec.statement, t, pos).accepted for t in BASELINE_TACTICS)
                first_seen = d_start + dt.timedelta(days=int(span * (i + 1) / (len(recs) + 1)))
                o = {
                    "id": f"{p.name}@{commit[k]}:{rec.name}",
                    "repo": p.name, "commit": commit[k], "snapshot_id": sid,
                    "module": p.name.replace("-", "_"), "decl": rec.name, "kind": "theorem",
                    "statement": rec.statement.show(),
                    "src_up_to_decl": wk.source(upto=pos, with_proofs=False),
                    "scope": f"project {p.name}; prefix {p.prefix}",
                    "deps_project": deps, "deps_library": [],
                    "n_deps_direct": len(deps), "n_deps_transitive": len(trans),
                    "proof": rec.proof,
                    "first_seen": first_seen.isoformat(), "window": k,
                    "split": "test" if rec.name in test_names else "wake",
                    "closed_by_baseline_tactics": bool(trivial),
                }
                validate_obligation(o)
                obligations.append(o)
        for k in (1, 2):
            sid = f"{p.name}@w{k}"
            wk = worlds[sid]
            for fname in wk.funcs:
                for dst in graph[fname]:
                    edges.append({"snapshot_id": sid, "src": fname, "dst": dst, "type": "unfolds_to"})
            for name, st in wk.lemmas:
                for dst in sorted((symbols(st.lhs) | symbols(st.rhs)) & set(wk.funcs)):
                    edges.append({"snapshot_id": sid, "src": name, "dst": dst, "type": "statement_uses"})
                for dst in proof_lemmas(wk.proofs[name]):
                    edges.append({"snapshot_id": sid, "src": name, "dst": dst, "type": "proof_uses"})
    for e in edges:
        validate_edge(e)
    cfg = {"seed": seed, "sizes": [len(p.lemmas) for p in projects], "test_fraction": TEST_FRACTION}
    return Stream(projects, prelude, obligations, edges, worlds, stages, cfg)


def write_stream(s: Stream, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "obligations.jsonl"), "w") as f:
        for o in s.obligations:
            f.write(json.dumps(o, sort_keys=True) + "\n")
    with open(os.path.join(out_dir, "edges.jsonl"), "w") as f:
        for e in s.edges:
            f.write(json.dumps(e, sort_keys=True) + "\n")
    with open(os.path.join(out_dir, "stream.json"), "w") as f:
        json.dump({"stages": [[sid, d.isoformat()] for sid, d in s.stages], "config": s.config,
                   "data_version": s.data_version()}, f, indent=1)
    os.makedirs(os.path.join(out_dir, "projects"), exist_ok=True)
    for p in s.projects + [s.prelude]:
        with open(os.path.join(out_dir, "projects", p.name + ".toyeq"), "w") as f:
            f.write(p.world.source(with_proofs=True) + "\n")
    return {"obligations": len(s.obligations), "edges": len(s.edges), "data_version": s.data_version()}



# =============================================================================================== Lean
# C1-prep: the same stream, built from a Lean repository's history (plan 5.2) instead of the toy generator.
# Nothing above this line changed. The Lean functions below use nothing from ToyEq (the module still imports
# it at the top for the toy path).
#
# Inputs are history records of bench/vsb_history.py (one per repository): the first-parent commit sequence
# up to the pinned commit c*_r followed by the forward commits, and for every declaration key (qualified name,
# statement hash) its presence intervals over that sequence. Texts (statement, proof, file prefix) come from a
# materializer: PlaceholderMaterializer (metadata only, synthetic Lean-like texts; used by the tests and
# before Lean exists), eval/c1prep.SourceScanMaterializer (local clones, source-level regex; used for the
# guard's real-source check, nothing written to the repository) or, at C1, the Lean extractor.
#
# Definitions (5.2): snapshot S_k = declarations live after commit k. Window j covers commits
# (k_{j-1}, k_j], k_W = c*. A declaration is new in window j if live in S_{k_j} and not in S_{k_{j-1}}
# (S_{k_0} = empty), by qualified name and statement hash. Its entry commit is the start of the presence
# interval containing k_j. The last 20% of a window's new declarations by entry commit date (ties: commit
# position, then key) are its test set, the rest its wake set. Forward set F_r: live at the last forward
# commit and not live at c*. Cuts give the windows roughly equal numbers of new declarations: W = 2 exactly
# (minimise the difference); W >= 3 greedily (each cut closest to an equal share of what is left).
# Held out (5.4): in-window test sets, the benchmark tasks (read from verisoftbench.jsonl; a history
# declaration with a benchmark task's qualified name is replaced by the benchmark record and never enters a
# wake set, in any window) and the forward sets. Materializers redact every held-out proof in file prefixes.

LEAN_W_STAGE_C = 2


def load_history(path: str) -> dict:
    import gzip
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def _live_at(intervals, k) -> int:
    """Index of the interval containing commit k, or -1."""
    for i, (a, b) in enumerate(intervals):
        if a <= k and (b is None or b > k):
            return i
    return -1


def _counts_live(hist, keys_mask=None):
    """n[k] = number of declarations (optionally restricted to keys_mask) live after commit k."""
    n = len(hist["commits"])
    diff = [0] * (n + 1)
    for i, d in enumerate(hist["decls"]):
        if keys_mask is not None and not keys_mask[i]:
            continue
        for a, b in d[4]:
            diff[a] += 1
            diff[n if b is None else b] -= 1
    out, run = [], 0
    for k in range(n):
        run += diff[k]
        out.append(run)
    return out


def window_cuts(hist: dict, W: int = LEAN_W_STAGE_C) -> list:
    """End commit index of each window (the last is c*). ValueError if the history is too short."""
    nh = hist["n_history_commits"]
    last = nh - 1
    if nh < W:
        raise ValueError(f"{hist['repo']}: {nh} commits before c*, cannot cut {W} windows")
    in_final = [_live_at(d[4], last) >= 0 for d in hist["decls"]]
    n_final = sum(in_final)
    final_and_k = _counts_live(hist, in_final)             # live at k and at c*
    cuts, prev = [], -1
    for j in range(1, W):
        not_prev = [not (prev >= 0 and _live_at(d[4], prev) >= 0) for d in hist["decls"]]
        new_cnt = _counts_live(hist, not_prev)              # live at k and not at prev
        best = None
        for k in range(prev + 1, last - (W - j) + 1):
            a = new_cnt[k]
            rest = n_final - final_and_k[k]
            score = abs(a - rest) if W - j == 1 else abs(a - (a + rest) / (W - j + 1))
            if best is None or score < best[0]:
                best = (score, k)
        prev = best[1]
        cuts.append(prev)
    cuts.append(last)
    return cuts


def window_members(hist: dict, cuts: list) -> list:
    """For each window: [(decl_index, entry_commit_index)] of its new declarations."""
    out, prev = [], -1
    for k in cuts:
        mem = []
        for i, d in enumerate(hist["decls"]):
            ii = _live_at(d[4], k)
            if ii < 0 or (prev >= 0 and _live_at(d[4], prev) >= 0):
                continue
            mem.append((i, d[4][ii][0]))
        out.append(mem)
        prev = k
    return out


def forward_members(hist: dict) -> list:
    nh, n = hist["n_history_commits"], len(hist["commits"])
    if n == nh:
        return []
    out = []
    for i, d in enumerate(hist["decls"]):
        ii = _live_at(d[4], n - 1)
        if ii >= 0 and _live_at(d[4], nh - 1) < 0:
            out.append((i, d[4][ii][0]))
    return out


def split_window(hist: dict, members: list, test_fraction: float = TEST_FRACTION):
    """(wake, test): members ordered by entry commit date (ties: position, key); the last 20% are test."""
    commits, decls = hist["commits"], hist["decls"]
    order = sorted(members, key=lambda m: (commits[m[1]][1], m[1], decls[m[0]][0], decls[m[0]][1]))
    n_test = int(round(test_fraction * len(order)))
    if n_test == 0 and len(order) >= 2:
        n_test = 1
    return order[: len(order) - n_test], order[len(order) - n_test:]


def repo_windows(hist: dict, W: int = LEAN_W_STAGE_C, test_fraction: float = TEST_FRACTION,
                 exclude_names=()) -> dict:
    """Window summary of one repository (counts only). exclude_names: name hashes held out as benchmark tasks
    (removed before the wake/test split, counted in n_benchmark)."""
    cuts = window_cuts(hist, W)
    mem = window_members(hist, cuts)
    commits, decls = hist["commits"], hist["decls"]
    ex = set(exclude_names)
    out = {"repo": hist["repo"], "W": W, "windows": []}
    prev = -1
    for j, (k, m) in enumerate(zip(cuts, mem), 1):
        keep = [x for x in m if decls[x[0]][0] not in ex]
        wake, test = split_window(hist, keep, test_fraction)
        out["windows"].append({
            "window": j, "first_commit": commits[prev + 1][0], "end_commit": commits[k][0], "end_index": k,
            "start_date": commits[prev + 1][1], "end_date": commits[k][1], "n_commits": k - prev,
            "n_new": len(m), "n_benchmark": len(m) - len(keep), "n_wake": len(wake), "n_test": len(test),
            "n_new_testlike_path": sum(decls[i][3] for i, _ in m),
        })
        prev = k
    fw = forward_members(hist)
    out["forward"] = {"n_commits": hist["n_forward_commits"], "n_new": len(fw),
                      "head": commits[-1][0] if hist["n_forward_commits"] else None,
                      "head_date": commits[-1][1] if hist["n_forward_commits"] else None}
    return out


def merge_stages(window_summaries: list) -> list:
    """5.2: windows of all repositories in one sequence by window end date, ties by repository name."""
    st = [(f"{w['repo']}@w{x['window']}", x["end_date"], w["repo"], x["window"]) for w in window_summaries
          for x in w["windows"]]
    st.sort(key=lambda s: (s[1], s[2], s[3]))
    return [(sid, end) for sid, end, _, _ in st]


# ------------------------------------------------------------------ materializers (texts of obligations)
def _ns(repo):
    return "".join(c if c.isalnum() else "_" for c in repo)


class PlaceholderMaterializer:
    """Synthetic, deterministic Lean-like texts from the hashes only (no repository source), marked by the
    '«ph»' token. A fifth of the declarations (by name hash) get the generic proof ':= by simp', so that the
    guard's non-distinctive-proof path is exercised. Prefixes are synthetic, so there is nothing to redact."""
    kind = "placeholder"

    def texts(self, repo, commit, decl_rows, held_keys):
        out = {}
        for d in decl_rows:
            nh, sh = d[0], d[1]
            v = "xyzabn"[int(sh[0], 16) % 6]
            name = f"{_ns(repo)}.d_{nh}"
            stmt = (f"theorem {name} ({v} : T_{sh[:4]}) (h : P_{sh[4:8]} {v}) :\n"
                    f"    f_{sh[8:12]} ({v} + 1) = g_{sh[12:16]} {v} «ph»")
            if int(nh[0], 16) % 5 == 0:
                proof = ":= by simp"
            else:
                proof = (f":= by\n  induction {v} with\n  | zero => simp [f_{sh[8:12]}, g_{sh[12:16]}]\n"
                         f"  | succ n ih =>\n    rw [lem_{nh[:6]}, ← lem_{nh[6:12]} h]\n"
                         f"    simp_all [ih, aux_{sh[:6]}_{nh[12:]}]")
            deps = [f"lem_{nh[:6]}", f"lem_{nh[6:12]}", f"aux_{sh[:6]}_{nh[12:]}"]
            out[(nh, sh)] = {"decl": name, "statement": stmt, "proof": proof, "module": f"{_ns(repo)}.M_{d[2][:6]}",
                             "src_up_to_decl": f"import {_ns(repo)}.Prelude «ph»\n",
                             "scope": f"namespace {_ns(repo)}", "deps_project": deps, "deps_library": []}
        return out


def lean_obligation(repo, commit, snapshot_id, t, window, split, first_seen, entered_commit, entered,
                    n_deps_transitive=0, materializer="placeholder"):
    """An Appendix B obligation in the internal field names (schemas.to_benchmark_names gives the stored,
    benchmark-named record). entered_commit/entered/materializer are C1-prep provenance extras."""
    o = {
        "id": f"{repo}@{commit}:{t['decl']}", "repo": repo, "commit": commit, "snapshot_id": snapshot_id,
        "module": t["module"], "decl": t["decl"], "kind": "theorem", "statement": t["statement"],
        "src_up_to_decl": t["src_up_to_decl"], "scope": t["scope"],
        "deps_project": list(t["deps_project"]), "deps_library": list(t["deps_library"]),
        "n_deps_direct": len(t["deps_project"]) + len(t["deps_library"]), "n_deps_transitive": n_deps_transitive,
        "proof": t["proof"], "first_seen": first_seen, "window": window, "split": split,
        "closed_by_baseline_tactics": False,
        "entered_commit": entered_commit, "entered": entered, "materializer": materializer,
    }
    if t.get("proof_states"):
        o["proof_states"] = list(t["proof_states"])
    validate_obligation(o)
    return o


def benchmark_statement(task: dict) -> str:
    """thm_stmt without the proof: on some tasks thm_stmt carries the whole ground_truth_proof after the
    signature (counted in reports/C1prep.md); a statement shown in a view or stored must not (5.4 L1)."""
    st, gt = task["thm_stmt"], (task.get("ground_truth_proof") or "").strip()
    if gt:
        i = st.find(gt[: min(len(gt), 60)])
        if i > 0:
            return st[:i].rstrip()
    return st


def benchmark_obligation(task: dict, commit: str, snapshot_id: str, window: int) -> dict:
    """A VeriSoftBench task as an obligation (split 'benchmark'); its fields are the benchmark's own."""
    deps_p = [x["name"] for f in ("used_repo_defs", "repo_lemmas", "used_local_defs", "used_local_lemmas")
              for x in task.get(f) or []]
    deps_l = [x["name"] for f in ("used_lib_defs", "lib_lemmas") for x in task.get(f) or []]
    t = {"decl": task["thm_name"], "statement": benchmark_statement(task), "proof": task["ground_truth_proof"],
         "module": task["rel_path"][:-5].replace("/", ".") if task["rel_path"].endswith(".lean") else task["rel_path"],
         "src_up_to_decl": task.get("local_ctx") or "", "scope": "\n".join(task.get("imports") or []),
         "deps_project": deps_p, "deps_library": deps_l}
    o = lean_obligation(task["lean_root"], commit, snapshot_id, t, window, "benchmark", "", "", "",
                        n_deps_transitive=int(task.get("transitive_dep_count") or 0), materializer="verisoftbench")
    o["benchmark_task_id"] = task["id"]
    return o


@dataclass
class LeanStream:
    """Same interface as Stream (obligations, edges, stages, config, backend(), by_id(), test_sets(),
    data_version()) for Lean repositories. stages: [(snapshot_id, end date 'YYYY-MM-DDTHH:MM:SSZ')]."""
    repos: list
    obligations: list
    edges: list
    snapshots: dict                 # snapshot_id -> {repo, commit, window, end_date, end_index}
    stages: list
    windows: list                   # repo_windows summaries
    config: dict

    def backend(self, kind: str = "vsb"):
        if kind == "repl":
            from ..backends.lean_repl import LeanReplBackend
            return LeanReplBackend(self.snapshots, self.obligations)
        from ..backends.lean_vsb import LeanVSBBackend
        return LeanVSBBackend(self.snapshots, self.obligations)

    def by_id(self):
        return {o["id"]: o for o in self.obligations}

    def test_sets(self) -> dict:
        """Stage index j (1-based) -> in-window test obligation ids."""
        out = {}
        for j, (sid, _) in enumerate(self.stages, 1):
            out[j] = [o["id"] for o in self.obligations if o["snapshot_id"] == sid and o["split"] == "test"]
        return out

    def held_out(self, split: str) -> list:
        return [o["id"] for o in self.obligations if o["split"] == split]

    def data_version(self) -> str:
        h = hashlib.sha256()
        for o in self.obligations:
            h.update(json.dumps(o, sort_keys=True).encode())
        return h.hexdigest()[:16]


def name_hash(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:16]


def resolve_statement_collisions(obligations: list, stages: list):
    """The guard compares statements up to normalisation (holdout.lean_statement_key), across repositories. Two
    obligations with equal keys are one theorem for the guard, so a key must not be both trained and held out.
    The first appearance decides (stage order; forward and benchmark after every window): if any member is a
    benchmark task, or the first member is held out (test), every wake member is dropped; if the first member
    is a wake obligation, later test members are not new and become wake, and forward members are dropped.
    Returns (obligations, counts)."""
    from .holdout import lean_statement_key
    order = {sid: i for i, (sid, _) in enumerate(stages)}
    late = len(order) + 1

    def stage(o):
        return late if o["split"] in ("forward", "benchmark") else order.get(o["snapshot_id"], late)
    groups = {}
    for i, o in enumerate(obligations):
        groups.setdefault(lean_statement_key(o["statement"]), []).append(i)
    drop, counts = set(), {"wake_dropped": 0, "test_to_wake": 0, "forward_dropped": 0, "groups": 0}
    for idx in groups.values():
        splits = {obligations[i]["split"] for i in idx}
        if len(idx) < 2 or "wake" not in splits or splits == {"wake"}:
            continue
        counts["groups"] += 1
        first = min(idx, key=lambda i: (stage(obligations[i]), obligations[i]["split"] != "test"))
        if "benchmark" in splits or obligations[first]["split"] != "wake":
            for i in idx:
                if obligations[i]["split"] == "wake":
                    drop.add(i)
                    counts["wake_dropped"] += 1
        else:
            for i in idx:
                if obligations[i]["split"] == "test":
                    obligations[i] = dict(obligations[i], split="wake")
                    counts["test_to_wake"] += 1
                elif obligations[i]["split"] == "forward":
                    drop.add(i)
                    counts["forward_dropped"] += 1
    return [o for i, o in enumerate(obligations) if i not in drop], counts


def build_lean_stream(histories: list, W: int = LEAN_W_STAGE_C, materializer=None, benchmark_tasks=None,
                      test_fraction: float = TEST_FRACTION, include_forward: bool = True) -> LeanStream:
    """histories: vsb_history records. benchmark_tasks: VeriSoftBench task dicts (benchmark field names)."""
    mat = materializer or PlaceholderMaterializer()
    obligations, snapshots, summaries, edges = [], {}, [], []
    matched = {}
    dup_names = {}
    for hist in histories:
        repo = hist["repo"]
        commits, decls = hist["commits"], hist["decls"]
        tasks = [t for t in (benchmark_tasks or []) if t["lean_root"] == repo]
        bench_names = {name_hash(t["thm_name"]) for t in tasks}
        summ = repo_windows(hist, W, test_fraction, exclude_names=bench_names)
        summaries.append(summ)
        cuts = [w["end_index"] for w in summ["windows"]]
        mem = window_members(hist, cuts)
        groups = []
        for j, (k, m) in enumerate(zip(cuts, mem), 1):
            sid = f"{repo}@w{j}"
            snapshots[sid] = {"repo": repo, "commit": commits[k][0], "window": j, "end_date": commits[k][1],
                              "end_index": k}
            keep = [x for x in m if decls[x[0]][0] not in bench_names]
            wake, test = split_window(hist, keep, test_fraction)
            groups.append((sid, commits[k][0], j, [(x, "wake") for x in wake] + [(x, "test") for x in test]))
        if include_forward and hist["n_forward_commits"]:
            sid = f"{repo}@fwd"
            snapshots[sid] = {"repo": repo, "commit": commits[-1][0], "window": 0, "end_date": commits[-1][1],
                              "end_index": len(commits) - 1}
            groups.append((sid, commits[-1][0], 0, [(x, "forward") for x in forward_members(hist)]))
        held_keys = {(decls[i][0], decls[i][1]) for _, _, _, rows in groups for (i, _), s in rows if s != "wake"}
        held_keys |= {(decls[i][0], decls[i][1]) for i, d in enumerate(decls) if d[0] in bench_names}
        matched[repo] = {"tasks": len(tasks), "found_in_history": len({d[0] for d in decls} & bench_names)}
        dup_names[repo] = 0
        for sid, commit, j, rows in groups:
            texts = mat.texts(repo, commit, [decls[i] for (i, _), _ in rows], held_keys)
            seen = set()
            for (i, entry), split in rows:
                d = decls[i]
                t = texts.get((d[0], d[1]))
                if t is None:
                    continue            # the materializer could not locate it (real-source path only)
                if t["decl"] in seen:
                    # one qualified name with two statements in one snapshot (private declarations in different
                    # files, or a namespace the scanner got wrong): keep both, disambiguated by statement hash
                    t = dict(t, decl=f"{t['decl']}~{d[1][:8]}")
                    dup_names[repo] += 1
                seen.add(t["decl"])
                o = lean_obligation(repo, commit, sid, t, j, split, commits[d[4][0][0]][1][:10], commits[entry][0],
                                    commits[entry][1], materializer=mat.kind)
                obligations.append(o)
                for dep in o["deps_project"]:
                    edges.append({"snapshot_id": sid, "src": o["decl"], "dst": dep, "type": "proof_uses"})
        last_sid = f"{repo}@w{W}"
        names = [t["thm_name"] for t in tasks]
        for t in tasks:
            o = benchmark_obligation(t, snapshots[last_sid]["commit"], last_sid, W)
            if names.count(t["thm_name"]) > 1:
                # two benchmark tasks with one thm_name (different files): keep both, disambiguated by task id
                o = dict(o, decl=f"{o['decl']}~task{t['id']}", id=f"{o['id']}~task{t['id']}")
                dup_names[repo] += 1
            obligations.append(o)
    stages = merge_stages(summaries)
    obligations, collisions = resolve_statement_collisions(obligations, stages)
    kept_decls = {(o["repo"], o["decl"]) for o in obligations}
    edges = [e for e in edges if (e["snapshot_id"].split("@")[0], e["src"]) in kept_decls]
    for e in edges:
        validate_edge(e)
    cfg = {"W": W, "test_fraction": test_fraction, "materializer": mat.kind,
           "repos": [h["repo"] for h in histories], "benchmark": matched, "collisions": collisions,
           "duplicate_names": dup_names,
           "scan_version": sorted({h.get("scan_version", "") for h in histories})}
    return LeanStream([h["repo"] for h in histories], obligations, edges, snapshots, stages, summaries, cfg)


def write_lean_stream(s: LeanStream, out_dir: str) -> dict:
    """Obligations are written with the benchmark's field names (5.3; schemas.to_benchmark_names)."""
    from .schemas import to_benchmark_names
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "obligations.jsonl"), "w") as f:
        for o in s.obligations:
            f.write(json.dumps(to_benchmark_names(o), sort_keys=True) + "\n")
    with open(os.path.join(out_dir, "stream.json"), "w") as f:
        json.dump({"stages": s.stages, "snapshots": s.snapshots, "windows": s.windows, "config": s.config,
                   "data_version": s.data_version()}, f, indent=1, sort_keys=True)
    return {"obligations": len(s.obligations), "stages": len(s.stages), "data_version": s.data_version()}
