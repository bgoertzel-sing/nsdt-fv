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
