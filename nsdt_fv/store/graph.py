"""Toy store for Stage S (plan 3.3, A.3 Store): statements, dependency edges and verdict history.

Holds, for every snapshot seen so far, the statements of its non-test declarations (the repository
statements the DT arms may re-derive), the proof_uses edges after HoldoutGuard.filter_edges (L2),
the verdict history of each statement from sleep attempts, and the dream library (statements proved
in conjecture sleep). It holds no proof text: proofs are never read from the store for training
(3.3). Test statements are not offered as sleep targets (L1, L3).

Sampling weights of re-derivation sleep (A.4): 1 + 4 * fading + centrality + 2 * near_change, each
term in [0, 1]:
  fading       1 if the statement was proved at some earlier sleep attempt and its most recent
               attempt failed (A.3 Store.fading), else 0. An arm whose learner does not see kernel
               verdicts (D2) records none, so its fading term is 0.
  centrality   in-degree of the statement in the proof_uses graph of its project's latest seen
               snapshot (number of later lemmas whose proof cites it), divided by the maximum over
               the pool (0 when the maximum is 0).
  near_change  1 for statements of the current window, 0.5 for statements of the same project from
               an earlier window, 0 otherwise (the toy reading of "declarations touched in the window").
Dream-library entries have centrality 0.
"""
from __future__ import annotations

from collections import defaultdict


class ToyStore:
    def __init__(self, stream, guard):
        self.stream = stream
        self.guard = guard
        self.by_sid = defaultdict(list)
        for o in stream.obligations:
            if o["split"] != "test":
                self.by_sid[o["snapshot_id"]].append(o)
        self.entries = {}            # key -> entry dict
        self.order = []
        self.latest_sid = {}         # project -> latest snapshot seen
        self.indeg = {}              # (project, decl) -> proof_uses in-degree in the latest snapshot
        self.history = defaultdict(list)   # key -> [(stage, accepted)]
        self.library = []

    @staticmethod
    def project(sid):
        return sid.split("@")[0]

    def add_snapshot(self, sid, stage):
        p = self.project(sid)
        self.latest_sid[p] = sid
        edges = [e for e in self.stream.edges if e["snapshot_id"] == sid]
        deg = defaultdict(int)
        for e in self.guard.filter_edges(edges):
            if e["type"] == "proof_uses":
                deg[e["dst"]] += 1
        for k in [k for k in self.indeg if k[0] == p]:
            del self.indeg[k]
        for d, n in deg.items():
            self.indeg[(p, d)] = n
        added = 0
        for o in self.by_sid[sid]:
            key = o["id"]
            if key in self.entries:
                continue
            if self.guard.is_test_statement(o["statement"]):
                continue                       # an alpha-equivalent of a test statement is never a target
            self.entries[key] = {"key": key, "kind": "repo", "obligation_id": o["id"], "snapshot_id": sid,
                                 "project": p, "decl": o["decl"], "statement": o["statement"], "stage": stage,
                                 "trivial": bool(o["closed_by_baseline_tactics"])}
            self.order.append(key)
            added += 1
        return added

    def add_dream(self, key, statement, sid, stage, trivial):
        if key in self.entries:
            return False
        self.entries[key] = {"key": key, "kind": "dream", "obligation_id": None, "snapshot_id": sid,
                             "project": self.project(sid), "decl": None, "statement": statement, "stage": stage,
                             "trivial": bool(trivial)}
        self.order.append(key)
        self.library.append(key)
        return True

    def record_verdict(self, key, accepted, stage):
        self.history[key].append((stage, bool(accepted)))

    def fading(self, key):
        h = self.history.get(key)
        if not h or h[-1][1]:
            return 0.0
        return 1.0 if any(a for _, a in h[:-1]) else 0.0

    def statements(self, project=None, kind=None):
        return [self.entries[k] for k in self.order
                if (project is None or self.entries[k]["project"] == project)
                and (kind is None or self.entries[k]["kind"] == kind)]

    def weights(self, entries, current_sid):
        cur_p = self.project(current_sid)
        cents = [self.indeg.get((e["project"], e["decl"]), 0) if e["kind"] == "repo" else 0 for e in entries]
        mx = max(cents) if cents else 0
        out = []
        for e, c in zip(entries, cents):
            near = 1.0 if e["snapshot_id"] == current_sid else (0.5 if e["project"] == cur_p else 0.0)
            cen = c / mx if mx > 0 else 0.0
            out.append({"fading": self.fading(e["key"]), "centrality": cen, "near_change": near,
                        "w": 1.0 + 4.0 * self.fading(e["key"]) + cen + 2.0 * near})
        return out
