"""Symbolic channels A-E on ToyEq (plan section 4, Table 2, Appendix C.5) behind the A.3 Channel interface.

    class Channel(Protocol):
        id: str                                    # "A" .. "E"
        def wake_examples(self, stage) -> Iterable[Example]: ...
        def nrem_targets(self, stage, budget) -> list[Target]: ...
        def rem_seeds(self, stage, budget) -> list[Conjecture]: ...
        def annotate(self, obligation) -> str: ...

Every channel runs once per stage on the stage's snapshot (ChannelBase.run_stage) and reads only:
the snapshot's definitions and library statements, the window's NON-TEST obligations (statements and,
for the proof-mining channels A, D and E, their human proofs), and the store (statement keys,
proof_uses edges after L2). It never reads a test obligation's proof; test lemmas are not even
available to its reference-prover calls (ChannelBase.private_world drops them from the library).

Guard rails common to all channels (section 4, "Guard rails"):
  * every emitted item passes HoldoutGuard.check before it is offered to anything (R4); a refused
    item is counted (yield table column 'guard-refused'), logged with the rule, and dropped;
  * a per-stage volume cap per item kind (configs/S3.yaml, channels.<id>.cap_*), applied after the
    channel's relevance ranking;
  * machine proofs are never training targets, with the one exception that C.5 builds into channel
    D (its repair triples' target is the reference prover's repair; reports/CHANGES.md, S3).

Item records (one per candidate, logged to channels.jsonl.gz by the S3 driver):
  stage, channel, item (id), what (statement | nextstate | premise | unfold | repair | plan | priority),
  origin (sub-generator), statement, guard (pass | rule), accepted (checker verdict of the item's
  check, see each channel), trivial (statement closed by the baseline tactics), dup_store (statement
  key equals the key of a store statement), proof (machine proof checked at emission, if any),
  emitted (True once past the guard and the cap).
"""
from __future__ import annotations

import hashlib
import random

from ...backends.toyeq import World, statement_key
from ...data.holdout import LeakError
from ...data.stream import BASELINE_TACTICS
from ...data.toyeq_prover import RefProver


def sha12(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


class ChannelBase:
    id = "?"
    statement_kinds = ("statement", "priority")   # item kinds whose 'statement' is a claim (dup/trivial apply)

    def __init__(self, cfg: dict, stream, guard, backend, tok=None, max_len=192, seed=0):
        self.cfg = cfg or {}
        self.stream = stream
        self.guard = guard
        self.backend = backend                # the stream's CheckerBackend (the only acceptance authority)
        self.tok = tok
        self.max_len = max_len
        self.seed = seed
        self.full = stream.by_id()
        self._wake, self._seeds, self._prio = {}, {}, {}
        self.records = []                     # all item records of the last run_stage
        self._pw = {}

    # ------------------------------------------------------------------ A.3 interface
    def wake_examples(self, stage):
        return list(self._wake.get(stage, []))

    def nrem_targets(self, stage, budget):
        return list(self._prio.get(stage, []))[:budget]

    def rem_seeds(self, stage, budget):
        return list(self._seeds.get(stage, []))[:budget]

    def annotate(self, obligation) -> str:
        return ""                             # prompt annotations are not used in S3 (eval prompts = S1)

    # ------------------------------------------------------------------ helpers
    def private_world(self, sid) -> World:
        """The snapshot with test lemmas removed from the library (definitions unchanged). Reference
        prover searches run here, so no machine proof can cite a test lemma."""
        if sid not in self._pw:
            w = self.stream.worlds[sid]
            test = {self.full[i]["decl"] for i in self.guard.test_ids if self.full[i]["snapshot_id"] == sid}
            pw = World(sid)
            for fd in w.funcs.values():
                pw.add_def(fd)
            for name, st in w.lemmas:
                if name not in test:
                    pw.add_lemma(name, st)
            self._pw[sid] = pw
        return self._pw[sid]

    def priv_pos(self, sid, decl):
        """Number of test-free library lemmas that precede decl in snapshot sid (its 'avail' there)."""
        w = self.stream.worlds[sid]
        pw = self.private_world(sid)
        full_pos = w.lemma_pos[decl]
        return sum(1 for n, _ in pw.lemmas if w.lemma_pos[n] < full_pos)

    def window(self, sid):
        """Non-test obligations of the window whose snapshot is sid (statements and human proofs)."""
        return [o for o in self.stream.obligations if o["snapshot_id"] == sid and o["split"] == "wake"]

    def check_stmt(self, sid, stmt, proof):
        return self.backend.check_statement(sid, "", stmt, proof, 120)

    def trivial(self, sid, stmt):
        return any(self.check_stmt(sid, stmt, t).accepted for t in BASELINE_TACTICS)

    def prove(self, sid, stmt_text, hints=(), node_limit=None):
        """A checker-accepted machine proof of stmt_text in snapshot sid, or None. Tries the hints,
        the baseline tactics, then the reference prover on the private (test-free) world. Every
        returned proof was accepted by backend.check_statement (the checker is the only authority)."""
        for h in list(hints) + list(BASELINE_TACTICS):
            if h and self.check_stmt(sid, stmt_text, h).accepted:
                return h
        pw = self.private_world(sid)
        try:
            st = pw.statement(stmt_text)
        except Exception:
            return None
        p = RefProver(pw, len(pw.lemmas), node_limit=node_limit or int(self.cfg.get("prover_nodes", 2000))).prove(st)
        if p is not None and self.check_stmt(sid, stmt_text, p).accepted:
            return p
        return None

    def key(self, sid, stmt_text):
        return statement_key(self.stream.worlds[sid].statement(stmt_text))

    def fits(self, prompt, target):
        if self.tok is None:
            return True
        return len(self.tok.encode(prompt)) + len(self.tok.encode(target)) + 3 <= self.max_len

    def guard_ok(self, ex):
        try:
            self.guard.check(ex)
            return "pass"
        except LeakError as e:
            return e.rule

    # ------------------------------------------------------------------ the stage
    def run_stage(self, stage, sid, store_keys, store=None):
        """Run the channel on snapshot sid at stage 'stage'. store_keys: statement keys of the store
        (repo statements of every snapshot seen so far and the dream library). Fills the A.3 outputs
        for this stage and returns the item records."""
        self.records = []
        self.store_keys = store_keys
        self.store = store
        self.rng = random.Random(f"{self.id}|{self.seed}|{stage}")
        self._wake[stage], self._seeds[stage], self._prio[stage] = [], [], []
        self.produce(stage, sid)
        return self.records

    def produce(self, stage, sid):
        raise NotImplementedError

    def offer(self, stage, sid, what, origin, ex, rank=0.0, statement=None, accepted=None, proof=None,
              extra=None):
        """Record one candidate item and run the guard on it. Returns the record (rec['guard'] == 'pass'
        when it may be emitted). The cap is applied later by finish()."""
        g = self.guard_ok(ex)
        rec = {"stage": stage, "channel": self.id, "what": what, "origin": origin, "guard": g,
               "statement": statement, "accepted": accepted, "proof": proof, "rank": rank,
               "source_ids": list(ex.get("source_ids", [])), "emitted": False}
        if statement is not None and what in self.statement_kinds:
            try:
                k = self.key(sid, statement)
                rec["dup_store"] = k in self.store_keys
                rec["skey"] = sha12(k)
            except Exception:
                rec["dup_store"] = None
        if extra:
            rec.update(extra)
        rec["item"] = f"{self.id}|{stage}|{what}|{len(self.records)}"
        rec["_ex"] = ex
        self.records.append(rec)
        return rec

    def finish(self, stage, what, cap, sink):
        """Apply the volume cap to guard-passed, eligible records of kind 'what' in rank order (higher
        first; ties by insertion order) and hand the emitted ones to sink ('wake' | 'seed' | 'prio')."""
        cands = [r for r in self.records if r["what"] == what and r["guard"] == "pass" and r.get("eligible", True)]
        cands.sort(key=lambda r: -r["rank"])           # stable
        for r in cands[:cap]:
            r["emitted"] = True
            if sink == "wake":
                self._wake[stage].append(r["_ex"])
            elif sink == "seed":
                self._seeds[stage].append({"statement": r["statement"], "item": r["item"], "trivial": r.get("trivial"),
                                           "dup_store": r.get("dup_store"), "accepted": r["accepted"]})
            elif sink == "prio":
                self._prio[stage].append(r["target_key"])
        for r in cands[cap:]:
            r["capped"] = True


def public(rec):
    """The loggable part of an item record (drops the example payload)."""
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def make_channel(cid, cfg, stream, guard, backend, tok=None, max_len=192, seed=0):
    from .a_artifacts import ChannelA
    from .b_compose import ChannelB
    from .c_explore import ChannelC
    from .d_repair import ChannelD
    from .e_plans import ChannelE
    cls = {"A": ChannelA, "B": ChannelB, "C": ChannelC, "D": ChannelD, "E": ChannelE}[cid]
    return cls(cfg, stream, guard, backend, tok=tok, max_len=max_len, seed=seed)
