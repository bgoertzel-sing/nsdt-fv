"""LeanReplBackend: the fast path of Stage C (plan A.2). Elaborates an obligation's file prefix once, caches the
environment, and checks candidates against it (leanprover-community/repl, pickled environments).

C1-prep skeleton: no Lean is installed, so methods that need Lean raise NotImplementedError. Before the fast path
serves any reported number it must agree with LeanVSBBackend on verdict category for 500 sampled candidates
(A.2); agreement() below is that comparison and runs on recorded verdicts without Lean.
"""
from __future__ import annotations

from collections import Counter

from .base import EvalResult, Verdict
from .lean_vsb import NEEDS_LEAN, SnapshotSpec

AGREEMENT_SAMPLE = 500


class LeanReplBackend:
    def __init__(self, snapshots: dict, obligations: list, repl_cmd: tuple = ("lake", "exe", "repl")):
        self.specs = {sid: SnapshotSpec(sid, s["repo"], s["commit"], s["window"], s["end_date"])
                      for sid, s in snapshots.items()}
        self.obligations = {o["id"]: o for o in obligations}
        self.repl_cmd = tuple(repl_cmd)
        self.env_cache: dict = {}          # (snapshot_id, prefix sha) -> REPL env id

    def snapshots(self) -> list[str]:
        return sorted(self.specs)

    def prefix_key(self, obligation_id: str) -> tuple:
        import hashlib
        o = self.obligations[obligation_id]
        return (o["snapshot_id"], hashlib.sha256(o["src_up_to_decl"].encode()).hexdigest()[:16])

    def check(self, snapshot_id: str, obligation_id: str, candidate: str, timeout_s: int) -> Verdict:
        raise NotImplementedError(f"LeanReplBackend.check: {NEEDS_LEAN}")

    def check_statement(self, snapshot_id: str, context: str, statement: str, candidate: str,
                        timeout_s: int) -> Verdict:
        raise NotImplementedError(f"LeanReplBackend.check_statement: {NEEDS_LEAN}")

    def evaluate_term(self, snapshot_id: str, context: str, term: str, timeout_s: int) -> EvalResult:
        raise NotImplementedError(f"LeanReplBackend.evaluate_term: {NEEDS_LEAN}")


def agreement(reference: list, fast: list) -> dict:
    """reference, fast: verdict categories for the same candidates in the same order."""
    if len(reference) != len(fast):
        raise ValueError("verdict lists differ in length")
    same = sum(a == b for a, b in zip(reference, fast))
    return {"n": len(reference), "agree": same, "all_agree": same == len(reference),
            "confusion": dict(Counter(f"{a}->{b}" for a, b in zip(reference, fast) if a != b)),
            "enough": len(reference) >= AGREEMENT_SAMPLE}
