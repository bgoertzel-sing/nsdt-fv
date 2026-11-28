"""LeanVSBBackend: the reference checker of Stage C (plan A.2): wraps VeriSoftBench's core/lean_interface.py.

C1-prep skeleton. No Lean is installed (order t-2026-11-28-nsdt-fv-c1-prep), so every method that needs Lean
raises NotImplementedError. What is real already: construction from a LeanStream's snapshots and obligations,
snapshots(), the A.3 types, and the mapping of a candidate to the benchmark's check input
(ground_truth_proof-shaped text placed after thm_stmt, as verify_ground_truth.py does).

To finish at C0/C1:
  1. per snapshot, a built checkout (scripts/setup_repos.sh at the window's end commit) and its toolchain;
  2. check(): write src_up_to_decl + statement + candidate to a scratch file inside the snapshot, run the
     benchmark's lean_interface check with timeout_s, and map its result to Verdict categories (R8: 'sorry',
     'axiom' via #print axioms, 'changed_statement' when the elaborated type differs);
  3. check_statement() / evaluate_term(): the same with a free statement or '#eval term'.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base import CATEGORIES, EvalResult, Verdict   # noqa: F401  (re-exported for callers)

NEEDS_LEAN = "needs Lean (elan + the snapshot's toolchain); not installed until Stage C is cleared"


@dataclass
class SnapshotSpec:
    snapshot_id: str
    repo: str
    commit: str
    window: int
    end_date: str
    checkout_dir: str | None = None       # set at C0 when the snapshot is built
    toolchain: str | None = None


class LeanVSBBackend:
    """CheckerBackend (A.3) over VeriSoftBench's Lean interface."""

    def __init__(self, snapshots: dict, obligations: list, vsb_root: str | None = None):
        self.specs = {sid: SnapshotSpec(sid, s["repo"], s["commit"], s["window"], s["end_date"])
                      for sid, s in snapshots.items()}
        self.obligations = {o["id"]: o for o in obligations}
        self.vsb_root = vsb_root

    def snapshots(self) -> list[str]:
        return sorted(self.specs)

    def check_input(self, obligation_id: str, candidate: str) -> str:
        """The Lean text that check() will elaborate (no Lean needed to build it)."""
        o = self.obligations[obligation_id]
        body = candidate if candidate.lstrip().startswith(":=") else ":= " + candidate
        return f"{o['src_up_to_decl']}\n{o['statement']} {body}\n"

    def check(self, snapshot_id: str, obligation_id: str, candidate: str, timeout_s: int) -> Verdict:
        raise NotImplementedError(f"LeanVSBBackend.check: {NEEDS_LEAN}")

    def check_statement(self, snapshot_id: str, context: str, statement: str, candidate: str,
                        timeout_s: int) -> Verdict:
        raise NotImplementedError(f"LeanVSBBackend.check_statement: {NEEDS_LEAN}")

    def evaluate_term(self, snapshot_id: str, context: str, term: str, timeout_s: int) -> EvalResult:
        raise NotImplementedError(f"LeanVSBBackend.evaluate_term: {NEEDS_LEAN}")
