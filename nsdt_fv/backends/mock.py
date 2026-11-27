"""MockLeanBackend: a CheckerBackend that knows nothing about ToyEq.

It stands in for a Lean backend (lean_vsb.py / lean_repl.py, Stage C): snapshots look like
'<repo>@<commit>', candidates are Lean-style proof scripts, and verdicts come from a fixed table
of accepted proofs plus Lean's acceptance rules of R8 (no sorry/admit, no new axioms, unchanged
statement). The harness tests run the same runner and metrics code over this backend and over
ToyEqBackend; nothing outside backends/ changes (gate G0, last condition)."""
from __future__ import annotations

import re
import time

from .base import EvalResult, Verdict


class MockLeanBackend:
    def __init__(self, table: dict, statements: dict | None = None, evals: dict | None = None):
        """table: {(snapshot_id, obligation_id): set of accepted candidate texts}
        statements: {(snapshot_id, statement): set of accepted candidates} for check_statement
        evals: {(snapshot_id, term): value} for evaluate_term"""
        self.table = {k: set(v) for k, v in table.items()}
        self.statements = {k: set(v) for k, v in (statements or {}).items()}
        self.evals = dict(evals or {})
        self.calls = 0

    def snapshots(self):
        return sorted({s for s, _ in self.table} | {s for s, _ in self.statements})

    @staticmethod
    def _rules(candidate):
        if re.search(r"\b(sorry|admit)\b", candidate):
            return "sorry", "declaration uses 'sorry'"
        if re.search(r"^\s*axiom\b", candidate, re.M):
            return "axiom", "new axiom declared"
        return None, None

    def _verdict(self, accepted_set, candidate, t0):
        self.calls += 1
        cat, msg = self._rules(candidate)
        if cat:
            return Verdict(False, cat, [{"severity": "error", "text": msg, "line": 1, "column": 0,
                                         "goal_state": ""}], time.monotonic() - t0)
        if candidate.strip() in accepted_set:
            return Verdict(True, "ok", [], time.monotonic() - t0)
        return Verdict(False, "error", [{"severity": "error", "text": "unsolved goals", "line": 1,
                                         "column": 0, "goal_state": "⊢ ?"}], time.monotonic() - t0)

    def check(self, snapshot_id, obligation_id, candidate, timeout_s):
        t0 = time.monotonic()
        key = (snapshot_id, obligation_id)
        if key not in self.table:
            raise KeyError(f"unknown obligation {obligation_id} in {snapshot_id}")
        return self._verdict(self.table[key], candidate, t0)

    def check_statement(self, snapshot_id, context, statement, candidate, timeout_s):
        t0 = time.monotonic()
        return self._verdict(self.statements.get((snapshot_id, statement), set()), candidate, t0)

    def evaluate_term(self, snapshot_id, context, term, timeout_s):
        key = (snapshot_id, term)
        if key in self.evals:
            return EvalResult(True, self.evals[key], "ok", "", 0.0)
        return EvalResult(False, None, "error", "failed to compile definition, consider marking it as 'noncomputable'", 0.0)
